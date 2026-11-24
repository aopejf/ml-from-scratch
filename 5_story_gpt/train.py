"""
Step 2 of the story GPT: train it.

    python train.py 7            train for 7 hours, then stop and write the report
    python train.py 7 --resume   carry on from the last checkpoint
    python train.py 0 --resume   no more training, just write the report

Training runs to a time limit, not a step count, so it finishes when you
expect it to. Every 10 minutes it saves a checkpoint, so nothing is lost if
the laptop sleeps or the run is stopped.

It prints the loss every 10 steps with a short sample of its writing, and
checks itself on stories it has never seen every 250 steps.

It saves:
    checkpoint.pt   everything needed to carry on
    model.pt        the finished model
    samples.md      the same prompt, continued at different points in training
    result.png      the loss curve
"""

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import CONFIG, new_model, write
from tokenizer import Tokenizer

torch.manual_seed(0)
here = Path(__file__).parent

if sys.platform == "win32":
    # Ask Windows not to go to sleep while this is running. It ends when the script ends.
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 7
RESUME = "--resume" in sys.argv
BATCH = 32
TOP_LEARNING_RATE = 1e-3
WARMUP_STEPS = 200
CHECK_EVERY = 250
SAVE_EVERY_MINUTES = 10
PROMPT = "Once upon a time, there was a little dinosaur who"
KEEP_SAMPLES_AT_MINUTES = [0, 5, 15, 30, 60, 120, 240, 360]

tokenizer = Tokenizer.load(here / "tokenizer.json")
train_data = np.memmap(here / "data" / "train.bin", dtype=np.uint16, mode="r")
unseen_data = np.memmap(here / "data" / "valid.bin", dtype=np.uint16, mode="r")
print(f"Stories: {len(train_data):,} word pieces to train on, {len(unseen_data):,} held back")

model = new_model(tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=TOP_LEARNING_RATE, weight_decay=0.01)
context = CONFIG["context"]
step, seconds_done = 0, 0.0
history = {"step": [], "minutes": [], "train": [], "unseen": []}
kept_samples = []

if RESUME:
    saved = torch.load(here / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    step, seconds_done = saved["step"], saved["seconds_done"]
    history, kept_samples = saved["history"], saved["kept_samples"]
    print(f"Carrying on from step {step} ({seconds_done / 3600:.2f} hours already done)")

print(f"Model: {sum(p.numel() for p in model.parameters()):,} adjustable numbers")
print(f"Time limit: {HOURS} hours in total\n", flush=True)
budget = HOURS * 3600


def get_batch(data):
    starts = np.random.randint(0, len(data) - context - 1, BATCH)
    x = np.stack([data[s : s + context] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + context + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def loss_on(data, batches=12):
    model.eval()
    total = sum(model(*get_batch(data))[1].item() for _ in range(batches))
    model.train()
    return total / batches


def save_checkpoint(seconds):
    temporary = here / "checkpoint.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": CONFIG,
            "step": step,
            "seconds_done": seconds,
            "history": history,
            "kept_samples": kept_samples,
        },
        temporary,
    )
    temporary.replace(here / "checkpoint.pt")  # swap in one move, so a crash cannot leave half a file


np.random.seed(step)
recent = []
session_start = time.time()
last_save = session_start
while True:
    elapsed = seconds_done + time.time() - session_start
    if elapsed >= budget:
        break

    # Learning rate: ramp up briefly, then glide down to a tenth as time runs out.
    progress = elapsed / budget
    rate = TOP_LEARNING_RATE * min(1, (step + 1) / WARMUP_STEPS) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    for group in optimizer.param_groups:
        group["lr"] = rate

    minutes = elapsed / 60
    if len(kept_samples) < len(KEEP_SAMPLES_AT_MINUTES) and minutes >= KEEP_SAMPLES_AT_MINUTES[len(kept_samples)]:
        torch.manual_seed(1)
        kept_samples.append((step, round(minutes), write(model, tokenizer, PROMPT, 120)[0]))
        torch.manual_seed(step)

    if step % CHECK_EVERY == 0:
        train_loss, unseen_loss = loss_on(train_data), loss_on(unseen_data)
        for key, value in zip(history, (step, minutes, train_loss, unseen_loss)):
            history[key].append(value)
        print(f"\n=== step {step}   {minutes:.0f} min of {budget / 60:.0f}   loss on training stories {train_loss:.3f}   on unseen stories {unseen_loss:.3f} ===")
        print(write(model, tokenizer, PROMPT, 100)[0] + "\n", flush=True)

    x, y = get_batch(train_data)
    _, loss = model(x, y)  # guess and score
    optimizer.zero_grad()
    loss.backward()  # work out the nudge
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()  # nudge
    step += 1
    recent.append(loss.item())

    if step % 10 == 0:
        sample = write(model, tokenizer, PROMPT, 25)[1].replace("\n", " ")
        print(f"step {step:6d}  {minutes:4.0f} min  loss {np.mean(recent[-10:]):.3f}  |{sample}", flush=True)

    if time.time() - last_save > SAVE_EVERY_MINUTES * 60:
        save_checkpoint(seconds_done + time.time() - session_start)
        last_save = time.time()

seconds_done += time.time() - session_start
save_checkpoint(seconds_done)

# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
train_loss, unseen_loss = loss_on(train_data, 50), loss_on(unseen_data, 50)
print(f"\nFinished: {step:,} steps in {seconds_done / 3600:.2f} hours")
print(f"Loss on training stories {train_loss:.3f}, on unseen stories {unseen_loss:.3f}")
torch.save({"model": model.state_dict(), "config": CONFIG, "step": step, "hours": seconds_done / 3600, "train_loss": train_loss, "unseen_loss": unseen_loss}, here / "model.pt")

torch.manual_seed(1)
final_story = write(model, tokenizer, PROMPT, 180)[0]
with open(here / "samples.md", "w", encoding="utf-8") as f:
    f.write("# The same prompt, as training goes on\n\n")
    f.write(f"Prompt: **{PROMPT}**\n\n")
    for at_step, at_minute, text in kept_samples:
        f.write(f"## After {at_minute} minutes (step {at_step:,})\n\n{text}\n\n")
    f.write(f"## Finished, {seconds_done / 3600:.1f} hours (step {step:,})\n\n{final_story}\n")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history["minutes"], history["train"], label="stories it trained on", color="#2a6fdb")
ax.plot(history["minutes"], history["unseen"], label="stories it never saw", color="#e0662b")
ax.set_xlabel("minutes of training")
ax.set_ylabel("loss (lower is better)")
ax.set_title("Story GPT: loss during training")
ax.legend()
fig.tight_layout()
fig.savefig(here / "result.png", dpi=110)
print("\nSaved: model.pt, samples.md, result.png")
print("\n" + final_story)
