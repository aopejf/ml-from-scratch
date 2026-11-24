"""
Train the mini GPT on Shakespeare and watch its writing improve.

    python train.py

Progress prints as it goes, with a writing sample at several points, so you
can watch it in real time. It saves:
    samples.md    the writing samples, from gibberish to fake Shakespeare
    result.png    the loss curve and a picture of what attention looks at
    model.pt      the trained model, so generate.py can write more without retraining
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gpt import MiniGPT

torch.manual_seed(0)
here = Path(__file__).parent

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BATCH = 32  # how many snippets of text the model studies per step
CONTEXT = 128  # how many characters back the model can see
LEARNING_RATE = 1e-3
KEEP_SAMPLES_AT = {0, 100, 300, 1000, 2000, STEPS}  # the ones saved to samples.md
CHECK_EVERY = 100  # full check: loss on unseen text and a long sample
QUICK_LOOK_EVERY = 10  # quick look: a short sample

# ---------------------------------------------------------------------------
# 1. Tokenize: turn every character into a number.
# ---------------------------------------------------------------------------
text = (here / "data" / "shakespeare.txt").read_text(encoding="utf-8")
characters = sorted(set(text))
to_number = {ch: i for i, ch in enumerate(characters)}
to_text = lambda numbers: "".join(characters[i] for i in numbers)

data = torch.tensor([to_number[ch] for ch in text])
split = int(0.9 * len(data))
train_data, unseen_data = data[:split], data[split:]  # last 10% is held back, as always

print(f"Text: {len(text):,} characters, {len(characters)} different ones")
print(f"Example: {text[:18]!r} becomes {data[:18].tolist()}")


def random_batch(source):
    """Pick random snippets. The target is the same snippet shifted one character on."""
    starts = torch.randint(len(source) - CONTEXT - 1, (BATCH,))
    inputs = torch.stack([source[s : s + CONTEXT] for s in starts])
    targets = torch.stack([source[s + 1 : s + CONTEXT + 1] for s in starts])
    return inputs, targets


@torch.no_grad()
def average_loss(model, source, batches=20):
    model.eval()
    total = sum(model(*random_batch(source))[1].item() for _ in range(batches))
    model.train()
    return total / batches


# ---------------------------------------------------------------------------
# 2. Build the model.
# ---------------------------------------------------------------------------
model = MiniGPT(vocab_size=len(characters), context=CONTEXT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # Adam, as built in project 1
n_weights = sum(p.numel() for p in model.parameters())
print(f"Model: {n_weights:,} adjustable numbers\n")

# ---------------------------------------------------------------------------
# 3. Train: guess, score, work out the nudge, nudge. The same loop as project 0.
# ---------------------------------------------------------------------------
history = []  # (step, loss on training text, loss on unseen text)
samples = []
start = time.time()

for step in range(STEPS + 1):
    if step % CHECK_EVERY == 0 or step == STEPS:
        train_loss, unseen_loss = average_loss(model, train_data), average_loss(model, unseen_data)
        history.append((step, train_loss, unseen_loss))
        print(f"step {step:5d}   loss {train_loss:.3f}   on unseen text {unseen_loss:.3f}   ({time.time() - start:.0f}s)", flush=True)

    if step % CHECK_EVERY == 0 or step == STEPS:  # write a sample at every check
        prompt = torch.tensor([[to_number["\n"]]])
        written = to_text(model.generate(prompt, 300)[0].tolist())[1:]
        if step in KEEP_SAMPLES_AT:
            samples.append((step, history[-1][2], written))
        print(f"\n----- what it writes at step {step} -----\n{written}\n-----\n", flush=True)

    elif step % QUICK_LOOK_EVERY == 0:
        # A quick look between full checks: a short sample and the latest training loss.
        prompt = torch.tensor([[to_number["\n"]]])
        written = to_text(model.generate(prompt, 150)[0].tolist())[1:]
        print(f"step {step:5d}   loss {loss.item():.3f}   ({time.time() - start:.0f}s)\n{written}\n", flush=True)

    if step == STEPS:
        break
    inputs, targets = random_batch(train_data)
    _, loss = model(inputs, targets)  # guess and score
    optimizer.zero_grad()
    loss.backward()  # project 2's engine, industrial strength
    optimizer.step()  # project 1's Adam

print(f"Training took {(time.time() - start) / 60:.1f} minutes")

# ---------------------------------------------------------------------------
# 4. Save everything.
# ---------------------------------------------------------------------------
torch.save({"weights": model.state_dict(), "characters": characters, "context": CONTEXT}, here / "model.pt")

with open(here / "samples.md", "w", encoding="utf-8") as f:
    f.write("# What the model writes as it trains\n\n")
    f.write("Every sample starts from nothing and is 300 characters long.\n")
    for step, unseen_loss, written in samples:
        f.write(f"\n## Step {step} (loss on unseen text: {unseen_loss:.2f})\n\n```\n{written}\n```\n")

# The picture: loss curve, and what one attention head looks at.
fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))
steps_done, train_curve, unseen_curve = zip(*history)
left.plot(steps_done, train_curve, color="#2a6fdb", linewidth=2, label="text it trained on")
left.plot(steps_done, unseen_curve, color="#c62828", linewidth=2, label="text it never saw")
left.set_title("The loss falls as the model learns")
left.set_xlabel("training step")
left.set_ylabel("loss (how wrong), lower is better")
left.legend()

phrase = "First Citizen:\nWe are accounted poor"
model.eval()
with torch.no_grad():
    model(torch.tensor([[to_number[ch] for ch in phrase]]))
weights = model.blocks[-1].attention.last_weights[0].mean(dim=0)  # last layer, averaged over heads
labels = [ch if ch not in " \n" else ("␣" if ch == " " else "↵") for ch in phrase]
right.imshow(weights, cmap="Blues")
right.set_xticks(range(len(labels)), labels, fontsize=8)
right.set_yticks(range(len(labels)), labels, fontsize=8)
right.set_title("Attention: each row is a character, darker = looked at more")
right.set_xlabel("...looks back at this character")
right.set_ylabel("when guessing what follows this character...")

fig.tight_layout()
fig.savefig(here / "result.png", dpi=120)
print("Saved: samples.md, result.png, model.pt")
