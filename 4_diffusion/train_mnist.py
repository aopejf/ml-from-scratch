"""
Train the upgraded diffusion model on MNIST: 60,000 handwritten digits, 28x28 pixels.

    python train_mnist.py            full run
    python train_mnist.py 200        short run with 200 steps (for testing)

Every 1,000 steps it pauses to generate one of each digit, saves that as
preview.png, and saves the model, so you can watch it improve and nothing
is lost if the run is stopped.

It saves:
    result_mnist.png      ten of each digit, made on request
    progress_mnist.png    the same ten digits at each point in training
    generating_mnist.gif  100 digits appearing out of static
    model_mnist.pt        the trained network
"""

import copy
import gzip
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter

from conditional import NO_LABEL, ConditionalNoisePredictor, generate
from diffusion import N_STEPS, add_noise

torch.manual_seed(0)
here = Path(__file__).parent

if sys.platform == "win32":
    # Ask Windows not to go to sleep while this is running. It ends when the script ends.
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
BATCH = 64
LEARNING_RATE = 1e-3
HIDE_LABEL_CHANCE = 0.1
PREVIEW_EVERY = 1000 if STEPS >= 2000 else max(STEPS // 2, 1)

# ---------------------------------------------------------------------------
# 1. Load MNIST straight from its original files.
# ---------------------------------------------------------------------------
with gzip.open(here / "data" / "train-images-idx3-ubyte.gz") as f:
    pixels = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
with gzip.open(here / "data" / "train-labels-idx1-ubyte.gz") as f:
    digit_labels = np.frombuffer(f.read(), np.uint8, offset=8)

images = torch.tensor(pixels / 255.0 * 2 - 1, dtype=torch.float32)  # -1 black, 1 white
labels = torch.tensor(digit_labels.astype(np.int64))
train_images, train_labels = images[:59000], labels[:59000]
unseen_images = images[59000:]  # 1,000 held back for the copying check
print(f"Digits: {len(train_images):,} to train on, {len(unseen_images):,} held back, each 28x28 pixels")

# ---------------------------------------------------------------------------
# 2. Train.
# ---------------------------------------------------------------------------
model = ConditionalNoisePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# A second copy of the model that changes slowly: a running average of the
# first one's weights. It gives cleaner images than the jittery latest weights.
smooth_model = copy.deepcopy(model)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} adjustable numbers\n")


def grid_of(batch, per_row):
    batch = batch.squeeze(1).numpy()
    size = batch.shape[-1]
    rows = len(batch) // per_row
    grid = np.full((rows * (size + 2) - 2, per_row * (size + 2) - 2), -1.0)
    for i, img in enumerate(batch[: rows * per_row]):
        r, c = divmod(i, per_row)
        grid[r * (size + 2) : r * (size + 2) + size, c * (size + 2) : c * (size + 2) + size] = img
    return grid


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)


losses, previews = [], []
start = time.time()
for step in range(1, STEPS + 1):
    pick = torch.randint(len(train_images), (BATCH,))
    batch, batch_labels = train_images[pick], train_labels[pick].clone()
    batch_labels[torch.rand(BATCH) < HIDE_LABEL_CHANCE] = NO_LABEL  # sometimes hide the label
    t = torch.randint(N_STEPS, (BATCH,))
    static = torch.randn_like(batch)

    guessed_static = model(add_noise(batch, t, static), t, batch_labels)  # guess
    loss = F.mse_loss(guessed_static, static)  # score
    optimizer.zero_grad()
    loss.backward()  # work out the nudge
    optimizer.step()  # nudge
    losses.append(loss.item())

    with torch.no_grad():  # update the slow-moving copy
        blend = min(0.999, step / (step + 10))
        for slow, fast in zip(smooth_model.parameters(), model.parameters()):
            slow.mul_(blend).add_(fast, alpha=1 - blend)

    if step % 100 == 0 or step == 1:
        print(f"step {step:5d}   loss {np.mean(losses[-100:]):.4f}   ({time.time() - start:.0f}s)", flush=True)

    if step % PREVIEW_EVERY == 0 or step == STEPS:
        torch.manual_seed(1)  # same starting static each time, so previews are comparable
        preview = generate(smooth_model, list(range(10)))
        torch.manual_seed(step)
        previews.append((step, preview))
        fig, ax = plt.subplots(figsize=(10, 1.5))
        show(ax, grid_of(preview, 10), f"Asked for 0 to 9, after {step} training steps")
        fig.tight_layout()
        fig.savefig(here / "preview.png", dpi=110)
        plt.close(fig)
        torch.save(smooth_model.state_dict(), here / "model_mnist.pt")
        print(f"           saved preview.png and model_mnist.pt   ({time.time() - start:.0f}s)", flush=True)

print(f"Training took {(time.time() - start) / 60:.1f} minutes")

# ---------------------------------------------------------------------------
# 3. Generate ten of each digit, on request.
# ---------------------------------------------------------------------------
print("\nGenerating 100 images, ten of each digit...", flush=True)
requested = [digit for _ in range(10) for digit in range(10)]  # each row is 0,1,...,9
generated, journey = generate(smooth_model, requested, keep_every=25)

# ---------------------------------------------------------------------------
# 4. Copying check, as in the small version.
# ---------------------------------------------------------------------------
def distance_to_nearest(candidates, reference):
    return torch.cdist(candidates.flatten(1), reference.flatten(1)).min(dim=1)


generated_gap, nearest_index = distance_to_nearest(generated, train_images)
unseen_gap, _ = distance_to_nearest(unseen_images, train_images)
print("\nCopying check: distance from each image to its closest training image")
print(f"  generated digits:        typical {generated_gap.median():.2f}   smallest {generated_gap.min():.2f}")
print(f"  real held-back digits:   typical {unseen_gap.median():.2f}   smallest {unseen_gap.min():.2f}")

# ---------------------------------------------------------------------------
# 5. Pictures.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
real_by_digit = torch.stack([train_images[train_labels == d][row] for row in range(10) for d in range(10)])
show(axes[0], grid_of(real_by_digit, 10), "Real handwriting (columns are 0 to 9)")
show(axes[1], grid_of(generated, 10), "Made by the model on request (columns are 0 to 9)")
smooth = np.convolve(losses, np.ones(100) / 100, mode="valid") if len(losses) > 100 else losses
axes[2].plot(smooth, color="#2a6fdb")
axes[2].set_title("Loss: how badly it guesses the static")
axes[2].set_xlabel("training step")
axes[2].set_ylim(0, 0.15)
fig.tight_layout()
fig.savefig(here / "result_mnist.png", dpi=110)
plt.close(fig)

fig, axes = plt.subplots(len(previews), 1, figsize=(10, 1.25 * len(previews)))
for ax, (step, preview) in zip(np.atleast_1d(axes), previews):
    show(ax, grid_of(preview, 10))
    ax.set_ylabel(f"step {step}", rotation=0, labelpad=32, va="center")
fig.suptitle("The same ten requests (0 to 9), as training goes on", fontsize=11)
fig.tight_layout()
fig.savefig(here / "progress_mnist.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6.3))
frames = journey + [journey[-1]] * 12


def draw(i):
    ax.clear()
    left, batch = frames[i]
    show(ax, grid_of(batch.clamp(-1, 1), 10), f"{left} denoising steps left")


FuncAnimation(fig, draw, frames=len(frames)).save(here / "generating_mnist.gif", writer=PillowWriter(fps=10), dpi=80)
print("\nSaved: result_mnist.png, progress_mnist.png, generating_mnist.gif, model_mnist.pt")
