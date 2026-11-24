"""
Train the diffusion model on handwritten digits, then generate new ones.

    python train.py

The images are 1,797 real handwritten digits, each 8 by 8 pixels, collected
from 43 people (the "Optical Recognition of Handwritten Digits" set that
ships with scikit-learn). They are tiny, which is what lets this train on a
laptop in a few minutes.

It saves:
    result.png      real digits next to generated ones, and the loss curve
    process.png     the forward process (image to static) and the reverse (static to image)
    generating.gif  64 images appearing out of static
    model.pt        the trained network
"""

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
from sklearn.datasets import load_digits

from diffusion import N_STEPS, NoisePredictor, add_noise, generate

torch.manual_seed(0)
here = Path(__file__).parent

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
BATCH = 128
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# 1. Load the real digits. Pixel values are rescaled to run from -1 (black) to 1 (white).
# ---------------------------------------------------------------------------
pixels = torch.tensor(load_digits().images, dtype=torch.float32)  # 1797 x 8 x 8, values 0..16
images = (pixels / 16.0 * 2 - 1).unsqueeze(1)
shuffled = images[torch.randperm(len(images))]
train_images, unseen_images = shuffled[:1597], shuffled[1597:]  # 200 held back
print(f"Digits: {len(train_images)} to train on, {len(unseen_images)} held back, each {images.shape[-1]}x{images.shape[-1]} pixels")

# ---------------------------------------------------------------------------
# 2. Train. Each step:
#      pick some real digits, pick a random noise level for each, add that much static,
#      ask the network what the static was, and score how close it got.
# ---------------------------------------------------------------------------
model = NoisePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} adjustable numbers\n")

losses = []
start = time.time()
for step in range(1, STEPS + 1):
    batch = train_images[torch.randint(len(train_images), (BATCH,))]
    t = torch.randint(N_STEPS, (BATCH,))
    static = torch.randn_like(batch)
    guessed_static = model(add_noise(batch, t, static), t)  # guess
    loss = F.mse_loss(guessed_static, static)  # score: how far off was the guess?
    optimizer.zero_grad()
    loss.backward()  # work out the nudge
    optimizer.step()  # nudge
    losses.append(loss.item())
    if step % 500 == 0 or step == 1:
        recent = np.mean(losses[-200:])
        print(f"step {step:5d}   loss {recent:.4f}   ({time.time() - start:.0f}s)", flush=True)

print(f"Training took {(time.time() - start) / 60:.1f} minutes")
torch.save(model.state_dict(), here / "model.pt")

# ---------------------------------------------------------------------------
# 3. Generate 64 brand-new digits from pure static.
# ---------------------------------------------------------------------------
print("\nGenerating 64 images (1,000 denoising steps each)...", flush=True)
generated, journey = generate(model, 64, images.shape[-1], keep_every=20)
generated = generated.clamp(-1, 1)

# ---------------------------------------------------------------------------
# 4. Is it inventing, or copying its training images?
#    For each generated digit, find the most similar training digit and measure
#    how different they are. Do the same for the 200 held-back REAL digits.
#    If generated digits sit much closer to the training set than real unseen
#    digits do, the model is copying.
# ---------------------------------------------------------------------------
def distance_to_nearest(candidates, reference):
    gaps = torch.cdist(candidates.flatten(1), reference.flatten(1))
    return gaps.min(dim=1)


generated_gap, nearest_index = distance_to_nearest(generated, train_images)
unseen_gap, _ = distance_to_nearest(unseen_images, train_images)
print("\nCopying check: distance from each image to its closest training image")
print(f"  generated digits:        typical {generated_gap.median():.2f}   smallest {generated_gap.min():.2f}")
print(f"  real held-back digits:   typical {unseen_gap.median():.2f}   smallest {unseen_gap.min():.2f}")


# ---------------------------------------------------------------------------
# 5. Pictures.
# ---------------------------------------------------------------------------
def as_grid(batch, per_row=8):
    """Arrange a batch of images into one big image, with thin gaps."""
    batch = batch.squeeze(1).numpy()
    size = batch.shape[-1]
    rows = len(batch) // per_row
    grid = np.full((rows * (size + 1) - 1, per_row * (size + 1) - 1), -1.0)
    for i, img in enumerate(batch[: rows * per_row]):
        r, c = divmod(i, per_row)
        grid[r * (size + 1) : r * (size + 1) + size, c * (size + 1) : c * (size + 1) + size] = img
    return grid


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


# result.png: real vs generated, the copying check, and the loss.
fig, axes = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw={"width_ratios": [1, 1, 0.55, 1]})
show(axes[0], as_grid(train_images[:64]), "64 real handwritten digits")
show(axes[1], as_grid(generated), "64 digits the model made up")
pairs = torch.cat([torch.stack([generated[i], train_images[nearest_index[i]]]) for i in range(6)])
show(axes[2], as_grid(pairs, per_row=2), "made up  |  closest real one")
smooth = np.convolve(losses, np.ones(100) / 100, mode="valid")
axes[3].plot(smooth, color="#2a6fdb")
axes[3].set_title("Loss: how badly it guesses the static")
axes[3].set_xlabel("training step")
axes[3].set_ylim(0, min(0.5, smooth.max()))
fig.tight_layout()
fig.savefig(here / "result.png", dpi=120)

# process.png: forward on top, reverse underneath.
moments = [0, 50, 150, 300, 500, 750, 999]
fig, axes = plt.subplots(2, len(moments), figsize=(2.1 * len(moments), 4.8))
one_digit = train_images[3:4]
fixed_static = torch.randn_like(one_digit)
for ax, t in zip(axes[0], moments):
    noisy = one_digit if t == 0 else add_noise(one_digit, torch.tensor([t]), fixed_static)
    show(ax, noisy[0, 0], f"step {t}")
by_step = dict(journey)
for ax, t in zip(axes[1], [1000, 760, 500, 300, 160, 60, 0]):
    show(ax, by_step[t][0, 0].clamp(-1, 1), f"{t} steps left")
axes[0, 0].set_ylabel("FORWARD\nimage to static", fontsize=10)
axes[1, 0].set_ylabel("REVERSE\nstatic to image", fontsize=10)
fig.suptitle("Top: a real digit drowned in static (no learning). Bottom: the model pulling a new digit out of static.", fontsize=11)
fig.tight_layout()
fig.savefig(here / "process.png", dpi=120)

# generating.gif: all 64 appearing at once.
fig, ax = plt.subplots(figsize=(5.5, 5.8))
frames = journey + [journey[-1]] * 12


def draw(i):
    ax.clear()
    left, batch = frames[i]
    show(ax, as_grid(batch.clamp(-1, 1)), f"{left} denoising steps left")


FuncAnimation(fig, draw, frames=len(frames)).save(here / "generating.gif", writer=PillowWriter(fps=10), dpi=80)
print("\nSaved: result.png, process.png, generating.gif, model.pt")
