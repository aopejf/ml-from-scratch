"""
Project 4: a small diffusion model.

The idea in two halves:

  Forward (no learning). Take a real image and add a little random static,
  1,000 times, until nothing is left but static. This is a fixed recipe.

  Reverse (the learning). Train a network to look at a noisy image and say
  "this part is the static". Subtract a little of that, repeat 1,000 times,
  and pure static turns into an image that never existed.

The network is never asked to draw. It is only ever asked to spot noise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STEPS = 1000


# ---------------------------------------------------------------------------
# The noise schedule: how much static is added at each of the 1,000 steps.
# A little at first (0.0001), more towards the end (0.02).
# ---------------------------------------------------------------------------
noise_added = torch.linspace(1e-4, 0.02, N_STEPS)
image_kept = 1.0 - noise_added
# How much of the ORIGINAL image survives after t steps (all the "kept" fractions multiplied).
original_left = torch.cumprod(image_kept, dim=0)


def add_noise(images, t, static):
    """
    Jump straight to step t of the forward process in one go.
    The result is a blend: some of the original image, some static.
    Early t: mostly image. Late t: mostly static.
    """
    keep = original_left[t].sqrt().view(-1, 1, 1, 1)
    noise = (1 - original_left[t]).sqrt().view(-1, 1, 1, 1)
    return keep * images + noise * static


# ---------------------------------------------------------------------------
# The network. Input: a noisy image and which step it is at.
# Output: its guess at the static in that image.
# ---------------------------------------------------------------------------
def step_embedding(t, width):
    """Turn the step number into a list of numbers the network can read (waves of different speeds)."""
    half = width // 2
    speeds = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t.float().view(-1, 1) * speeds.view(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class Block(nn.Module):
    """Two rounds of "look at each pixel's neighbourhood", with the step number mixed in."""

    def __init__(self, channels_in, channels_out, step_width):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.step = nn.Linear(step_width, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.shortcut = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x, step_info):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.step(step_info).view(-1, h.size(1), 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class NoisePredictor(nn.Module):
    """
    A small U-shaped network: look at the image in detail, shrink it to see
    the overall shape, then grow it back, reusing the detail from the way down.
    Stable Diffusion's core network has the same U shape, much bigger.
    """

    def __init__(self, width=64, step_width=128):
        super().__init__()
        self.step_width = step_width
        self.read_step = nn.Sequential(nn.Linear(step_width, step_width), nn.SiLU(), nn.Linear(step_width, step_width))
        self.first = nn.Conv2d(1, width, 3, padding=1)
        self.down = Block(width, width, step_width)
        self.middle1 = Block(width, 2 * width, step_width)
        self.middle2 = Block(2 * width, 2 * width, step_width)
        self.up = Block(3 * width, width, step_width)
        self.last = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, noisy_images, t):
        step_info = self.read_step(step_embedding(t, self.step_width))
        detail = self.down(self.first(noisy_images), step_info)  # full size
        small = F.avg_pool2d(detail, 2)  # half size: the big picture
        small = self.middle2(self.middle1(small, step_info), step_info)
        big = F.interpolate(small, scale_factor=2, mode="nearest")  # back to full size
        return self.last(self.up(torch.cat([big, detail], dim=1), step_info))


# ---------------------------------------------------------------------------
# Generating: start from pure static and walk the 1,000 steps backwards.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, n_images, size, keep_every=None):
    """
    Returns the finished images. If keep_every is set, also returns the
    in-between images so the static-to-image process can be shown.
    """
    model.eval()
    images = torch.randn(n_images, 1, size, size)  # pure static
    kept = []
    for step in reversed(range(N_STEPS)):
        if keep_every and step % keep_every == keep_every - 1:
            kept.append((step + 1, images.clone()))
        t = torch.full((n_images,), step)
        guessed_static = model(images, t)
        # Remove this step's share of the static the network pointed at.
        share = noise_added[step] / (1 - original_left[step]).sqrt()
        images = (images - share * guessed_static) / image_kept[step].sqrt()
        if step > 0:
            # Add a small fresh dose of randomness, so every run gives different images.
            images = images + noise_added[step].sqrt() * torch.randn_like(images)
    model.train()
    if keep_every:
        kept.append((0, images.clone()))
        return images, kept
    return images
