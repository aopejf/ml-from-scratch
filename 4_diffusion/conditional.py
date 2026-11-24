"""
The upgraded diffusion model: bigger images, and you choose the digit.

Three changes from diffusion.py:

  1. Size. It works on 28x28 images, so the U goes two levels deep
     (28 -> 14 -> 7 pixels and back) where the small one went one.

  2. Conditioning. During training the network is also told which digit the
     image is. So it learns "the static in a noisy 7" separately from "the
     static in a noisy 3", and at generation time you can ask for a 7.
     Text-to-image tools work the same way, with a sentence in place of a digit.

  3. Guidance. One time in ten the label is hidden during training, so the
     network also learns to denoise without knowing the digit. When generating,
     it makes both guesses (with and without the label) and pushes further in
     the direction the label suggests. That makes the digits clearer.
     (Its technical name is classifier-free guidance.)

The noise schedule and the building block are reused from diffusion.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import N_STEPS, Block, image_kept, noise_added, original_left, step_embedding

NO_LABEL = 10  # an 11th "digit" that means "not telling you"


class ConditionalNoisePredictor(nn.Module):
    def __init__(self, width=32, step_width=128):
        super().__init__()
        self.step_width = step_width
        self.read_step = nn.Sequential(nn.Linear(step_width, step_width), nn.SiLU(), nn.Linear(step_width, step_width))
        self.read_label = nn.Embedding(11, step_width)  # digits 0-9, plus NO_LABEL
        self.first = nn.Conv2d(1, width, 3, padding=1)
        self.down1 = Block(width, width, step_width)  # 28x28
        self.down2 = Block(width, 2 * width, step_width)  # 14x14
        self.middle1 = Block(2 * width, 4 * width, step_width)  # 7x7
        self.middle2 = Block(4 * width, 2 * width, step_width)
        self.up2 = Block(4 * width, width, step_width)  # 14x14
        self.up1 = Block(2 * width, width, step_width)  # 28x28
        self.last = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, noisy_images, t, labels):
        # The step number and the digit label are mixed into one note for the network.
        note = self.read_step(step_embedding(t, self.step_width)) + self.read_label(labels)
        detail = self.down1(self.first(noisy_images), note)
        medium = self.down2(F.avg_pool2d(detail, 2), note)
        small = self.middle2(self.middle1(F.avg_pool2d(medium, 2), note), note)
        x = F.interpolate(small, scale_factor=2, mode="nearest")
        x = self.up2(torch.cat([x, medium], dim=1), note)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(torch.cat([x, detail], dim=1), note)
        return self.last(x)


@torch.no_grad()
def generate(model, labels, guidance=2.0, keep_every=None, size=28):
    """
    Make one image per entry in `labels` (a list of digits 0-9), starting from static.
    guidance = how hard to push towards the requested digit. 0 ignores the label.
    """
    model.eval()
    labels = torch.as_tensor(labels)
    n = len(labels)
    no_labels = torch.full_like(labels, NO_LABEL)
    images = torch.randn(n, 1, size, size)
    kept = []
    for step in reversed(range(N_STEPS)):
        if keep_every and step % keep_every == keep_every - 1:
            kept.append((step + 1, images.clone()))
        t = torch.full((n,), step)
        # Two guesses in one pass: knowing the digit, and not knowing it.
        both = model(torch.cat([images, images]), torch.cat([t, t]), torch.cat([labels, no_labels]))
        with_label, without_label = both[:n], both[n:]
        guessed_static = without_label + guidance * (with_label - without_label)

        share = noise_added[step] / (1 - original_left[step]).sqrt()
        images = (images - share * guessed_static) / image_kept[step].sqrt()
        if step > 0:
            images = images + noise_added[step].sqrt() * torch.randn_like(images)
    model.train()
    images = images.clamp(-1, 1)
    if keep_every:
        kept.append((0, images.clone()))
        return images, kept
    return images
