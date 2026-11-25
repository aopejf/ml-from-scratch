"""
Imagined futures for the stock market, made by the diffusion model from project 4.

Black-Scholes assumes each day's move is a draw from the same bell curve. Real markets do not
behave that way: crashes are far more common than a bell curve allows, and wild days come in
clusters. So here a model learns what three months of the S&P 500 look like from every
three-month stretch since 1950, and then makes new ones.

The diffusion recipe is project 4's, imported and used as it is:
    diffusion.add_noise       bury real data in static, a chosen amount
    diffusion.generate        start from pure static and remove it in 1,000 steps
    diffusion.step_embedding  tell the network which step it is on
In project 4 the "image" was a handwritten digit. Here it is an 8 x 8 grid holding 64 daily
moves, read row by row. Only the network inside is new, because a run of days is a
sequence, not a picture: it looks along the days, near and far, in place of across pixels.

    python scenarios.py train       learn (about 8 minutes on a laptop processor)
    python scenarios.py generate    make 6,000 futures (about 35 minutes: each one takes 1,000 passes through the network)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "4_diffusion"))
import diffusion  # noqa: E402  (project 4)

SIDE = 8
DAYS = SIDE * SIDE  # 64 trading days: about three months
TRADING_DAYS_A_YEAR = 252
YEARS = DAYS / TRADING_DAYS_A_YEAR


def daily_moves():
    """Each day's move in the S&P 500 since 1950, as a log return (0.01 is about +1%)."""
    closes = pd.read_csv(here / "data" / "sp500.csv").close.to_numpy()
    return np.diff(np.log(closes))


def windows(moves):
    """Every stretch of 64 days in a row. Returns (how many, 64)."""
    return np.lib.stride_tricks.sliding_window_view(moves, DAYS).copy()


class Block(nn.Module):
    """Look at each day together with days a set distance either side of it, with the step number mixed in."""

    def __init__(self, width, step_width, reach):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=reach, dilation=reach)
        self.step = nn.Linear(step_width, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)

    def forward(self, x, step_info):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.step(step_info).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SequenceNoisePredictor(nn.Module):
    """
    Takes the same input as project 4's network (noisy 8 x 8 grids, and the step number) and
    gives the same output (its guess at the static), so project 4's generate() can drive it unchanged.
    Inside, the grid is unrolled into 64 days in order. The blocks look 1, 2, 4, 8 and 16 days
    to either side, so a quiet or wild spell anywhere in the three months can be noticed.
    """

    def __init__(self, width=32, step_width=64):
        super().__init__()
        self.step_width = step_width
        self.read_step = nn.Sequential(nn.Linear(step_width, step_width), nn.SiLU(), nn.Linear(step_width, step_width))
        self.first = nn.Conv1d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList([Block(width, step_width, reach) for reach in (1, 2, 4, 8, 16, 1)])
        self.last = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1))

    def forward(self, noisy, t):
        step_info = self.read_step(diffusion.step_embedding(t, self.step_width))
        h = self.first(noisy.reshape(noisy.size(0), 1, -1))
        for block in self.blocks:
            h = block(h, step_info)
        return self.last(h).reshape(noisy.shape)


def train(steps=5000, batch=128, seed=0, report=500):
    torch.manual_seed(seed)
    moves = daily_moves()
    typical = float(moves.std())  # one typical daily move, about 1%. Everything is measured in these.
    data = torch.tensor(windows(moves) / typical, dtype=torch.float32).view(-1, 1, SIDE, SIDE)
    model = SequenceNoisePredictor()
    smooth = SequenceNoisePredictor()  # a running average of the weights, which makes steadier samples
    smooth.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, began = [], time.time()
    for step in range(steps):
        real = data[torch.randint(0, len(data), (batch,))]
        t = torch.randint(0, diffusion.N_STEPS, (batch,))
        static = torch.randn_like(real)
        # The whole training signal: bury the real data in static, ask the network to point at the static.
        loss = F.mse_loss(model(diffusion.add_noise(real, t, static), t), static)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for a, b in zip(smooth.parameters(), model.parameters()):
                a.mul_(0.999).add_(b, alpha=0.001)
        losses.append(loss.item())
        if (step + 1) % report == 0:
            print(f"  step {step + 1:>5}: loss {np.mean(losses[-report:]):.4f}  ({time.time() - began:.0f}s)", flush=True)
    torch.save(dict(weights=smooth.state_dict(), typical=typical, steps=steps,
                    loss_curve=[float(np.mean(losses[i:i + 100])) for i in range(0, steps, 100)]), here / "scenario_model.pt")
    print("Saved scenario_model.pt")


def load_model():
    saved = torch.load(here / "scenario_model.pt", weights_only=True)
    model = SequenceNoisePredictor()
    model.load_state_dict(saved["weights"])
    return model, saved


def generate(n=6000, batch=1500, seed=0):
    """Make n new three-month stretches with project 4's generate(), and save them as daily moves."""
    torch.manual_seed(seed)
    model, saved = load_model()
    made, began = [], time.time()
    for start in range(0, n, batch):
        grids = diffusion.generate(model, min(batch, n - start), SIDE)
        made.append(grids.view(len(grids), DAYS).numpy() * saved["typical"])
        print(f"  {start + len(grids):>6} of {n} made  ({time.time() - began:.0f}s)", flush=True)
    moves = np.concatenate(made).astype(np.float32)
    np.savez_compressed(here / "data" / "generated_moves.npz", moves=moves)
    print("Saved data/generated_moves.npz")
    return moves


BIGGEST_BELIEVABLE_DAY = 0.25  # the worst day in the history, in October 1987, was 0.229 on this scale (a 20.5% fall)


def generated_moves(with_count=False):
    """
    The futures the model made. A very few come out broken, with a single day far beyond anything
    that has ever happened (one had a day of -89%). Those are thrown away, and the number is reported.
    """
    moves = np.load(here / "data" / "generated_moves.npz")["moves"].astype(float)
    believable = np.abs(moves).max(axis=1) <= BIGGEST_BELIEVABLE_DAY
    return (moves[believable], int((~believable).sum())) if with_count else moves[believable]


def bell_curve_moves(n, typical, average=0.0, seed=0):
    """What Black-Scholes assumes: every day an independent draw from one bell curve."""
    return average + np.random.default_rng(seed).standard_normal((n, DAYS)) * typical


def fingerprint(moves):
    """
    The features of real markets that a bell curve gets wrong, measured on a set of 64-day stretches.
    """
    flat = moves.ravel()
    typical = flat.std()
    size = np.abs(moves)
    centred = size - size.mean(axis=1, keepdims=True)

    def echo(lag):  # do big days follow big days? (autocorrelation of the size of moves)
        return float((centred[:, lag:] * centred[:, :-lag]).mean() / (centred**2).mean())

    total = moves.sum(axis=1)
    return dict(
        yearly_volatility=float(typical * np.sqrt(TRADING_DAYS_A_YEAR)),
        fat_tails=float(((flat - flat.mean()) ** 4).mean() / typical**4 - 3),  # excess kurtosis: 0 for a bell curve
        days_beyond_3=float((np.abs(flat) > 3 * typical).mean()),  # a bell curve says 0.27%
        days_beyond_5=float((np.abs(flat) > 5 * typical).mean()),  # a bell curve says 0.00006%
        echo_1_day=echo(1), echo_5_days=echo(5),
        three_month_fall_10=float((np.exp(total) - 1 < -0.10).mean()),  # share of stretches losing over 10%
        three_month_fall_20=float((np.exp(total) - 1 < -0.20).mean()),
        three_month_worst_1_in_100=float(np.exp(np.quantile(total, 0.01)) - 1),
        three_month_lean=float(((total - total.mean()) ** 3).mean() / total.std() ** 3),  # skew: negative = falls are bigger than rises
        calm_to_wild=float(np.quantile(moves.std(axis=1), 0.95) / np.quantile(moves.std(axis=1), 0.05)),
    )


def nearest_real(generated, real, sample=300, seed=0):
    """
    Did the model just memorise history? For a sample of generated stretches, find the closest real one.
    Compared with how close real stretches are to each other (ones at least a year apart, so they do not overlap).
    """
    rng = np.random.default_rng(seed)
    g = generated[rng.choice(len(generated), sample, replace=False)]
    to_real = np.sqrt(((g[:, None, :] - real[None]) ** 2).sum(axis=2)).min(axis=1)
    picks = rng.choice(len(real), sample, replace=False)
    between = []
    for i in picks:
        d = np.sqrt(((real - real[i]) ** 2).sum(axis=1))
        d[max(0, i - TRADING_DAYS_A_YEAR): i + TRADING_DAYS_A_YEAR] = np.inf
        between.append(d.min())
    return dict(generated_to_nearest_real=float(np.median(to_real)), real_to_nearest_real=float(np.median(between)),
                closest_any=float(to_real.min()))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "train":
        train()
    elif command == "generate":
        generate()
    else:
        print(__doc__)
