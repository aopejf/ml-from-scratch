"""
Project 1: three optimizers from scratch.

An optimizer is the rule for "walking downhill" on the loss.
Here the loss is a simple landscape with two adjustable numbers (x and y):
a long, narrow valley. It is steep across the valley and nearly flat along it,
which is exactly the shape that makes plain gradient descent struggle.

All three walkers start at the same spot and get the same number of steps.
The lowest point of the valley is at (0, 0).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEEPNESS = 100.0  # how much steeper the valley walls are than the valley floor


def loss(p):
    """Height of the landscape at position p = (x, y)."""
    x, y = p
    return 0.5 * (x**2 + STEEPNESS * y**2)


def gradient(p):
    """The uphill direction at p. Walking against it goes downhill."""
    x, y = p
    return np.array([x, STEEPNESS * y])


# ---------------------------------------------------------------------------
# The three optimizers. Each takes a start point and returns every
# position it visited.
# ---------------------------------------------------------------------------
def gradient_descent(start, learning_rate, steps):
    """Plain version: take a small step straight downhill. Repeat."""
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p -= learning_rate * gradient(p)
        path.append(p.copy())
    return np.array(path)


def momentum(start, learning_rate, steps, memory=0.9):
    """
    Remembers where it was already heading, like a ball rolling.
    Side-to-side wobbles cancel out; steady progress along the valley builds up.
    """
    p = np.array(start, dtype=float)
    velocity = np.zeros(2)
    path = [p.copy()]
    for _ in range(steps):
        velocity = memory * velocity - learning_rate * gradient(p)
        p += velocity
        path.append(p.copy())
    return np.array(path)


def adam(start, learning_rate, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Keeps two running averages: of the gradient (which way, on average)
    and of its square (how big it usually is). Dividing one by the other
    gives every direction its own step size: careful on the steep walls,
    bold along the flat floor.
    """
    p = np.array(start, dtype=float)
    avg_grad = np.zeros(2)
    avg_sq_grad = np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = gradient(p)
        avg_grad = beta1 * avg_grad + (1 - beta1) * g
        avg_sq_grad = beta2 * avg_sq_grad + (1 - beta2) * g**2
        # The averages start at zero, so early on they read too small. Correct for that.
        corrected_grad = avg_grad / (1 - beta1**t)
        corrected_sq = avg_sq_grad / (1 - beta2**t)
        p -= learning_rate * corrected_grad / (np.sqrt(corrected_sq) + eps)
        path.append(p.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Race them.
# ---------------------------------------------------------------------------
START = (-9.0, 2.0)
STEPS = 60

runs = {
    "Gradient descent": (gradient_descent(START, 0.018, STEPS), "#c62828"),
    "Momentum": (momentum(START, 0.03, STEPS, memory=0.8), "#2a6fdb"),
    "Adam": (adam(START, 0.8, STEPS), "#1a7f37"),
}

print(f"All start at {START}, loss {loss(START):.2f}. Lowest possible loss is 0.\n")
print(f"{'optimizer':<18}{'loss after 20':>15}{'loss after 60':>15}")
for name, (path, _) in runs.items():
    print(f"{name:<18}{loss(path[20]):>15.4f}{loss(path[-1]):>15.6f}")

# What "training explodes" means: same plain gradient descent, step size too big.
exploded = gradient_descent(START, 0.021, 12)
print("\nGradient descent with the step size too big (0.021 instead of 0.018):")
print("  loss every 3 steps:", "  ".join(f"{loss(p):.0f}" for p in exploded[::3]))


# ---------------------------------------------------------------------------
# Draw the picture.
# ---------------------------------------------------------------------------
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

gx, gy = np.meshgrid(np.linspace(-10, 3, 300), np.linspace(-4.5, 4.5, 300))
left.contourf(gx, gy, loss((gx, gy)), levels=30, cmap="Greys", alpha=0.55)
for name, (path, color) in runs.items():
    left.plot(path[:, 0], path[:, 1], "-o", color=color, markersize=3, linewidth=1.5, label=name)
left.plot(*START, "k*", markersize=14)
left.plot(0, 0, "kx", markersize=12, markeredgewidth=3)
left.annotate("start", START, textcoords="offset points", xytext=(6, 10))
left.annotate("lowest point", (0, 0), textcoords="offset points", xytext=(8, -16))
left.set_title("Three walks down the same valley (darker = higher)")
left.set_xlabel("x")
left.set_ylabel("y")
left.legend(loc="upper right")

for name, (path, color) in runs.items():
    right.plot([loss(p) for p in path], color=color, linewidth=2, label=name)
right.set_yscale("log")
right.set_title("Loss at each step (lower is better)")
right.set_xlabel("step")
right.set_ylabel("loss, log scale")
right.legend()

out = Path(__file__).with_name("result.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"\nSaved picture: {out.name}")
