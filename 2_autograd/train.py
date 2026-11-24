"""
Train a small neural network with the engine, and show why you would want one.

The data is two interlocking crescents of points, red and blue. No straight
line can separate them. One neuron (logistic regression, project 0) can only
draw a straight line. A network of neurons can bend the line.

    python train.py          trains both, saves result.png and training.gif
    python train.py --live   same, and shows a window that updates as it trains

This is practice data made by a formula (scikit-learn's make_moons), chosen
because you can see the whole problem in one picture. checks.py runs the
engine on the real tumour data.
"""

import sys
import time
from pathlib import Path

import matplotlib

LIVE = "--live" in sys.argv
if not LIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons

from nn import Adam, Network, cross_entropy, optimizer_step, zero_grads

STEPS = 600
SNAPSHOT_EVERY = 10

# ---------------------------------------------------------------------------
# Data: 300 points, 200 to train on and 100 held back.
# ---------------------------------------------------------------------------
points, colour = make_moons(n_samples=300, noise=0.2, random_state=0)
train_x, test_x = points[:200], points[200:]
train_y, test_y = colour[:200].astype(float), colour[200:].astype(float)
rows, truths = train_x.tolist(), train_y.tolist()

# ---------------------------------------------------------------------------
# Two models, same loss, same optimizer, same number of steps.
# ---------------------------------------------------------------------------
models = {
    "One neuron (logistic regression)": Network(2, [1], seed=0),
    "Network: 2 inputs, 8 neurons, 8 neurons, 1 output": Network(2, [8, 8, 1], seed=0),
}
optimizers = {name: Adam(learning_rate=0.05) for name in models}
losses = {name: [] for name in models}
snapshots = {name: [] for name in models}  # the boundary at regular moments, for the animation

grid_a, grid_b = np.meshgrid(np.linspace(-1.8, 2.8, 120), np.linspace(-1.4, 1.9, 120))
grid_points = np.column_stack([grid_a.ravel(), grid_b.ravel()])


def fast_predict(model, x):
    """
    The same forward pass as the network, done with NumPy on many points at once.
    Used only for drawing and scoring, where no gradients are needed.
    """
    for layer in model.layers:
        w = np.array([[v.data for v in neuron.weights] for neuron in layer.neurons])
        b = np.array([neuron.bias.data for neuron in layer.neurons])
        x = x @ w.T + b
        x = np.tanh(x) if layer.neurons[0].squash == "tanh" else 1.0 / (1.0 + np.exp(-x))
    return x[:, 0]


def accuracy(model, x, y):
    return np.mean((fast_predict(model, x) > 0.5) == y)


# ---------------------------------------------------------------------------
# The picture (set up before training so --live can update it).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
loss_ax, panels = axes[0], dict(zip(models, axes[1:]))
line_colors = dict(zip(models, ["#c62828", "#1a7f37"]))


def draw(snapshot_index):
    step = snapshot_index * SNAPSHOT_EVERY
    loss_ax.clear()
    for name in models:
        shown = losses[name][: step + 1]
        loss_ax.plot(shown, color=line_colors[name], linewidth=2.5, label=name.split(":")[0].split(" (")[0])
    loss_ax.set_xlim(0, STEPS)
    loss_ax.set_ylim(0, 0.8)
    loss_ax.set_xlabel("training step")
    loss_ax.set_ylabel("loss (how wrong), lower is better")
    loss_ax.set_title(f"Step {step} of {STEPS}")
    loss_ax.legend(loc="upper right")

    for name, ax in panels.items():
        prob, test_acc = snapshots[name][snapshot_index]
        ax.clear()
        ax.contourf(grid_a, grid_b, prob.reshape(grid_a.shape), levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.65)
        ax.contour(grid_a, grid_b, prob.reshape(grid_a.shape), levels=[0.5], colors="black", linewidths=2)
        ax.scatter(*test_x[test_y == 0].T, color="#c62828", edgecolor="white", s=30)
        ax.scatter(*test_x[test_y == 1].T, color="#1f5fbf", edgecolor="white", s=30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\n{test_acc:.0%} right on 100 unseen points", fontsize=10.5)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Train: guess, score, work out the nudge (the engine does this), nudge.
# ---------------------------------------------------------------------------
if LIVE:
    plt.ion()

start = time.time()
print(f"{'step':>5}   " + "   ".join(f"{name.split(':')[0].split(' (')[0]:>12}" for name in models) + "   (loss)")
for step in range(STEPS + 1):
    for name, model in models.items():
        params = model.parameters()
        loss = cross_entropy([model(row) for row in rows], truths)  # guess and score
        losses[name].append(loss.data)
        if step % SNAPSHOT_EVERY == 0:
            snapshots[name].append((fast_predict(model, grid_points), accuracy(model, test_x, test_y)))
        if step < STEPS:
            zero_grads(params)
            loss.backward()  # the engine works out every weight's nudge
            optimizer_step(optimizers[name], params)  # Adam from project 1 applies it
    if step % 100 == 0:
        print(f"{step:>5}   " + "   ".join(f"{losses[name][-1]:>12.4f}" for name in models))
    if LIVE and step % SNAPSHOT_EVERY == 0:
        draw(step // SNAPSHOT_EVERY)
        plt.pause(0.01)

print(f"\nTraining took {time.time() - start:.0f} seconds")
print("\nResults")
for name, model in models.items():
    print(f"  {name}")
    print(f"    weights: {len(model.parameters())}")
    print(f"    right on training points: {accuracy(model, train_x, train_y):.1%}")
    print(f"    right on unseen points:   {accuracy(model, test_x, test_y):.1%}")

# ---------------------------------------------------------------------------
# Save the final picture and the animation.
# ---------------------------------------------------------------------------
here = Path(__file__).parent
n_snapshots = STEPS // SNAPSHOT_EVERY + 1
draw(n_snapshots - 1)
fig.savefig(here / "result.png", dpi=120)
print("\nSaved picture: result.png")

hold = 12  # keep the last frame on screen for a moment
animation = FuncAnimation(fig, lambda i: draw(min(i, n_snapshots - 1)), frames=n_snapshots + hold)
animation.save(here / "training.gif", writer=PillowWriter(fps=10), dpi=70)
print("Saved animation: training.gif")

if LIVE:
    plt.ioff()
    draw(n_snapshots - 1)
    plt.show()
