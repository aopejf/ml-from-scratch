"""
A neural network in pure NumPy, and a map of the hill it rolls down.

    python neural_net.py

Part 1 builds a small network with no ML library: the forward pass, the
backward pass (backpropagation, worked out by hand with matrices) and
gradient descent are all written here.

Part 2 draws the loss landscape. The network has 337 adjustable numbers, so
its loss is a surface in 337 dimensions, which nobody can draw. The trick:
record every position training passed through, find the two directions it
moved along most (PCA), and slice the landscape along just those two. Then
the path gradient descent took can be drawn on top.

It saves:
    result.png      the task, the loss curve, the landscape with the path on it
    landscape.gif   the 3D landscape, turning, with the path
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

here = Path(__file__).parent
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# The task: two spirals wound around each other. No straight line separates them.
# ---------------------------------------------------------------------------
def make_spirals(n_each, noise=0.08):
    distance = np.sqrt(rng.random(n_each)) * 1.0 + 0.1
    angle = distance * 2.6 * np.pi
    arm = np.c_[distance * np.cos(angle), distance * np.sin(angle)]
    points = np.r_[arm, -arm] + rng.normal(0, noise * 0.5, (2 * n_each, 2))
    return points, np.r_[np.zeros(n_each), np.ones(n_each)]


train_x, train_y = make_spirals(200)
test_x, test_y = make_spirals(200)

# ---------------------------------------------------------------------------
# Part 1: the network.
# All its weights live in one long list of numbers, so "where the network is
# on the landscape" is simply that list.
# ---------------------------------------------------------------------------
SIZES = [2, 16, 16, 1]  # 2 inputs -> 16 neurons -> 16 neurons -> 1 answer
SHAPES = [(a, b) for a, b in zip(SIZES, SIZES[1:])]
N_WEIGHTS = sum(a * b + b for a, b in SHAPES)


def unpack(weights):
    """Cut the long list back into a weight matrix and a bias list per layer."""
    layers, i = [], 0
    for a, b in SHAPES:
        layers.append((weights[i : i + a * b].reshape(a, b), weights[i + a * b : i + a * b + b]))
        i += a * b + b
    return layers


def starting_weights():
    return np.concatenate([np.r_[rng.normal(0, np.sqrt(1 / a), a * b), np.zeros(b)] for a, b in SHAPES])


def forward(weights, x):
    """Guess. Returns the prediction and what each layer put out on the way (needed for backprop)."""
    outputs = [x]
    layers = unpack(weights)
    for W, b in layers[:-1]:
        outputs.append(np.tanh(outputs[-1] @ W + b))
    W, b = layers[-1]
    prediction = 1 / (1 + np.exp(-(outputs[-1] @ W + b)[:, 0]))  # sigmoid, as in project 0
    return prediction, outputs


def loss_of(weights, x, y):
    """Score. Cross-entropy, as in project 0."""
    p = np.clip(forward(weights, x)[0], 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradient(weights, x, y):
    """
    Work out the nudge: backpropagation.
    Start from the error at the output and pass the blame backwards, layer by layer.
    """
    prediction, outputs = forward(weights, x)
    layers = unpack(weights)
    blame = ((prediction - y) / len(y))[:, None]  # project 0's "prediction - truth"
    pieces = []
    for (W, b), layer_input in zip(reversed(layers), reversed(outputs)):
        pieces.append(np.r_[(layer_input.T @ blame).ravel(), blame.sum(axis=0)])
        blame = (blame @ W.T) * (1 - layer_input**2)  # pass blame back through tanh
    return np.concatenate(pieces[::-1])


def accuracy(weights, x, y):
    return np.mean((forward(weights, x)[0] > 0.5) == y)


# ----- check the hand-written backprop against brute force ------------------
weights = starting_weights()
analytic = gradient(weights, train_x, train_y)
numeric = np.zeros(N_WEIGHTS)
for i in range(N_WEIGHTS):
    bump = np.zeros(N_WEIGHTS)
    bump[i] = 1e-6
    numeric[i] = (loss_of(weights + bump, train_x, train_y) - loss_of(weights - bump, train_x, train_y)) / 2e-6
print(f"Network: {N_WEIGHTS} adjustable numbers")
print(f"Backprop check: largest difference from the brute-force gradient is {np.abs(analytic - numeric).max():.1e}\n")

# ----- train with plain gradient descent -----------------------------------
LEARNING_RATE = 1.0
STEPS = 20000
path, losses = [weights.copy()], [loss_of(weights, train_x, train_y)]
for step in range(1, STEPS + 1):
    weights = weights - LEARNING_RATE * gradient(weights, train_x, train_y)  # nudge
    path.append(weights.copy())
    losses.append(loss_of(weights, train_x, train_y))
    if step % 2500 == 0:
        print(f"step {step:6d}   loss {losses[-1]:.4f}   right on training points {accuracy(weights, train_x, train_y):.1%}")
path = np.array(path)
print(f"\nFinal: {accuracy(weights, train_x, train_y):.1%} on training points, {accuracy(weights, test_x, test_y):.1%} on new points")

# ---------------------------------------------------------------------------
# Part 2: the loss landscape.
# ---------------------------------------------------------------------------
# The two directions training moved along most, measured from where it ended.
moves = path - path[-1]
_, strength, directions = np.linalg.svd(moves[::20], full_matrices=False)
direction_1, direction_2 = directions[0], directions[1]
captured = (strength[:2] ** 2).sum() / (strength**2).sum()
print(f"\nThe 2D slice captures {captured:.1%} of the path's movement")

path_1, path_2 = moves @ direction_1, moves @ direction_2  # the path, seen in that slice
pad_1, pad_2 = 0.25 * np.ptp(path_1), 0.25 * np.ptp(path_2)
grid_1 = np.linspace(path_1.min() - pad_1, path_1.max() + pad_1, 70)
grid_2 = np.linspace(path_2.min() - pad_2, path_2.max() + pad_2, 70)
landscape = np.array([[loss_of(path[-1] + a * direction_1 + b * direction_2, train_x, train_y) for a in grid_1] for b in grid_2])
A, B = np.meshgrid(grid_1, grid_2)
# The path's true loss is not exactly the slice's loss (the path wanders slightly out of the slice),
# so for drawing, the path is laid on the surface.
path_on_surface = np.array([loss_of(path[-1] + a * direction_1 + b * direction_2, train_x, train_y) for a, b in zip(path_1[::50], path_2[::50])])

# ---------------------------------------------------------------------------
# Pictures.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(19, 11))

ax = fig.add_subplot(2, 3, 1)
span = np.linspace(-1.3, 1.3, 300)
GX, GY = np.meshgrid(span, span)
ax.contourf(GX, GY, forward(weights, np.c_[GX.ravel(), GY.ravel()])[0].reshape(GX.shape), levels=20, cmap="RdBu", alpha=0.75)
ax.scatter(*test_x.T, c=test_y, cmap="RdBu", edgecolors="white", s=22)
ax.set_title(f"What it learned ({accuracy(weights, test_x, test_y):.0%} right on these new points)")
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(2, 3, 4)
ax.semilogy(losses, color="#2a6fdb")
ax.set_xlabel("gradient descent step")
ax.set_ylabel("loss")
ax.set_title("Loss during training")
ax.grid(alpha=0.25)

ax = fig.add_subplot(2, 3, (2, 5))
levels = np.geomspace(landscape.min(), landscape.max(), 30)
ax.contourf(A, B, landscape, levels=levels, cmap="viridis", norm=matplotlib.colors.LogNorm())
ax.contour(A, B, landscape, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
ax.plot(path_1, path_2, color="#ff5a36", lw=2)
ax.scatter(path_1[0], path_2[0], color="white", edgecolors="black", s=90, zorder=3, label="start")
ax.scatter(path_1[-1], path_2[-1], color="#ff5a36", edgecolors="black", marker="*", s=260, zorder=3, label="end")
ax.set_title("The loss landscape from above, with the path gradient descent took\n(dark = low loss)")
ax.set_xlabel("direction training moved along most")
ax.set_ylabel("second most")
ax.legend()

ax3d = fig.add_subplot(2, 3, (3, 6), projection="3d")


def draw_3d(ax, turn=0):
    ax.plot_surface(A, B, np.log10(landscape), cmap="viridis", alpha=0.85, linewidth=0, rcount=70, ccount=70)
    ax.plot(path_1[::50], path_2[::50], np.log10(path_on_surface) + 0.03, color="#ff5a36", lw=2.5, zorder=10)
    ax.set_zlabel("loss (powers of 10)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=38, azim=-55 + turn)
    ax.set_box_aspect(None, zoom=0.85)  # leaves room for the axis label


draw_3d(ax3d)
ax3d.set_title("The same landscape in 3D")
fig.tight_layout()
fig.savefig(here / "result.png", dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(6.4, 5.6))
ax = fig.add_subplot(projection="3d")
fig.subplots_adjust(left=0, right=0.95, bottom=0, top=1)


def frame(i):
    ax.clear()
    draw_3d(ax, turn=i * 4)


FuncAnimation(fig, frame, frames=90).save(here / "landscape.gif", writer=PillowWriter(fps=15), dpi=70)
print("Saved: result.png, landscape.gif")
