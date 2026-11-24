"""
Watch three optimizers learn, live.

Three copies of the tumour classifier from project 0 start out identical,
knowing nothing. Each one is trained by a different optimizer from optim.py,
all with the same step size. The window updates after every training step.

    python watch_training.py          opens a live window
    python watch_training.py --save   writes training.gif instead

What you see:
  Left   The loss for each optimizer, falling as it learns.
  Right  One row per optimizer. Every dot is a patient the model never
         trained on, placed at the model's "chance of cancer" for them.
         Everyone starts at 50%. Learning pulls the red dots (cancer)
         right and the green dots (no cancer) left.
"""

import sys
from pathlib import Path

import matplotlib

SAVE = "--save" in sys.argv
if SAVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import load_breast_cancer

from optim import Adam, GradientDescent, Momentum

STEPS = 150
LEARNING_RATE = 0.05  # the same for all three, so the comparison is fair


# ---------------------------------------------------------------------------
# Data: the same patients, split and scaling as project 0.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
data = load_breast_cancer()
inputs, is_cancer = data.data, (data.target == 0).astype(float)
order = rng.permutation(len(inputs))
inputs, is_cancer = inputs[order], is_cancer[order]
n_train = int(0.8 * len(inputs))
train_x, test_x = inputs[:n_train], inputs[n_train:]
train_y, test_y = is_cancer[:n_train], is_cancer[n_train:]
mean, spread = train_x.mean(axis=0), train_x.std(axis=0)
train_x, test_x = (train_x - mean) / spread, (test_x - mean) / spread
PENALTY = 1.0 / n_train


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# One learner = one model + the optimizer training it.
# ---------------------------------------------------------------------------
class Learner:
    def __init__(self, name, optimizer, color):
        self.name, self.optimizer, self.color = name, optimizer, color
        self.weights = np.zeros(train_x.shape[1])
        self.bias = np.zeros(1)
        self.losses = []
        self.record()

    def record(self):
        p = sigmoid(train_x @ self.weights + self.bias)
        wrongness = -np.mean(train_y * np.log(p + 1e-12) + (1 - train_y) * np.log(1 - p + 1e-12))
        self.losses.append(wrongness + 0.5 * PENALTY * np.sum(self.weights**2))

    def train_one_step(self):
        p = sigmoid(train_x @ self.weights + self.bias)  # guess
        error = p - train_y  # prediction minus truth
        grads = [train_x.T @ error / n_train + PENALTY * self.weights, np.array([error.mean()])]
        self.optimizer.step([self.weights, self.bias], grads)  # nudge
        self.record()  # score

    def test_probabilities(self):
        return sigmoid(test_x @ self.weights + self.bias)


learners = [
    Learner("Gradient descent", GradientDescent(LEARNING_RATE), "#c62828"),
    Learner("Momentum", Momentum(LEARNING_RATE), "#2a6fdb"),
    Learner("Adam", Adam(LEARNING_RATE), "#1a7f37"),
]


# ---------------------------------------------------------------------------
# The picture.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 6.5))
grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.15], hspace=0.75, wspace=0.18)

loss_ax = fig.add_subplot(grid[:, 0])
loss_ax.set_xlim(0, STEPS)
loss_ax.set_ylim(0.05, 0.8)
loss_ax.set_yscale("log")
loss_ax.set_yticks([0.05, 0.1, 0.2, 0.4, 0.7], ["0.05", "0.1", "0.2", "0.4", "0.7"])
loss_ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
loss_ax.set_xlabel("training step")
loss_ax.set_ylabel("loss (how wrong), lower is better")
loss_lines = {L.name: loss_ax.plot([], [], color=L.color, linewidth=2.5, label=L.name)[0] for L in learners}
loss_ax.legend(loc="upper right")

has_cancer = test_y == 1
jitter = rng.uniform(-0.35, 0.35, len(test_y))  # spreads dots vertically so they don't overlap
dot_colors = np.where(has_cancer, "#c62828", "#1a7f37")
strips = {}
for row, L in enumerate(learners):
    ax = fig.add_subplot(grid[row, 1])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    dots = ax.scatter(np.full(len(test_y), 0.5), jitter, c=dot_colors, s=22, alpha=0.75, edgecolor="white", linewidth=0.4)
    strips[L.name] = (ax, dots)
strips[learners[-1].name][0].set_xlabel("model's chance of cancer for each unseen patient   (red = really cancer, green = not)")


def draw(frame):
    step = min(frame, STEPS)
    for L in learners:
        while len(L.losses) <= step:  # train live, one step per frame
            L.train_one_step()
        loss_lines[L.name].set_data(range(len(L.losses)), L.losses)

        prob = L.test_probabilities()
        ax, dots = strips[L.name]
        dots.set_offsets(np.column_stack([prob, jitter]))
        missed = int(np.sum((prob <= 0.5) & has_cancer))
        false_alarms = int(np.sum((prob > 0.5) & ~has_cancer))
        ax.set_title(
            f"{L.name}:  loss {L.losses[-1]:.3f}   cancers missed {missed}   false alarms {false_alarms}",
            color=L.color,
            fontsize=11,
            loc="left",
        )
    loss_ax.set_title(f"Step {step} of {STEPS}   (same step size for all three)")


HOLD = 25  # extra frames at the end so the final state stays on screen
animation = FuncAnimation(fig, draw, frames=STEPS + 1 + HOLD, interval=70, repeat=False)

if SAVE:
    out = Path(__file__).with_name("training.gif")
    animation.save(out, writer=PillowWriter(fps=15), dpi=80)
    print(f"Saved animation: {out.name}")
    print(f"\n{'after 150 steps':<18}{'loss':>8}")
    for L in learners:
        print(f"{L.name:<18}{L.losses[STEPS]:>8.4f}")
    fig.savefig(Path(__file__).with_name("training_final.png"), dpi=110)
else:
    plt.show()
