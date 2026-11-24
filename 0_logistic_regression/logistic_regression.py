"""
Project 0: logistic regression from zero.

The job: predict whether a student passes (1) or fails (0) an exam
from two numbers: hours studied and hours slept.

No machine-learning library is used. Only NumPy (for arithmetic on lists
of numbers) and Matplotlib (for the picture at the end).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)  # fixed seed, so every run gives the same result


# ---------------------------------------------------------------------------
# 1. Make some data.
#    400 pretend students. Students who study and sleep more tend to pass,
#    with some randomness so the two groups overlap a little, like real life.
# ---------------------------------------------------------------------------
n_students = 400
hours_studied = rng.uniform(0, 10, n_students)
hours_slept = rng.uniform(3, 10, n_students)

hidden_score = 1.2 * hours_studied + 0.8 * hours_slept - 11 + rng.normal(0, 1.5, n_students)
passed = (hidden_score > 0).astype(float)  # 1.0 = pass, 0.0 = fail

inputs = np.column_stack([hours_studied, hours_slept])  # one row per student

# Hold back 100 students the model never trains on, to test it honestly.
train_x, test_x = inputs[:300], inputs[300:]
train_y, test_y = passed[:300], passed[300:]

# Put both inputs on the same scale (centred on 0, similar spread).
# This makes learning smoother. The scale is measured on training data only.
mean, spread = train_x.mean(axis=0), train_x.std(axis=0)
train_x = (train_x - mean) / spread
test_x = (test_x - mean) / spread


# ---------------------------------------------------------------------------
# 2. The model.
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Squash any number into the range 0 to 1, so it can be read as a probability."""
    return 1.0 / (1.0 + np.exp(-z))


def predict_probability(x, weights, bias):
    """Weighted sum of the inputs, then squash. Returns 'chance of passing'."""
    return sigmoid(x @ weights + bias)


def cross_entropy(probability, truth):
    """
    How wrong are we? This is the loss.
    Being confidently wrong (saying 99% pass for a student who failed)
    is punished far more than being unsure.
    """
    eps = 1e-12  # avoids log(0)
    return -np.mean(truth * np.log(probability + eps) + (1 - truth) * np.log(1 - probability + eps))


def gradients(x, probability, truth):
    """
    Which way should each weight move to reduce the loss?
    After all the calculus, the answer is just (prediction - truth),
    multiplied by the input and averaged over the students.
    """
    error = probability - truth
    return x.T @ error / len(truth), error.mean()


# ---------------------------------------------------------------------------
# 3. Check the famous shortcut.
#    Compare the "prediction minus truth" formula against a brute-force
#    measurement: nudge a weight a tiny bit and see how much the loss moves.
# ---------------------------------------------------------------------------
check_w, check_b = rng.normal(size=2), 0.3
formula_grad, _ = gradients(train_x, predict_probability(train_x, check_w, check_b), train_y)

measured_grad = np.zeros(2)
tiny = 1e-6
for i in range(2):
    up, down = check_w.copy(), check_w.copy()
    up[i] += tiny
    down[i] -= tiny
    loss_up = cross_entropy(predict_probability(train_x, up, check_b), train_y)
    loss_down = cross_entropy(predict_probability(train_x, down, check_b), train_y)
    measured_grad[i] = (loss_up - loss_down) / (2 * tiny)

print("Gradient check")
print("  formula (prediction - truth):", np.round(formula_grad, 6))
print("  measured by nudging:         ", np.round(measured_grad, 6))
print("  biggest difference:          ", f"{np.abs(formula_grad - measured_grad).max():.2e}")


# ---------------------------------------------------------------------------
# 4. Train: guess, score, nudge, repeat.
# ---------------------------------------------------------------------------
weights = np.zeros(2)  # start knowing nothing
bias = 0.0
learning_rate = 0.5  # how big each nudge is
steps = 500
loss_history = []

print("\nTraining")
for step in range(steps + 1):
    probability = predict_probability(train_x, weights, bias)  # guess
    loss = cross_entropy(probability, train_y)  # score
    loss_history.append(loss)
    grad_w, grad_b = gradients(train_x, probability, train_y)  # which way is downhill
    weights -= learning_rate * grad_w  # nudge
    bias -= learning_rate * grad_b
    if step % 100 == 0:
        print(f"  step {step:3d}   loss {loss:.4f}")


# ---------------------------------------------------------------------------
# 5. Test on the 100 students the model has never seen.
#    Above 0.5 we say "pass", below we say "fail".
# ---------------------------------------------------------------------------
def accuracy(x, y):
    return np.mean((predict_probability(x, weights, bias) > 0.5) == y)


print("\nResults")
print(f"  accuracy on training students: {accuracy(train_x, train_y):.1%}")
print(f"  accuracy on unseen students:   {accuracy(test_x, test_y):.1%}")


# ---------------------------------------------------------------------------
# 6. Draw the picture.
# ---------------------------------------------------------------------------
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

left.plot(loss_history, color="#2a6fdb")
left.set_title("The loss falls as the model learns")
left.set_xlabel("training step")
left.set_ylabel("loss (how wrong)")

# Shade the background by the model's predicted chance of passing.
grid_s, grid_z = np.meshgrid(np.linspace(0, 10, 200), np.linspace(3, 10, 200))
grid = (np.column_stack([grid_s.ravel(), grid_z.ravel()]) - mean) / spread
grid_prob = predict_probability(grid, weights, bias).reshape(grid_s.shape)

shading = right.contourf(grid_s, grid_z, grid_prob, levels=20, cmap="RdYlGn", alpha=0.6)
right.contour(grid_s, grid_z, grid_prob, levels=[0.5], colors="black", linewidths=2)
raw_test = inputs[300:]
right.scatter(*raw_test[test_y == 1].T, color="#1a7f37", edgecolor="white", label="passed")
right.scatter(*raw_test[test_y == 0].T, color="#c62828", edgecolor="white", marker="X", label="failed")
right.set_title("Unseen students. Black line = the 50/50 boundary")
right.set_xlabel("hours studied")
right.set_ylabel("hours slept")
right.legend(loc="lower left")
fig.colorbar(shading, ax=right, label="predicted chance of passing")

out = Path(__file__).with_name("result.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"\nSaved picture: {out.name}")
