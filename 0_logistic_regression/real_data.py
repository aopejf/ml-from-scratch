"""
Project 0, real-data version: is this tumour cancerous?

Same model as logistic_regression.py. Only the data is different.

The data is the Breast Cancer Wisconsin (Diagnostic) dataset: 569 real
tumour samples. For each one, 30 measurements were taken from a microscope
image of its cells (size, texture, smoothness and so on), and a doctor
recorded whether it was malignant (cancer) or benign (not cancer).

scikit-learn is used for two things only: loading the dataset, and
checking our answer at the end. The model and the training are still ours.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Load the real data.
# ---------------------------------------------------------------------------
data = load_breast_cancer()
inputs = data.data  # 569 rows, 30 measurements each
is_cancer = (data.target == 0).astype(float)  # 1.0 = malignant, 0.0 = benign
measurement_names = data.feature_names

# Shuffle, then hold back 20% of patients the model never trains on.
order = rng.permutation(len(inputs))
inputs, is_cancer = inputs[order], is_cancer[order]
n_train = int(0.8 * len(inputs))
train_x, test_x = inputs[:n_train], inputs[n_train:]
train_y, test_y = is_cancer[:n_train], is_cancer[n_train:]

# Put all 30 measurements on the same scale (measured on training data only).
mean, spread = train_x.mean(axis=0), train_x.std(axis=0)
train_x = (train_x - mean) / spread
test_x = (test_x - mean) / spread

print(f"Patients: {len(train_y)} to train on, {len(test_y)} held back for testing")
print(f"Share with cancer: {is_cancer.mean():.0%}")


# ---------------------------------------------------------------------------
# 2. The model. Identical to the pretend-student version, with one addition.
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_probability(x, weights, bias):
    return sigmoid(x @ weights + bias)


# The addition: a small penalty for large weights.
# With 30 measurements the model can become wildly overconfident by pushing
# its weights to huge values. This penalty keeps them modest.
# (Its technical name is L2 regularization.)
PENALTY = 1.0 / len(train_y)


def cross_entropy(probability, truth, weights):
    eps = 1e-12
    wrongness = -np.mean(truth * np.log(probability + eps) + (1 - truth) * np.log(1 - probability + eps))
    return wrongness + 0.5 * PENALTY * np.sum(weights**2)


def gradients(x, probability, truth, weights):
    error = probability - truth  # still just prediction minus truth
    return x.T @ error / len(truth) + PENALTY * weights, error.mean()


# ---------------------------------------------------------------------------
# 3. Train: guess, score, nudge, repeat.
# ---------------------------------------------------------------------------
def train(x, y, steps=20000, learning_rate=0.5, verbose=False):
    """Start from zero weights and learn from the patients in x, y."""
    weights = np.zeros(x.shape[1])  # 30 weights now, one per measurement
    bias = 0.0
    history = []
    for step in range(steps + 1):
        probability = predict_probability(x, weights, bias)
        loss = cross_entropy(probability, y, weights)
        history.append(loss)
        grad_w, grad_b = gradients(x, probability, y, weights)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        if verbose and step in (0, 10, 100, 1000, steps):
            print(f"  step {step:5d}   loss {loss:.4f}")
    return weights, bias, history


print("\nTraining")
weights, bias, loss_history = train(train_x, train_y, verbose=True)


# ---------------------------------------------------------------------------
# 4. Choose the cut-off.
#    So far: above 50% we say "cancer". That treats a missed cancer and a
#    false alarm as equally bad, and they are not. A lower cut-off flags more
#    patients: more cancers caught, more false alarms.
#
#    The cut-off must be chosen WITHOUT looking at the test patients,
#    otherwise the final score is rigged. So we rehearse on training patients:
#    split them into 5 groups, and for each group train a model on the other
#    four and score the group it did not see. Every training patient ends up
#    with a prediction from a model that never saw them.
#    (This is called cross-validation.)
# ---------------------------------------------------------------------------
TARGET_CATCH_RATE = 0.99  # we want to catch at least 99% of cancers in rehearsal

rehearsal_prob = np.zeros(len(train_y))
for group in np.array_split(np.arange(len(train_y)), 5):
    others = np.setdiff1d(np.arange(len(train_y)), group)
    w, b, _ = train(train_x[others], train_y[others])
    rehearsal_prob[group] = predict_probability(train_x[group], w, b)

# Highest cut-off that still catches the target share of rehearsal cancers.
cancer_scores = np.sort(rehearsal_prob[train_y == 1])
allowed_misses = int(np.floor((1 - TARGET_CATCH_RATE) * len(cancer_scores)))
cutoff = float(cancer_scores[allowed_misses]) - 1e-9

rehearsal_caught = np.mean(rehearsal_prob[train_y == 1] > cutoff)
rehearsal_alarms = np.mean(rehearsal_prob[train_y == 0] > cutoff)
print("\nChoosing the cut-off on training patients only")
print(f"  chosen cut-off:              {cutoff:.3f}  (was 0.500)")
print(f"  rehearsal cancers caught:    {rehearsal_caught:.1%}")
print(f"  rehearsal false-alarm rate:  {rehearsal_alarms:.1%}")


# ---------------------------------------------------------------------------
# 5. Test on patients the model has never seen, with both cut-offs.
# ---------------------------------------------------------------------------
test_prob = predict_probability(test_x, weights, bias)
has_cancer = test_y == 1


def count_outcomes(threshold):
    said = test_prob > threshold
    return {
        "caught": int(np.sum(said & has_cancer)),
        "missed": int(np.sum(~said & has_cancer)),
        "false_alarm": int(np.sum(said & ~has_cancer)),
        "cleared": int(np.sum(~said & ~has_cancer)),
    }


before, after = count_outcomes(0.5), count_outcomes(cutoff)
n_cancer, n_healthy = int(has_cancer.sum()), int((~has_cancer).sum())

print(f"\nResults on {len(test_y)} unseen patients ({n_cancer} with cancer, {n_healthy} without)")
print(f"  {'':<18}{'cut-off 0.500':>15}{f'cut-off {cutoff:.3f}':>15}")
print(f"  {'cancers caught':<18}{before['caught']:>15}{after['caught']:>15}")
print(f"  {'cancers missed':<18}{before['missed']:>15}{after['missed']:>15}")
print(f"  {'false alarms':<18}{before['false_alarm']:>15}{after['false_alarm']:>15}")
acc_before = (before["caught"] + before["cleared"]) / len(test_y)
acc_after = (after["caught"] + after["cleared"]) / len(test_y)
print(f"  {'accuracy':<18}{acc_before:>15.1%}{acc_after:>15.1%}")

missed_scores = np.sort(test_prob[has_cancer & (test_prob <= 0.5)])
print("  model's cancer probability for the cancers missed at 0.5:", np.round(missed_scores, 3))

said_cancer = test_prob > 0.5  # used for the library comparison below
caught, missed = after["caught"], after["missed"]
false_alarm, correctly_cleared = after["false_alarm"], after["cleared"]
test_acc = acc_after


# ---------------------------------------------------------------------------
# 6. Check our work against scikit-learn, a library used in industry.
#    Same data, same penalty. If our from-scratch code is right, both should
#    arrive at nearly the same weights.
# ---------------------------------------------------------------------------
library_model = LogisticRegression(C=1.0, tol=1e-10, max_iter=100000).fit(train_x, train_y)
library_acc = library_model.score(test_x, test_y)
biggest_gap = np.abs(weights - library_model.coef_[0]).max()
agree = np.mean(library_model.predict(test_x) == said_cancer)

print("\nCheck against scikit-learn")
print(f"  its accuracy on unseen patients:  {library_acc:.1%}")
print(f"  patients where both models agree: {agree:.1%}")
print(f"  biggest difference in any weight: {biggest_gap:.5f}")


# ---------------------------------------------------------------------------
# 7. Draw the picture.
# ---------------------------------------------------------------------------
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5))

a.plot(loss_history, color="#2a6fdb")
a.set_xscale("symlog")
a.set_title("The loss falls as the model learns")
a.set_xlabel("training step (log scale)")
a.set_ylabel("loss (how wrong)")

grid = np.array([[caught, missed], [false_alarm, correctly_cleared]])
labels = [["caught", "MISSED"], ["false alarm", "correctly cleared"]]
tint = np.array([[0.35, 0.9], [0.7, 0.35]])
b.imshow(tint, cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.6)
for row in range(2):
    for col in range(2):
        b.text(col, row, f"{grid[row, col]}\n{labels[row][col]}", ha="center", va="center", fontsize=14)
b.set_xticks([0, 1], ["model said: cancer", "model said: no cancer"])
b.set_yticks([0, 1], ["really cancer", "really not"])
b.set_title(f"{len(test_y)} unseen patients, cut-off {cutoff:.2f} ({test_acc:.1%} correct)")

top = np.argsort(np.abs(weights))[-10:]
c.barh(
    [measurement_names[i] for i in top],
    weights[top],
    color=["#c62828" if w > 0 else "#1a7f37" for w in weights[top]],
)
c.axvline(0, color="black", linewidth=0.8)
c.set_title("The 10 measurements the model leans on most")
c.set_xlabel("weight  (right = points to cancer, left = points away)")

out = Path(__file__).with_name("result_real_data.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"\nSaved picture: {out.name}")
