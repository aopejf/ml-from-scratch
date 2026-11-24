"""
Who is about to cancel their membership?

    python make_data.py     (once, to create members.csv)
    python churn.py

The business question: you can afford to contact 10% of your members this
month with a "please stay" offer. Which 10%?

Three ways of choosing are compared on members the models never saw:
  1. At random                       (no model)
  2. A simple rule                   (lowest usage first, what most people would do)
  3. Logistic regression             (project 0's model)
  4. Gradient-boosted decision trees (the usual winner on table data like this)

It saves result.png and prints a money estimate.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

here = Path(__file__).parent
CAN_CONTACT = 0.10  # share of members you can reach each month
OFFER_COST = 5.00  # cost of the "please stay" offer, per member contacted
OFFER_WORKS = 0.30  # share of would-be cancellers the offer saves (an assumption)
MONTHS_KEPT = 6  # extra months a saved member stays (an assumption)

# ---------------------------------------------------------------------------
# 1. Load the members and hold some back for testing.
# ---------------------------------------------------------------------------
members = pd.read_csv(here / "members.csv")
answers = members.pop("cancelled")
table = pd.get_dummies(members, columns=["plan", "joined_through"], dtype=int)
train_x, test_x, train_y, test_y = train_test_split(table, answers, test_size=0.25, random_state=0, stratify=answers)
print(f"Members: {len(train_x):,} to learn from, {len(test_x):,} held back")
print(f"Cancelled this month: {answers.mean():.1%}\n")

# ---------------------------------------------------------------------------
# 2. Four ways to score "how likely is this member to cancel?"
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train_x, train_y)
trees = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0).fit(train_x, train_y)

risk = {
    "At random": rng.random(len(test_x)),
    "Simple rule: fewest washes": -test_x["washes_last_month"].to_numpy() + rng.random(len(test_x)) * 0.01,
    "Logistic regression": logistic.predict_proba(test_x)[:, 1],
    "Boosted trees": trees.predict_proba(test_x)[:, 1],
}

# ---------------------------------------------------------------------------
# 3. Score them the way the business would.
# ---------------------------------------------------------------------------
n_contact = int(len(test_x) * CAN_CONTACT)
truth = test_y.to_numpy()
prices = members.loc[test_x.index, "monthly_price"].to_numpy()
print(f"If you contact the riskiest {CAN_CONTACT:.0%} ({n_contact} of {len(test_x):,} members):\n")
print(f"{'Method':30s} {'Cancellers reached':>19s} {'Share of all':>13s} {'Ranking score':>14s} {'Est. profit':>12s}")
results = {}
for name, scores in risk.items():
    chosen = np.argsort(-scores)[:n_contact]
    reached = truth[chosen].sum()
    # Money: each saved member keeps paying for MONTHS_KEPT months; everyone contacted costs an offer.
    saved_income = (truth[chosen] * prices[chosen]).sum() * OFFER_WORKS * MONTHS_KEPT
    profit = saved_income - n_contact * OFFER_COST
    results[name] = (reached, reached / truth.sum(), roc_auc_score(truth, scores), profit)
    print(f"{name:30s} {reached:>10d} of {truth.sum():<5d} {reached / truth.sum():>12.0%} {results[name][2]:>14.3f} {profit:>11,.0f}$")
print("\n'Ranking score' is ROC AUC: 0.5 is a coin flip, 1.0 is perfect.")

# ---------------------------------------------------------------------------
# 4. Which facts does the best model lean on?
# ---------------------------------------------------------------------------
importance = permutation_importance(trees, test_x, test_y, scoring="roc_auc", n_repeats=5, random_state=0)
order = np.argsort(importance.importances_mean)
print("\nWhat the boosted trees rely on (how much the ranking score drops if that column is scrambled):")
for i in order[::-1]:
    print(f"  {table.columns[i]:28s} {importance.importances_mean[i]:+.3f}")

# ---------------------------------------------------------------------------
# 5. Are its percentages honest? (When it says 30%, do about 30% cancel?)
# ---------------------------------------------------------------------------
predicted = risk["Boosted trees"]
bins = np.quantile(predicted, np.linspace(0, 1, 11))
group = np.clip(np.digitize(predicted, bins[1:-1]), 0, 9)
said = [predicted[group == g].mean() for g in range(10)]
happened = [truth[group == g].mean() for g in range(10)]

# ---------------------------------------------------------------------------
# 6. Pictures.
# ---------------------------------------------------------------------------
colours = {"At random": "#9a948a", "Simple rule: fewest washes": "#d9a441", "Logistic regression": "#2a6fdb", "Boosted trees": "#c0392b"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

share = np.arange(1, len(truth) + 1) / len(truth)
for name, scores in risk.items():
    found = np.cumsum(truth[np.argsort(-scores)]) / truth.sum()
    axes[0].plot(share * 100, found * 100, label=name, color=colours[name], lw=2)
axes[0].axvline(CAN_CONTACT * 100, color="black", ls=":", lw=1)
axes[0].set_xlabel("% of members contacted (riskiest first)")
axes[0].set_ylabel("% of all cancellers reached")
axes[0].set_title("How many cancellers you reach")
axes[0].legend(loc="lower right")

axes[1].barh([table.columns[i].replace("_", " ") for i in order], importance.importances_mean[order], color="#c0392b")
axes[1].set_title("What the model relies on")
axes[1].set_xlabel("drop in ranking score when scrambled")

axes[2].plot([0, max(said)], [0, max(said)], color="black", ls=":", lw=1, label="perfectly honest")
axes[2].plot(said, happened, "o-", color="#c0392b", label="boosted trees")
axes[2].set_xlabel("chance of cancelling the model gave")
axes[2].set_ylabel("share who really cancelled")
axes[2].set_title("Are its percentages honest?")
axes[2].legend()

fig.tight_layout()
fig.savefig(here / "result.png", dpi=110)
print("\nSaved result.png")
