"""
The same question as churn.py, on real data.

    python telco.py

The made-up car wash list proved the method. This runs it on a real, public
dataset: 7,043 customers of a phone and internet company, published by IBM,
with whether each one left.

Same business question: you can contact 10% of customers. Which 10%?

Every customer is scored by a model that never saw them during training
(5-fold cross-validation), so all 7,043 count as test customers.

Saves result_telco.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

here = Path(__file__).parent
CAN_CONTACT = 0.10

# ---------------------------------------------------------------------------
# 1. Load and tidy. Real data is messier than made-up data.
# ---------------------------------------------------------------------------
customers = pd.read_csv(here / "data" / "telco.csv").drop(columns="customerID")
left = (customers.pop("Churn") == "Yes").astype(int)
# Total charges is blank for 11 brand-new customers who have not been billed yet.
customers["TotalCharges"] = pd.to_numeric(customers["TotalCharges"], errors="coerce").fillna(0)
numbers = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
words = [c for c in customers.columns if c not in numbers]
table = pd.get_dummies(customers, columns=words, drop_first=True, dtype=int)
print(f"Customers: {len(table):,} real ones, {left.mean():.1%} left. {len(customers.columns)} facts about each.\n")

# ---------------------------------------------------------------------------
# 2. Score every customer with a model that did not train on them.
# ---------------------------------------------------------------------------
folds = StratifiedKFold(5, shuffle=True, random_state=0)
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
trees = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0)
rng = np.random.default_rng(0)
monthly = (customers["Contract"] == "Month-to-month").to_numpy()
risk = {
    "At random": rng.random(len(table)),
    "Simple rule: no contract, newest first": monthly * 1000.0 - customers["tenure"].to_numpy(),
    "Logistic regression": cross_val_predict(logistic, table, left, cv=folds, method="predict_proba")[:, 1],
    "Boosted trees": cross_val_predict(trees, table, left, cv=folds, method="predict_proba")[:, 1],
}

# ---------------------------------------------------------------------------
# 3. Score the four methods.
# ---------------------------------------------------------------------------
truth = left.to_numpy()
n_contact = int(len(truth) * CAN_CONTACT)
print(f"If you contact the riskiest {CAN_CONTACT:.0%} ({n_contact} of {len(truth):,} customers):\n")
print(f"{'Method':40s} {'Leavers reached':>17s} {'Hit rate':>9s} {'Ranking score':>14s}")
for name, scores in risk.items():
    reached = truth[np.argsort(-scores)[:n_contact]].sum()
    print(f"{name:40s} {reached:>8d} of {truth.sum():<5d} {reached / n_contact:>8.0%} {roc_auc_score(truth, scores):>14.3f}")
print("\n'Hit rate': of the customers you contacted, the share who really were about to leave.")
print("'Ranking score' is ROC AUC: 0.5 is a coin flip, 1.0 is perfect.")

# ---------------------------------------------------------------------------
# 4. What does the model lean on? Scramble one original fact at a time.
# ---------------------------------------------------------------------------
half = np.arange(len(table)) % 4 == 0  # hold a quarter back for this check
fitted = trees.fit(table[~half], left[~half])
base = roc_auc_score(left[half], fitted.predict_proba(table[half])[:, 1])
importance = {}
for fact in customers.columns:
    columns = [c for c in table.columns if c == fact or c.startswith(fact + "_")]
    drops = []
    for seed in range(5):
        shuffled = table[half].copy()
        shuffled[columns] = shuffled[columns].sample(frac=1, random_state=seed).to_numpy()
        drops.append(base - roc_auc_score(left[half], fitted.predict_proba(shuffled)[:, 1]))
    importance[fact] = np.mean(drops)
importance = pd.Series(importance).sort_values()
print("\nWhat the boosted trees rely on most:")
for fact, drop in importance[::-1][:6].items():
    print(f"  {fact:18s} {drop:+.3f}")

# ---------------------------------------------------------------------------
# 5. Pictures.
# ---------------------------------------------------------------------------
colours = dict(zip(risk, ["#9a948a", "#d9a441", "#2a6fdb", "#c0392b"]))
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

share = np.arange(1, len(truth) + 1) / len(truth)
for name, scores in risk.items():
    axes[0].plot(share * 100, np.cumsum(truth[np.argsort(-scores)]) / truth.sum() * 100, label=name, color=colours[name], lw=2)
axes[0].axvline(CAN_CONTACT * 100, color="black", ls=":", lw=1)
axes[0].set_xlabel("% of customers contacted (riskiest first)")
axes[0].set_ylabel("% of all leavers reached")
axes[0].set_title("Real data: how many leavers you reach")
axes[0].legend(loc="lower right")

axes[1].barh(importance.index[-10:], importance.values[-10:], color="#c0392b")
axes[1].set_title("What the model relies on (top 10)")
axes[1].set_xlabel("drop in ranking score when scrambled")

predicted = risk["Boosted trees"]
group = np.clip(np.digitize(predicted, np.quantile(predicted, np.linspace(0, 1, 11))[1:-1]), 0, 9)
said = [predicted[group == g].mean() for g in range(10)]
happened = [truth[group == g].mean() for g in range(10)]
axes[2].plot([0, 0.8], [0, 0.8], color="black", ls=":", lw=1, label="perfectly honest")
axes[2].plot(said, happened, "o-", color="#c0392b", label="boosted trees")
axes[2].set_xlabel("chance of leaving the model gave")
axes[2].set_ylabel("share who really left")
axes[2].set_title("Are its percentages honest?")
axes[2].legend()

fig.tight_layout()
fig.savefig(here / "result_telco.png", dpi=110)
print("\nSaved result_telco.png")
