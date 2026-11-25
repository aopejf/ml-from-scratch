"""
Loan default predictor: who is likely to miss their next payment, and who should the bank keep lending to?

The model is a small neural network built on this repository's own parts:
    2_autograd/engine.py   works out the adjustments (backpropagation), written by hand
    2_autograd/nn.py       the neurons and the loss
    1_optimizers/optim.py  Adam, the rule for how far to move each number

Nothing here calls a machine-learning library to do the learning. scikit-learn is used
only in analysis.py, as a referee: two standard models to compare against.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "2_autograd"))
from engine import Value  # noqa: E402
import nn  # noqa: E402  (this also brings in Adam from 1_optimizers)

# What the model is allowed to look at: only how the customer has handled credit.
# Sex, age, marital status and education are in the data and are deliberately left out.
# Lending law in most countries bans deciding on the first three, and education can stand in for them.
FEATURES = {
    "late_now": "Months behind on the latest bill",
    "late_before": "Months behind the month before",
    "late_worst": "Worst lateness in the four months before that",
    "months_late": "How many of the last six months had a late payment",
    "used": "Share of the credit limit in use",
    "paid_share": "Share of the last bill that was paid",
    "paid_share_usual": "Share of the bill usually paid, over five months",
    "limit": "Size of the credit limit",
    "balance_change": "Change in the balance over five months, as a share of the limit",
    "dormant": "Card not used last month",
}
LEFT_OUT = ["sex", "age", "marriage", "education"]
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]  # newest first

# The lending sums. All are assumptions, shown on the page and easy to change.
MARGIN = 0.15  # what a good customer earns the bank in a year, as a share of their balance
LOSS_IF_DEFAULT = 0.75  # the share of the balance lost when a customer defaults (card debt has nothing behind it)


def load():
    return pd.read_csv(here / "data" / "loans.csv")


def build_features(loans):
    """Turn the raw monthly columns into the ten figures the model reads."""
    late = np.column_stack([loans[f"late_{m}"].clip(lower=0) for m in MONTHS])  # 0 = on time
    bills = np.column_stack([loans[f"bill_{m}"] for m in MONTHS]).astype(float)
    paid = np.column_stack([loans[f"paid_{m}"] for m in MONTHS]).astype(float)
    limit = loans.credit_limit.to_numpy(float)
    # The payment made in a month goes against the bill from the month before.
    owed, settled = bills[:, 1:], paid[:, :-1]
    share = np.where(owed > 0, np.clip(settled / np.where(owed > 0, owed, 1), 0, 1), 1.0)
    table = pd.DataFrame({
        "late_now": np.minimum(late[:, 0], 4),
        "late_before": np.minimum(late[:, 1], 4),
        "late_worst": np.minimum(late[:, 2:].max(axis=1), 4),
        "months_late": (late > 0).sum(axis=1),
        "used": np.clip(bills[:, 0] / limit, 0, 1.5),
        "paid_share": share[:, 0],
        "paid_share_usual": share.mean(axis=1),
        "limit": np.log10(limit),
        "balance_change": np.clip((bills[:, 0] - bills[:, -1]) / limit, -1, 1),
        "dormant": (loans.late_sep == -2).astype(float),
    }, index=loans.index)
    return table[list(FEATURES)]


def split(loans, seed=0):
    """60% to learn from, 20% to choose settings on, 20% locked away for the final score."""
    order = np.random.default_rng(seed).permutation(len(loans))
    a, b = int(0.6 * len(loans)), int(0.8 * len(loans))
    return order[:a], order[a:b], order[b:]


def scaler(table):
    """Centre and scale each figure using the training customers only."""
    return dict(mean=table.mean().to_numpy(), spread=table.std().to_numpy())


def scale(table, s):
    return (table.to_numpy(float) - s["mean"]) / s["spread"]


# ---------------------------------------------------------------------------
# Learning, on the hand-built engine
# ---------------------------------------------------------------------------
def train(x, y, layers, steps, batch=64, learning_rate=0.01, seed=0, report=None):
    """
    Each step: pick a batch of customers, let the network guess, measure how wrong it was,
    have the engine work out which way to nudge every weight, and let Adam do the nudging.
    """
    model = nn.Network(x.shape[1], layers, seed=seed)
    params = model.parameters()
    optimizer = nn.Adam(learning_rate)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        rows = rng.integers(0, len(x), batch)
        guesses = [model([float(v) for v in x[r]]) for r in rows]
        loss = nn.cross_entropy(guesses, [float(y[r]) for r in rows])
        nn.zero_grads(params)
        loss.backward()
        nn.optimizer_step(optimizer, params)
        losses.append(loss.data)
        if report and (step + 1) % report == 0:
            print(f"  step {step + 1:>5}: loss {np.mean(losses[-report:]):.4f}", flush=True)
    return model, losses


def weights_of(model):
    """The learned numbers as plain lists, so they can be saved and used without the engine."""
    return [dict(weights=[[w.data for w in n.weights] for n in layer.neurons], bias=[n.bias.data for n in layer.neurons],
                 squash=layer.neurons[0].squash) for layer in model.layers]


def predict(weights, x):
    """The same sums as the network, done on every customer at once."""
    for layer in weights:
        x = x @ np.array(layer["weights"]).T + np.array(layer["bias"])
        x = np.tanh(x) if layer["squash"] == "tanh" else 1.0 / (1.0 + np.exp(-x))
    return x[:, 0]


def network_from(weights):
    """Rebuild an engine network from saved weights."""
    model = nn.Network(len(weights[0]["weights"][0]), [len(layer["bias"]) for layer in weights])
    for layer, saved in zip(model.layers, weights):
        for neuron, w, b in zip(layer.neurons, saved["weights"], saved["bias"]):
            for value, number in zip(neuron.weights, w):
                value.data = number
            neuron.bias.data = b
    return model


def reasons(weights, row, steps=20):
    """
    Why did this customer get this score? Start from the average customer (every input at 0) and
    walk in a straight line to this one. At each point on the walk, run the engine and ask for the
    slope of the risk against every input: the same backpropagation used for learning, pointed at
    the inputs. Average slope x distance walked is how much each figure moved the risk.
    The pieces add up to exactly (this customer's risk - the average customer's risk).
    The official name is integrated gradients. Lenders call the output reason codes, and must
    give them to anyone they decline.
    Returns (risk, risk of the average customer, pushes by feature, biggest first).
    """
    model = network_from(weights)
    slopes = np.zeros(len(row))
    for k in range(steps):
        inputs = [Value(float(v) * (k + 0.5) / steps) for v in row]
        model(inputs).backward()
        slopes += [v.grad for v in inputs]
    pushes = dict(zip(FEATURES, (slopes / steps * row).tolist()))
    risk = model([Value(float(v)) for v in row]).data
    average = model([Value(0.0) for _ in row]).data
    return risk, average, dict(sorted(pushes.items(), key=lambda kv: -abs(kv[1])))


# ---------------------------------------------------------------------------
# How good is the ranking?
# ---------------------------------------------------------------------------
def auc(y, score):
    """Pick one defaulter and one good customer at random. How often does the model rank the defaulter as riskier?"""
    ranks = pd.Series(score).rank().to_numpy()
    bad = y == 1
    return float((ranks[bad].sum() - bad.sum() * (bad.sum() + 1) / 2) / (bad.sum() * (~bad).sum()))


def ks(y, score):
    """The widest gap between "share of defaulters below this score" and "share of good customers below it"."""
    order = np.argsort(score)
    bad = np.cumsum(y[order] == 1) / (y == 1).sum()
    good = np.cumsum(y[order] == 0) / (y == 0).sum()
    return float(np.max(np.abs(bad - good)))


def calibration(y, score, bins=10):
    """Sort customers into ten equal groups by predicted risk. Does a "30% risk" group really default about 30% of the time?"""
    groups = pd.qcut(pd.Series(score).rank(method="first"), bins, labels=False)
    table = pd.DataFrame(dict(group=groups, predicted=score, actual=y)).groupby("group").agg(
        predicted=("predicted", "mean"), actual=("actual", "mean"), customers=("actual", "size"))
    return table.reset_index()


# ---------------------------------------------------------------------------
# From a risk score to a lending decision
# ---------------------------------------------------------------------------
def break_even_risk(margin=MARGIN, loss=LOSS_IF_DEFAULT):
    """Lend while the expected gain beats the expected loss: (1 - risk) x margin > risk x loss."""
    return margin / (margin + loss)


def profit(balance, defaulted_or_risk, approved, margin=MARGIN, loss=LOSS_IF_DEFAULT):
    """Money made on the approved customers. Pass real outcomes (0/1) for what happened, or risks for what to expect."""
    each = balance * ((1 - defaulted_or_risk) * margin - defaulted_or_risk * loss)
    return float((each * approved).sum())


def policy_curve(balance, outcome, risk, cutoffs, margin=MARGIN, loss=LOSS_IF_DEFAULT):
    """For each cutoff: lend to everyone whose risk is below it, and see what happens."""
    rows = []
    for c in cutoffs:
        ok = risk < c
        rows.append(dict(cutoff=float(c), approved=float(ok.mean()), profit=profit(balance, outcome, ok, margin, loss),
                         default_rate=float((outcome * ok).sum() / max(ok.sum(), 1)),
                         balance_lent=float((balance * ok).sum())))
    return rows


def stress(risk, multiplier):
    """
    Make every customer riskier, so that the average risk rises by the multiplier, while keeping
    the order of customers the same. Done by adding one number to everyone's log-odds, found by halving the search range.
    """
    target = min(risk.mean() * multiplier, 0.99)
    log_odds = np.log(risk / (1 - risk))
    low, high = 0.0, 10.0
    for _ in range(60):
        middle = (low + high) / 2
        if (1 / (1 + np.exp(-(log_odds + middle)))).mean() < target:
            low = middle
        else:
            high = middle
    return 1 / (1 + np.exp(-(log_odds + (low + high) / 2)))


def save_weights(weights, s, extra):
    payload = dict(features=list(FEATURES), mean=list(s["mean"]), spread=list(s["spread"]), layers=weights, **extra)
    (here / "weights.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_weights():
    saved = json.loads((here / "weights.json").read_text(encoding="utf-8"))
    return saved, dict(mean=np.array(saved["mean"]), spread=np.array(saved["spread"]))
