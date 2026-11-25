"""
Score the loan model, run the recession stress test, print the findings, and save everything the other files use.

    python analysis.py

Needs weights.json (from train.py) and data/ (from fetch_data.py). Writes:
    results.json   every number, for the README, the web page and the tests
    data.js        what the web page loads
    result.png     the four charts in the README
    and copies the web page to docs/loans/ so GitHub Pages serves it
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import loan
import recession

here = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. The model, on customers it has never seen
# ---------------------------------------------------------------------------
loans = loan.load()
table = loan.build_features(loans)
learn, choose, test = loan.split(loans)
saved, s = loan.load_weights()
x, y = loan.scale(table, s), loans.defaulted.to_numpy()
risk = np.round(loan.predict(saved["layers"], x), 6)  # six decimal places, the same figures the web page is given
balance = loans.bill_sep.clip(lower=0).to_numpy(float)

one_neuron = loan.predict(saved["candidates"]["one neuron"]["layers"], x)
referee_simple = LogisticRegression(max_iter=1000).fit(x[learn], y[learn]).predict_proba(x[test])[:, 1]
referee_strong = HistGradientBoostingClassifier(random_state=0).fit(x[learn], y[learn]).predict_proba(x[test])[:, 1]
everything = loans.drop(columns=["id", "defaulted"]).to_numpy(float)  # every raw column, including sex, age, marriage and education
referee_all = HistGradientBoostingClassifier(random_state=0).fit(everything[learn], y[learn]).predict_proba(everything[test])[:, 1]

models = {
    "This project: network on the hand-built engine": loan.auc(y[test], risk[test]),
    "This project: one neuron on the hand-built engine": loan.auc(y[test], one_neuron[test]),
    "scikit-learn logistic regression": loan.auc(y[test], referee_simple),
    "scikit-learn gradient boosting": loan.auc(y[test], referee_strong),
    "Gradient boosting, also given sex, age, marriage, education": loan.auc(y[test], referee_all),
}
calibration = loan.calibration(y[test], risk[test])

# ---------------------------------------------------------------------------
# 2. From a score to a lending decision
# ---------------------------------------------------------------------------
line = loan.break_even_risk()
cutoffs = np.round(np.arange(0.02, 1.0001, 0.02), 2)
normal_curve = loan.policy_curve(balance[test], y[test], risk[test], cutoffs)  # what really happened to these customers
approved = risk[test] < line
policy = dict(
    break_even_risk=line, margin=loan.MARGIN, loss_if_default=loan.LOSS_IF_DEFAULT, customers=int(len(test)),
    balance=float(balance[test].sum()), default_rate_all=float(y[test].mean()),
    lend_to_all=loan.profit(balance[test], y[test], np.ones(len(test))),
    with_model=loan.profit(balance[test], y[test], approved), approved=float(approved.mean()),
    default_rate_approved=float(y[test][approved].mean()), default_rate_declined=float(y[test][~approved].mean()),
    best_possible=loan.profit(balance[test], y[test], y[test] == 0),  # a crystal ball: lend only to those who paid
)

# ---------------------------------------------------------------------------
# 3. The recession detector, and what a recession does to card losses
# ---------------------------------------------------------------------------
economy = pd.read_csv(here / "data" / "economy.csv", index_col="month")
card_losses = pd.read_csv(here / "data" / "card_losses.csv", index_col="quarter")
moods, months = recession.run(economy)
marks = dict(hindsight=recession.score(months, "hindsight"), live=recession.score(months, "live"))
by_mood, loss_table = recession.losses_by_mood(months, card_losses)
called = months.hindsight > 0.5
spells = [dict(start=a, end=b) for a, b in zip(months.index[called & ~called.shift(1, fill_value=False)],
                                                months.index[called & ~called.shift(-1, fill_value=False)])]
official = months.official == 1
official_spells = [dict(start=a, end=b) for a, b in zip(months.index[official & ~official.shift(1, fill_value=False)],
                                                         months.index[official & ~official.shift(-1, fill_value=False)])]
live_now = float(months.live.iloc[-1])

# ---------------------------------------------------------------------------
# 4. The stress test: the same customers, in a recession
# ---------------------------------------------------------------------------
multiplier = by_mood["multiplier"]
hard_risk = np.full(len(risk), np.nan)
hard_risk[test] = loan.stress(risk[test], multiplier)  # every test customer riskier, same order
# In a recession the rule is the same (lend while the expected gain beats the expected loss), but the risks are higher.
# So the cutoff on today's score has to fall. This is the fallback cutoff.
fallback_cutoff = float(risk[test][hard_risk[test] >= line].min())  # the safest customer who no longer clears the line
keeps = risk[test] < fallback_cutoff
expected = lambda r, ok: loan.profit(balance[test], r, ok)
stress_test = dict(
    multiplier=multiplier, calm_written_off=by_mood["calm"], recession_written_off=by_mood["stressed"],
    average_risk_normal=float(risk[test].mean()), average_risk_recession=float(hard_risk[test].mean()),
    normal_cutoff=line, fallback_cutoff=fallback_cutoff, approved_normal=float(approved.mean()), approved_fallback=float(keeps.mean()),
    expected_normal=expected(risk[test], approved),
    recession_without_fallback=expected(hard_risk[test], approved),
    recession_with_fallback=expected(hard_risk[test], keeps),
    recession_lend_to_all=expected(hard_risk[test], np.ones(len(test))),
    expected_loss_normal=float((balance[test] * risk[test] * loan.LOSS_IF_DEFAULT * approved).sum()),
    expected_loss_recession=float((balance[test] * hard_risk[test] * loan.LOSS_IF_DEFAULT * approved).sum()),
    live_reading=live_now, cutoff_in_use="fallback" if live_now > 0.5 else "normal", latest_month=months.index[-1],
)
stress_test["fallback_saves"] = stress_test["recession_with_fallback"] - stress_test["recession_without_fallback"]
recession_curve = [dict(cutoff=float(c), approved=float((risk[test] < c).mean()), profit=expected(hard_risk[test], risk[test] < c)) for c in cutoffs]
normal_expected_curve = [dict(cutoff=float(c), profit=expected(risk[test], risk[test] < c)) for c in cutoffs]

# ---------------------------------------------------------------------------
# 5. Three customers, and why each got their score
# ---------------------------------------------------------------------------
examples = []
for label, target in (("Low risk", 0.05), ("On the line", line), ("High risk", 0.60)):
    i = test[np.argmin(np.abs(risk[test] - target))]
    score, average, pushes = loan.reasons(saved["layers"], x[i])
    examples.append(dict(label=label, risk=score, average_customer_risk=average, recession_risk=float(hard_risk[i]), defaulted=int(y[i]), balance=float(balance[i]),
                         decision="Lend" if score < line else "Decline",
                         facts={name: float(table.iloc[i][name]) for name in loan.FEATURES},
                         reasons=[dict(feature=name, label=loan.FEATURES[name], push=float(p)) for name, p in pushes.items()]))

results = dict(
    note="Real, public data: 30,000 credit card customers in Taiwan in 2005 (UCI), and US economic figures (FRED). A worked method, not a production credit model.",
    data=dict(customers=int(len(loans)), default_rate=float(y.mean()), learn=int(len(learn)), choose=int(len(choose)), test=int(len(test)),
              features=loan.FEATURES, left_out=loan.LEFT_OUT),
    model=dict(chosen=saved["chosen"], steps=saved["steps"], numbers=sum(len(l["bias"]) + sum(len(w) for w in l["weights"]) for l in saved["layers"]),
               auc=models["This project: network on the hand-built engine"], gini=2 * models["This project: network on the hand-built engine"] - 1,
               ks=loan.ks(y[test], risk[test]), comparison=models, seconds={n: c["seconds"] for n, c in saved["candidates"].items()},
               loss_curve={n: c["loss_curve"] for n, c in saved["candidates"].items()}),
    calibration=json.loads(calibration.to_json(orient="records")),
    policy=policy, normal_curve=normal_curve, normal_expected_curve=normal_expected_curve, recession_curve=recession_curve,
    recession=dict(features=recession.FEATURES, moods=recession.MOODS, typical=moods["mean"].tolist(),
                   stay=np.diag(moods["stay"]).tolist(), marks=marks, spells=spells, official_spells=official_spells,
                   first_month=months.index[0], last_month=months.index[-1]),
    stress=stress_test, examples=examples,
)
(here / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

# The web page also gets each test customer (risk, balance, what happened) and the monthly recession readings.
page = dict(results, customers=dict(risk=np.round(risk[test], 6).tolist(), balance=balance[test].astype(int).tolist(), defaulted=y[test].tolist()),
            months=dict(month=[m[:7] for m in months.index], hindsight=np.round(months.hindsight, 3).tolist(), live=np.round(months.live, 3).tolist()),
            losses=dict(quarter=[str(q.date())[:7] for q in loss_table.index], written_off=loss_table.written_off.tolist()))
(here / "data.js").write_text("window.DATA = " + json.dumps(page) + ";\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Print the findings
# ---------------------------------------------------------------------------
m = lambda v: f"NT${v / 1e6:,.1f}m"
print(f"{len(loans):,} real credit card customers, {y.mean():.1%} missed the next payment. Scored on {len(test):,} the model never saw.")
print("\n1. HOW WELL DOES IT RANK CUSTOMERS? (AUC: 0.5 is a coin flip, 1.0 is perfect)")
for name, score in models.items():
    print(f"   {score:.3f}  {name}")
print(f"   Gini {results['model']['gini']:.2f}, KS {results['model']['ks']:.2f}. The network has {results['model']['numbers']} adjustable numbers.")
print("\n2. ARE ITS PERCENTAGES HONEST? Ten equal groups, safest to riskiest: predicted against actual")
print("   " + "  ".join(f"{r.predicted:.0%}/{r.actual:.0%}" for r in calibration.itertuples()))
print(f"\n3. WHO TO LEND TO\n   A good customer earns {loan.MARGIN:.0%} of their balance a year; a default loses {loan.LOSS_IF_DEFAULT:.0%}. Break-even risk: {line:.1%}.")
print(f"   Lend to everyone: {m(policy['lend_to_all'])}. Lend below the line ({policy['approved']:.0%} of customers): {m(policy['with_model'])}. A crystal ball: {m(policy['best_possible'])}.")
print(f"   Approved customers defaulted {policy['default_rate_approved']:.1%} of the time; declined ones {policy['default_rate_declined']:.1%}.")
print(f"\n4. THE RECESSION DETECTOR (never shown the official dates)\n   Caught {marks['hindsight']['recessions_caught']} of {marks['hindsight']['official_recessions']} official recessions since {months.index[0][:4]}, "
      f"with {marks['hindsight']['false_alarms']} false alarm(s). Agrees with the official label in {marks['hindsight']['months_agreeing']:.0%} of months.")
print(f"   Using only what was known at the time: {marks['live']['recessions_caught']} of {marks['live']['official_recessions']} caught, {marks['live']['false_alarms']} false alarm(s).")
print(f"   Latest month ({months.index[-1][:7]}): {live_now:.0%} chance the economy is shrinking. Cutoff in use: {stress_test['cutoff_in_use']}.")
print(f"\n5. THE STRESS TEST\n   US banks wrote off {by_mood['calm']:.1f}% of card debt a year in calm times and {by_mood['stressed']:.1f}% in and just after recessions: {multiplier:.2f} times as much.")
print(f"   Apply that here: average risk {stress_test['average_risk_normal']:.1%} becomes {stress_test['average_risk_recession']:.1%}.")
print(f"   Expected profit in normal times {m(stress_test['expected_normal'])}. In a recession, same lending: {m(stress_test['recession_without_fallback'])}.")
print(f"   Fall back to a cutoff of {fallback_cutoff:.1%} ({stress_test['approved_fallback']:.0%} approved, down from {stress_test['approved_normal']:.0%}): "
      f"{m(stress_test['recession_with_fallback'])}, which saves {m(stress_test['fallback_saves'])}.")
print("\n6. WHY? THREE CUSTOMERS")
for e in examples:
    print(f"   {e['label']}: risk {e['risk']:.1%} -> {e['decision']} (the average customer scores {e['average_customer_risk']:.1%}). "
          + "; ".join(f"{r['label']} ({r['push']:+.1%})" for r in e["reasons"][:3]))

# ---------------------------------------------------------------------------
# Charts for the README
# ---------------------------------------------------------------------------
INK, QUIET, RULE = "#111111", "#5c5c5c", "#e4e4e4"
BLUE, RED, GREEN, GREY = "#0b35d6", "#c00000", "#0b7a53", "#8c8c8c"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": QUIET, "axes.labelsize": 9.5,
                     "xtick.color": QUIET, "ytick.color": QUIET, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "ytick.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": QUIET,
                     "lines.solid_capstyle": "round"})


def heading(ax, words):
    """A chart title set like a line of the write-up: serif, plain weight, flush left."""
    ax.set_title(words, loc="left", pad=12, fontfamily=["Georgia", "DejaVu Serif"], fontsize=12.5, fontweight="normal", color=INK)


fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))

ax = axes[0, 0]
names = ["My network, on my own engine", "A single neuron, on my own engine", "scikit-learn logistic regression", "scikit-learn gradient boosting",
         "Boosting, also told sex, age,\nmarital status and education"]
scores = list(models.values())
rows = list(range(len(scores)))[::-1]
ax.hlines(rows, 0.74, scores, color=RULE, lw=1)
ax.scatter(scores[:2], rows[:2], color=INK, s=52, zorder=3)
ax.scatter(scores[2:], rows[2:], facecolor="white", edgecolor=INK, s=52, linewidth=1.3, zorder=3)
for y, v in zip(rows, scores):
    ax.text(v + 0.0015, y, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(rows, names)
ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK)
ax.set_xlim(0.74, 0.782)
ax.set_ylim(-0.6, len(scores) - 0.4)
ax.grid(False)
ax.set_xlabel("How often it ranks a defaulter above a good customer (AUC)")
heading(ax, "Mine against the standard tools, on customers neither had seen")

ax = axes[0, 1]
ax.plot([0, 0.75], [0, 0.75], color=GREY, lw=1.6, ls="--")
ax.plot(calibration.predicted, calibration.actual, "o-", color=INK, ms=5.5, lw=2)
ax.text(0.56, 0.52, "on this line a predicted\n30% really is 30%", fontsize=9, color=QUIET, va="top")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_xlabel("Risk the model predicted, customers in ten equal groups")
ax.set_ylabel("Share who defaulted")
heading(ax, "Its percentages mean what they say")

ax = axes[1, 0]
ax.axhline(0, color=INK, lw=0.8)
ax.plot([r["cutoff"] for r in normal_curve], [r["profit"] / 1e6 for r in normal_curve], color=INK, lw=2, label="Calm, which is what happened")
ax.plot([r["cutoff"] for r in recession_curve], [r["profit"] / 1e6 for r in recession_curve], color=RED, lw=2, label="Recession, as a stress test")
low, top = ax.get_ylim()
ax.vlines(line, low, top, color=INK, lw=0.8, ls=":")
ax.vlines(fallback_cutoff, low, top, color=RED, lw=0.8, ls=":")
ax.text(line + 0.012, top, f"calm rule: below {line:.1%}", fontsize=9, color=INK, va="top")
ax.text(fallback_cutoff - 0.012, top, f"recession rule:\nbelow {fallback_cutoff:.1%}", fontsize=9, color=RED, va="top", ha="right")
ax.set_ylim(low, top)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.yaxis.set_major_formatter(lambda v, _: f"({-v:,.0f})" if v < 0 else f"{v:,.0f}")
ax.set_xlabel("Lend to everyone whose risk is below")
ax.set_ylabel("Profit, NT$ millions, losses in (brackets)")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.93))
heading(ax, "Where to draw the line, and where it moves")

ax = axes[1, 1]
dates = pd.to_datetime(months.index)
for spell in official_spells:
    ax.axvspan(pd.Timestamp(spell["start"]), pd.Timestamp(spell["end"]) + pd.offsets.MonthEnd(1), color="#d9d9d9", lw=0)
ax.plot(dates, months.hindsight, color=INK, lw=1.2)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_ylabel("Chance the economy is shrinking")
ax.set_xlabel("Line: my model, which was never shown a recession. Grey: the official ones.")
heading(ax, "It found the recessions by itself")

fig.tight_layout(h_pad=2.5, w_pad=3)
fig.savefig(here / "result.png", dpi=130, facecolor="white")

site = here.parent / "docs" / "loans"
site.mkdir(parents=True, exist_ok=True)
shutil.copy(here / "page.html", site / "index.html")
for file in ("data.js", "policy.js"):
    shutil.copy(here / file, site / file)
print("\nSaved results.json, data.js and result.png, and copied the web page to docs/loans")
