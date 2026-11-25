"""
Run the whole finance pack on the made-up chain.

    python analysis.py

Prints the findings, and saves:
    results.json     every number, for the dashboard, the Excel model and the memo
    data.js          the same numbers in the form the dashboard page loads
    result.png       four charts for the README
It also copies the dashboard into ../docs/finance, the folder the website is served from.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import finance as f

here = Path(__file__).parent
YEAR = 2025
data = f.load()
m = data["monthly"]

# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
card, card_info = f.scorecard(data, YEAR)
members = f.membership(data, YEAR)
var = f.variance(data, YEAR)
var_by_site = {int(s): f.variance(data, YEAR, site_id=s) for s in data["sites"].site_id}
fc = f.forecast(data)
test = f.backtest(data)
s24 = f.site28(data)

by_year = m.groupby("year").agg(revenue=("revenue", "sum"), profit=("profit", "sum"), cars=("cars", "sum"))
overhead = data["corporate"].assign(year=data["corporate"].month.str[:4].astype(int)).groupby("year").overhead.sum()
by_month = m.groupby("month").agg(revenue=("revenue", "sum"), profit=("profit", "sum"), cars=("cars", "sum"), members=("members_end", "sum"))
fc_month = fc.groupby("month").agg(revenue=("revenue", "sum"), cars=("cars", "sum"), members=("members_end", "sum"))
fc_site = fc.groupby("site_id").agg(revenue=("revenue", "sum"), cars=("cars", "sum"))

chain_labour_share = (m.groupby("month").labour.sum() / m.groupby("month").revenue.sum())
site_monthly = {}
for sid, g in m.groupby("site_id"):
    site_monthly[int(sid)] = dict(
        months=g.month.tolist(), revenue=g.revenue.round().tolist(), profit=g.profit.round().tolist(),
        cars=g.cars.round().tolist(), members=g.members_end.tolist(),
        labour_share=(g.labour / g.revenue).round(4).tolist(),
        chemicals_per_car=(g.chemicals / g.cars).round(4).tolist(),
        churn=(g.cancels / g.members_start.replace(0, np.nan)).round(4).fillna(0).tolist(),
    )

results = dict(
    note="All data is made up. No real company, site or customer.",
    year=YEAR,
    chain=dict(
        by_year={int(y): dict(revenue=float(r.revenue), site_profit=float(r.profit), cars=float(r.cars),
                              head_office=float(overhead[y]), profit_after_head_office=float(r.profit - overhead[y]))
                 for y, r in by_year.iterrows()},
        months=by_month.index.tolist(), revenue=by_month.revenue.round().tolist(), profit=by_month.profit.round().tolist(),
        cars=by_month.cars.round().tolist(), members=by_month.members.tolist(),
        labour_share=chain_labour_share.round(4).tolist(),
    ),
    scorecard=json.loads(card.round(4).to_json(orient="records")),
    scorecard_info=card_info,
    site_monthly=site_monthly,
    membership=members,
    variance=var,
    variance_by_site=var_by_site,
    forecast=dict(months=fc_month.index.tolist(), revenue=fc_month.revenue.round().tolist(), cars=fc_month.cars.round().tolist(),
                  members=fc_month.members.round().tolist(),
                  by_site={int(k): dict(revenue=float(v.revenue), cars=float(v.cars)) for k, v in fc_site.iterrows()},
                  backtest=test),
    site28=s24,
)
text = json.dumps(results, indent=1, default=float)
(here / "results.json").write_text(text, encoding="utf-8")
(here / "data.js").write_text("window.DATA = " + json.dumps(results, default=float) + ";\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Print the findings
# ---------------------------------------------------------------------------
y = results["chain"]["by_year"][YEAR]
print(f"{YEAR}: revenue ${y['revenue'] / 1e6:.1f}m, site profit ${y['site_profit'] / 1e6:.1f}m ({y['site_profit'] / y['revenue']:.0%}), "
      f"after head office ${y['profit_after_head_office'] / 1e6:.1f}m ({y['profit_after_head_office'] / y['revenue']:.0%}).\n")

print("1. SITE SCORECARD (best margin first)")
print(f"{'Site':9s}{'Revenue':>11s}{'Margin':>8s}{'Labour':>8s}{'Chem/car':>10s}{'Churn':>7s}  Flag")
for r in card.itertuples():
    worth = f" (fixing it is worth about ${r.fix_worth_per_year:,.0f} a year)" if r.fix_worth_per_year else ""
    print(f"{r.name:9s}{r.revenue:>11,.0f}{r.margin:>8.0%}{r.labour_share:>8.0%}{r.chemicals_recent:>10.2f}{r.churn:>7.1%}  {r.flags}{worth}")

print(f"\n2. MEMBERSHIP\n   {members['members']:,} members ({members['members'] - members['members_a_year_ago']:+,} in the year), "
      f"paying ${members['fee']:.2f} a month: ${members['monthly_recurring_revenue'] / 1e6:.2f}m a month of recurring revenue.")
print(f"   {members['churn']:.1%} cancel each month. Of new members, {members['retention'][12]:.0%} are still members a year later.")
print(f"   A new member is worth ${members['ltv']:,.0f} (LTV) and costs ${members['cac']:,.0f} in marketing to win (CAC): "
      f"{members['ltv_to_cac']:.0f} to 1, paid back in {members['payback_months']:.1f} months.")

print(f"\n3. BUDGET AGAINST ACTUAL\n   Profit was ${var['actual_profit'] / 1e6:.2f}m against a budget of ${var['budget_profit'] / 1e6:.2f}m: "
      f"${(var['actual_profit'] - var['budget_profit']) / 1e3:+,.0f}k.")
for name, value in var["effects"].items():
    print(f"   {name:24s}{value / 1e3:>+9,.0f}k")

print(f"\n4. FORECAST\n   Tested on 2025 using only 2023-24, on {test['sites_tested']} established sites: monthly cars were off by "
      f"{test['monthly_miss_model']:.1%} on average.\n   The simple rule 'same as this month last year' was off by {test['monthly_miss_same_month_last_year']:.1%}.")
print(f"   2026 forecast: revenue ${fc.revenue.sum() / 1e6:.1f}m ({fc.revenue.sum() / y['revenue'] - 1:+.1%}), {fc.cars.sum() / 1e6:.2f}m cars.")

print(f"\n5. SITE 28\n   Volume model from {s24['model']['sites_used']} established sites: each extra 1,000 cars of daily traffic adds "
      f"{s24['model']['per_1000_traffic']:,.0f} washes a year,\n   each nearby competitor takes away {-s24['model']['per_competitor']:,.0f}.")
print(f"   {'Candidate':20s}{'Cars/yr':>9s}{'NPV low':>11s}{'NPV expected':>14s}{'NPV high':>11s}{'IRR':>7s}{'Payback':>9s}")
for c in s24["candidates"]:
    e = c["cases"]["Expected"]
    payback = f"{e['payback_years']:.1f} yrs" if e["payback_years"] else "never"
    print(f"   {c['candidate']:20s}{c['cars_expected']:>9,.0f}{c['cases']['Low']['npv'] / 1e6:>10.2f}m{e['npv'] / 1e6:>13.2f}m"
          f"{c['cases']['High']['npv'] / 1e6:>10.2f}m{e['irr']:>7.1%}{payback:>9s}")

# ---------------------------------------------------------------------------
# Charts for the README
# ---------------------------------------------------------------------------
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, GREY, GOOD, BAD = "#2a78d6", "#eb6834", "#c3c2b7", "#008300", "#d03b3b"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

ax = axes[0, 0]
c = card.iloc[::-1]
colours = [GREY if "filling" in fl else ORANGE if fl else BLUE for fl in c["flags"]]
ax.barh(c["name"], c.margin * 100, color=colours, height=0.7)
for name, margin, flag in zip(c["name"], c.margin, c["flags"]):
    if flag:
        ax.text(max(margin * 100, 0) + 1, name, flag, va="center", fontsize=8.5, color=SOFT)
ax.axvline(0, color=GREY, lw=1)
ax.set_xlim(-6, 62)
ax.set_xlabel("Profit margin, 2025 (%)")
ax.set_title("1. Site scorecard: who earns what, and what is wrong")
ax.tick_params(axis="y", labelsize=8.5)

ax = axes[0, 1]
steps = [("Budget", var["budget_profit"])] + sorted(var["effects"].items(), key=lambda kv: kv[1]) + [("Actual", var["actual_profit"])]
level, low = 0, None
for i, (name, value) in enumerate(steps):
    if name in ("Budget", "Actual"):
        ax.bar(i, value / 1e6, color=BLUE, width=0.7)
        level = value
        label_y = value
    else:
        ax.bar(i, value / 1e6, bottom=level / 1e6, color=GOOD if value > 0 else BAD, width=0.7)
        level += value
        label_y = max(level, level - value)
    ax.text(i, label_y / 1e6 + 0.03, f"{value / 1e6:.2f}" if name in ("Budget", "Actual") else f"{value / 1e3:+,.0f}k",
            ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(steps)), [n for n, _ in steps], rotation=40, ha="right", fontsize=8.5)
floor = round(min(var["budget_profit"], var["actual_profit"]) / 1e6 - 0.9, 1)
ax.set_ylim(floor, max(var["budget_profit"], var["actual_profit"]) / 1e6 + 0.3)
ax.set_ylabel(f"Site profit, $ millions (axis starts at {floor})")
ax.set_title("2. Why 2025 profit missed the budget")

ax = axes[1, 0]
curve = members["retention"]
ax.plot(list(curve), [v * 100 for v in curve.values()], color=BLUE, lw=2)
for k in (1, 6, 12, 24):
    ax.plot(k, curve[k] * 100, "o", color=BLUE, ms=6)
    ax.annotate(f"{curve[k]:.0%}", (k, curve[k] * 100), textcoords="offset points", xytext=(6, 6), fontsize=9, color=INK)
ax.set_ylim(0, 105)
ax.set_xlabel("Months since joining")
ax.set_ylabel("Still a member (%)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_title("3. Membership: how long new members stay")

ax = axes[1, 1]
names = [c["candidate"] for c in s24["candidates"]]
for i, cnd in enumerate(s24["candidates"]):
    lo, mid, hi = (cnd["cases"][k]["npv"] / 1e6 for k in ("Low", "Expected", "High"))
    ax.plot([lo, hi], [i, i], color=GREY, lw=6, solid_capstyle="round")
    ax.plot(mid, i, "o", color=BLUE if mid > 0 else BAD, ms=11)
    ax.text(mid, i + 0.22, f"{"+" if mid >= 0 else "-"}${abs(mid):.1f}m, {cnd['cases']['Expected']['irr']:.0%} a year", ha="center", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(3), names)
ax.set_ylim(-0.6, 2.6)
ax.invert_yaxis()
ax.set_xlabel("Value created over 10 years, $ millions (NPV). Dot: expected. Bar: low to high case.")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_title("4. Site 28: three locations compared")

fig.suptitle("A made-up 27-site car wash chain: the finance pack", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(here / "result.png", dpi=130)

# The copy the website serves at .../finance/
site = here.parent / "docs" / "finance"
site.mkdir(parents=True, exist_ok=True)
shutil.copy(here / "dashboard.html", site / "index.html")
for name in ("data.js", "site28.js"):
    shutil.copy(here / name, site / name)
print("\nSaved results.json, data.js and result.png, and copied the dashboard to docs/finance")
