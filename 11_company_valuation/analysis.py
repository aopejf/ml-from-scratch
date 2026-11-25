"""
Run the whole valuation, print the findings, and save everything the other files use.

    python analysis.py

Writes:
    results.json   every number, for the Excel model, the web page and the tests
    data.js        the same numbers in a form the web page can load
    result.png     the four charts in the README
    and copies the web page to docs/valuation/ so GitHub Pages serves it
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import valuation as v

here = Path(__file__).parent
r = v.run()
h, a, base, capital, market, start = r["history"], r["assumptions"], r["base"], r["capital"], r["market"], r["start"]
name = market["name"]

results = dict(
    note="Real published figures from SEC filings. A worked valuation method, not advice to buy or sell.",
    company=dict(name=name, ticker=market["ticker"], price=market["price"], price_date=market["price_date"],
                 shares_outstanding=market["shares_outstanding"], risk_free_date=market["risk_free_date"], sources=market["sources"]),
    history=json.loads(h.fillna(0).to_json(orient="records")),
    capital=capital, start=start, assumptions=a, base=base,
    sensitivity=r["sensitivity"], scenarios=r["scenarios"], implied=r["implied"], ratios=r["ratios"],
)
(here / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
(here / "data.js").write_text("window.DATA = " + json.dumps(results) + ";\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Print the findings
# ---------------------------------------------------------------------------
bn = lambda x: f"${x / 1e9:,.1f}bn"
last = h.iloc[-1]
print(f"{name} ({market['ticker']}), financial year {int(last.fiscal_year)}: sales {bn(last.revenue)}, operating profit {bn(last.operating_income)} "
      f"({last.margin:.1%} of sales), free cash flow {bn(last.free_cash_flow)}.")
print(f"\n1. THE RETURN INVESTORS REQUIRE\n   Government bonds pay {capital['risk_free_rate']:.2%}. The share moves {capital['beta']:.2f}% when the market moves 1% (beta).\n"
      f"   Shareholders need {capital['cost_of_equity']:.1%}, lenders {capital['cost_of_debt']:.1%}. Blended (WACC): {capital['wacc']:.1%}.")
print(f"\n2. THE FORECAST\n   Sales growth starts at {a['growth_start']:.1%} and fades to {a['terminal_growth']:.1%}. Operating margin {a['margin']:.1%}.")
print("   Year   Sales     Free cash   Worth today")
for row in base["rows"]:
    print(f"   {row['year']}  {bn(row['revenue']):>9}  {bn(row['free_cash_flow']):>9}  {bn(row['present_value']):>9}")
print(f"\n3. THE ANSWER\n   Ten years of cash: {bn(base['forecast_present'])}. Everything after: {bn(base['terminal_present'])} ({base['terminal_share']:.0%} of the total).\n"
      f"   Whole business {bn(base['enterprise_value'])}, less debt {bn(start['debt'])}, plus cash {bn(start['cash'])}: {bn(base['equity_value'])} for shareholders.\n"
      f"   Value per share ${base['value_per_share']:,.0f}. The share price is ${base['price']:,.2f} ({base['gap']:+.0%}).")
print("\n4. HOW SURE?\n   Value per share by discount rate (down) and for-ever growth (across):")
s = r["sensitivity"]
print("          " + "".join(f"{g:>8.1%}" for g in s["terminal_growths"]))
for w, line in zip(s["waccs"], s["value_per_share"]):
    print(f"   {w:>6.1%} " + "".join(f"{x:>8,.0f}" for x in line))
for case, c in r["scenarios"].items():
    print(f"   {case:<11} growth {c['growth_start']:.1%}, margin {c['margin']:.1%}: ${c['value_per_share']:,.0f} a share")
i = r["implied"]
fmt = lambda x: "no value in range" if x is None else f"{x:.1%}"
print(f"\n5. WHAT THE SHARE PRICE ASSUMES\n   To be worth ${base['price']:,.0f}, holding everything else at the base case, you need ONE of:\n"
      f"   sales growth starting at {fmt(i['growth_start'])} (base {a['growth_start']:.1%}), or an operating margin of {fmt(i['margin'])} (base {a['margin']:.1%}),\n"
      f"   or to accept a return of only {fmt(i['wacc'])} a year (base {a['wacc']:.1%}).\n"
      f"   Today the market pays {r['ratios']['price_to_earnings']:.0f} times last year's profit.")

# ---------------------------------------------------------------------------
# Charts for the README
# ---------------------------------------------------------------------------
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

ax = axes[0, 0]
years = [int(y) for y in h.fiscal_year] + [row["year"] for row in base["rows"]]
sales = list(h.revenue / 1e9) + [row["revenue"] / 1e9 for row in base["rows"]]
ax.bar(years[:len(h)], sales[:len(h)], color=BLUE, width=0.7, label="Reported")
ax.bar(years[len(h):], sales[len(h):], color=GREY, width=0.7, label="Forecast")
ax.set_ylabel("Sales, $ billions")
ax.legend(frameon=False, loc="upper left")
ax.set_xticks(years[1::3])
ax.set_title("1. Sales: ten years reported, ten years forecast")

ax = axes[0, 1]
fcf = h.dropna(subset=["free_cash_flow"])
ax.bar(fcf.fiscal_year, fcf.free_cash_flow / 1e9, color=BLUE, width=0.7)
ax.bar([row["year"] for row in base["rows"]], [row["free_cash_flow"] / 1e9 for row in base["rows"]], color=GREY, width=0.7)
ax.plot([row["year"] for row in base["rows"]], [row["present_value"] / 1e9 for row in base["rows"]], "o-", color=ORANGE, ms=4, lw=1.5)
ax.annotate("worth in today's money", (base["rows"][-1]["year"], base["rows"][-1]["present_value"] / 1e9), textcoords="offset points",
            xytext=(0, -16), ha="right", fontsize=9, color=SOFT)
ax.set_ylabel("Free cash flow, $ billions")
ax.set_xticks(years[1::3])
ax.set_title("2. Free cash flow, and what the forecast years are worth today")

ax = axes[1, 0]
parts = [("Ten years\nof cash", base["forecast_present"]), ("Everything\nafter", base["terminal_present"]), ("Less debt", -start["debt"]),
         ("Plus cash", start["cash"]), ("Worth to\nshareholders", base["equity_value"]), ("What the\nmarket pays", r["ratios"]["market_value"])]
level = 0
for k, (label, value) in enumerate(parts):
    if k >= 4:
        ax.bar(k, value / 1e9, color=BLUE if k == 4 else ORANGE, width=0.7)
        top = value
    else:
        ax.bar(k, value / 1e9, bottom=level / 1e9, color=GREY, width=0.7)
        level += value
        top = max(level, level - value)
    ax.text(k, top / 1e9 + 6, bn(abs(value)), ha="center", fontsize=9, color=INK)
ax.set_xticks(range(len(parts)), [p[0] for p in parts], fontsize=9)
ax.set_ylabel("$ billions")
ax.set_title("3. Adding it up: the model against the market")

ax = axes[1, 1]
grid = s["value_per_share"]
image = ax.imshow(grid, cmap="Blues", aspect="auto")
for y, line in enumerate(grid):
    for x, value in enumerate(line):
        middle = y == len(grid) // 2 and x == len(line) // 2
        ax.text(x, y, f"${value:,.0f}", ha="center", va="center", fontsize=10, fontweight="bold" if middle else "normal",
                color="white" if image.norm(value) > 0.55 else INK)
ax.set_xticks(range(len(s["terminal_growths"])), [f"{g:.1%}" for g in s["terminal_growths"]])
ax.set_yticks(range(len(s["waccs"])), [f"{w:.1%}" for w in s["waccs"]])
ax.set_xlabel("Growth for ever after year ten")
ax.set_ylabel("Return investors require (WACC)")
ax.set_title(f"4. Value per share under other assumptions (price today: ${base['price']:,.0f})")
for side in ax.spines.values():
    side.set_visible(False)

fig.suptitle(f"What is {name} worth? A discounted cash flow valuation from its filed accounts", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(here / "result.png", dpi=130)

# The copy the website serves at .../valuation/
site = here.parent / "docs" / "valuation"
site.mkdir(parents=True, exist_ok=True)
if (here / "page.html").exists():
    shutil.copy(here / "page.html", site / "index.html")
    for file in ("data.js", "dcf.js"):
        shutil.copy(here / file, site / file)
print("\nSaved results.json, data.js and result.png, and copied the web page to docs/valuation")
