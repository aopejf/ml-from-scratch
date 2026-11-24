"""
A one-page dashboard for a handful of stocks.

    python dashboard.py                      the default six
    python dashboard.py NVDA KO WMT TSLA     your own choice

It downloads real daily prices from Yahoo Finance and answers, on one page:
  - How much did each one grow?                (returns)
  - How bumpy was the ride?                    (volatility)
  - How bad did it get at its worst?           (drawdown)
  - Do they move together?                     (correlation)
  - Was the bumpiness worth it?                (risk versus return)

It also builds an "equal mix": the same amount of money in each stock,
topped back up to equal every day, to show what spreading money out does.

This describes the past. It is not a forecast and not investment advice.

It saves dashboard.png and prints the summary table.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import MIX, compute, load_prices

here = Path(__file__).parent
TICKERS = [t.upper() for t in sys.argv[1:]] or ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "SPY"]
START, END = "2021-11-22", "2026-11-21"  # fixed dates, so the numbers can be reproduced

# ---------------------------------------------------------------------------
# 1. Prices and numbers. The calculations are in metrics.py.
# ---------------------------------------------------------------------------
prices = load_prices(TICKERS, START, END)
print(f"Prices: {len(prices):,} trading days, {prices.index[0].date()} to {prices.index[-1].date()}, for {', '.join(TICKERS)}\n")
numbers = compute(prices)
daily, growth, yearly_return, volatility = numbers["daily"], numbers["growth"], numbers["yearly_return"], numbers["volatility"]
drawdown, correlation, rolling_volatility, summary = numbers["drawdown"], numbers["correlation"], numbers["rolling_volatility"], numbers["summary"]

as_text = summary.copy()
as_text["$1 became"] = summary["$1 became"].map("${:.2f}".format)
as_text["Return / volatility"] = summary["Return / volatility"].map("{:.2f}".format)
for column in ["Return per year", "Volatility", "Worst drop", "Worst day", "Best day"]:
    as_text[column] = summary[column].map("{:+.1%}".format if column != "Volatility" else "{:.1%}".format)
print(as_text.to_string())

average_volatility = volatility[TICKERS].mean()
print(f"\nSpreading out: the stocks' average volatility is {average_volatility:.1%}; the equal mix's is {volatility[MIX]:.1%}.")
print(f"Average correlation between pairs: {correlation.values[np.triu_indices(len(TICKERS), 1)].mean():.2f}")

# ---------------------------------------------------------------------------
# 3. The dashboard.
# ---------------------------------------------------------------------------
palette = ["#2a6fdb", "#c0392b", "#2e9e6b", "#d9a441", "#7d4fc4", "#3aa7b8", "#d0679d", "#6b7280"]
colour = {name: palette[i % len(palette)] for i, name in enumerate(TICKERS)}
colour[MIX] = "black"
style = lambda name: dict(color=colour[name], lw=2.4 if name == MIX else 1.3, ls="--" if name == MIX else "-")

fig = plt.figure(figsize=(18, 13.5))
grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.62], hspace=0.32, wspace=0.22)
fig.suptitle(f"Portfolio dashboard: {', '.join(TICKERS)}   ({prices.index[0].date()} to {prices.index[-1].date()})", fontsize=16, fontweight="bold", y=0.975)

ax = fig.add_subplot(grid[0, :2])
for name in growth:
    ax.plot(growth[name], label=name, **style(name))
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4], ["$0.50", "$1", "$2", "$4"])
ax.minorticks_off()
ax.set_title("Growth of $1 (dividends included)")
ax.legend(ncol=4, frameon=False)
ax.grid(alpha=0.25)

ax = fig.add_subplot(grid[0, 2])
for name in summary.index:
    ax.scatter(volatility[name], yearly_return[name], s=130, color=colour[name], marker="D" if name == MIX else "o", zorder=3)
    ax.annotate(name, (volatility[name], yearly_return[name]), xytext=(8, 5), textcoords="offset points", fontsize=10)
ax.axhline(0, color="black", lw=0.6)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.margins(0.18)
ax.set_xlabel("volatility (bumpier to the right)")
ax.set_ylabel("return per year")
ax.set_title("Risk versus return (up and left is better)")
ax.grid(alpha=0.25)

ax = fig.add_subplot(grid[1, 0])
for name in drawdown:
    ax.plot(drawdown[name], **style(name))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Drawdown: how far below its previous peak")
ax.grid(alpha=0.25)

ax = fig.add_subplot(grid[1, 1])
for name in rolling_volatility:
    ax.plot(rolling_volatility[name], **style(name))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Volatility over the previous 3 months")
ax.grid(alpha=0.25)

ax = fig.add_subplot(grid[1, 2])
image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(TICKERS)), TICKERS)
ax.set_yticks(range(len(TICKERS)), TICKERS)
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        value = correlation.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.6 else "black")
ax.set_title("Correlation of daily moves (1 = always together)")
fig.colorbar(image, ax=ax, fraction=0.046)

ax = fig.add_subplot(grid[2, :])
ax.axis("off")
table = ax.table(cellText=as_text.values, rowLabels=as_text.index, colLabels=as_text.columns, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)
for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#d0d0d0")
    if row == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#f0f0f0")
ax.set_title("Summary, best return-for-risk first.  Describes the past; not a forecast.", fontsize=11)

fig.savefig(here / "dashboard.png", dpi=100, bbox_inches="tight")
print("\nSaved dashboard.png")
