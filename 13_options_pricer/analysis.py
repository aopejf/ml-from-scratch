"""
Options pricer: the full run.

    python analysis.py

1. The Greeks from the hand-built autograd engine, checked against the textbook formulas.
2. Three ways to price the same option (formula, tree, imagined futures) and whether they agree.
3. Real option prices: the volatility each one implies, and the "smile" that Black-Scholes cannot explain.
4. Futures imagined by the diffusion model: do they behave like real markets where a bell curve does not?
5. Options priced on those futures: how much of the smile do they explain?

Writes results.json, data.js and result.png, and copies the web page to docs/options.
"""

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import options
import scenarios

here = Path(__file__).parent
market = json.loads((here / "data" / "market.json").read_text())
chain = pd.read_csv(here / "data" / "chain.csv")

spot, rate = market["fund_price"], market["rate"]
years = market["days_to_expiry"] / 365
discount = math.exp(-rate * years)

# ---------------------------------------------------------------------------
# 0. What the market is assuming about dividends, from the prices themselves
# ---------------------------------------------------------------------------
# A call minus a put at the same strike is worth the share (less dividends) minus the strike in today's money.
# That rule (put-call parity) holds whatever you believe about volatility, so it reveals the dividend rate.
chain["mid"] = (chain.bid + chain.ask) / 2
quoted = chain[(chain.bid > 0) & (chain.ask > chain.bid)]
pairs = quoted[quoted.kind == "call"].merge(quoted[quoted.kind == "put"], on="strike", suffixes=("_call", "_put"))
near = pairs[(pairs.strike / spot - 1).abs() < 0.05]
dividend = float(np.median(-np.log((near.mid_call - near.mid_put + near.strike * discount) / spot) / years))
forward = spot * math.exp((rate - dividend) * years)

# ---------------------------------------------------------------------------
# 1. The Greeks: engine against textbook
# ---------------------------------------------------------------------------
at_the_money = float(pairs.strike.iloc[(pairs.strike - spot).abs().argmin()])
example = dict(price=spot, strike=at_the_money, years=years, rate=rate, volatility=0.16, dividend=dividend)
greek_table, worst_gap = {}, 0.0
for kind in ("call", "put"):
    engine, book = options.greeks(**example, kind=kind), options.textbook_greeks(**example, kind=kind)
    greek_table[kind] = dict(value=engine["value"], **{g: dict(engine=engine[g], textbook=book[g]) for g in book})
rng = np.random.default_rng(0)
for _ in range(500):  # 500 random options, to be sure the example is not a lucky one
    case = dict(price=100.0, strike=float(rng.uniform(60, 150)), years=float(rng.uniform(0.02, 3)), rate=float(rng.uniform(0, 0.08)),
                volatility=float(rng.uniform(0.05, 0.8)), dividend=float(rng.uniform(0, 0.05)))
    for kind in ("call", "put"):
        engine, book = options.greeks(**case, kind=kind), options.textbook_greeks(**case, kind=kind)
        worst_gap = max(worst_gap, max(abs(engine[g] - book[g]) for g in book))

# Forty of them are kept, with a small tree each, so tests/check_options.js can hold the web page's sums to the same answers.
spot_checks = []
for _ in range(40):
    case = dict(price=float(rng.uniform(50, 900)), years=float(rng.uniform(0.02, 2)), rate=float(rng.uniform(0, 0.08)),
                volatility=float(rng.uniform(0.08, 0.7)), dividend=float(rng.uniform(0, 0.04)), kind="call" if rng.uniform() < 0.5 else "put")
    case["strike"] = case["price"] * float(rng.uniform(0.7, 1.3))
    spot_checks.append(dict(case=case, engine=options.greeks(**case), tree=options.tree(**case, steps=200),
                            american=options.tree(**case, steps=200, american=True)))

# ---------------------------------------------------------------------------
# 2. Three methods, one option
# ---------------------------------------------------------------------------
formula = {kind: options.value(**example, kind=kind) for kind in ("call", "put")}
tree_steps = [10, 25, 50, 100, 250, 500, 1000, 2000]
tree_curve = [dict(steps=n, value=options.tree(**example, kind="put", steps=n)) for n in tree_steps]
endings = options.bell_curve_endings(200_000, years, example["volatility"])
paths_value, paths_error = options.from_paths(endings, spot, at_the_money, years, rate, dividend, "put")
parity_gap = formula["call"] - formula["put"] - (spot * math.exp(-dividend * years) - at_the_money * discount)
american = {kind: options.tree(**example, kind=kind, steps=1000, american=True) for kind in ("call", "put")}
methods = dict(example=dict(example, kind="put"), formula=formula["put"], tree=tree_curve[-1]["value"], paths=paths_value, paths_error=paths_error,
               paths_used=len(endings), tree_curve=tree_curve, parity_gap=parity_gap, call=formula["call"],
               american_put=american["put"], american_call=american["call"], early_use_worth=american["put"] - formula["put"])

# ---------------------------------------------------------------------------
# 3. Real prices: the volatility each one implies
# ---------------------------------------------------------------------------
solved, how = [], {"newton": 0, "adam": 0, "no answer": 0}
for q in quoted.itertuples():
    vol, method = options.implied_volatility(q.mid, spot, q.strike, years, rate, dividend, q.kind)
    how[method] += 1
    solved.append(dict(kind=q.kind, strike=q.strike, mid=q.mid, bid=q.bid, ask=q.ask, volatility=vol, method=method))
solved = pd.DataFrame(solved)
# The smile uses the option on the cheaper side of each strike (puts below, calls above): those are traded most and priced tightest.
side = np.where(solved.strike < forward, "put", "call")
smile = solved[(solved.kind == side) & solved.volatility.notna() & ((solved.ask - solved.bid) / solved.mid < 0.5)
               & (solved.strike / spot).between(0.75, 1.15)].sort_values("strike").reset_index(drop=True)
market_atm = float(np.interp(forward, smile.strike, smile.volatility))
down_10 = float(np.interp(0.9 * spot, smile.strike, smile.volatility))
up_5 = float(np.interp(1.05 * spot, smile.strike, smile.volatility))

# Today's quotes are well behaved. To test the fallback, make 1,000 awkward options with a known volatility
# (far from the strike, a few days to many years, very calm to very wild), price them, and ask for the volatility back.
hard, hard_how, hard_worst = np.random.default_rng(1), {"newton": 0, "adam": 0, "no answer": 0}, 0.0
for _ in range(1000):
    true_vol, strike, time = float(hard.uniform(0.05, 1.5)), float(100 * math.exp(hard.uniform(-0.7, 0.7))), float(math.exp(hard.uniform(math.log(0.01), math.log(5))))
    kind = "call" if hard.uniform() < 0.5 else "put"
    target = options.value(100.0, strike, time, 0.03, true_vol, 0.01, kind)
    found, method = options.implied_volatility(target, 100.0, strike, time, 0.03, 0.01, kind)
    hard_how[method] += 1
    if method != "no answer":  # "no answer" = the price is all built-in value, so no volatility can be read from it
        hard_worst = max(hard_worst, abs(options.value(100.0, strike, time, 0.03, found, 0.01, kind) - target))

# What Black-Scholes would charge with one volatility for every strike, against what the market charges.
crash_strike = float(smile.strike.iloc[(smile.strike - 0.9 * spot).abs().argmin()])
crash_row = smile[smile.strike == crash_strike].iloc[0]
crash_flat = options.value(spot, crash_strike, years, rate, market_atm, dividend, "put")

# ---------------------------------------------------------------------------
# 4. Imagined futures: real, diffusion, bell curve
# ---------------------------------------------------------------------------
real = scenarios.windows(scenarios.daily_moves())
made, discarded = scenarios.generated_moves(with_count=True)  # broken futures are dropped and counted
typical = float(scenarios.daily_moves().std())
bell = scenarios.bell_curve_moves(len(made), typical, float(scenarios.daily_moves().mean()))  # same average and same swing as history
prints = dict(real=scenarios.fingerprint(real), diffusion=scenarios.fingerprint(made), bell_curve=scenarios.fingerprint(bell))
copied = scenarios.nearest_real(made, real)
_, saved = scenarios.load_model()

# ---------------------------------------------------------------------------
# 5. Options priced on the imagined futures
# ---------------------------------------------------------------------------
strikes = smile.strike[(smile.strike / spot).between(0.80, 1.10)].to_numpy()


def smile_from(moves):
    """Price each strike by averaging over the futures, then ask what volatility Black-Scholes would need to agree."""
    growth = np.exp(moves.sum(axis=1))
    rows = []
    for k in strikes:
        kind = "put" if k < forward else "call"
        value, error = options.from_paths(growth, spot, k, years, rate, dividend, kind)
        vol = [options.implied_volatility(v, spot, k, years, rate, dividend, kind)[0] for v in (value - error, value, value + error)]
        rows.append(dict(strike=float(k), kind=kind, value=value, error=error, low=vol[0], volatility=vol[1], high=vol[2]))
    return pd.DataFrame(rows)


raw = smile_from(made)
model_atm = float(np.interp(forward, raw.strike, raw.volatility))
# History's average volatility is not today's. Stretch every imagined move by one number so the middle of the
# smile sits where the market's does, then compare the shape.
stretch = market_atm / model_atm
for _ in range(3):
    matched = smile_from(made * stretch)
    stretch *= market_atm / float(np.interp(forward, matched.strike, matched.volatility))
matched = smile_from(made * stretch)
bell_smile = smile_from(bell * market_atm / (typical * math.sqrt(scenarios.TRADING_DAYS_A_YEAR)))


def tilt(table):
    """How much dearer the 10%-down strike is than the middle, in volatility points."""
    return float(np.interp(0.9 * spot, table.strike, table.volatility) - np.interp(forward, table.strike, table.volatility))


shape = dict(market=tilt(smile), diffusion=tilt(matched), bell_curve=tilt(bell_smile))
shape["share_explained"] = shape["diffusion"] / shape["market"]
crash_model = float(np.interp(crash_strike, matched.strike, matched.value))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results = dict(
    note="Real, public prices (S&P 500 since 1950, one day of SPY option quotes). A worked method, not advice to trade anything.",
    market=dict(market, dividend=dividend, forward=forward, years=years, quotes=int(len(chain)), usable=int(len(quoted)), smile_points=int(len(smile))),
    greeks=dict(example=example, table=greek_table, worst_gap=worst_gap, options_checked=1000),
    methods=methods,
    spot_checks=spot_checks,
    implied=dict(how=how, hard=dict(how=hard_how, worst_price_gap=hard_worst), atm=market_atm, down_10=down_10, up_5=up_5,
                 crash=dict(strike=crash_strike, market=float(crash_row.mid), flat=crash_flat, times=float(crash_row.mid / crash_flat))),
    scenarios=dict(days=scenarios.DAYS, history_windows=int(len(real)), made=int(len(made)), discarded=discarded, training_steps=int(saved["steps"]),
                   numbers=int(sum(p.numel() for p in scenarios.SequenceNoisePredictor().parameters())),
                   fingerprints=prints, copied=copied, loss_curve=saved["loss_curve"]),
    smile=dict(history_atm=model_atm, stretch=stretch, shape=shape, crash_model=crash_model),
)
(here / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

pick = np.random.default_rng(3)


def lines(moves, n):
    rows = moves[pick.choice(len(moves), n, replace=False)]
    return np.round(100 * np.exp(np.cumsum(np.column_stack([np.zeros(n), rows]), axis=1)), 2).tolist()


def spread_of(moves, bins):
    return (np.histogram(100 * (np.exp(moves.sum(axis=1)) - 1), bins=bins)[0] / len(moves)).round(5).tolist()


bins = np.arange(-40, 41, 2.0)
page = dict(
    results,
    smile=dict(results["smile"],
               market=dict(strike=smile.strike.tolist(), volatility=smile.volatility.round(5).tolist(), kind=smile.kind.tolist(), mid=smile.mid.round(3).tolist()),
               diffusion={c: matched[c].round(5).tolist() for c in ("strike", "low", "volatility", "high")},
               diffusion_raw=dict(strike=raw.strike.tolist(), volatility=raw.volatility.round(5).tolist()),
               bell_curve=dict(strike=bell_smile.strike.tolist(), volatility=bell_smile.volatility.round(5).tolist())),
    paths=dict(real=lines(real, 30), diffusion=lines(made, 30), bell_curve=lines(bell, 30)),
    endings=dict(middle=((bins[:-1] + bins[1:]) / 2).tolist(), real=spread_of(real, bins), diffusion=spread_of(made, bins), bell_curve=spread_of(bell, bins)),
)
(here / "data.js").write_text("window.DATA = " + json.dumps(page) + ";\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
f = prints
print(f"SPY at ${spot:.2f} on {market['price_date']}. Options ending {market['expiry']} ({market['days_to_expiry']} days). Safe rate {rate:.2%}, dividend rate from the prices {dividend:.2%}.")
print("\n1. THE GREEKS, FROM THE AUTOGRAD ENGINE")
for g in ("delta", "gamma", "vega", "theta", "rho"):
    row = greek_table["call"][g]
    print(f"   {g:6} engine {row['engine']:+.8f}   textbook {row['textbook']:+.8f}")
print(f"   Largest gap over 1,000 random options: {worst_gap:.1e}")
print("\n2. THREE WAYS TO PRICE ONE PUT")
print(f"   Formula ${methods['formula']:.4f}. Tree (2,000 steps) ${methods['tree']:.4f}. {len(endings):,} imagined futures ${paths_value:.4f} +/- {paths_error:.4f}.")
print(f"   Call minus put matches the no-arbitrage rule to {abs(parity_gap):.1e}. The right to use the put early is worth ${methods['early_use_worth']:.2f}.")
print("\n3. WHAT THE MARKET IMPLIES")
print(f"   {len(quoted)} quotes solved: {how['newton']} by Newton, {how['adam']} needed the Adam fallback, {how['no answer']} had no answer.")
print(f"   1,000 awkward made-up options: {hard_how['newton']} by Newton, {hard_how['adam']} by the Adam fallback, {hard_how['no answer']} with no volatility to find. Worst price gap {hard_worst:.1e}.")
print(f"   Implied volatility: {down_10:.1%} at 10% below today, {market_atm:.1%} at today's price, {up_5:.1%} at 5% above.")
print(f"   A put at ${crash_strike:.0f} costs ${crash_row.mid:.2f}. One flat volatility says ${crash_flat:.2f}: the market pays {crash_row.mid / crash_flat:.1f} times that.")
print("\n4. IMAGINED FUTURES: real history / diffusion model / bell curve")
for key, label, style in (("yearly_volatility", "Yearly volatility", ".1%"), ("fat_tails", "Fat tails (0 for a bell curve)", ".1f"),
                          ("days_beyond_3", "Days beyond 3 typical moves", ".2%"), ("echo_1_day", "Big days follow big days", ".3f"),
                          ("calm_to_wild", "Wildest stretches / calmest", ".1f"), ("three_month_fall_10", "Three-month falls over 10%", ".1%"), ("three_month_fall_20", "Three-month falls over 20%", ".2%"),
                          ("three_month_worst_1_in_100", "Worst 1-in-100 three months", ".1%"), ("three_month_lean", "Lean to the downside", ".2f")):
    print(f"   {label:32} {format(f['real'][key], style):>8} {format(f['diffusion'][key], style):>8} {format(f['bell_curve'][key], style):>8}")
print(f"   {len(made):,} futures kept, {discarded} thrown away for a day bigger than any in history.")
print(f"   Nearest real stretch to a generated one: {copied['generated_to_nearest_real']:.3f}. Real stretches to each other: {copied['real_to_nearest_real']:.3f}. (Not copies.)")
print("\n5. THE SMILE")
print(f"   Extra volatility at 10% down, over the middle: market {shape['market']:+.1%}, diffusion {shape['diffusion']:+.1%}, bell curve {shape['bell_curve']:+.1%}.")
print(f"   The diffusion model's futures explain {shape['share_explained']:.0%} of the market's tilt.")

# ---------------------------------------------------------------------------
# Chart
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
prices = np.linspace(0.8 * spot, 1.2 * spot, 81)
ax.plot(prices, [options.value(p, at_the_money, years, rate, 0.16, dividend, "call") for p in prices], color=INK, lw=2, label="Worth today")
ax.plot(prices, np.maximum(prices - at_the_money, 0), color=GREY, lw=1.6, ls="--", label="Worth on the last day")
d = greek_table["call"]["delta"]["engine"]
ax.plot([spot - 45, spot + 45], [formula["call"] - 45 * d, formula["call"] + 45 * d], color=BLUE, lw=1.4)
ax.scatter([spot], [formula["call"]], color=BLUE, zorder=3, s=36, edgecolor="white", linewidth=1.4)
ax.annotate(f"at \\${spot:.0f} it gains \\${d:.2f} for each\n\\$1 the share rises (delta)", (spot, formula["call"]), xytext=(-150, 58),
            textcoords="offset points", fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkB=5))
ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
ax.set_xlabel("Share price")
ax.legend(loc="upper left")
heading(ax, f"A call at ${at_the_money:.0f} with {market['days_to_expiry']} days to run")

ax = axes[0, 1]
ax.axhline(methods["formula"], color=GREY, lw=1.6, ls="--", label=f"The formula, ${methods['formula']:.3f}")
ax.plot([r["steps"] for r in tree_curve], [r["value"] for r in tree_curve], color=INK, lw=2, marker="o", ms=4.5, label="A tree, as I give it more steps")
ax.errorbar([3500], [paths_value], yerr=[2 * paths_error], color=GREEN, marker="o", ms=5, capsize=4, lw=1.6, ls="none", label="200,000 imagined futures")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.yaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
ax.set_xlabel("Steps in the tree")
ax.legend(loc="lower right")
heading(ax, "Three ways to price the same put")

ax = axes[1, 0]
middle = (bins[:-1] + bins[1:]) / 2
for moves, colour, dash, width, label in ((bell, GREY, "--", 1.6, "A bell curve, which the formula assumes"), (made, GREEN, "-", 2, "My diffusion model"),
                                          (real, INK, "-", 2, "Real history since 1950")):
    ax.plot(middle, np.maximum(np.histogram(100 * (np.exp(moves.sum(axis=1)) - 1), bins=bins)[0] / len(moves), 1e-5), color=colour, lw=width, ls=dash, label=label)
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.3)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{100 * v:g}%")
ax.set_xlabel("Change over three months")
ax.set_ylabel("Share of three-month stretches")
ax.legend(loc="lower center")
heading(ax, "Where three months can end up")

ax = axes[1, 1]
ax.plot(smile.strike / spot * 100, smile.volatility * 100, color=INK, lw=2, label="The market, from real option prices")
ax.fill_between(matched.strike / spot * 100, matched.low * 100, matched.high * 100, color=GREEN, alpha=0.15, lw=0)
ax.plot(matched.strike / spot * 100, matched.volatility * 100, color=GREEN, lw=2, label="Futures from my diffusion model")
ax.plot(bell_smile.strike / spot * 100, bell_smile.volatility * 100, color=GREY, lw=1.6, ls="--", label="Bell-curve futures")
ax.set_xlim(79, 111)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("Strike, as a share of today's price")
ax.set_ylabel("Volatility the price implies")
ax.legend(loc="upper right")
heading(ax, "The formula says this line should be flat")

fig.tight_layout(h_pad=2.5, w_pad=3)
fig.savefig(here / "result.png", dpi=130, facecolor="white")

site = here.parent / "docs" / "options"
site.mkdir(parents=True, exist_ok=True)
if (here / "page.html").exists():
    shutil.copy(here / "page.html", site / "index.html")
    shutil.copy(here / "data.js", site / "data.js")
    shutil.copy(here / "pricer.js", site / "pricer.js")
print("\nSaved results.json, data.js and result.png, and copied the web page to docs/options")
