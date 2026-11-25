"""
What is a company worth? A discounted cash flow (DCF) valuation, step by step.

The idea in one line: a business is worth the cash it will hand its owners in the
future, shrunk back to today's money because cash later is worth less than cash now.

    1. history()        what the company has actually done, from its filed accounts
    2. cost_of_capital  the yearly return investors require (WACC), from market data
    3. dcf()            forecast ten years of cash, add a value for everything after, discount it all
    4. sensitivity()    how much the answer moves when the two touchiest assumptions move
    5. implied_growth() turn it round: what growth must you believe to justify today's share price?

This is a worked method on public data. It is not advice to buy or sell anything.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
YEARS = 10
EQUITY_RISK_PREMIUM = 0.05  # extra yearly return investors want from shares over government bonds. An assumption.
TERMINAL_GROWTH = 0.025  # growth for ever after year ten: about the long-run growth of the whole economy
AVERAGE_OVER = 3  # recent years averaged to set the starting assumptions


def load():
    return dict(
        financials=pd.read_csv(here / "data" / "financials.csv"),
        prices=pd.read_csv(here / "data" / "prices.csv"),
        market=json.loads((here / "data" / "market.json").read_text(encoding="utf-8")),
    )


# ---------------------------------------------------------------------------
# 1. History
# ---------------------------------------------------------------------------
def history(financials):
    """The filed numbers, plus the ratios the forecast is built from."""
    h = financials.copy()
    h["growth"] = h.revenue.pct_change()
    h["margin"] = h.operating_income / h.revenue
    h["tax_rate"] = h.income_tax / h.pretax_income
    h["depreciation_share"] = h.depreciation / h.revenue
    h["capex_share"] = h.capex / h.revenue
    # Working capital: money tied up in stock and unpaid bills, less what the company owes suppliers.
    # Cash and borrowings are left out because they are counted separately at the end.
    h["working_capital"] = (h.current_assets - h.cash - h.short_term_investments) - (h.current_liabilities - h.debt_current)
    h["working_capital_share"] = h.working_capital / h.revenue
    h["debt"] = h.debt_current + h.debt_long_term + h.finance_leases
    h["profit_after_tax"] = h.operating_income * (1 - h.tax_rate)
    # Free cash flow: what is left for lenders and owners after tax and after reinvesting in the business.
    h["free_cash_flow"] = h.profit_after_tax + h.depreciation - h.capex - h.working_capital.diff()
    return h


# ---------------------------------------------------------------------------
# 2. The return investors require
# ---------------------------------------------------------------------------
def beta(prices, months=60):
    """How much the share moves when the whole market moves 1%. 1.0 means in step with the market."""
    returns = prices[["company", "market"]].pct_change().dropna().tail(months)
    return float(np.cov(returns.company, returns.market)[0, 1] / np.var(returns.market, ddof=1))


def cost_of_capital(h, prices, market):
    """WACC: the blend of what shareholders and lenders each require, weighted by how much each has put in."""
    last = h.iloc[-1]
    b = beta(prices)
    cost_of_equity = market["risk_free_rate"] + b * EQUITY_RISK_PREMIUM  # the capital asset pricing model (CAPM)
    tax_rate = float(h.tax_rate.tail(AVERAGE_OVER).mean())
    cost_of_debt = float(last.interest_expense / h.debt.tail(2).mean())
    equity_value = market["price"] * market["shares_outstanding"]
    equity_weight = equity_value / (equity_value + last.debt)
    wacc = equity_weight * cost_of_equity + (1 - equity_weight) * cost_of_debt * (1 - tax_rate)
    return dict(risk_free_rate=market["risk_free_rate"], beta=b, equity_risk_premium=EQUITY_RISK_PREMIUM,
                cost_of_equity=cost_of_equity, cost_of_debt=cost_of_debt, tax_rate=tax_rate,
                equity_value=equity_value, debt=float(last.debt), equity_weight=float(equity_weight), wacc=float(wacc))


# ---------------------------------------------------------------------------
# 3. The valuation
# ---------------------------------------------------------------------------
def starting_point(h, market):
    """The facts the forecast starts from. None of these are opinions."""
    last = h.iloc[-1]
    return dict(fiscal_year=int(last.fiscal_year), revenue=float(last.revenue),
                cash=float(last.cash + last.short_term_investments), debt=float(last.debt),
                shares=float(market["shares_outstanding"]), price=float(market["price"]))


def base_assumptions(h, wacc):
    """The opinions. Each starts at the company's own recent average, so none is plucked from the air."""
    recent = h.tail(AVERAGE_OVER)
    return dict(
        growth_start=float(recent.growth.mean()),  # sales growth next year; it then fades evenly to terminal_growth
        terminal_growth=TERMINAL_GROWTH,
        margin=float(recent.margin.mean()),  # operating profit as a share of sales
        tax_rate=float(recent.tax_rate.mean()),
        depreciation_share=float(recent.depreciation_share.mean()),
        capex_share=float(recent.capex_share.mean()),  # spending on buildings and equipment, as a share of sales
        working_capital_share=float(recent.working_capital_share.mean()),
        wacc=float(wacc),
        years=YEARS,
    )


def growth_path(a):
    """Sales growth for each forecast year: starts at growth_start, ends at terminal_growth, straight line between."""
    n = a["years"]
    return [a["growth_start"] + (a["terminal_growth"] - a["growth_start"]) * i / (n - 1) for i in range(n)]


def dcf(start, a):
    """
    Forecast, discount, add up.

    Returns the year-by-year table and the answer. The same sums are in dcf.js (for the web page)
    and in the Excel model, and the three are checked against each other.
    """
    rows, revenue = [], start["revenue"]
    for year, growth in enumerate(growth_path(a), start=1):
        previous, revenue = revenue, revenue * (1 + growth)
        operating_profit = revenue * a["margin"]
        after_tax = operating_profit * (1 - a["tax_rate"])
        depreciation = revenue * a["depreciation_share"]
        capex = revenue * a["capex_share"]
        working_capital_change = a["working_capital_share"] * (revenue - previous)
        free_cash_flow = after_tax + depreciation - capex - working_capital_change
        discount = 1 / (1 + a["wacc"]) ** year
        rows.append(dict(year=start["fiscal_year"] + year, growth=growth, revenue=revenue, operating_profit=operating_profit,
                         after_tax=after_tax, depreciation=depreciation, capex=capex, working_capital_change=working_capital_change,
                         free_cash_flow=free_cash_flow, discount=discount, present_value=free_cash_flow * discount))
    last = rows[-1]
    # Everything after year ten, as one number: cash growing at terminal_growth for ever (the Gordon growth formula).
    terminal_value = last["free_cash_flow"] * (1 + a["terminal_growth"]) / (a["wacc"] - a["terminal_growth"])
    terminal_present = terminal_value * last["discount"]
    forecast_present = sum(r["present_value"] for r in rows)
    enterprise_value = forecast_present + terminal_present  # the whole business, whoever financed it
    equity_value = enterprise_value - start["debt"] + start["cash"]  # what is left for shareholders
    per_share = equity_value / start["shares"]
    return dict(
        rows=rows, forecast_present=forecast_present, terminal_value=terminal_value, terminal_present=terminal_present,
        enterprise_value=enterprise_value, equity_value=equity_value, value_per_share=per_share,
        price=start["price"], gap=per_share / start["price"] - 1,
        terminal_share=terminal_present / enterprise_value,
        # A cross-check people use: the terminal value as a multiple of that year's profit before depreciation.
        terminal_multiple=terminal_value / (last["operating_profit"] + last["depreciation"]),
    )


# ---------------------------------------------------------------------------
# 4. How sure is the answer?
# ---------------------------------------------------------------------------
def sensitivity(start, a, wacc_steps=(-0.01, -0.005, 0, 0.005, 0.01), growth_steps=(-0.01, -0.005, 0, 0.005, 0.01)):
    """Value per share for a grid of discount rates and for-ever growth rates."""
    waccs = [a["wacc"] + s for s in wacc_steps]
    growths = [a["terminal_growth"] + s for s in growth_steps]
    grid = [[dcf(start, {**a, "wacc": w, "terminal_growth": g})["value_per_share"] for g in growths] for w in waccs]
    return dict(waccs=waccs, terminal_growths=growths, value_per_share=grid)


def scenarios(start, a):
    """Three stories. Each changes growth and margin together, the way good and bad years really arrive."""
    cases = {
        "Cautious": dict(growth_start=a["growth_start"] - 0.03, margin=a["margin"] - 0.004),
        "Base": {},
        "Optimistic": dict(growth_start=a["growth_start"] + 0.03, margin=a["margin"] + 0.004),
    }
    out = {}
    for name, change in cases.items():
        used = {**a, **change}
        out[name] = dict(growth_start=used["growth_start"], margin=used["margin"], value_per_share=dcf(start, used)["value_per_share"])
    return out


def solve(start, a, name, low, high, target=None):
    """The value of one assumption that makes the model's answer equal the share price. None if no value in range does."""
    target = start["price"] if target is None else target
    value = lambda x: dcf(start, {**a, name: x})["value_per_share"] - target
    if value(low) * value(high) > 0:
        return None
    rising = value(high) > value(low)
    for _ in range(80):
        mid = (low + high) / 2
        if (value(mid) < 0) == rising:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def implied(start, a):
    """Reverse DCF: hold everything else at the base case and ask what the share price needs."""
    return dict(
        growth_start=solve(start, a, "growth_start", -0.05, 0.60),
        margin=solve(start, a, "margin", 0.001, 0.30),
        wacc=solve(start, a, "wacc", a["terminal_growth"] + 0.002, 0.25),
    )


def market_ratios(h, market):
    """What the market pays today for each dollar the company earned last year."""
    last = h.iloc[-1]
    value = market["price"] * market["shares_outstanding"]
    enterprise = value + last.debt - last.cash - last.short_term_investments
    return dict(market_value=float(value), enterprise_value=float(enterprise),
                price_to_earnings=float(value / last.net_income),
                ev_to_ebitda=float(enterprise / (last.operating_income + last.depreciation)),
                free_cash_flow_yield=float(last.free_cash_flow / value))


def run(data=None):
    data = data or load()
    h = history(data["financials"])
    capital = cost_of_capital(h, data["prices"], data["market"])
    start = starting_point(h, data["market"])
    a = base_assumptions(h, capital["wacc"])
    return dict(history=h, capital=capital, start=start, assumptions=a, base=dcf(start, a), sensitivity=sensitivity(start, a),
                scenarios=scenarios(start, a), implied=implied(start, a), ratios=market_ratios(h, data["market"]), market=data["market"])


if __name__ == "__main__":
    r = run()
    a, b = r["assumptions"], r["base"]
    print({k: round(v, 4) for k, v in a.items()})
    print({k: round(v, 4) for k, v in r["capital"].items()})
    print(f"Value per share ${b['value_per_share']:,.0f} against a price of ${b['price']:,.0f} ({b['gap']:+.0%}).")
    print(f"Terminal value is {b['terminal_share']:.0%} of the total, {b['terminal_multiple']:.1f} times profit before depreciation.")
    print("Implied:", r["implied"], "\nRatios:", r["ratios"], "\nScenarios:", r["scenarios"])
    print(r["history"][["fiscal_year", "growth", "margin", "tax_rate", "capex_share", "working_capital_share", "free_cash_flow"]].round(4).to_string(index=False))
