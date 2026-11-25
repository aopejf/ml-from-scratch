"""
Download a real company's published numbers and save them in data/.

    python fetch_data.py

You do not need to run this: the files it writes are already in data/, so everything
else works offline. Run it again only to refresh the numbers.

Three public sources, none of which needs an account:
    SEC EDGAR     the figures every US listed company files with the regulator (annual reports, form 10-K)
    Yahoo Finance month-end share prices, for the company and for the S&P 500
    FRED          the yield on 10-year US government bonds (the "risk-free rate")

Writes:
    data/financials.csv   one row per financial year: sales, profit, cash flow, balance sheet
    data/prices.csv       month-end prices for the company and the S&P 500 (to measure beta)
    data/market.json      today's share price, share count and the 10-year government bond yield
"""

import io
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

COMPANY = dict(name="Costco Wholesale", ticker="COST", cik="0000909832")
MARKET_TICKER = "^GSPC"  # the S&P 500
YEARS_KEPT = 10
HEADERS = {"User-Agent": "company-valuation-study-script/1.0"}

here = Path(__file__).parent
out = here / "data"
out.mkdir(exist_ok=True)

# Companies do not all use the same label for the same thing, so each line lists the labels to try, in order.
# "flow" lines cover a whole year (sales). "stock" lines are a snapshot on the last day of the year (cash).
LINES = {
    "revenue": ("flow", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]),
    "operating_income": ("flow", ["OperatingIncomeLoss"]),
    "pretax_income": ("flow", ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]),
    "income_tax": ("flow", ["IncomeTaxExpenseBenefit"]),
    "net_income": ("flow", ["NetIncomeLoss"]),
    "depreciation": ("flow", ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndOther"]),
    "capex": ("flow", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "cash_from_operations": ("flow", ["NetCashProvidedByUsedInOperatingActivities"]),
    "share_based_pay": ("flow", ["ShareBasedCompensation"]),
    "interest_expense": ("flow", ["InterestExpense", "InterestExpenseNonoperating"]),
    "diluted_shares": ("flow", ["WeightedAverageNumberOfDilutedSharesOutstanding"]),
    "cash": ("stock", ["CashAndCashEquivalentsAtCarryingValue"]),
    "short_term_investments": ("stock", ["ShortTermInvestments", "MarketableSecuritiesCurrent"]),
    "current_assets": ("stock", ["AssetsCurrent"]),
    "current_liabilities": ("stock", ["LiabilitiesCurrent"]),
    "debt_current": ("stock", ["LongTermDebtCurrent"]),
    "debt_long_term": ("stock", ["LongTermDebtNoncurrent"]),
    "finance_leases": ("stock", ["FinanceLeaseLiabilityNoncurrent"]),
    "operating_leases": ("stock", ["OperatingLeaseLiabilityNoncurrent"]),
    "equity": ("stock", ["StockholdersEquity"]),
}


def yearly_values(facts, kind, labels):
    """{year end date: value} from annual reports, taking the most recently filed figure for each year."""
    found = {}
    for label in labels:
        for rows in facts.get(label, {}).get("units", {}).values():
            for r in rows:
                if r.get("form") not in ("10-K", "10-K/A"):
                    continue
                if kind == "flow":
                    days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
                    if not 350 <= days <= 380:  # a whole year, not a quarter
                        continue
                elif "start" in r:
                    continue
                if r["end"] not in found or r["filed"] > found[r["end"]][0]:
                    found[r["end"]] = (r["filed"], r["val"])
        if found:
            break  # the first label that has data wins
    return {end: value for end, (_, value) in found.items()}


def fetch_financials():
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{COMPANY['cik']}.json"
    facts = requests.get(url, headers=HEADERS, timeout=60).json()
    table = pd.DataFrame({name: yearly_values(facts["facts"]["us-gaap"], kind, labels) for name, (kind, labels) in LINES.items()})
    table = table.sort_index().dropna(subset=["revenue", "operating_income", "cash"]).tail(YEARS_KEPT)
    table.index.name = "year_end"
    table.insert(0, "fiscal_year", [int(d[:4]) for d in table.index])
    shares = facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
    latest = max(shares, key=lambda r: r["end"])
    return table.fillna(0).reset_index(), dict(shares_outstanding=latest["val"], shares_as_of=latest["end"])


def fetch_prices():
    import yfinance as yf

    raw = yf.download([COMPANY["ticker"], MARKET_TICKER], period="6y", interval="1mo", auto_adjust=True, progress=False)["Close"]
    monthly = raw.dropna().rename(columns={COMPANY["ticker"]: "company", MARKET_TICKER: "market"})
    last = yf.download(COMPANY["ticker"], period="5d", interval="1d", auto_adjust=False, progress=False)["Close"].dropna()
    price, price_date = float(last.iloc[-1].iloc[0]), str(last.index[-1].date())
    monthly.index = [str(d.date()) for d in monthly.index]
    monthly.index.name = "month"
    return monthly.round(4).reset_index(), dict(price=round(price, 2), price_date=price_date)


def fetch_government_bond_yield():
    text = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", headers=HEADERS, timeout=60).text
    series = pd.read_csv(io.StringIO(text))
    series.columns = ["day", "percent"]
    series = series[pd.to_numeric(series.percent, errors="coerce").notna()]
    return dict(risk_free_rate=round(float(series.percent.iloc[-1]) / 100, 4), risk_free_date=str(series.day.iloc[-1]))


if __name__ == "__main__":
    financials, shares = fetch_financials()
    prices, price = fetch_prices()
    market = dict(**COMPANY, **price, **shares, **fetch_government_bond_yield(),
                  sources=["SEC EDGAR company facts", "Yahoo Finance", "FRED series DGS10"])
    financials.to_csv(out / "financials.csv", index=False)
    prices.to_csv(out / "prices.csv", index=False)
    (out / "market.json").write_text(json.dumps(market, indent=2), encoding="utf-8")
    print(financials[["fiscal_year", "revenue", "operating_income", "capex", "cash"]].assign(
        revenue=lambda t: (t.revenue / 1e9).round(1), operating_income=lambda t: (t.operating_income / 1e9).round(2),
        capex=lambda t: (t.capex / 1e9).round(2), cash=lambda t: (t.cash / 1e9).round(1)).to_string(index=False))
    print(json.dumps(market, indent=2))
