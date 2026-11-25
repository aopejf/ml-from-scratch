"""
Download the real, public market data this project uses. No account is needed.

    python fetch_data.py

Writes:
    data/sp500.csv     the S&P 500 index's closing level every trading day since 1950 (Yahoo Finance)
    data/chain.csv     today's quoted prices for options on SPY, the fund that tracks the S&P 500,
                       for the contract date closest to three months away (Yahoo Finance)
    data/market.json   the fund's price, the contract date, and the 3-month interest rate (FRED)
"""

import io
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "options-study-script/1.0"}
INDEX, FUND = "^GSPC", "SPY"
TARGET_DAYS = 91  # about three months, to match the 64 trading days the scenario model makes


def fetch_index():
    closes = yf.download(INDEX, start="1950-01-01", progress=False, auto_adjust=True)["Close"].dropna()
    table = closes.reset_index()
    table.columns = ["day", "close"]
    table["day"] = table.day.dt.strftime("%Y-%m-%d")
    return table.round(2)


def fetch_chain():
    fund = yf.Ticker(FUND)
    today = date.today()
    expiry = min(fund.options, key=lambda d: abs((date.fromisoformat(d) - today).days - TARGET_DAYS))
    chain = fund.option_chain(expiry)
    rows = []
    for kind, table in (("call", chain.calls), ("put", chain.puts)):
        keep = table[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
        keep.columns = ["strike", "bid", "ask", "last", "volume", "open_interest"]
        keep.insert(0, "kind", kind)
        rows.append(keep)
    last = yf.download(FUND, period="5d", interval="1d", auto_adjust=False, progress=False)["Close"].dropna()
    return pd.concat(rows, ignore_index=True), dict(
        fund=FUND, fund_price=round(float(last.iloc[-1].iloc[0]), 2), price_date=str(last.index[-1].date()),
        expiry=expiry, fetched=str(today), days_to_expiry=(date.fromisoformat(expiry) - today).days)


def fetch_rate():
    text = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO", headers=HEADERS, timeout=60).text
    series = pd.read_csv(io.StringIO(text))
    series.columns = ["day", "percent"]
    series = series[pd.to_numeric(series.percent, errors="coerce").notna()]
    return dict(rate=round(float(series.percent.iloc[-1]) / 100, 4), rate_date=str(series.day.iloc[-1]))


if __name__ == "__main__":
    index = fetch_index()
    index.to_csv(out / "sp500.csv", index=False)
    print(f"sp500.csv: {len(index):,} trading days, {index.day.iloc[0]} to {index.day.iloc[-1]}")
    chain, market = fetch_chain()
    chain.to_csv(out / "chain.csv", index=False)
    market.update(fetch_rate(), sources=["Yahoo Finance", "FRED series DGS3MO"])
    (out / "market.json").write_text(json.dumps(market, indent=2), encoding="utf-8")
    print(f"chain.csv: {len(chain)} quotes for {market['expiry']}; {market['fund']} at {market['fund_price']}; rate {market['rate']:.2%}")
