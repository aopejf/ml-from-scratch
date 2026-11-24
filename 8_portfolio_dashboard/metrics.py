"""
The calculations behind the dashboard, shared by dashboard.py (the picture)
and app.py (the interactive page).

Everything comes from one thing: each day's percentage change in price.
"""

from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
TRADING_DAYS = 252  # per year
MIX = "Equal mix"


def load_prices(tickers, start, end):
    """Daily prices from Yahoo Finance. Downloaded once, then kept in a file."""
    saved = here / "data" / f"prices_{'_'.join(tickers)}_{start}_{end}.csv"
    if saved.exists():
        prices = pd.read_csv(saved, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        # auto_adjust folds dividends and share splits into the price, so growth is the true total.
        prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        missing = [t for t in tickers if t not in prices or prices[t].dropna().empty]
        if missing:
            raise ValueError(f"No prices found for: {', '.join(missing)}")
        saved.parent.mkdir(exist_ok=True)
        prices.to_csv(saved)
    return prices[tickers].dropna()


def compute(prices):
    tickers = list(prices.columns)
    daily = prices.pct_change().dropna()
    daily[MIX] = daily[tickers].mean(axis=1)  # the same money in each, topped back up to equal every day

    growth = (1 + daily).cumprod()  # what $1 became
    years = len(daily) / TRADING_DAYS
    yearly_return = growth.iloc[-1] ** (1 / years) - 1  # the steady yearly rate that gives the same end result
    volatility = daily.std() * np.sqrt(TRADING_DAYS)  # typical size of a year's swings
    drawdown = growth / growth.cummax() - 1  # how far below its own previous peak
    summary = pd.DataFrame(
        {
            "$1 became": growth.iloc[-1],
            "Return per year": yearly_return,
            "Volatility": volatility,
            "Return / volatility": yearly_return / volatility,  # Sharpe ratio with a 0% risk-free rate
            "Worst drop": drawdown.min(),
            "Worst day": daily.min(),
            "Best day": daily.max(),
        }
    ).sort_values("Return / volatility", ascending=False)
    return {
        "daily": daily,
        "growth": growth,
        "yearly_return": yearly_return,
        "volatility": volatility,
        "drawdown": drawdown,
        "correlation": daily[tickers].corr(),
        "rolling_volatility": daily.rolling(63).std() * np.sqrt(TRADING_DAYS),  # the last 3 months
        "summary": summary,
    }
