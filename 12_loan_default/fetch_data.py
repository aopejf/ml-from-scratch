"""
Download the two sets of real, public data this project uses. No account is needed for either.

    python fetch_data.py

1. Loans. "Default of credit card clients" from the UCI Machine Learning Repository:
   30,000 real credit card customers of a bank in Taiwan in 2005, and whether each one
   missed the next month's payment. Published by I-Cheng Yeh under the CC BY 4.0 licence.

2. The economy. Monthly US figures from FRED, the Federal Reserve's public data site:
   unemployment, factory output, jobs, interest rates, the official recession dates,
   and how much credit card debt banks wrote off each quarter.

Writes data/loans.csv, data/economy.csv and data/card_losses.csv.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "loan-default-study-script/1.0"}

LOANS_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# FRED's name for each series, and the plain name used in this project.
MONTHLY = {
    "UNRATE": "unemployment",  # percent of the workforce without a job
    "INDPRO": "factory_output",  # industrial production index
    "PAYEMS": "jobs",  # people on payrolls, thousands
    "GS10": "ten_year_rate",  # 10-year government bond yield
    "TB3MS": "three_month_rate",  # 3-month government bill yield
    "USREC": "official_recession",  # 1 in months the NBER later declared a recession. Never shown to the model.
}
QUARTERLY = {
    "CORCCACBS": "written_off",  # percent of credit card debt banks gave up on, yearly rate
    "DRCCLACBS": "behind",  # percent of credit card debt 30+ days late
}


def fred(series):
    text = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", headers=HEADERS, timeout=60).text
    table = pd.read_csv(io.StringIO(text))
    table.columns = ["month", series]
    table[series] = pd.to_numeric(table[series], errors="coerce")
    return table.set_index("month")[series]


def fetch_loans():
    raw = requests.get(LOANS_URL, headers=HEADERS, timeout=180).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        sheet = pd.read_excel(io.BytesIO(z.read(z.namelist()[0])), header=1)  # needs the xlrd package
    months = ["sep", "aug", "jul", "jun", "may", "apr"]  # the file's columns run from the newest month back
    names = {"ID": "id", "LIMIT_BAL": "credit_limit", "SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marriage",
             "AGE": "age", "default payment next month": "defaulted"}
    names["PAY_0"] = "late_sep"
    for k, m in enumerate(months[1:], start=2):
        names[f"PAY_{k}"] = f"late_{m}"
    for k, m in enumerate(months, start=1):
        names[f"BILL_AMT{k}"] = f"bill_{m}"
        names[f"PAY_AMT{k}"] = f"paid_{m}"
    return sheet.rename(columns=names)


if __name__ == "__main__":
    loans = fetch_loans()
    loans.to_csv(out / "loans.csv", index=False)
    print(f"loans.csv: {len(loans):,} customers, {loans.defaulted.mean():.1%} defaulted")

    economy = pd.DataFrame({name: fred(series) for series, name in MONTHLY.items()}).dropna()
    economy.to_csv(out / "economy.csv")
    print(f"economy.csv: {len(economy)} months, {economy.index[0]} to {economy.index[-1]}")

    losses = pd.DataFrame({name: fred(series) for series, name in QUARTERLY.items()}).dropna(how="all")
    losses.index.name = "quarter"
    losses.to_csv(out / "card_losses.csv")
    print(f"card_losses.csv: {len(losses)} quarters, {losses.index[0]} to {losses.index[-1]}")
