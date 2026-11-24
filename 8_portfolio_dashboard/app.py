"""
The dashboard as an interactive page.

    python app.py

Then open http://localhost:8050. Type any stock symbols, pick a time span,
and every chart updates. Hover over a chart to read exact values.

Uses only Python's built-in web server, plus the same metrics.py as the
static dashboard.
"""

import json
import os
import re
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from metrics import compute, load_prices

here = Path(__file__).parent
HOST, PORT = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", 8050))
SYMBOL = re.compile(r"^[A-Z0-9.\-^=]{1,10}$")


def dashboard_data(tickers, years):
    end = date.today()
    prices = load_prices(tickers, str(end - timedelta(days=round(365.25 * years))), str(end))
    if len(prices) < 80:
        raise ValueError("Not enough shared price history for these symbols over this time span.")
    n = compute(prices)
    names = list(n["growth"].columns)
    dates = [d.strftime("%Y-%m-%d") for d in n["growth"].index]
    series = lambda frame: {name: [None if v != v else round(float(v), 5) for v in frame[name]] for name in names}
    return {
        "tickers": tickers,
        "names": names,
        "dates": dates,
        "growth": series(n["growth"]),
        "drawdown": series(n["drawdown"]),
        "rolling_volatility": series(n["rolling_volatility"]),
        "correlation": n["correlation"].round(3).values.tolist(),
        "summary": [{"name": name, **{k: round(float(v), 4) for k, v in row.items()}} for name, row in n["summary"].iterrows()],
        "span": f"{dates[0]} to {dates[-1]}",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind, status=200):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send((here / "app.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/data":
            ask = parse_qs(url.query)
            try:
                tickers = list(dict.fromkeys(t for t in re.split(r"[\s,]+", ask.get("tickers", [""])[0].upper()) if t))
                if not 2 <= len(tickers) <= 8 or not all(SYMBOL.match(t) for t in tickers):
                    raise ValueError("Enter between 2 and 8 stock symbols, like AAPL MSFT KO.")
                years = min(max(int(ask.get("years", ["5"])[0]), 1), 10)
                self.send(json.dumps(dashboard_data(tickers, years)).encode(), "application/json")
            except ValueError as problem:
                self.send(json.dumps({"error": str(problem)}).encode(), "application/json", 400)
            except Exception:
                self.send(json.dumps({"error": "Could not load prices. Check the symbols and your connection."}).encode(), "application/json", 500)
        else:
            self.send_error(404)


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT} in a browser. Press Ctrl+C here to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
