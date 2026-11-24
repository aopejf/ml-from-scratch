"""
Run every query in queries.sql against data/chain.db and save the answers.

    python run_queries.py

Writes queries_output.md. Run make_data.py first: it builds the database.
"""

import sqlite3
from pathlib import Path

here = Path(__file__).parent
database = here / "data" / "chain.db"
if not database.exists():
    raise SystemExit("data/chain.db is missing. Run: python make_data.py")

blocks = (here / "queries.sql").read_text(encoding="utf-8").split("-- QUERY: ")[1:]
lines = ["# SQL answers", "", "Written by `run_queries.py` from `queries.sql`. All data is made up.", ""]
with sqlite3.connect(database) as db:
    for block in blocks:
        question, sql = block.split("\n", 1)
        cursor = db.execute(sql)
        names = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        lines += [f"## {question.strip()}", "", "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
        for row in rows:
            lines.append("| " + " | ".join(f"{v:,.0f}" if isinstance(v, float) and abs(v) >= 1000 else "" if v is None else str(v) for v in row) + " |")
        lines.append("")
        print(f"{question.strip()}: {len(rows)} rows")
db.close()
(here / "queries_output.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved queries_output.md")
