"""
Build the Excel valuation model, with live formulas.

    python analysis.py      (first: it writes results.json)
    python build_excel.py

Writes company_valuation_model.xlsx. Change any blue cell and the value per share updates.

Sheets:
    Read me     what this is and how to use it
    Valuation   assumptions, the ten-year forecast, and the answer
    WACC        how the return investors require is worked out
    History     ten years of filed figures and the ratios taken from them
    Checks      the Python answers, to compare with (check_excel.ps1 does this automatically)
"""

import json
from pathlib import Path

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

here = Path(__file__).parent
R = json.loads((here / "results.json").read_text(encoding="utf-8"))
H, A, S, CAP, BASE, CO = R["history"], R["assumptions"], R["start"], R["capital"], R["base"], R["company"]
M = 1e6  # the workbook is in $ millions

book = xlsxwriter.Workbook(here / "company_valuation_model.xlsx")
title = book.add_format({"bold": True, "font_size": 14})
head = book.add_format({"bold": True, "bottom": 1, "text_wrap": True, "valign": "bottom", "align": "right"})
head_left = book.add_format({"bold": True, "bottom": 1})
bold = book.add_format({"bold": True})
note = book.add_format({"italic": True, "font_color": "#595959"})
wrap = book.add_format({"text_wrap": True, "valign": "top"})
num = book.add_format({"num_format": "#,##0;[Red]-#,##0"})
num_bold = book.add_format({"num_format": "#,##0;[Red]-#,##0", "bold": True, "top": 1})
dec = book.add_format({"num_format": "0.000"})
pct = book.add_format({"num_format": "0.0%"})
pct2 = book.add_format({"num_format": "0.00%"})
usd = book.add_format({"num_format": "$#,##0.00"})
big_usd = book.add_format({"num_format": "$#,##0", "bold": True, "font_size": 13})
big_pct = book.add_format({"num_format": "+0%;[Red]-0%", "bold": True, "font_size": 13})
# Finance convention: blue for numbers you may change, black for formulas.
in_pct = book.add_format({"num_format": "0.00%", "font_color": "#0000FF", "bg_color": "#FFF9E5"})
in_num = book.add_format({"num_format": "#,##0.00", "font_color": "#0000FF", "bg_color": "#FFF9E5"})
in_whole = book.add_format({"num_format": "#,##0", "font_color": "#0000FF", "bg_color": "#FFF9E5"})

readme, val, wacc, hist, checks = (book.add_worksheet(n) for n in ("Read me", "Valuation", "WACC", "History", "Checks"))

# ---------------------------------------------------------------------------
# History: filed figures, with the ratios as formulas
# ---------------------------------------------------------------------------
w = hist
w.write("A1", f"{CO['name']}: ten years of filed figures ($ millions)", title)
w.write("A2", "Source: annual reports filed with the US regulator (SEC EDGAR).", note)
w.write(3, 0, "Financial year", head_left)
n = len(H)
first_col, last_col = xl_col_to_name(1), xl_col_to_name(n)
for i, row in enumerate(H):
    w.write_number(3, 1 + i, row["fiscal_year"], head)
filed = [("Sales", "revenue"), ("Operating profit", "operating_income"), ("Profit before tax", "pretax_income"), ("Income tax", "income_tax"),
         ("Net profit", "net_income"), ("Depreciation", "depreciation"), ("Spending on buildings and equipment", "capex"),
         ("Interest paid on debt", "interest_expense"), ("Cash", "cash"), ("Short-term investments", "short_term_investments"),
         ("Current assets", "current_assets"), ("Current liabilities", "current_liabilities"), ("Debt due within a year", "debt_current"),
         ("Long-term debt", "debt_long_term"), ("Finance leases", "finance_leases")]
ROW = {}
for k, (label, key) in enumerate(filed):
    r = 4 + k
    ROW[key] = r + 1  # Excel row number
    w.write(r, 0, label)
    for i, row in enumerate(H):
        w.write_number(r, 1 + i, row[key] / M, num)
w.write(20, 0, "Worked out from the figures above", bold)
ratios = [
    ("Sales growth", "growth", lambda c, p: f"={c}{ROW['revenue']}/{p}{ROW['revenue']}-1" if p else None, pct),
    ("Operating margin", "margin", lambda c, p: f"={c}{ROW['operating_income']}/{c}{ROW['revenue']}", pct2),
    ("Tax rate", "tax_rate", lambda c, p: f"={c}{ROW['income_tax']}/{c}{ROW['pretax_income']}", pct),
    ("Depreciation, share of sales", "depreciation_share", lambda c, p: f"={c}{ROW['depreciation']}/{c}{ROW['revenue']}", pct2),
    ("Buildings and equipment, share of sales", "capex_share", lambda c, p: f"={c}{ROW['capex']}/{c}{ROW['revenue']}", pct2),
    ("Working capital", "working_capital", lambda c, p: f"=({c}{ROW['current_assets']}-{c}{ROW['cash']}-{c}{ROW['short_term_investments']})-({c}{ROW['current_liabilities']}-{c}{ROW['debt_current']})", num),
    ("Working capital, share of sales", "working_capital_share", lambda c, p: f"={c}27/{c}{ROW['revenue']}", pct2),
    ("Debt", "debt", lambda c, p: f"={c}{ROW['debt_current']}+{c}{ROW['debt_long_term']}+{c}{ROW['finance_leases']}", num),
    ("Free cash flow", "free_cash_flow", lambda c, p: f"={c}{ROW['operating_income']}*(1-{c}24)+{c}{ROW['depreciation']}-{c}{ROW['capex']}-({c}27-{p}27)" if p else None, num),
]
RATIO = {}
for k, (label, key, formula, fmt) in enumerate(ratios):
    r = 21 + k
    RATIO[key] = r + 1
    w.write(r, 0, label)
    for i, row in enumerate(H):
        c, p = xl_col_to_name(1 + i), (xl_col_to_name(i) if i else None)
        f = formula(c, p)
        if f:
            scale = M if fmt is num else 1
            w.write_formula(r, 1 + i, f, fmt, row[key] / scale)
assert RATIO["tax_rate"] == 24 and RATIO["working_capital"] == 27
w.write(31, 0, "Free cash flow: operating profit after tax, plus depreciation, less buildings and equipment, less the rise in working capital.", note)
w.set_column(0, 0, 40)
w.set_column(1, n, 11)
recent = f"{xl_col_to_name(n - 2)}{{row}}:{last_col}{{row}}"  # the last three years
avg = lambda key: f"=AVERAGE(History!{recent.format(row=RATIO[key])})"

# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------
w = wacc
w.write("A1", "The return investors require (WACC)", title)
w.write("A2", "Shareholders and lenders each want a return. WACC blends the two by how much money each has in the business.", note)
lines = [
    ("What shareholders require", None, None),
    ("Yield on 10-year US government bonds (risk-free rate)", CAP["risk_free_rate"], in_pct),
    ("Beta: how far the share moves when the market moves 1%", CAP["beta"], in_num),
    ("Extra return wanted from shares over bonds (equity risk premium)", CAP["equity_risk_premium"], in_pct),
    ("Cost of equity = risk-free rate + beta x premium", "=B5+B6*B7", pct2),
    (None, None, None),
    ("What lenders require", None, None),
    ("Interest paid last year ($ millions)", f"=History!{last_col}{ROW['interest_expense']}", num),
    ("Average debt over the last two years ($ millions)", f"=AVERAGE(History!{xl_col_to_name(n - 1)}{RATIO['debt']}:{last_col}{RATIO['debt']})", num),
    ("Cost of debt = interest / debt", "=B11/B12", pct2),
    ("Tax rate (interest is paid before tax, so debt is cheaper than it looks)", avg("tax_rate"), pct),
    (None, None, None),
    ("The blend", None, None),
    ("Share price", CO["price"], in_num),
    ("Shares in issue (millions)", S["shares"] / M, in_whole),
    ("Market value of the shares ($ millions)", "=B17*B18", num),
    ("Debt ($ millions)", f"=History!{last_col}{RATIO['debt']}", num),
    ("Share of the money that is shareholders'", "=B19/(B19+B20)", pct),
    ("WACC", "=B21*B8+(1-B21)*B13*(1-B14)", book.add_format({"num_format": "0.00%", "bold": True, "font_size": 12})),
]
for k, (label, value, fmt) in enumerate(lines):
    r = 3 + k
    if label is None:
        continue
    if value is None:
        w.write(r, 0, label, bold)
    elif isinstance(value, str):
        w.write(r, 0, label)
        w.write_formula(r, 1, value, fmt)
    else:
        w.write(r, 0, label)
        w.write_number(r, 1, value, fmt)
w.write("A24", f"Beta is measured in Python from 60 months of share prices against the S&P 500. Bond yield as of {CO['risk_free_date']}, share price {CO['price_date']}.", note)
w.write("A25", "The equity risk premium is an assumption. Published estimates mostly sit between 4% and 6%.", note)
w.set_column(0, 0, 66)
w.set_column(1, 1, 16)

# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------
w = val
w.write("A1", f"What is {CO['name']} worth?  ($ millions unless it says otherwise)", title)
w.write("A2", "Blue cells are assumptions. Change any of them and the answer updates. They start at the company's own three-year averages.", note)
w.write("A4", "Assumptions", bold)
assumptions = [
    ("Sales growth next year (then fades evenly to the for-ever rate)", avg("growth"), "growth_start"),
    ("Growth for ever after year ten", A["terminal_growth"], "terminal_growth"),
    ("Operating margin", avg("margin"), "margin"),
    ("Tax rate", avg("tax_rate"), "tax_rate"),
    ("Depreciation, share of sales", avg("depreciation_share"), "depreciation_share"),
    ("Buildings and equipment, share of sales", avg("capex_share"), "capex_share"),
    ("Working capital, share of sales", avg("working_capital_share"), "working_capital_share"),
    ("Return investors require (WACC)", "=WACC!B22", "wacc"),
]
CELL = {}
for k, (label, value, key) in enumerate(assumptions):
    r = 4 + k
    CELL[key] = f"$B${r + 1}"
    w.write(r, 0, label)
    if isinstance(value, str):
        w.write_formula(r, 1, value, in_pct, A[key])
    else:
        w.write_number(r, 1, value, in_pct)
w.write("C5", "A formula: the average of the last three years on the History sheet. Type a number over it to test your own view.", note)
w.write("C12", "A formula: taken from the WACC sheet.", note)

w.write("A14", "Starting facts", bold)
facts = [("Sales last year", f"=History!{last_col}{ROW['revenue']}", num), ("Cash and short-term investments", f"=History!{last_col}{ROW['cash']}+History!{last_col}{ROW['short_term_investments']}", num),
         ("Debt", f"=History!{last_col}{RATIO['debt']}", num), ("Shares in issue (millions)", "=WACC!B18", num), ("Share price today ($)", "=WACC!B17", usd)]
for k, (label, formula, fmt) in enumerate(facts):
    w.write(14 + k, 0, label)
    w.write_formula(14 + k, 1, formula, fmt)
SALES0, CASH, DEBT, SHARES, PRICE = [f"$B${15 + k}" for k in range(5)]

top = 21  # zero-based row of the forecast header
w.write(top, 0, "The forecast", head_left)
labels = ["Year number", "Sales growth", "Sales", "Operating profit", "Profit after tax", "Plus depreciation", "Less buildings and equipment",
          "Less rise in working capital", "Free cash flow", "Discount factor", "Worth today"]
for k, label in enumerate(labels):
    w.write(top + 1 + k, 0, label, bold if label in ("Free cash flow", "Worth today") else None)
hr = top + 1  # Excel row number of the header
years = A["years"]
for y in range(1, years + 1):
    c, L, P = y, xl_col_to_name(y), xl_col_to_name(y - 1)
    row = BASE["rows"][y - 1]
    w.write_number(top, c, row["year"], head)
    w.write_number(top + 1, c, y, num)
    w.write_formula(top + 2, c, f"={CELL['growth_start']}+({CELL['terminal_growth']}-{CELL['growth_start']})*({L}{hr + 1}-1)/({years}-1)", pct, row["growth"])
    previous = SALES0 if y == 1 else f"{P}{hr + 3}"
    w.write_formula(top + 3, c, f"={previous}*(1+{L}{hr + 2})", num, row["revenue"] / M)
    w.write_formula(top + 4, c, f"={L}{hr + 3}*{CELL['margin']}", num, row["operating_profit"] / M)
    w.write_formula(top + 5, c, f"={L}{hr + 4}*(1-{CELL['tax_rate']})", num, row["after_tax"] / M)
    w.write_formula(top + 6, c, f"={L}{hr + 3}*{CELL['depreciation_share']}", num, row["depreciation"] / M)
    w.write_formula(top + 7, c, f"=-{L}{hr + 3}*{CELL['capex_share']}", num, -row["capex"] / M)
    w.write_formula(top + 8, c, f"=-{CELL['working_capital_share']}*({L}{hr + 3}-{previous})", num, -row["working_capital_change"] / M)
    w.write_formula(top + 9, c, f"=SUM({L}{hr + 5}:{L}{hr + 8})", num_bold, row["free_cash_flow"] / M)
    w.write_formula(top + 10, c, f"=1/(1+{CELL['wacc']})^{L}{hr + 1}", dec, row["discount"])
    w.write_formula(top + 11, c, f"={L}{hr + 9}*{L}{hr + 10}", num_bold, row["present_value"] / M)
END = xl_col_to_name(years)

w.write("A35", "The answer", title)
answers = [
    ("Ten years of cash, worth today", f"=SUM(B{hr + 11}:{END}{hr + 11})", num, BASE["forecast_present"] / M),
    ("Value of everything after year ten, in year ten", f"={END}{hr + 9}*(1+{CELL['terminal_growth']})/({CELL['wacc']}-{CELL['terminal_growth']})", num, BASE["terminal_value"] / M),
    ("Everything after year ten, worth today", f"=B37*{END}{hr + 10}", num, BASE["terminal_present"] / M),
    ("Value of the whole business (enterprise value)", "=B36+B38", num_bold, BASE["enterprise_value"] / M),
    ("Less debt", f"=-{DEBT}", num, -S["debt"] / M),
    ("Plus cash", f"={CASH}", num, S["cash"] / M),
    ("Value belonging to shareholders", "=SUM(B39:B41)", num_bold, BASE["equity_value"] / M),
    ("Value per share ($)", f"=B42/{SHARES}", big_usd, BASE["value_per_share"]),
    ("Share price today ($)", f"={PRICE}", usd, BASE["price"]),
    ("Model against price", "=B43/B44-1", big_pct, BASE["gap"]),
    (None, None, None, None),
    ("Share of the value that comes after year ten", "=B38/B39", pct, BASE["terminal_share"]),
    ("Year-ten value as a multiple of profit before depreciation", f"=B37/({END}{hr + 4}+{END}{hr + 6})", book.add_format({"num_format": '0.0 "times"'}), BASE["terminal_multiple"]),
]
for k, (label, formula, fmt, value) in enumerate(answers):
    if label:
        w.write(35 + k, 0, label, bold if fmt in (num_bold, big_usd, big_pct) else None)
        w.write_formula(35 + k, 1, formula, fmt, value)
w.write("C45", "Negative: the model is below the market. That is a statement about the assumptions, not advice to sell.", note)
w.set_column(0, 0, 58)
w.set_column(1, years, 12)

# ---------------------------------------------------------------------------
# Checks: the Python answers
# ---------------------------------------------------------------------------
w = checks
w.write("A1", "The same valuation worked out in Python (valuation.py)", title)
w.write("A2", "check_excel.ps1 opens this workbook in Excel and confirms every figure below matches the Valuation sheet.", note)
w.write_row("A4", ["Figure", "Python", "This workbook", "Difference"], head_left)
compare = [("Sales growth next year", A["growth_start"], "=Valuation!B5", pct2), ("Operating margin", A["margin"], "=Valuation!B7", pct2),
           ("WACC", A["wacc"], "=Valuation!B12", pct2), ("Enterprise value ($ millions)", BASE["enterprise_value"] / M, "=Valuation!B39", num),
           ("Value per share ($)", BASE["value_per_share"], "=Valuation!B43", usd), ("Share after year ten", BASE["terminal_share"], "=Valuation!B47", pct2)]
for k, (label, python, formula, fmt) in enumerate(compare):
    w.write(4 + k, 0, label)
    w.write_number(4 + k, 1, python, fmt)
    w.write_formula(4 + k, 2, formula, fmt)
    w.write_formula(4 + k, 3, f"=C{5 + k}-B{5 + k}", book.add_format({"num_format": "0.000000"}))
w.write("A12", "Value per share for other discount rates (down) and for-ever growth rates (across), from Python", bold)
sens = R["sensitivity"]
for j, g in enumerate(sens["terminal_growths"]):
    w.write_number(12, 1 + j, g, pct)
for i, rate in enumerate(sens["waccs"]):
    w.write_number(13 + i, 0, rate, pct)
    for j, value in enumerate(sens["value_per_share"][i]):
        w.write_number(13 + i, 1 + j, value, usd)
w.set_column(0, 0, 44)
w.set_column(1, 5, 16)

# ---------------------------------------------------------------------------
# Read me
# ---------------------------------------------------------------------------
readme.set_column(0, 0, 112)
text = [
    (f"Valuation model: {CO['name']} ({CO['ticker']})", title),
    ("A discounted cash flow (DCF) valuation built from the company's filed accounts. A worked method, NOT advice to buy or sell.", bold),
    ("", None),
    ("The idea: a business is worth the cash it will produce for its owners, shrunk back to today's money because cash later is worth less than cash now.", wrap),
    ("", None),
    ("How to use it", bold),
    ("Valuation: change any blue cell (growth, margin, WACC...) and the value per share updates. Start here.", wrap),
    ("WACC: how the return investors require is built from the bond yield, the share's beta and the company's debt.", wrap),
    ("History: ten years of filed figures. The starting assumptions are averages of the last three years, as formulas you can trace.", wrap),
    ("Checks: the same answers worked out separately in Python, side by side with this workbook's.", wrap),
    ("", None),
    ("Blue cells are assumptions. Black cells are formulas.", wrap),
    (f"Share price ${CO['price']:,.2f} as of {CO['price_date']}. Sources: {', '.join(CO['sources'])}.", wrap),
    ("", None),
    ("What it leaves out", bold),
    ("Leased shops are treated as a running cost, not as debt. Share-based pay is treated as a real cost. Cash flows are counted at each year end.", wrap),
    ("The answer moves a lot when the discount rate or the for-ever growth rate moves a little. The Checks sheet shows how much.", wrap),
]
for k, (line, fmt) in enumerate(text):
    readme.write(k, 0, line, fmt)

val.activate()
book.close()
print("Saved company_valuation_model.xlsx")
