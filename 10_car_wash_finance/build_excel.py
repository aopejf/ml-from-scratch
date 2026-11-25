"""
Build the Excel model: car_wash_finance_model.xlsx

    python build_excel.py

Run analysis.py first. The workbook holds the raw monthly data and then works everything
out with live Excel formulas, so anyone can change an assumption (blue cells) and watch
the answer move. check_excel.ps1 opens it in Excel and compares its answers with Python's.

Sheets:
    Read me            what is in the workbook
    Site 28            the investment model: pick a location and a case, change any assumption
    Scorecard          one row per site, built with SUMIFS over the monthly data
    Budget vs actual   every line of the 2025 budget against what happened, for the chain or one site
    Monthly data       36 months for 27 sites
    Budget             the 2025 budget
"""

import json
from pathlib import Path

import pandas as pd
import xlsxwriter

import finance as f

here = Path(__file__).parent
results = json.loads((here / "results.json").read_text(encoding="utf-8"))
monthly = pd.read_csv(here / "data" / "monthly.csv")
budget = pd.read_csv(here / "data" / "budget.csv")
s24 = results["site28"]

book = xlsxwriter.Workbook(here / "car_wash_finance_model.xlsx")
title = book.add_format({"bold": True, "font_size": 14})
head = book.add_format({"bold": True, "bottom": 1, "text_wrap": True, "valign": "bottom"})
bold = book.add_format({"bold": True})
note = book.add_format({"italic": True, "font_color": "#595959"})
wrap = book.add_format({"text_wrap": True, "valign": "top"})
money = book.add_format({"num_format": "#,##0;[Red]-#,##0"})
money_bold = book.add_format({"num_format": "#,##0;[Red]-#,##0", "bold": True, "top": 1})
money2 = book.add_format({"num_format": "0.00"})
pct = book.add_format({"num_format": "0.0%"})
pct_bold = book.add_format({"num_format": "0.0%", "bold": True, "top": 1})
number = book.add_format({"num_format": "#,##0"})
years1 = book.add_format({"num_format": '0.0 "years"', "bold": True})
# Finance convention: blue for numbers you may change, black for formulas.
in_money = book.add_format({"num_format": "#,##0", "font_color": "#0000FF", "bg_color": "#FFF9E5"})
in_pct = book.add_format({"num_format": "0.0%", "font_color": "#0000FF", "bg_color": "#FFF9E5"})
in_num = book.add_format({"num_format": "0.00", "font_color": "#0000FF", "bg_color": "#FFF9E5"})
in_text = book.add_format({"font_color": "#0000FF", "bg_color": "#FFF9E5", "bold": True})
big = book.add_format({"num_format": "#,##0;[Red]-#,##0", "bold": True, "font_size": 12})
big_pct = book.add_format({"num_format": "0.0%", "bold": True, "font_size": 12})

readme = book.add_worksheet("Read me")
s24_sheet = book.add_worksheet("Site 28")
score = book.add_worksheet("Scorecard")
bva = book.add_worksheet("Budget vs actual")
data_sheet = book.add_worksheet("Monthly data")
budget_sheet = book.add_worksheet("Budget")

# ---------------------------------------------------------------------------
# Monthly data, with named ranges so the formulas elsewhere read like sentences
# ---------------------------------------------------------------------------
cols = list(monthly.columns)  # site_id ... other
n = len(monthly)
data_sheet.write_row(0, 0, cols + ["year", "revenue", "cars", "costs", "profit"], head)
for r, row in enumerate(monthly.itertuples(index=False), start=1):
    data_sheet.write_row(r, 0, list(row))
    x = r + 1
    data_sheet.write_number(r, 19, int(row.month[:4]))
    data_sheet.write_formula(r, 20, f"=J{x}+K{x}", money, row.membership_revenue + row.retail_revenue)
    data_sheet.write_formula(r, 21, f"=H{x}+I{x}", number, row.member_washes + row.retail_cars)
    data_sheet.write_formula(r, 22, f"=SUM(L{x}:S{x})", money, sum(getattr(row, c) for c in f.COSTS))
    data_sheet.write_formula(r, 23, f"=U{x}-W{x}", money, row.membership_revenue + row.retail_revenue - sum(getattr(row, c) for c in f.COSTS))
data_sheet.freeze_panes(1, 2)
data_sheet.set_column(0, 23, 13)
letters = "ABCDEFGHIJKLMNOPQRSTUVWX"
for i, name in enumerate(cols + ["year", "revenue", "cars", "costs", "profit"]):
    book.define_name(f"m_{name}", f"='Monthly data'!${letters[i]}$2:${letters[i]}${n + 1}")

bcols = list(budget.columns)
budget_sheet.write_row(0, 0, bcols, head)
for r, row in enumerate(budget.itertuples(index=False), start=1):
    budget_sheet.write_row(r, 0, list(row))
budget_sheet.freeze_panes(1, 2)
budget_sheet.set_column(0, len(bcols), 13)
for i, name in enumerate(bcols):
    book.define_name(f"b_{name}", f"=Budget!${letters[i]}$2:${letters[i]}${len(budget) + 1}")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
score.write("A1", "Site scorecard", title)
score.write("A2", "Year")
score.write_number("B2", 2025, in_money)
score.write("C2", "Change the year (2023, 2024 or 2025) and every number updates.", note)
heads = ["Site", "Revenue", "Costs", "Profit", "Margin", "Cars washed", "Revenue per car", "Labour share of revenue",
         "Chemicals per car", "Members at year end", "Cancel rate per month"]
score.write_row(3, 0, heads, head)
score.set_row(3, 32)
N = len(results["scorecard"])
first, last = 5, 4 + N
for i in range(N):
    r = 4 + i
    x = r + 1
    score.write_number(r, 0, i + 1)
    score.write_formula(r, 1, f"=SUMIFS(m_revenue,m_site_id,$A{x},m_year,$B$2)", money)
    score.write_formula(r, 2, f"=SUMIFS(m_costs,m_site_id,$A{x},m_year,$B$2)", money)
    score.write_formula(r, 3, f"=B{x}-C{x}", money)
    score.write_formula(r, 4, f"=IF(B{x}=0,0,D{x}/B{x})", pct)
    score.write_formula(r, 5, f"=SUMIFS(m_cars,m_site_id,$A{x},m_year,$B$2)", number)
    score.write_formula(r, 6, f"=IF(F{x}=0,0,B{x}/F{x})", money2)
    score.write_formula(r, 7, f"=IF(B{x}=0,0,SUMIFS(m_labour,m_site_id,$A{x},m_year,$B$2)/B{x})", pct)
    score.write_formula(r, 8, f"=IF(F{x}=0,0,SUMIFS(m_chemicals,m_site_id,$A{x},m_year,$B$2)/F{x})", money2)
    score.write_formula(r, 9, f'=SUMIFS(m_members_end,m_site_id,$A{x},m_month,$B$2&"-12")', number)
    score.write_formula(r, 10, f"=IFERROR(SUMIFS(m_cancels,m_site_id,$A{x},m_year,$B$2)/SUMIFS(m_members_start,m_site_id,$A{x},m_year,$B$2),0)", pct)
t = last + 1
score.write(t - 1, 0, "Chain", bold)
for c, L in [(1, "B"), (2, "C"), (3, "D"), (5, "F"), (9, "J")]:
    score.write_formula(t - 1, c, f"=SUM({L}{first}:{L}{last})", money_bold)
score.write_formula(t - 1, 4, f"=D{t}/B{t}", pct_bold)
score.write_formula(t - 1, 6, f"=B{t}/F{t}", book.add_format({"num_format": "0.00", "bold": True, "top": 1}))
score.write_formula(t - 1, 7, f"=SUMIFS(m_labour,m_year,$B$2)/B{t}", pct_bold)
score.write_formula(t - 1, 8, f"=SUMIFS(m_chemicals,m_year,$B$2)/F{t}", book.add_format({"num_format": "0.00", "bold": True, "top": 1}))
score.write_formula(t - 1, 10, f"=SUMIFS(m_cancels,m_year,$B$2)/SUMIFS(m_members_start,m_year,$B$2)", pct_bold)
score.conditional_format(f"E{first}:E{last}", {"type": "data_bar", "bar_color": "#2A78D6", "bar_solid": True})
score.conditional_format(f"H{first}:H{last}", {"type": "3_color_scale", "min_color": "#FFFFFF", "mid_color": "#FFFFFF", "max_color": "#F4B6A0"})
score.conditional_format(f"I{first}:I{last}", {"type": "3_color_scale", "min_color": "#FFFFFF", "mid_color": "#FFFFFF", "max_color": "#F4B6A0"})
score.conditional_format(f"K{first}:K{last}", {"type": "3_color_scale", "min_color": "#FFFFFF", "mid_color": "#FFFFFF", "max_color": "#F4B6A0"})
score.set_column(0, 0, 8)
score.set_column(1, 10, 14)
score.freeze_panes(4, 1)

# ---------------------------------------------------------------------------
# Budget vs actual
# ---------------------------------------------------------------------------
bva.write("A1", "2025 budget against actual", title)
bva.write("A2", "Site")
bva.write("B2", "All", in_text)
bva.data_validation("B2", {"validate": "list", "source": ["All"] + list(range(1, N + 1))})
bva.write("C2", 'Pick "All" for the chain or a site number.', note)
bva.write_row(3, 0, ["Line", "Budget", "Actual", "Difference", "Difference %", "Good or bad for profit"], head)
lines = [("Membership revenue", "membership_revenue", 1), ("Retail revenue", "retail_revenue", 1)] + [(c.replace("_", " ").capitalize(), c, -1) for c in f.COSTS]
for i, (label, col, sign) in enumerate(lines):
    r = 4 + i
    x = r + 1
    bva.write(r, 0, label)
    bva.write_formula(r, 1, f'=IF($B$2="All",SUM(b_{col}),SUMIFS(b_{col},b_site_id,$B$2))', money)
    bva.write_formula(r, 2, f'=IF($B$2="All",SUMIFS(m_{col},m_year,2025),SUMIFS(m_{col},m_site_id,$B$2,m_year,2025))', money)
    bva.write_formula(r, 3, f"=C{x}-B{x}", money)
    bva.write_formula(r, 4, f"=IF(B{x}=0,0,D{x}/B{x})", pct)
    bva.write_formula(r, 5, f'=IF({sign}*D{x}>=0,"Good","Bad")')
r = 4 + len(lines)
x = r + 1
bva.write(r, 0, "Profit", bold)
for c, L in [(1, "B"), (2, "C")]:
    bva.write_formula(r, c, f"=SUM({L}5:{L}6)-SUM({L}7:{L}{x - 1})", money_bold)
bva.write_formula(r, 3, f"=C{x}-B{x}", money_bold)
bva.write_formula(r, 4, f"=IF(B{x}=0,0,D{x}/B{x})", pct_bold)
bva.write_formula(r, 5, f'=IF(D{x}>=0,"Good","Bad")', bold)
bva.write(r + 2, 0, "Why revenue moved: how many, and at what price", bold)
bva.write_row(r + 3, 0, ["", "Budget", "Actual", "Effect on revenue"], head)
y = r + 5  # first Excel row of this block
bva.write(y - 1, 0, "Member-months")
bva.write_formula(y - 1, 1, '=IF($B$2="All",SUM(b_members),SUMIFS(b_members,b_site_id,$B$2))', number)
bva.write_formula(y - 1, 2, '=IF($B$2="All",SUMIFS(m_members_start,m_year,2025)+SUMIFS(m_members_end,m_year,2025),SUMIFS(m_members_start,m_site_id,$B$2,m_year,2025)+SUMIFS(m_members_end,m_site_id,$B$2,m_year,2025))/2', number)
bva.write(y, 0, "Average fee")
bva.write_formula(y, 1, f"=IF(B{y}=0,0,B5/B{y})", money2)
bva.write_formula(y, 2, f"=IF(C{y}=0,0,C5/C{y})", money2)
bva.write(y + 1, 0, "Retail cars")
bva.write_formula(y + 1, 1, '=IF($B$2="All",SUM(b_retail_cars),SUMIFS(b_retail_cars,b_site_id,$B$2))', number)
bva.write_formula(y + 1, 2, '=IF($B$2="All",SUMIFS(m_retail_cars,m_year,2025),SUMIFS(m_retail_cars,m_site_id,$B$2,m_year,2025))', number)
bva.write(y + 2, 0, "Average price per retail car")
bva.write_formula(y + 2, 1, f"=IF(B{y + 2}=0,0,B6/B{y + 2})", money2)
bva.write_formula(y + 2, 2, f"=IF(C{y + 2}=0,0,C6/C{y + 2})", money2)
bva.write(y + 4, 0, "Members: how many")
bva.write_formula(y + 4, 3, f"=(C{y}-B{y})*B{y + 1}", money)
bva.write(y + 5, 0, "Members: price")
bva.write_formula(y + 5, 3, f"=(C{y + 1}-B{y + 1})*C{y}", money)
bva.write(y + 6, 0, "Retail: how many cars")
bva.write_formula(y + 6, 3, f"=(C{y + 2}-B{y + 2})*B{y + 3}", money)
bva.write(y + 7, 0, "Retail: price per car")
bva.write_formula(y + 7, 3, f"=(C{y + 3}-B{y + 3})*C{y + 2}", money)
bva.write(y + 8, 0, "Total (matches the revenue difference above)", bold)
bva.write_formula(y + 8, 3, f"=SUM(D{y + 5}:D{y + 8})", money_bold)
bva.write(y + 10, 0, "This split uses yearly totals. The Python analysis does it month by month and site by site, so its pieces differ a little; both add up to the same total.", note)
bva.set_column(0, 0, 42)
bva.set_column(1, 5, 16)

# ---------------------------------------------------------------------------
# Site 28
# ---------------------------------------------------------------------------
w = s24_sheet
a, econ, model = s24["assumptions"], s24["economics"], s24["model"]
w.write("A1", "Where should site 28 go?", title)
w.write("A2", "Blue cells are assumptions. Change any of them and the answer updates.", note)

w.write("A4", "Location", bold)
w.write("B4", s24["candidates"][1]["candidate"], in_text)
w.data_validation("B4", {"validate": "list", "source": [c["candidate"] for c in s24["candidates"]]})
w.write("A5", "Volume case", bold)
w.write("B5", "Expected", in_text)
w.data_validation("B5", {"validate": "list", "source": ["Low", "Expected", "High"]})

w.write("A7", "Assumptions", bold)
inputs = [
    ("Return the owners require (discount rate)", a["discount_rate"], in_pct),
    ("Share of full volume, year 1", a["ramp"][0], in_pct),
    ("Share of full volume, year 2", a["ramp"][1], in_pct),
    ("Share of full volume, year 3 onward", a["ramp"][2], in_pct),
    ("Yearly rise in prices and costs", a["growth"], in_pct),
    ("Share of revenue set aside to replace equipment", a["upkeep_share"], in_pct),
    ("Sale value after year 10, times that year's profit", a["exit_multiple"], in_num),
    ("Opening cost in year 1", a["opening_cost"], in_money),
    ("Low and high case: this many typical misses from expected", 1.3, in_num),
]
for i, (label, value, fmt) in enumerate(inputs):
    w.write(7 + i, 0, label)
    w.write_number(7 + i, 1, value, fmt)
RATE, R1, R2, R3, GROW, UPKEEP, EXIT, OPEN, SPREAD = [f"$B${8 + i}" for i in range(9)]

w.write("A18", "From the chain's own books (established sites, 2025)", bold)
econ_rows = [("Revenue per car", econ["revenue_per_car"], in_num), ("Cost of washing one more car", econ["cost_per_car"], in_num),
             ("Fixed running cost per year, before rent", econ["fixed_cost_per_year"], in_money), ("Card fees, share of revenue", econ["card_fee_share"], in_pct)]
for i, (label, value, fmt) in enumerate(econ_rows):
    w.write(18 + i, 0, label)
    w.write_number(18 + i, 1, value, fmt)
REV_CAR, COST_CAR, FIXED, CARD = [f"$B${19 + i}" for i in range(4)]

w.write("A24", "Volume model: washes a year, learned from " + str(model["sites_used"]) + " established sites", bold)
coef_rows = [("Starting point", model["intercept"]), ("For each 1,000 cars passing a day", model["per_1000_traffic"]),
             ("For each competitor within 3 miles", model["per_competitor"]), ("For each $1,000 of local household income", model["per_1000_income"])]
for i, (label, value) in enumerate(coef_rows):
    w.write(24 + i, 0, label)
    w.write_number(24 + i, 1, value, in_money)
C0, C_TRAFFIC, C_COMP, C_INCOME = [f"$B${25 + i}" for i in range(4)]

# The three locations.
w.write_row("D4", ["Location", "Cars passing a day", "Competitors", "Household income", "Rent per month", "Build cost", "Washes a year, expected", "Typical miss"], head)
w.set_row(3, 32)
for i, c in enumerate(s24["candidates"]):
    x = 5 + i
    w.write(x - 1, 3, c["candidate"])
    w.write_number(x - 1, 4, c["traffic"], in_money)
    w.write_number(x - 1, 5, c["competitors"], in_money)
    w.write_number(x - 1, 6, c["income"], in_money)
    w.write_number(x - 1, 7, c["rent"], in_money)
    w.write_number(x - 1, 8, c["build_cost"], in_money)
    w.write_formula(x - 1, 9, f"={C0}+{C_TRAFFIC}*E{x}/1000+{C_COMP}*F{x}+{C_INCOME}*G{x}/1000", number, c["cars_expected"])
    w.write_number(x - 1, 10, c["cars_typical_miss"], in_money)
w.write("D8", "Typical miss: how far off the volume model usually is for one new site. It comes from the Python analysis.", note)

# The picked location.
w.write("D10", "The location picked", bold)
picked = [("Build cost", "=INDEX($I$5:$I$7,MATCH($B$4,$D$5:$D$7,0))"), ("Rent per month", "=INDEX($H$5:$H$7,MATCH($B$4,$D$5:$D$7,0))"),
          ("Washes a year once full", f'=INDEX($J$5:$J$7,MATCH($B$4,$D$5:$D$7,0))+(MATCH($B$5,{{"Low","Expected","High"}},0)-2)*{SPREAD}*INDEX($K$5:$K$7,MATCH($B$4,$D$5:$D$7,0))')]
for i, (label, formula) in enumerate(picked):
    w.write(10 + i, 3, label)
    w.write_formula(10 + i, 4, formula, money)
BUILD, RENT, CARS = "$E$11", "$E$12", "$E$13"

# Year by year, across the page.
top = 15  # zero-based row of the header
w.write(top, 3, "Year", head)
labels = ["Share of full volume", "Price level", "Revenue", "Running costs", "Profit", "Equipment replacement", "Sale of the site", "Cash flow", "Running total, without the sale", "Payback helper"]
for i, label in enumerate(labels):
    w.write(top + 1 + i, 3, label, bold if label in ("Profit", "Cash flow") else None)
col = lambda year: xlsxwriter.utility.xl_col_to_name(4 + year)
hr = top + 1  # Excel row number of the header
for year in range(0, 11):
    L, c = col(year), 4 + year
    w.write_number(top, c, year, head)
    if year == 0:
        w.write_formula(top + 8, c, f"=-{BUILD}", money)
        w.write_formula(top + 9, c, f"={L}{hr + 8}", money)
        continue
    P = col(year - 1)
    w.write_formula(top + 1, c, f"=IF({L}{hr}=1,{R1},IF({L}{hr}=2,{R2},{R3}))", pct)
    w.write_formula(top + 2, c, f"=(1+{GROW})^({L}{hr}-1)", money2)
    w.write_formula(top + 3, c, f"={CARS}*{L}{hr + 1}*{REV_CAR}*{L}{hr + 2}", money)
    w.write_formula(top + 4, c, f"=({CARS}*{L}{hr + 1}*{COST_CAR}+{FIXED}+{RENT}*12)*{L}{hr + 2}+{L}{hr + 3}*{CARD}", money)
    w.write_formula(top + 5, c, f"={L}{hr + 3}-{L}{hr + 4}" + (f"-{OPEN}" if year == 1 else ""), money)
    w.write_formula(top + 6, c, f"={UPKEEP}*{L}{hr + 3}", money)
    w.write_formula(top + 7, c, f"={EXIT}*{L}{hr + 5}" if year == 10 else "=0", money)
    w.write_formula(top + 8, c, f"={L}{hr + 5}-{L}{hr + 6}+{L}{hr + 7}", money)
    w.write_formula(top + 9, c, f"={P}{hr + 9}+{L}{hr + 5}-{L}{hr + 6}", money)
    w.write_formula(top + 10, c, f"=IF(AND({P}{hr + 9}<0,{L}{hr + 9}>=0),{L}{hr}-1-{P}{hr + 9}/({L}{hr + 5}-{L}{hr + 6}),0)", money2)
FLOWS = f"$F${hr + 8}:$O${hr + 8}"

w.write("D28", "The answer", title)
answers = [
    ("Value created over 10 years (NPV)", f"=$E${hr + 8}+NPV({RATE},{FLOWS})", big),
    ("Yearly return (IRR)", f"=IFERROR(IRR($E${hr + 8}:$O${hr + 8}),0)", big_pct),
    ("Years to earn back the build cost", f'=IF(SUM($F${hr + 10}:$O${hr + 10})=0,"over 10 years",SUM($F${hr + 10}:$O${hr + 10}))', years1),
    ("Profit a year once full (year 3)", f"=$H${hr + 5}", big),
    ("Verdict", f'=IF($E$29>0,"Clears the required return","Does not clear the required return")', bold),
]
for i, (label, formula, fmt) in enumerate(answers):
    w.write(28 + i, 3, label, bold)
    w.write_formula(28 + i, 4, formula, fmt)

w.write("D35", "Cross-check: the same nine cases worked out in Python (finance.py)", bold)
w.write_row("D36", ["Location", "Case", "Washes a year", "NPV", "IRR", "Payback years"], head)
r = 36
for c in s24["candidates"]:
    for case in ("Low", "Expected", "High"):
        k = c["cases"][case]
        w.write(r, 3, c["candidate"])
        w.write(r, 4, case)
        w.write_number(r, 5, k["cars"], number)
        w.write_number(r, 6, k["npv"], money)
        w.write_number(r, 7, k["irr"], pct)
        if k["payback_years"]:
            w.write_number(r, 8, k["payback_years"], money2)
        else:
            w.write(r, 8, "over 10 years")
        r += 1
w.set_column(0, 0, 52)
w.set_column(1, 1, 16)
w.set_column(2, 2, 3)
w.set_column(3, 3, 34)
w.set_column(4, 14, 14)

# ---------------------------------------------------------------------------
# Read me
# ---------------------------------------------------------------------------
readme.set_column(0, 0, 110)
text = [
    ("Finance model for a 27-site car wash chain", title),
    ("ALL DATA IN THIS WORKBOOK IS MADE UP. No real company, site or customer.", bold),
    ("", None),
    ("Site 28: the investment model. Pick a location and a volume case, change any blue cell, and the NPV, IRR and payback update.", wrap),
    ("Scorecard: one row per site. Every number is a SUMIFS over the Monthly data sheet. Change the year in B2.", wrap),
    ("Budget vs actual: each line of the 2025 budget against what happened, for the whole chain or one site, and why revenue moved.", wrap),
    ("Monthly data, Budget: the raw numbers the other sheets add up.", wrap),
    ("", None),
    ("Conventions: blue cells on a yellow background are inputs. Black cells are formulas. Named ranges (m_revenue, m_site_id, b_labour and so on) point at columns of the data sheets.", wrap),
    ("Profit means site-level profit before head-office cost, interest, tax and depreciation (four-wall EBITDA).", wrap),
    ("NPV: future cash discounted back to today at the required return, minus the build cost. Above zero means the project beats the required return.", wrap),
    ("IRR: the yearly return the project itself earns. Payback: years of cash needed to earn back the build cost, leaving out the sale value.", wrap),
    ("", None),
    ("The workbook is written by build_excel.py. check_excel.ps1 opens it in Excel and confirms its answers match the Python analysis.", wrap),
]
for i, (line, fmt) in enumerate(text):
    readme.write(i, 0, line, fmt)

s24_sheet.activate()
book.close()
print("Saved car_wash_finance_model.xlsx")
