"""
Invent the books of a 27-site car wash chain.

    python make_data.py

EVERYTHING HERE IS MADE UP. No real company, site or customer is involved.
The numbers are sized to look like an express car wash business, and a few
problems are hidden in them on purpose (listed at the bottom of this
docstring) so the analysis can be checked: did it find what was planted?

Writes into data/:
    sites.csv        one row per site: region, opening date, traffic, competitors
    monthly.csv      one row per site per month, 2023 to 2025: members, cars, revenue, costs
    budget.csv       the 2025 budget, written the way a finance team would in late 2024
    cohorts.csv      for members who joined in a given month, how many are still members later
    corporate.csv    head-office cost per month
    candidates.csv   three possible locations for site 28
    chain.db         the same tables as a SQLite database, for the SQL queries

Planted on purpose:
    Site 07   is overstaffed: labour runs about a third above what its volume needs
    Site 11   had a competitor open next door in September 2024
    Site 14   has used far too much chemical per car since June 2025
    Site 18   sits on a quiet road with four competitors
    Site 27   opened in April 2025 (the budget assumed February) and is still filling up
    All sites raised the membership price by $2 in March 2025 (the budget assumed January)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
out = here / "data"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

N_SITES = 27
FIRST_KEPT, LAST = pd.Period("2023-01", "M"), pd.Period("2025-12", "M")
BURN_IN_FROM = pd.Period("2015-01", "M")  # simulate earlier years so 2023 starts with a realistic member base

# How busy each month is compared with an average month, by region.
SEASON = {
    "North": [1.18, 1.15, 1.12, 1.02, 0.98, 0.93, 0.90, 0.90, 0.93, 0.97, 0.98, 1.08],  # road salt in winter
    "Central": [1.05, 1.04, 1.08, 1.06, 1.02, 0.97, 0.94, 0.94, 0.96, 0.98, 0.97, 0.99],
    "South": [0.98, 1.00, 1.10, 1.10, 1.04, 0.97, 0.94, 0.93, 0.96, 0.99, 1.00, 0.99],  # pollen in spring
}
NORMAL_RAIN = {
    "North": [9, 8, 10, 11, 11, 10, 9, 9, 8, 9, 10, 10],
    "Central": [8, 8, 10, 10, 11, 9, 8, 7, 7, 8, 8, 8],
    "South": [9, 8, 9, 7, 8, 11, 12, 12, 9, 6, 7, 8],
}
EXPECTED_MONTHS_AS_MEMBER = 35  # what the cancel rates below work out to


def monthly_cancel_rate(months_as_member):
    """New members cancel fastest. Those who stay a year mostly keep staying."""
    age = np.asarray(months_as_member)
    return np.select([age <= 1, age <= 3, age <= 12], [0.09, 0.06, 0.035], 0.022)


# ---------------------------------------------------------------------------
# 1. The sites
# ---------------------------------------------------------------------------
regions = ["North"] * 9 + ["Central"] * 9 + ["South"] * 9
sites = pd.DataFrame({
    "site_id": np.arange(1, N_SITES + 1),
    "name": [f"Site {i:02d}" for i in range(1, N_SITES + 1)],
    "region": regions,
    "traffic": rng.integers(22, 44, N_SITES) * 1000,  # cars passing per day
    "competitors": rng.choice([0, 1, 1, 2, 2, 3], N_SITES),  # other washes within 3 miles
    "income": rng.integers(58, 108, N_SITES) * 1000,  # typical household income nearby
})
opened = [pd.Period(f"{rng.integers(2012, 2021)}-{rng.integers(1, 13):02d}", "M") for _ in range(N_SITES)]
opened[-3], opened[-2], opened[-1] = pd.Period("2023-03", "M"), pd.Period("2024-02", "M"), pd.Period("2025-04", "M")
sites["opened"] = [str(p) for p in opened]
sites.loc[sites.site_id == 3, ["traffic", "competitors"]] = [43000, 0]  # the star
sites.loc[sites.site_id == 18, ["traffic", "competitors"]] = [23000, 4]  # the quiet road
sites.loc[sites.site_id == 11, "competitors"] = 1  # becomes 2 in September 2024

# What drives volume. The site-28 model has to rediscover these from the results.
luck = rng.normal(0, 800, N_SITES)
sites["mature_cars_hidden"] = (3500 + 0.30 * sites.traffic - 1100 * sites.competitors + 0.035 * (sites.income - 80000) + luck).round()
sites["rent"] = (6500 + 0.33 * sites.traffic + rng.normal(0, 1200, N_SITES)).round(-2)

COMPETITOR_OPENS = (11, pd.Period("2024-09", "M"))
PRICE_RISE = pd.Period("2025-03", "M")

# ---------------------------------------------------------------------------
# 2. Month by month
# ---------------------------------------------------------------------------
months = pd.period_range(BURN_IN_FROM, LAST, freq="M")
region_rain = {  # weather is shared across a region, with a little local difference
    r: {m: rng.normal(0, 2.2) for m in months} for r in SEASON
}
rows, cohort_rows = [], []
for s in sites.itertuples():
    season, normal_rain = SEASON[s.region], NORMAL_RAIN[s.region]
    target_members = 0.26 * s.mature_cars_hidden
    joins_per_month = target_members / EXPECTED_MONTHS_AS_MEMBER
    retail_mature = 0.32 * s.mature_cars_hidden
    base_fee = 27.0 + 0.00005 * (s.income - 80000)  # richer areas buy the dearer plans
    base_ticket = 12.6 + 0.00003 * (s.income - 80000)
    labour_bloat = 1.35 if s.site_id == 7 else 1.0
    start = max(pd.Period(s.opened, "M"), BURN_IN_FROM)
    cohorts = {}  # month joined -> members still active
    members = 0
    for m in months:
        if m < start:
            continue
        age = (m - start).n
        year_index = m.year - 2023 + (m.month - 1) / 12  # 0 at January 2023
        busy = season[m.month - 1]
        rain = max(0, round(normal_rain[m.month - 1] + region_rain[s.region][m] + rng.normal(0, 1.0)))
        wet = rain - normal_rain[m.month - 1]
        rival = s.site_id == COMPETITOR_OPENS[0] and m >= COMPETITOR_OPENS[1]
        price_shock = m in (PRICE_RISE, PRICE_RISE + 1)

        # Members: some cancel, some join.
        members_start = members
        cancels = 0
        for joined in list(cohorts):
            rate = float(monthly_cancel_rate((m - joined).n)) * (1.6 if rival else 1.0) * (1.15 if price_shock else 1.0)
            lost = rng.binomial(cohorts[joined], min(rate, 1.0))
            cohorts[joined] -= lost
            cancels += lost
        grand_opening = 3.0 if age < 9 else 1.0
        joins = rng.poisson(joins_per_month * (0.6 + 0.4 * busy) * grand_opening * (0.75 if rival else 1.0))
        cohorts[m] = joins
        members = members_start - cancels + joins

        # Cars.
        filling_up = min(1.0, 0.45 + 0.55 * age / 12)
        retail_cars = retail_mature * busy * filling_up * (1 - 0.025 * wet) * (0.82 if rival else 1.0) * rng.normal(1, 0.04)
        visits_per_member = 2.6 * busy * (1 - 0.012 * wet) * rng.normal(1, 0.02)
        avg_members = (members_start + members) / 2
        member_washes = avg_members * visits_per_member
        cars = retail_cars + member_washes

        # Money in.
        fee = base_fee + 0.9 * (m.year - 2023) + (2.0 if m >= PRICE_RISE else 0.0)
        ticket = base_ticket * 1.03 ** year_index * rng.normal(1, 0.01)
        membership_revenue = avg_members * fee
        retail_revenue = retail_cars * ticket
        revenue = membership_revenue + retail_revenue

        # Money out.
        wages = 1.04 ** year_index
        chem_per_car = 0.62 * 1.02 ** year_index * rng.normal(1, 0.04)
        if s.site_id == 14 and m >= pd.Period("2025-06", "M"):
            chem_per_car *= 1.55
        heating = 1.25 if s.region == "North" and m.month in (12, 1, 2) else 1.0
        breakdown = 22000 * rng.random() + 8000 if rng.random() < 0.02 else 0
        launch_ads = 6000 if age < 6 else 0

        if m >= FIRST_KEPT:
            rows.append(dict(
                site_id=s.site_id, month=str(m), rain_days=rain,
                members_start=members_start, joins=joins, cancels=cancels, members_end=members,
                member_washes=round(member_washes), retail_cars=round(retail_cars),
                membership_revenue=round(membership_revenue, 2), retail_revenue=round(retail_revenue, 2),
                labour=round((17500 + 0.85 * cars) * wages * labour_bloat * rng.normal(1, 0.03), 2),
                chemicals=round(chem_per_car * cars, 2),
                utilities=round((2500 + 0.55 * cars) * heating * 1.03 ** year_index * rng.normal(1, 0.04), 2),
                maintenance=round((5500 + 0.25 * cars) * rng.normal(1, 0.10) + breakdown, 2),
                rent=float(s.rent),
                marketing=round(4500 * rng.normal(1, 0.10) + launch_ads, 2),
                card_fees=round(0.026 * revenue, 2),
                other=round(3000 * rng.normal(1, 0.08), 2),
            ))
            for joined, active in cohorts.items():
                if joined >= FIRST_KEPT:
                    cohort_rows.append((s.site_id, str(joined), (m - joined).n, active))

monthly = pd.DataFrame(rows)
cohorts = pd.DataFrame(cohort_rows, columns=["site_id", "joined", "months_later", "active"])
COSTS = ["labour", "chemicals", "utilities", "maintenance", "rent", "marketing", "card_fees", "other"]

# ---------------------------------------------------------------------------
# 3. The 2025 budget, written in late 2024 from what was known then
# ---------------------------------------------------------------------------
m24 = monthly[monthly.month.str.startswith("2024")].copy()
budget_rows = []
for s in sites.itertuples():
    last_year = m24[m24.site_id == s.site_id].set_index("month")
    for month_number in range(1, 13):
        month = f"2025-{month_number:02d}"
        if s.site_id == N_SITES:  # the new site: planned to open in February and fill up over a year
            age = month_number - 2
            if age < 0:
                continue
            typical = m24.groupby("month").mean(numeric_only=True).loc[f"2024-{month_number:02d}"]
            fill = min(1.0, 0.45 + 0.55 * age / 12)
            members = 0.26 * 11500 * min(1.0, (age + 1) / 24) * 1.6
            line = dict(members=members, fee=30.5, retail_cars=typical.retail_cars * fill, ticket=13.6)
            costs = {c: typical[c] * 1.04 for c in COSTS}
            costs["rent"], costs["marketing"] = float(s.rent), 4700 + (6000 if age < 6 else 0)
        else:
            same_month = f"2024-{month_number:02d}"
            if same_month in last_year.index:
                base = last_year.loc[same_month]
            else:  # site 26 opened in February 2024, so it has no January: use its December
                base = last_year.loc["2024-12"]
            december = last_year.loc["2024-12"]
            fee_2024 = last_year.membership_revenue.sum() / ((last_year.members_start + last_year.members_end) / 2).sum()
            line = dict(
                members=december.members_end * 1.004 ** month_number,  # steady growth
                fee=fee_2024 + 0.90 + 2.0,  # the price rise was planned for January
                retail_cars=base.retail_cars * 1.03,
                ticket=base.retail_revenue / base.retail_cars * 1.03,
            )
            costs = {c: base[c] * 1.04 for c in COSTS}
            costs["rent"] = float(s.rent)
        membership_revenue = line["members"] * line["fee"]
        retail_revenue = line["retail_cars"] * line["ticket"]
        costs["card_fees"] = 0.026 * (membership_revenue + retail_revenue)
        budget_rows.append(dict(
            site_id=s.site_id, month=month, members=round(line["members"], 1), member_fee=round(line["fee"], 2),
            retail_cars=round(line["retail_cars"]), retail_ticket=round(line["ticket"], 2),
            membership_revenue=round(membership_revenue, 2), retail_revenue=round(retail_revenue, 2),
            **{c: round(v, 2) for c, v in costs.items()},
        ))
budget = pd.DataFrame(budget_rows)

corporate = pd.DataFrame({
    "month": [str(m) for m in pd.period_range(FIRST_KEPT, LAST, freq="M")],
})
corporate["overhead"] = [round(285000 * 1.045 ** ((pd.Period(m, "M").year - 2023)) * rng.normal(1, 0.02), 2) for m in corporate.month]

# ---------------------------------------------------------------------------
# 4. Three places site 28 could go
# ---------------------------------------------------------------------------
candidates = pd.DataFrame({
    "candidate": ["A: Retail park", "B: Commuter road", "C: Edge of town"],
    "region": ["Central", "North", "South"],
    "traffic": [41000, 34000, 24000],
    "competitors": [3, 1, 0],
    "income": [84000, 91000, 66000],
    "rent": [27500, 22000, 17000],
    "build_cost": [4_600_000, 4_100_000, 3_500_000],  # building and equipment; the land is leased
})

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
public_sites = sites.drop(columns="mature_cars_hidden")
public_sites.loc[public_sites.site_id == COMPETITOR_OPENS[0], "competitors"] += 1  # today's count
tables = dict(sites=public_sites, monthly=monthly, budget=budget, cohorts=cohorts, corporate=corporate, candidates=candidates)
(out / "chain.db").unlink(missing_ok=True)
with sqlite3.connect(out / "chain.db") as db:
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        table.to_sql(name, db, index=False)
db.close()

revenue = monthly.membership_revenue + monthly.retail_revenue
y25 = monthly.month.str.startswith("2025")
print(f"{N_SITES} sites, {monthly.month.nunique()} months, {len(monthly):,} site-months, {len(cohorts):,} cohort rows.")
print(f"2025 revenue ${revenue[y25].sum() / 1e6:.1f} million. Members at the end of 2025: {monthly[monthly.month == '2025-12'].members_end.sum():,}.")
print(f"2025 site-level profit margin: {1 - monthly.loc[y25, COSTS].sum().sum() / revenue[y25].sum():.0%}.")
