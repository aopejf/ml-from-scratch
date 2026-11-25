"""
The finance calculations, as plain functions. analysis.py calls them and the tests check them.

Terms used throughout:
    profit            site-level profit before head-office cost, interest, tax and depreciation.
                      The official name is four-wall EBITDA.
    margin            profit as a share of revenue
    churn             the share of members who cancel in a month
    LTV               lifetime value: what one new member is worth over their whole stay
    CAC               customer acquisition cost: marketing spent per new member
    NPV               net present value: future cash, discounted back to today, minus the cost
    IRR               internal rate of return: the yearly return the project earns
"""

from pathlib import Path

import numpy as np
import pandas as pd

COSTS = ["labour", "chemicals", "utilities", "maintenance", "rent", "marketing", "card_fees", "other"]
VOLUME_COSTS = ["labour", "chemicals", "utilities", "maintenance"]  # the ones that rise with cars washed
MATURE_AFTER_MONTHS = 24


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(folder=None):
    folder = Path(folder) if folder else Path(__file__).parent / "data"
    data = {name: pd.read_csv(folder / f"{name}.csv") for name in ["sites", "monthly", "budget", "cohorts", "corporate", "candidates"]}
    data["monthly"] = add_totals(data["monthly"])
    return data


def add_totals(monthly):
    m = monthly.copy()
    m["revenue"] = m.membership_revenue + m.retail_revenue
    m["cars"] = m.member_washes + m.retail_cars
    m["costs"] = m[COSTS].sum(axis=1)
    m["profit"] = m.revenue - m.costs
    m["avg_members"] = (m.members_start + m.members_end) / 2
    m["year"] = m.month.str[:4].astype(int)
    return m


def months_open(sites, as_of="2025-12"):
    return {s.site_id: (pd.Period(as_of, "M") - pd.Period(s.opened, "M")).n + 1 for s in sites.itertuples()}


def straight_line(x, y):
    """Best-fit line y = fixed + per_unit * x. Returns (fixed, per_unit)."""
    per_unit, fixed = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(fixed), float(per_unit)


# ---------------------------------------------------------------------------
# 1. Site scorecard
# ---------------------------------------------------------------------------
def scorecard(data, year=2025):
    """One row per site for the year, with what looks wrong and what fixing it is worth."""
    sites, m = data["sites"], data["monthly"]
    y = m[m.year == year]
    age = months_open(sites, f"{year}-12")
    t = y.groupby("site_id").agg(
        revenue=("revenue", "sum"), profit=("profit", "sum"), cars=("cars", "sum"),
        labour=("labour", "sum"), chemicals=("chemicals", "sum"), utilities=("utilities", "sum"),
        cancels=("cancels", "sum"), member_months=("members_start", "sum"), members=("members_end", "last"),
        months=("month", "count"),
    )
    t["margin"] = t.profit / t.revenue
    t["revenue_per_car"] = t.revenue / t.cars
    t["labour_share"] = t.labour / t.revenue
    t["chemicals_per_car"] = t.chemicals / t.cars
    t["churn"] = t.cancels / t.member_months
    t["months_open"] = pd.Series(age)
    t["mature"] = t.months_open >= MATURE_AFTER_MONTHS
    t = sites[["site_id", "name", "region", "opened", "traffic", "competitors"]].merge(t, left_on="site_id", right_index=True)

    # What labour should a site of this volume need? Learn it from every site's months, then compare.
    fixed, per_car = straight_line(y.cars, y.labour)
    t["labour_expected"] = fixed * t.months + per_car * t.cars
    t["labour_excess"] = t.labour - t.labour_expected

    # Chemical use in the most recent six months, against the chain's typical figure.
    recent = y[y.month >= f"{year}-07"].groupby("site_id").agg(chem=("chemicals", "sum"), cars=("cars", "sum"))
    recent_rate = recent.chem / recent.cars
    typical_chem = float(recent_rate.median())
    t["chemicals_recent"] = t.site_id.map(recent_rate)
    t["chemicals_excess_year"] = (t.chemicals_recent - typical_chem) * t.site_id.map(recent.cars) * 2  # six months, doubled

    typical_churn = float(t[t.mature].churn.median())
    flags, worth = [], []
    for r in t.itertuples():
        f, w = [], 0.0
        if not r.mature:
            f.append("Still filling up")
        else:
            if r.labour_excess > 0.15 * r.labour_expected:
                f.append("Labour high for its volume")
                w += r.labour_excess
            if r.churn > 1.25 * typical_churn:
                f.append("Members leaving fast")
            if r.margin < 0.15:
                f.append("Barely profitable")
        if r.chemicals_recent > 1.2 * typical_chem:
            f.append("Chemical use high")
            w += r.chemicals_excess_year
        flags.append("; ".join(f))
        worth.append(round(w))
    t["flags"], t["fix_worth_per_year"] = flags, worth
    t = t.sort_values("margin", ascending=False).reset_index(drop=True)
    return t, dict(labour_fixed_per_month=fixed, labour_per_car=per_car, typical_chemicals_per_car=typical_chem, typical_churn=typical_churn)


# ---------------------------------------------------------------------------
# 2. Membership
# ---------------------------------------------------------------------------
def retention_curve(cohorts, max_months=24):
    """Of everyone who joined, the share still a member N months later."""
    start = cohorts[cohorts.months_later == 0].set_index(["site_id", "joined"]).active.rename("joined_count")
    c = cohorts.join(start, on=["site_id", "joined"])
    c = c[(c.joined_count > 0) & (c.months_later <= max_months)]
    g = c.groupby("months_later").agg(active=("active", "sum"), joined=("joined_count", "sum"))
    return (g.active / g.joined).round(4)


def membership(data, year=2025):
    m = data["monthly"]
    y = m[m.year == year]
    mature_ids = [k for k, v in months_open(data["sites"], f"{year}-12").items() if v >= MATURE_AFTER_MONTHS]
    ym = y[y.site_id.isin(mature_ids)]
    last = y[y.month == f"{year}-12"]

    fee = float(last.membership_revenue.sum() / last.avg_members.sum())
    churn = float(y.cancels.sum() / y.members_start.sum())
    visits = float(y.member_washes.sum() / y.avg_members.sum())
    # Cost of washing one more car, learned from how costs move with volume.
    cost_per_car = straight_line(ym.cars, ym[VOLUME_COSTS].sum(axis=1))[1]
    card_fee = 0.026 * fee
    monthly_margin = fee - visits * cost_per_car - card_fee
    lifetime_months = 1 / churn
    ltv = monthly_margin * lifetime_months
    cac = float(ym.marketing.sum() / ym.joins.sum())
    curve = retention_curve(data["cohorts"])
    return dict(
        members=int(last.members_end.sum()),
        members_a_year_ago=int(m[m.month == f"{year - 1}-12"].members_end.sum()),
        monthly_recurring_revenue=float(last.members_end.sum() * fee),
        membership_share_of_revenue=float(y.membership_revenue.sum() / y.revenue.sum()),
        fee=fee, churn=churn, visits_per_member=visits, cost_per_car=cost_per_car,
        monthly_margin_per_member=monthly_margin, lifetime_months=lifetime_months,
        ltv=ltv, cac=cac, ltv_to_cac=ltv / cac, payback_months=cac / monthly_margin,
        retention={int(k): float(v) for k, v in curve.items()},
        churn_by_month={k: float(v) for k, v in (m.groupby("month").cancels.sum() / m.groupby("month").members_start.sum()).items()},
        members_by_month={k: int(v) for k, v in m.groupby("month").members_end.sum().items()},
    )


# ---------------------------------------------------------------------------
# 3. Budget against actual
# ---------------------------------------------------------------------------
def variance(data, year=2025, site_id=None):
    """
    Why profit differs from budget, split so the pieces add up exactly.

    For each revenue line:   volume effect = (actual units - budget units) x budget price
                             price effect  = (actual price - budget price) x actual units
    For each cost line:      effect = budget cost - actual cost   (spending less is good)
    """
    a = data["monthly"][data["monthly"].year == year]
    b = data["budget"]
    if site_id is not None:
        a, b = a[a.site_id == site_id], b[b.site_id == site_id]
    j = b.merge(a, on=["site_id", "month"], how="outer", suffixes=("_b", "_a")).fillna(0.0)
    units_a, units_b = j.avg_members, j.members
    price_b = np.nan_to_num(np.where(units_b > 0, j.membership_revenue_b / units_b.replace(0, np.nan), 0.0))
    ticket_b = np.nan_to_num(np.where(j.retail_cars_b > 0, j.retail_revenue_b / j.retail_cars_b.replace(0, np.nan), 0.0))
    price_a = np.where(units_a > 0, j.membership_revenue_a / units_a.replace(0, np.nan), 0.0)
    retail_price_a = np.where(j.retail_cars_a > 0, j.retail_revenue_a / j.retail_cars_a.replace(0, np.nan), 0.0)
    effects = {
        "Members: how many": float(((units_a - units_b) * price_b).sum()),
        "Members: price": float(((np.nan_to_num(price_a) - price_b) * units_a).sum()),
        "Retail: how many cars": float(((j.retail_cars_a - j.retail_cars_b) * ticket_b).sum()),
        "Retail: price per car": float(((np.nan_to_num(retail_price_a) - ticket_b) * j.retail_cars_a).sum()),
    }
    for c in COSTS:
        effects[c.replace("_", " ").capitalize()] = float((j[f"{c}_b"] - j[f"{c}_a"]).sum())
    budget_revenue = float((j.membership_revenue_b + j.retail_revenue_b).sum())
    actual_revenue = float((j.membership_revenue_a + j.retail_revenue_a).sum())
    budget_profit = budget_revenue - float(sum(j[f"{c}_b"].sum() for c in COSTS))
    actual_profit = actual_revenue - float(sum(j[f"{c}_a"].sum() for c in COSTS))
    return dict(budget_revenue=budget_revenue, actual_revenue=actual_revenue,
                budget_profit=budget_profit, actual_profit=actual_profit, effects=effects)


# ---------------------------------------------------------------------------
# 4. Forecast
# ---------------------------------------------------------------------------
def _seasonal_index(frame, column):
    """How each calendar month compares with a region's average month."""
    f = frame.copy()
    f["m"] = f.month.str[5:].astype(int)
    f["ratio"] = f[column] / f.groupby(["site_id", "year"])[column].transform("mean")
    return f.groupby(["region", "m"]).ratio.mean()


def forecast(data, history_to="2025-12", months_ahead=12):
    """
    Cars and revenue for each site for the next year, built from parts:
      members next month = members now - cancellations + new joins
      retail cars        = the site's recent level x the usual seasonal pattern
    Only uses months up to history_to, so it can be tested on a year it did not see.
    """
    sites = data["sites"]
    m = data["monthly"][data["monthly"].month <= history_to].merge(sites[["site_id", "region"]], on="site_id")
    end = pd.Period(history_to, "M")
    age = months_open(sites, history_to)
    full_years = m[m.groupby(["site_id", "year"]).month.transform("count") == 12]
    full_years = full_years[full_years.site_id.map(age) >= MATURE_AFTER_MONTHS + 12]
    season = {c: _seasonal_index(full_years, c) for c in ["retail_cars", "joins"]}
    visits = full_years.assign(m=full_years.month.str[5:].astype(int), v=full_years.member_washes / full_years.avg_members).groupby(["region", "m"]).v.mean()

    rows = []
    for s in sites.itertuples():
        h = m[m.site_id == s.site_id].sort_values("month")
        if h.empty:
            continue
        young = age[s.site_id] < 15
        recent = h.tail(6 if young else 12)
        idx_now = np.mean([season["retail_cars"][(s.region, int(mm[5:]))] for mm in recent.month])
        retail_level = recent.retail_cars.mean() / idx_now  # the site's level with the season taken out
        j_recent = h.tail(3) if young else h.tail(12)
        join_level = j_recent.joins.mean() / np.mean([season["joins"][(s.region, int(mm[5:]))] for mm in j_recent.month])
        if young:
            join_level *= 0.5  # opening offers end; assume joins settle at half the launch pace
        churn = h.tail(6).cancels.sum() / h.tail(6).members_start.sum()
        last = h.iloc[-1]
        fee = last.membership_revenue / last.avg_members
        ticket = h.tail(12).retail_revenue.sum() / h.tail(12).retail_cars.sum()
        members = float(last.members_end)
        for k in range(1, months_ahead + 1):
            p = end + k
            start_members = members
            members = start_members * (1 - churn) + join_level * season["joins"][(s.region, p.month)]
            avg_members = (start_members + members) / 2
            retail = retail_level * season["retail_cars"][(s.region, p.month)]
            washes = avg_members * visits[(s.region, p.month)]
            rows.append(dict(site_id=s.site_id, month=str(p), members_end=members, retail_cars=retail,
                             member_washes=washes, cars=retail + washes,
                             revenue=avg_members * fee + retail * ticket * 1.03))
    return pd.DataFrame(rows)


def backtest(data):
    """Forecast 2025 using only 2023-2024, then compare with what happened and with a simple rule."""
    f = forecast(data, history_to="2024-12")
    actual = data["monthly"][data["monthly"].year == 2025][["site_id", "month", "cars", "members_end"]]
    last_year = data["monthly"][data["monthly"].year == 2024][["site_id", "month", "cars"]].copy()
    last_year["month"] = "2025" + last_year.month.str[4:]
    j = actual.merge(f, on=["site_id", "month"], suffixes=("", "_model")).merge(last_year, on=["site_id", "month"], suffixes=("", "_naive"))
    age = months_open(data["sites"], "2024-12")
    j = j[j.site_id.map(age) >= MATURE_AFTER_MONTHS]

    def miss(prediction):
        return float((np.abs(prediction - j.cars) / j.cars).mean())

    yearly = j.groupby("site_id")[["cars", "cars_model", "cars_naive"]].sum()
    return dict(
        sites_tested=int(j.site_id.nunique()),
        monthly_miss_model=miss(j.cars_model), monthly_miss_same_month_last_year=miss(j.cars_naive),
        yearly_miss_model=float((np.abs(yearly.cars_model - yearly.cars) / yearly.cars).mean()),
        yearly_miss_same_month_last_year=float((np.abs(yearly.cars_naive - yearly.cars) / yearly.cars).mean()),
        chain_total_miss_model=float(yearly.cars_model.sum() / yearly.cars.sum() - 1),
    )


# ---------------------------------------------------------------------------
# 5. Site 28
# ---------------------------------------------------------------------------
def npv(rate, cash_flows):
    """cash_flows[0] is today (usually the negative build cost), then one per year."""
    return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows)))


def irr(cash_flows):
    """The discount rate at which NPV is zero, found by halving the search range."""
    low, high = -0.99, 1.0
    if npv(low, cash_flows) < 0 or npv(high, cash_flows) > 0:
        return float("nan")
    for _ in range(100):
        mid = (low + high) / 2
        low, high = (mid, high) if npv(mid, cash_flows) > 0 else (low, mid)
    return (low + high) / 2


def payback_years(cash_flows):
    """Years until the build cost is earned back (with part-years), or None if never."""
    total = cash_flows[0]
    for year, cf in enumerate(cash_flows[1:], start=1):
        if total + cf >= 0:
            return year - 1 + (-total / cf)
        total += cf
    return None


SITE28_ASSUMPTIONS = dict(
    discount_rate=0.10,       # the return the owners require
    years=10,
    ramp=[0.55, 0.85, 1.0],   # share of full volume in years 1, 2, 3 onward
    growth=0.02,              # yearly rise in prices and costs after year 1
    upkeep_share=0.015,       # share of revenue set aside each year to replace equipment
    exit_multiple=6.0,        # the site is valued at this many times its year-10 profit
    opening_cost=250_000,     # launch marketing and training, year 1
)


def volume_model(data, year=2025):
    """Learn yearly cars from traffic, competitors and income, using the mature sites."""
    t = data["monthly"][data["monthly"].year == year].groupby("site_id").cars.sum().rename("cars")
    s = data["sites"].merge(t, left_on="site_id", right_index=True)
    s = s[s.site_id.map(months_open(data["sites"], f"{year}-12")) >= MATURE_AFTER_MONTHS + 12]
    X = np.column_stack([np.ones(len(s)), s.traffic / 1000, s.competitors, s.income / 1000])
    y = s.cars.to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    sigma = float(np.sqrt(resid @ resid / dof))
    return dict(coef=coef, cov=np.linalg.inv(X.T @ X), sigma=sigma, n=len(y),
                r2=float(1 - resid @ resid / ((y - y.mean()) ** 2).sum()))


def predict_cars(model, traffic, competitors, income):
    """Expected yearly cars, and the typical size of the miss for one new site."""
    x = np.array([1.0, traffic / 1000, competitors, income / 1000])
    return float(x @ model["coef"]), float(model["sigma"] * np.sqrt(1 + x @ model["cov"] @ x))


def site_economics(data, year=2025):
    """What a mature site earns and spends per car and per year, from the chain's own books."""
    m = data["monthly"][data["monthly"].year == year]
    mature = m[m.site_id.map(months_open(data["sites"], f"{year}-12")) >= MATURE_AFTER_MONTHS + 12]
    not_rent = [c for c in COSTS if c not in ("rent", "card_fees")]
    fixed, per_car = straight_line(mature.cars, mature[not_rent].sum(axis=1))
    return dict(revenue_per_car=float(mature.revenue.sum() / mature.cars.sum()),
                cost_per_car=per_car, fixed_cost_per_year=fixed * 12, card_fee_share=0.026)


def site28_cash_flows(cars, rent_per_month, build_cost, econ, a=SITE28_ASSUMPTIONS):
    flows, profits = [-build_cost], []
    for year in range(1, a["years"] + 1):
        fill = a["ramp"][min(year, len(a["ramp"])) - 1]
        g = (1 + a["growth"]) ** (year - 1)
        revenue = cars * fill * econ["revenue_per_car"] * g
        costs = (cars * fill * econ["cost_per_car"] + econ["fixed_cost_per_year"] + rent_per_month * 12) * g + revenue * econ["card_fee_share"]
        profit = revenue - costs - (a["opening_cost"] if year == 1 else 0)
        profits.append(profit)
        flows.append(profit - a["upkeep_share"] * revenue)
    flows[-1] += a["exit_multiple"] * profits[-1]
    return flows, profits


def site28(data, a=SITE28_ASSUMPTIONS):
    model, econ = volume_model(data), site_economics(data)
    results = []
    for c in data["candidates"].itertuples():
        expected, miss = predict_cars(model, c.traffic, c.competitors, c.income)
        cases = {}
        for label, cars in [("Low", expected - 1.3 * miss), ("Expected", expected), ("High", expected + 1.3 * miss)]:
            flows, profits = site28_cash_flows(cars, c.rent, c.build_cost, econ, a)
            cases[label] = dict(cars=cars, npv=npv(a["discount_rate"], flows), irr=irr(flows),
                                payback_years=payback_years(flows[:-1] + [flows[-1] - a["exit_multiple"] * profits[-1]]),
                                mature_profit=profits[2], flows=flows)
        # The volume at which the project just breaks even (NPV of zero).
        low, high = 0.0, 400_000.0
        for _ in range(60):
            mid = (low + high) / 2
            low, high = (mid, high) if npv(a["discount_rate"], site28_cash_flows(mid, c.rent, c.build_cost, econ, a)[0]) < 0 else (low, mid)
        results.append(dict(candidate=c.candidate, region=c.region, traffic=int(c.traffic), competitors=int(c.competitors),
                            income=int(c.income), rent=float(c.rent), build_cost=float(c.build_cost),
                            cars_expected=expected, cars_typical_miss=miss, breakeven_cars=high, cases=cases))
    return dict(candidates=results, economics=econ, assumptions=a,
                model=dict(intercept=float(model["coef"][0]), per_1000_traffic=float(model["coef"][1]),
                           per_competitor=float(model["coef"][2]), per_1000_income=float(model["coef"][3]),
                           r2=model["r2"], sites_used=model["n"], typical_miss=model["sigma"]))
