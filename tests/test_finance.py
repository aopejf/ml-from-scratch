"""
Checks for the car wash finance pack (10_car_wash_finance).

    python -m pytest tests/test_finance.py
"""

import pytest

from conftest import load

finance = load("10_car_wash_finance", "finance")


@pytest.fixture(scope="module")
def data():
    return finance.load()


# --- The investment sums, against answers worked out by hand ----------------


def test_npv_of_a_textbook_project():
    # Pay 1,000 today, receive 600 a year for two years, at 10%: -1000 + 600/1.1 + 600/1.21
    assert finance.npv(0.10, [-1000, 600, 600]) == pytest.approx(41.3223, abs=1e-4)


def test_irr_is_the_rate_where_npv_is_zero():
    flows = [-1000, 600, 600]
    rate = finance.irr(flows)
    assert rate == pytest.approx(0.130662, abs=1e-5)
    assert finance.npv(rate, flows) == pytest.approx(0, abs=1e-6)


def test_payback_counts_part_years_and_admits_never():
    assert finance.payback_years([-1000, 400, 400, 400]) == pytest.approx(2.5)
    assert finance.payback_years([-1000, 100, 100]) is None


def test_site_24_is_worth_more_with_more_cars(data):
    econ = finance.site_economics(data)
    values = [finance.npv(0.10, finance.site24_cash_flows(cars, 20000, 4_000_000, econ)[0]) for cars in (100_000, 130_000, 160_000)]
    assert values[0] < values[1] < values[2]


# --- Budget against actual ----------------------------------------------------


def test_budget_bridge_adds_up_exactly(data):
    for site in (None, 3, 11, 23):  # the chain, an ordinary site, and the site that opened late
        v = finance.variance(data, site_id=site)
        assert v["budget_profit"] + sum(v["effects"].values()) == pytest.approx(v["actual_profit"], abs=0.01)


# --- The scorecard finds what was planted -------------------------------------


def test_scorecard_finds_the_planted_problems(data):
    card, _ = finance.scorecard(data)
    flags = dict(zip(card.site_id, card["flags"]))
    assert "Labour high" in flags[7]
    assert "leaving fast" in flags[11]
    assert "Chemical use high" in flags[14]
    assert "Barely profitable" in flags[18]
    assert flags[23] == "Still filling up"  # a new site is not called a failure
    clean = [s for s in range(1, 21) if s not in (7, 11, 14, 18)]
    assert all(flags[s] == "" for s in clean)  # and no false alarms


# --- Membership ----------------------------------------------------------------


def test_retention_starts_at_everyone_and_only_falls(data):
    curve = finance.retention_curve(data["cohorts"])
    assert curve.iloc[0] == 1.0
    assert (curve.diff().dropna() <= 0).all()


def test_member_counts_reconcile(data):
    m = data["monthly"]
    assert (m.members_start + m.joins - m.cancels == m.members_end).all()


# --- Forecast -------------------------------------------------------------------


def test_forecast_cannot_see_the_year_it_is_tested_on(data):
    changed = dict(data)
    monthly = data["monthly"].copy()
    monthly.loc[monthly.year == 2025, ["retail_cars", "joins", "cancels", "members_end"]] *= 3
    changed["monthly"] = monthly
    before = finance.forecast(data, history_to="2024-12")
    after = finance.forecast(changed, history_to="2024-12")
    assert before.cars.tolist() == after.cars.tolist()


def test_forecast_beats_the_simple_rule(data):
    result = finance.backtest(data)
    assert result["monthly_miss_model"] < result["monthly_miss_same_month_last_year"]
    assert result["monthly_miss_model"] < 0.08
