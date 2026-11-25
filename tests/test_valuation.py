"""
Checks for the company valuation model (11_company_valuation).

    python -m pytest tests/test_valuation.py
"""

import pytest

from conftest import load

v = load("11_company_valuation", "valuation")


@pytest.fixture(scope="module")
def result():
    return v.run()


def steady(growth, wacc, years=10):
    """A made-up company that grows at one rate for ever, so the right answer is known from a formula."""
    start = dict(fiscal_year=2025, revenue=1000.0, cash=0.0, debt=0.0, shares=1.0, price=1.0)
    a = dict(growth_start=growth, terminal_growth=growth, margin=0.10, tax_rate=0.25, depreciation_share=0.02,
             capex_share=0.03, working_capital_share=0.0, wacc=wacc, years=years)
    return start, a


# --- The sums, against answers known in advance ------------------------------------


def test_a_company_growing_at_one_rate_matches_the_textbook_formula():
    # Cash next year / (required return - growth). Ten forecast years plus "everything after" must give the same.
    start, a = steady(growth=0.03, wacc=0.09)
    cash_next_year = 1000 * 1.03 * (0.10 * 0.75 + 0.02 - 0.03)
    assert v.dcf(start, a)["enterprise_value"] == pytest.approx(cash_next_year / (0.09 - 0.03), rel=1e-12)


def test_the_answer_does_not_depend_on_how_many_years_are_forecast():
    start, a = steady(growth=0.02, wacc=0.08)
    five, twenty = v.dcf(start, {**a, "years": 5}), v.dcf(start, {**a, "years": 20})
    assert five["enterprise_value"] == pytest.approx(twenty["enterprise_value"], rel=1e-12)


def test_growth_fades_in_a_straight_line(result):
    path = v.growth_path(result["assumptions"])
    assert path[0] == result["assumptions"]["growth_start"] and path[-1] == pytest.approx(result["assumptions"]["terminal_growth"])
    steps = [b - a for a, b in zip(path, path[1:])]
    assert max(steps) - min(steps) < 1e-12


def test_debt_comes_off_and_cash_goes_on(result):
    start, a = result["start"], result["assumptions"]
    more_debt = v.dcf({**start, "debt": start["debt"] + 1e9}, a)
    assert result["base"]["equity_value"] - more_debt["equity_value"] == pytest.approx(1e9)


# --- The valuation behaves the way finance says it should ----------------------------


def test_value_moves_the_right_way(result):
    start, a, base = result["start"], result["assumptions"], result["base"]["value_per_share"]
    assert v.dcf(start, {**a, "wacc": a["wacc"] + 0.01})["value_per_share"] < base  # a higher required return lowers value
    assert v.dcf(start, {**a, "margin": a["margin"] + 0.005})["value_per_share"] > base
    assert v.dcf(start, {**a, "growth_start": a["growth_start"] + 0.02})["value_per_share"] > base
    assert v.dcf(start, {**a, "capex_share": a["capex_share"] + 0.005})["value_per_share"] < base


def test_the_parts_add_up(result):
    b, start = result["base"], result["start"]
    assert sum(r["present_value"] for r in b["rows"]) + b["terminal_present"] == pytest.approx(b["enterprise_value"])
    assert b["enterprise_value"] - start["debt"] + start["cash"] == pytest.approx(b["equity_value"])
    for r in b["rows"]:
        assert r["after_tax"] + r["depreciation"] - r["capex"] - r["working_capital_change"] == pytest.approx(r["free_cash_flow"])


def test_reverse_valuation_really_hits_the_share_price(result):
    start, a = result["start"], result["assumptions"]
    for name, needed in result["implied"].items():
        if needed is not None:
            assert v.dcf(start, {**a, name: needed})["value_per_share"] == pytest.approx(start["price"], rel=1e-6)


# --- The real data is what it claims to be --------------------------------------------


def test_filed_figures_are_consistent(result):
    h = result["history"]
    assert len(h) == 10 and h.fiscal_year.is_monotonic_increasing
    assert (h.revenue > h.operating_income).all() and (h.operating_income > h.net_income).all()
    assert h.tax_rate.between(0.10, 0.40).all()
    # Free cash flow built here should be in the same region as the company's own cash-flow statement.
    reported = (h.cash_from_operations - h.capex).iloc[1:]
    assert ((h.free_cash_flow.iloc[1:] / reported).between(0.5, 1.5)).all()


def test_required_return_is_built_from_its_parts(result):
    c = result["capital"]
    assert c["cost_of_equity"] == pytest.approx(c["risk_free_rate"] + c["beta"] * c["equity_risk_premium"])
    assert c["cost_of_debt"] * (1 - c["tax_rate"]) < c["wacc"] < c["cost_of_equity"]
    assert 0.3 < c["beta"] < 2.0
