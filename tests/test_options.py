"""
Checks for the options pricer (13_options_pricer).

    python -m pytest tests/test_options.py

Every check uses an answer known in advance, so none needs the downloaded prices.
"""

import math

import numpy as np
import pytest

from conftest import ROOT, load

options = load("13_options_pricer", "options")
Value = options.Value

CASES = [
    dict(price=100.0, strike=100.0, years=1.0, rate=0.05, volatility=0.2, dividend=0.0),
    dict(price=100.0, strike=120.0, years=0.25, rate=0.03, volatility=0.35, dividend=0.02),
    dict(price=718.78, strike=650.0, years=94 / 365, rate=0.0437, volatility=0.16, dividend=0.012),
    dict(price=50.0, strike=40.0, years=2.0, rate=0.0, volatility=0.6, dividend=0.04),
]


# --- The three operations added to the engine ------------------------------------------


@pytest.mark.parametrize("operation", [options.exp, options.bell_area, options.bell_height_value])
def test_each_new_operation_reports_the_right_slope(operation):
    for x in (-2.0, -0.3, 0.0, 0.7, 1.9):
        v = Value(x)
        operation(v).backward()
        nudge = 1e-6
        assert v.grad == pytest.approx((operation(x + nudge).data - operation(x - nudge).data) / (2 * nudge), abs=1e-8)


def test_bell_area_at_known_points():
    assert options.bell_area(0.0).data == 0.5
    assert options.bell_area(1.959963984540054).data == pytest.approx(0.975, abs=1e-12)
    assert options.bell_area(-8.0).data + options.bell_area(8.0).data == pytest.approx(1.0)


# --- Black-Scholes ---------------------------------------------------------------------


def test_the_textbook_example():
    # Hull, "Options, Futures and Other Derivatives": share 42, strike 40, six months, 10% interest, 20% volatility.
    assert options.value(42, 40, 0.5, 0.10, 0.20, kind="call") == pytest.approx(4.76, abs=0.005)
    assert options.value(42, 40, 0.5, 0.10, 0.20, kind="put") == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("case", CASES)
def test_call_minus_put_obeys_the_no_free_money_rule(case):
    call, put = options.value(**case, kind="call"), options.value(**case, kind="put")
    share = case["price"] * math.exp(-case["dividend"] * case["years"])
    strike = case["strike"] * math.exp(-case["rate"] * case["years"])
    assert call - put == pytest.approx(share - strike, abs=1e-10)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_the_engine_greeks_match_the_textbook_formulas(case, kind):
    engine, book = options.greeks(**case, kind=kind), options.textbook_greeks(**case, kind=kind)
    for name in book:
        assert engine[name] == pytest.approx(book[name], abs=1e-10), name


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_the_engine_greeks_match_nudging_each_input(case, kind):
    """A check that uses no calculus at all: move one input a little and see how far the price moves."""
    g = options.greeks(**case, kind=kind)

    def moved(name, by):
        return options.value(**{**case, name: case[name] + by}, kind=kind)

    h = case["price"] * 1e-4
    assert g["delta"] == pytest.approx((moved("price", h) - moved("price", -h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((moved("price", h) - 2 * g["value"] + moved("price", -h)) / h**2, abs=1e-5)
    assert g["vega"] == pytest.approx((moved("volatility", 1e-5) - moved("volatility", -1e-5)) / 2e-5 / 100, abs=1e-6)
    assert g["rho"] == pytest.approx((moved("rate", 1e-5) - moved("rate", -1e-5)) / 2e-5 / 100, abs=1e-6)
    assert g["theta"] == pytest.approx(-(moved("years", 1e-5) - moved("years", -1e-5)) / 2e-5 / 365, abs=1e-6)


def test_common_sense():
    base = CASES[0]
    assert options.value(**{**base, "volatility": 0.3}) > options.value(**base)  # more uncertainty, worth more
    assert options.value(**{**base, "years": 2.0}) > options.value(**base)  # more time, worth more
    assert options.value(**{**base, "strike": 110.0}) < options.value(**base)  # a dearer strike, a cheaper call
    assert 0 < options.greeks(**base)["delta"] < 1 and -1 < options.greeks(**base, kind="put")["delta"] < 0


# --- The tree --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_the_tree_closes_in_on_the_formula(case):
    for kind in ("call", "put"):
        formula = options.value(**case, kind=kind)
        assert options.tree(**case, kind=kind, steps=2000) == pytest.approx(formula, abs=max(0.002 * formula, 0.002))


def test_using_an_option_early():
    case = CASES[0]  # no dividend
    assert options.tree(**case, kind="put", american=True) > options.tree(**case, kind="put") + 0.1  # worth something for a put
    assert options.tree(**case, kind="call", american=True) == pytest.approx(options.tree(**case, kind="call"), abs=1e-9)  # never for a call with no dividend
    deep = dict(case, price=50.0)  # a put far in the money: best used at once
    assert options.tree(**deep, kind="put", american=True) == pytest.approx(50.0, abs=1e-9)


# --- Imagined futures ------------------------------------------------------------------


def test_bell_curve_futures_give_the_formula_price():
    case = CASES[2]
    endings = options.bell_curve_endings(400_000, case["years"], case["volatility"], seed=3)
    for kind in ("call", "put"):
        price, error = options.from_paths(endings, case["price"], case["strike"], case["years"], case["rate"], case["dividend"], kind)
        assert abs(price - options.value(**case, kind=kind)) < 4 * error


def test_futures_are_recentred_so_an_opinion_about_the_market_cannot_change_the_price():
    case = CASES[0]
    endings = options.bell_curve_endings(100_000, 1.0, 0.2, seed=1)
    plain = options.from_paths(endings, case["price"], 100.0, 1.0, 0.05)[0]
    bullish = options.from_paths(endings * 1.5, case["price"], 100.0, 1.0, 0.05)[0]  # every future 50% better
    assert bullish == pytest.approx(plain, rel=1e-12)


# --- Implied volatility ----------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_volatility_gives_back_the_volatility_that_made_the_price(case, kind):
    found, method = options.implied_volatility(options.value(**case, kind=kind), case["price"], case["strike"], case["years"],
                                               case["rate"], case["dividend"], kind)
    assert method in ("newton", "adam")
    assert found == pytest.approx(case["volatility"], abs=1e-6)


def test_when_newton_fails_the_adam_fallback_finds_it():
    # Far from the strike, short-dated, very volatile: Newton's first step from a 20% guess leaves the allowed range.
    hard = dict(price=100.0, strike=148.0, years=0.021, rate=0.03, dividend=0.01)
    target = options.value(**hard, volatility=1.39, kind="call")
    found, method = options.implied_volatility(target, **hard, kind="call")
    assert method == "adam"
    assert found == pytest.approx(1.39, abs=1e-6)


def test_the_fallback_on_many_awkward_options():
    rng, used = np.random.default_rng(5), {"newton": 0, "adam": 0, "no answer": 0}
    for _ in range(150):
        vol, strike, years = float(rng.uniform(0.05, 1.5)), float(100 * math.exp(rng.uniform(-0.7, 0.7))), float(math.exp(rng.uniform(math.log(0.01), math.log(5))))
        target = options.value(100.0, strike, years, 0.03, vol, 0.01, "put")
        found, method = options.implied_volatility(target, 100.0, strike, years, 0.03, 0.01, "put")
        used[method] += 1
        if method != "no answer":
            assert options.value(100.0, strike, years, 0.03, found, 0.01, "put") == pytest.approx(target, abs=1e-6)
    assert used["newton"] > 20 and used["adam"] > 20  # both routes really are exercised


def test_a_price_no_volatility_can_explain_is_reported_as_such():
    found, method = options.implied_volatility(5.0, 100.0, 90.0, 0.5, 0.05, 0.0, "call")  # below what it pays if used today
    assert method == "no answer" and math.isnan(found)


# --- The diffusion scenarios -----------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    pytest.importorskip("torch")
    return load("13_options_pricer", "scenarios")


def test_the_new_network_fits_project_4s_generator(scenarios):
    import torch

    torch.manual_seed(0)
    model = scenarios.SequenceNoisePredictor()
    noisy, t = torch.randn(3, 1, scenarios.SIDE, scenarios.SIDE), torch.tensor([0, 500, 999])
    assert model(noisy, t).shape == noisy.shape
    made = scenarios.diffusion.generate(model, 2, scenarios.SIDE)  # project 4's own loop, unchanged, all 1,000 steps
    assert made.shape == (2, 1, scenarios.SIDE, scenarios.SIDE) and torch.isfinite(made).all()


def test_the_network_reads_days_in_order_not_as_a_picture(scenarios):
    """Changing day 40 must be able to reach day 39 and day 41 (neighbours in time), which sit on different rows of the grid."""
    import torch

    torch.manual_seed(0)
    model = scenarios.SequenceNoisePredictor().eval()
    grid = torch.zeros(1, 1, scenarios.SIDE, scenarios.SIDE)
    changed = grid.clone()
    changed.view(-1)[40] = 1.0
    t = torch.tensor([10])
    with torch.no_grad():
        moved = (model(changed, t) - model(grid, t)).view(-1).abs()
    assert moved[39] > 0 and moved[41] > 0


def test_fingerprints_of_a_bell_curve(scenarios):
    prints = scenarios.fingerprint(scenarios.bell_curve_moves(20_000, 0.01))
    assert prints["yearly_volatility"] == pytest.approx(0.01 * math.sqrt(252), rel=0.01)
    assert abs(prints["fat_tails"]) < 0.05 and abs(prints["echo_1_day"]) < 0.03 and abs(prints["three_month_lean"]) < 0.06
    assert prints["days_beyond_3"] == pytest.approx(0.0027, abs=0.0003)


def test_fingerprints_notice_wild_spells(scenarios):
    rng = np.random.default_rng(0)
    wild = rng.standard_normal((4000, scenarios.DAYS)) * 0.01
    wild[:400, 32:] *= 4  # in one stretch in ten, the second half is four times as wild
    prints = scenarios.fingerprint(wild)
    assert prints["fat_tails"] > 3 and prints["echo_1_day"] > 0.1 and prints["calm_to_wild"] > 2


def test_the_saved_model_loads_and_was_trained(scenarios):
    if not (ROOT / "13_options_pricer" / "scenario_model.pt").exists():
        pytest.skip("run 13_options_pricer/scenarios.py train first")
    _, saved = scenarios.load_model()
    assert saved["loss_curve"][-1] < 0.85 * saved["loss_curve"][0]
    assert 0.005 < saved["typical"] < 0.02  # a typical day on the S&P 500 is about 1%
