"""
Checks for the loan default predictor (12_loan_default).

    python -m pytest tests/test_loan.py

Every check but the last uses made-up data with an answer known in advance, so none needs the downloaded files.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, load

loan = load("12_loan_default", "loan")
recession = load("12_loan_default", "recession")


# --- The lending sums -------------------------------------------------------------


def test_break_even_is_where_expected_gain_equals_expected_loss():
    line = loan.break_even_risk(0.15, 0.75)
    assert line == pytest.approx(1 / 6)
    assert (1 - line) * 0.15 == pytest.approx(line * 0.75)


def test_profit_on_a_hand_made_book():
    balance = np.array([100.0, 100.0, 100.0])
    defaulted = np.array([0, 1, 0])
    approved = np.array([True, True, False])
    assert loan.profit(balance, defaulted, approved, 0.15, 0.75) == pytest.approx(15 - 75)


def test_lending_only_below_the_line_beats_every_other_cutoff_when_the_risks_are_right():
    rng = np.random.default_rng(0)
    risk, balance = rng.uniform(0.01, 0.9, 5000), rng.uniform(100, 1000, 5000)
    cutoffs = np.round(np.arange(0.02, 1.0, 0.02), 2)
    expected = [row["profit"] for row in loan.policy_curve(balance, risk, risk, cutoffs)]
    best = cutoffs[int(np.argmax(expected))]
    assert abs(best - loan.break_even_risk()) <= 0.02


# --- How good is the ranking? -----------------------------------------------------


def test_ranking_score_on_cases_with_known_answers():
    y = np.array([0, 0, 1, 1])
    assert loan.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0  # perfect
    assert loan.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0  # perfectly wrong
    assert loan.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # no idea
    assert loan.auc(y, np.array([0.1, 0.8, 0.2, 0.9])) == 0.75  # three of the four pairs the right way round
    assert loan.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_calibration_groups_are_equal_sized_and_honest_risks_come_out_calibrated():
    rng = np.random.default_rng(1)
    risk = rng.uniform(0, 1, 1000)
    table = loan.calibration((rng.uniform(0, 1, 1000) < risk).astype(int), risk)
    assert list(table.customers) == [100] * 10
    assert np.abs(table.predicted - table.actual).max() < 0.15


# --- The stress test --------------------------------------------------------------


def test_stress_raises_the_average_by_the_multiplier_and_keeps_the_order():
    risk = np.random.default_rng(2).uniform(0.01, 0.6, 2000)
    harder = loan.stress(risk, 1.48)
    assert harder.mean() == pytest.approx(risk.mean() * 1.48, rel=1e-9)
    assert (harder > risk).all()
    assert (np.argsort(harder) == np.argsort(risk)).all()


def test_a_multiplier_of_one_changes_nothing():
    risk = np.random.default_rng(3).uniform(0.01, 0.6, 500)
    assert loan.stress(risk, 1.0) == pytest.approx(risk, abs=1e-12)


# --- Learning, on the hand-built engine --------------------------------------------


@pytest.fixture(scope="module")
def planted():
    """Made-up customers where the true rule is known: the first figure raises risk, the second lowers it, the third is noise."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 3))
    truth = 1 / (1 + np.exp(-(2.0 * x[:, 0] - 1.5 * x[:, 1])))
    y = (rng.uniform(size=600) < truth).astype(int)
    model, losses = loan.train(x[:400], y[:400], [1], steps=300, batch=32, learning_rate=0.05)
    return x, y, loan.weights_of(model), losses


def test_the_network_learns_a_planted_pattern(planted):
    x, y, weights, losses = planted
    assert np.mean(losses[-50:]) < np.mean(losses[:20])
    assert loan.auc(y[400:], loan.predict(weights, x[400:])) > 0.85
    up, down, noise = weights[0]["weights"][0]
    assert up > 1 and down < -0.7 and abs(noise) < 0.5


def test_fast_scoring_matches_the_engine(planted):
    x, _, weights, _ = planted
    model = loan.network_from(weights)
    for row in x[:5]:
        assert model([float(v) for v in row]).data == pytest.approx(loan.predict(weights, row[None])[0], abs=1e-12)


def test_reasons_add_up_to_the_score():
    saved, _ = loan.load_weights()
    rng = np.random.default_rng(4)
    for _ in range(3):
        row = rng.standard_normal(len(loan.FEATURES))
        risk, average, pushes = loan.reasons(saved["layers"], row, steps=50)
        assert risk == pytest.approx(loan.predict(saved["layers"], row[None])[0], abs=1e-12)
        assert sum(pushes.values()) == pytest.approx(risk - average, abs=2e-3)


def test_protected_details_are_not_inputs():
    saved, _ = loan.load_weights()
    assert saved["features"] == list(loan.FEATURES)
    assert not set(loan.LEFT_OUT) & set(saved["features"])
    assert len(saved["layers"][0]["weights"][0]) == len(loan.FEATURES)


def test_features_from_a_hand_made_customer():
    months = ["sep", "aug", "jul", "jun", "may", "apr"]
    row = dict(credit_limit=100000, late_sep=2, late_aug=-1, late_jul=0, late_jun=0, late_may=3, late_apr=0)
    row.update({f"bill_{m}": 50000 for m in months})
    row.update({f"paid_{m}": 5000 for m in months})
    f = loan.build_features(pd.DataFrame([row])).iloc[0]
    assert f.late_now == 2 and f.late_before == 0 and f.late_worst == 3 and f.months_late == 2
    assert f.used == pytest.approx(0.5) and f.paid_share == pytest.approx(0.1) and f.limit == pytest.approx(5.0)
    assert f.balance_change == 0 and f.dormant == 0


# --- The recession detector --------------------------------------------------------


def planted_economy(seed=0):
    """A made-up economy: mostly calm, with five planted slumps. The detector is not told where they are."""
    rng = np.random.default_rng(seed)
    n = 600
    slump = np.zeros(n, bool)
    for start in (60, 170, 290, 400, 520):
        slump[start:start + 14] = True
    x = rng.standard_normal((n, 3)) * 0.5
    x[slump] += np.array([2.5, -2.5, -2.5])
    return x, slump


def mood_chances(model, x):
    return recession.forward_backward(recession.bell(x, model["mean"], model["spread"]), model["stay"], model["start"])


def test_the_detector_finds_planted_slumps_without_being_told():
    x, slump = planted_economy()
    model = recession.fit(x)
    worst = int(np.argmax(model["mean"][:, 0]))
    called = mood_chances(model, x)[0][:, worst] > 0.5
    assert (called == slump).mean() > 0.97


def test_each_round_of_fitting_explains_the_data_at_least_as_well():
    x, _ = planted_economy(1)
    history = recession.fit(x)["log_likelihood"]
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:]))


def test_mood_chances_are_proper_probabilities():
    x, _ = planted_economy(2)
    model = recession.fit(x)
    chance, _, ahead, _ = mood_chances(model, x)
    assert chance.sum(axis=1) == pytest.approx(1) and ahead.sum(axis=1) == pytest.approx(1)
    assert model["stay"].sum(axis=1) == pytest.approx(1)


def test_on_the_real_economy_every_official_recession_is_caught():
    path = ROOT / "12_loan_default" / "data" / "economy.csv"
    if not path.exists():
        pytest.skip("run 12_loan_default/fetch_data.py first")
    economy = pd.read_csv(path, index_col=0)
    _, months = recession.run(economy)
    marks = recession.score(months)
    assert marks["recessions_caught"] == marks["official_recessions"] >= 11
    assert marks["false_alarms"] <= 2 and marks["months_agreeing"] > 0.85
