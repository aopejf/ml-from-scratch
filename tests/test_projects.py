"""
Automatic checks. Each one tests a claim the projects depend on.

    python -m pytest
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import load


# --- 1. Optimizers -----------------------------------------------------------


@pytest.mark.parametrize("name, learning_rate", [("GradientDescent", 0.1), ("Momentum", 0.02), ("Adam", 0.1)])
def test_optimizer_finds_the_bottom_of_a_bowl(name, learning_rate):
    optim = load("1_optimizers", "optim")
    optimizer = getattr(optim, name)(learning_rate)
    target = np.array([3.0, -2.0])
    position = np.zeros(2)
    for _ in range(500):
        optimizer.step([position], [2 * (position - target)])  # slope of (position - target)^2
    assert np.allclose(position, target, atol=1e-2)


# --- 2. The backpropagation engine -------------------------------------------


def test_backprop_matches_nudging_each_input():
    Value = load("2_autograd", "engine").Value

    def formula(a, b, c):
        return ((a * b + c).tanh() * (a - c) ** 2 + (b * c).sigmoid() / (a + 5.0)).log()

    inputs = [0.7, -1.3, 2.1]
    values = [Value(v) for v in inputs]
    formula(*values).backward()

    for i, value in enumerate(values):
        up, down = list(inputs), list(inputs)
        up[i] += 1e-6
        down[i] -= 1e-6
        nudged = (formula(*map(Value, up)).data - formula(*map(Value, down)).data) / 2e-6
        assert value.grad == pytest.approx(nudged, rel=1e-5, abs=1e-8)


def test_backprop_counts_a_value_used_twice():
    Value = load("2_autograd", "engine").Value
    a = Value(3.0)
    (a * a + a).backward()  # slope of a^2 + a is 2a + 1
    assert a.grad == pytest.approx(7.0)


def test_scratch_network_learns():
    nn = load("2_autograd", "nn")
    Adam = load("1_optimizers", "optim").Adam
    points = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    answers = [0, 1, 1, 0]  # XOR: a single neuron cannot learn this, a small network can
    network = nn.Network(2, [4, 1], seed=1)
    params = network.parameters()
    optimizer = Adam(0.1)
    for _ in range(300):
        loss = nn.cross_entropy([network(p) for p in points], answers)
        nn.zero_grads(params)
        loss.backward()
        nn.optimizer_step(optimizer, params)
    assert [round(network(p).data) for p in points] == answers


# --- 3. The GPT --------------------------------------------------------------


def test_gpt_cannot_see_the_future():
    """Changing a later token must not change the predictions made before it."""
    MiniGPT = load("3_mini_gpt", "gpt").MiniGPT
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=50, context=16, width=32, n_heads=4, n_layers=2, dropout=0.0).eval()
    tokens = torch.randint(0, 50, (1, 16))
    changed = tokens.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 50
    with torch.no_grad():
        before, _ = model(tokens)
        after, _ = model(changed)
    assert torch.allclose(before[0, :10], after[0, :10], atol=1e-5)
    assert not torch.allclose(before[0, 10:], after[0, 10:], atol=1e-5)


def test_untrained_gpt_is_guessing_evenly():
    """Before training, the loss should be close to ln(vocabulary size): no better than chance."""
    MiniGPT = load("3_mini_gpt", "gpt").MiniGPT
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=200, context=32, width=64, n_heads=4, n_layers=2, dropout=0.0)
    tokens = torch.randint(0, 200, (8, 32))
    _, loss = model(tokens, tokens)
    assert abs(loss.item() - math.log(200)) < 0.5


def test_gpt_can_memorise_one_sentence():
    MiniGPT = load("3_mini_gpt", "gpt").MiniGPT
    torch.manual_seed(0)
    model = MiniGPT(vocab_size=30, context=16, width=32, n_heads=4, n_layers=2, dropout=0.0)
    tokens = torch.randint(0, 30, (1, 17))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


# --- 4. Diffusion ------------------------------------------------------------


def test_noise_schedule_goes_from_image_to_static():
    diffusion = load("4_diffusion", "diffusion")
    images, static = torch.ones(2, 1, 8, 8), torch.zeros(2, 1, 8, 8)
    early = diffusion.add_noise(images, torch.tensor([0, 0]), static)
    late = diffusion.add_noise(images, torch.tensor([diffusion.N_STEPS - 1] * 2), static)
    assert early.mean() > 0.99  # step 0: almost all image
    assert late.mean() < 0.01  # last step: almost none of the image is left


def test_digit_model_returns_an_image_shaped_guess():
    conditional = load("4_diffusion", "conditional")
    model = conditional.ConditionalNoisePredictor(width=16)
    guess = model(torch.randn(3, 1, 28, 28), torch.tensor([0, 500, 999]), torch.tensor([7, 2, conditional.NO_LABEL]))
    assert guess.shape == (3, 1, 28, 28)


# --- 5. The tokenizer --------------------------------------------------------


def test_tokenizer_gives_back_exactly_what_went_in():
    tokenizer_file = load("5_story_gpt", "tokenizer")
    text = "Once upon a time, there was a little cat. The cat liked to play. " * 20
    tokenizer = tokenizer_file.Tokenizer.train(text, vocab_size=300)
    for sample in [text, "A word it never saw: xylophone!", "Accents and symbols: café, naïve, 🙂", ""]:
        assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_tokenizer_squeezes_text_and_marks_story_ends():
    tokenizer_file = load("5_story_gpt", "tokenizer")
    text = "the cat sat on the mat. " * 50
    tokenizer = tokenizer_file.Tokenizer.train(text, vocab_size=300)
    assert len(tokenizer.encode(text)) < len(text.encode()) / 2  # learned pieces are longer than single letters
    pieces = tokenizer.encode("one" + tokenizer_file.END + "two")
    assert pieces.count(tokenizer.end_token) == 1


# --- 8. Portfolio numbers ----------------------------------------------------


def test_portfolio_numbers_on_prices_with_known_answers():
    metrics = load("8_portfolio_dashboard", "metrics")
    days = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"UP": [100, 110, 121, 121, 133.1], "DIP": [100, 100, 50, 75, 100]}, index=days)
    n = metrics.compute(prices)

    assert n["growth"]["UP"].iloc[-1] == pytest.approx(1.331)  # $1 became $1.331
    assert n["growth"]["DIP"].iloc[-1] == pytest.approx(1.0)
    assert n["drawdown"]["DIP"].min() == pytest.approx(-0.5)  # fell to half its peak
    assert n["drawdown"]["UP"].min() == pytest.approx(0.0)  # never below its peak
    assert n["daily"][metrics.MIX].iloc[1] == pytest.approx((0.10 - 0.50) / 2)  # the mix is the average move
    assert n["correlation"].loc["UP", "UP"] == pytest.approx(1.0)
