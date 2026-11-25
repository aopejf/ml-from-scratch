"""
Options pricer.

An option is the right, not the duty, to buy (a "call") or sell (a "put") something at a
fixed price (the "strike") on a future date. This file prices them three ways and measures
how the price reacts to everything it depends on.

    black_scholes     the famous formula
    tree              a tree of up and down moves; also handles options you can use early (American)
    from_paths        average the payoff over many imagined futures (Monte Carlo)

The reactions are called the Greeks. The textbook way is to work out five more formulas by
calculus. Here they are not worked out at all: the price formula is written with the
Value objects from 2_autograd/engine.py, and the engine's backpropagation hands back all of
them in one pass. This is the same code that trains the neural networks in this repository.

Finding the volatility that explains a market price (implied volatility) uses Newton's method
with the engine's slope, and falls back on the Adam optimizer from 1_optimizers when Newton fails.
"""

import math
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "2_autograd"))
sys.path.insert(0, str(here.parent / "1_optimizers"))
from engine import Value  # noqa: E402
from optim import Adam  # noqa: E402

# ---------------------------------------------------------------------------
# Three operations the engine did not have. Each one says what its answer is
# and how fast that answer moves when its input moves. That is all the engine needs.
# ---------------------------------------------------------------------------


def as_value(x):
    return x if isinstance(x, Value) else Value(x)


def exp(x):
    x = as_value(x)
    e = math.exp(x.data)
    return Value(e, (x,), (e,))  # the slope of e^x is e^x


def bell_height(z):
    """The height of the standard bell curve at z."""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def bell_area(x):
    """The share of a bell curve lying below x (the normal cumulative distribution, written N(x))."""
    x = as_value(x)
    return Value(0.5 * (1 + math.erf(x.data / math.sqrt(2))), (x,), (bell_height(x.data),))  # slope = the curve's height


def bell_height_value(x):
    """The bell curve's height as a Value, needed once: to differentiate delta and get gamma."""
    x = as_value(x)
    h = bell_height(x.data)
    return Value(h, (x,), (-x.data * h,))


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
INPUTS = ["price", "strike", "years", "rate", "volatility", "dividend"]


def black_scholes(price, strike, years, rate, volatility, dividend=0.0, kind="call"):
    """
    price       what the share costs today
    strike      the fixed price in the contract
    years       time until the contract ends
    rate        the yearly interest rate on safe money
    volatility  how much the share price swings in a year (0.20 = 20%)
    dividend    the yearly income the share pays, as a share of its price

    Give it plain numbers or Values. With Values, the answer remembers how it was made.
    """
    price, strike, years, rate, volatility, dividend = map(as_value, (price, strike, years, rate, volatility, dividend))
    swing = volatility * years**0.5  # the typical move by the end date
    d1 = ((price / strike).log() + (rate - dividend + volatility * volatility * 0.5) * years) / swing
    d2 = d1 - swing
    share_today = price * exp(-dividend * years)  # the share, less the dividends you will not receive
    strike_today = strike * exp(-rate * years)  # the strike, in today's money
    if kind == "call":
        return share_today * bell_area(d1) - strike_today * bell_area(d2)
    return strike_today * bell_area(-d2) - share_today * bell_area(-d1)


def value(price, strike, years, rate, volatility, dividend=0.0, kind="call"):
    """Just the number."""
    return black_scholes(price, strike, years, rate, volatility, dividend, kind).data


def greeks(price, strike, years, rate, volatility, dividend=0.0, kind="call"):
    """
    The option's price and how it reacts, from the engine. No Greek formula appears anywhere in this function.

        delta   change in the option when the share rises $1
        gamma   change in delta when the share rises $1
        vega    change in the option when volatility rises 1 point (20% to 21%)
        theta   change in the option when one day passes
        rho     change in the option when the interest rate rises 1 point
    """
    v = {name: Value(x) for name, x in zip(INPUTS, (price, strike, years, rate, volatility, dividend))}
    option = black_scholes(*v.values(), kind=kind)
    option.backward()  # one backward pass fills in the slope against every input
    answer = dict(value=option.data, delta=v["price"].grad, vega=v["volatility"].grad / 100,
                  theta=-v["years"].grad / 365, rho=v["rate"].grad / 100)

    # Gamma is a slope of a slope. Write delta itself with Values and ask the engine for its slope against the share price.
    w = {name: Value(x) for name, x in zip(INPUTS, (price, strike, years, rate, volatility, dividend))}
    swing = w["volatility"] * w["years"] ** 0.5
    d1 = ((w["price"] / w["strike"]).log() + (w["rate"] - w["dividend"] + w["volatility"] * w["volatility"] * 0.5) * w["years"]) / swing
    delta = exp(-w["dividend"] * w["years"]) * (bell_area(d1) if kind == "call" else bell_area(d1) - 1)
    delta.backward()
    answer["gamma"] = w["price"].grad
    return answer


def textbook_greeks(price, strike, years, rate, volatility, dividend=0.0, kind="call"):
    """The formulas from the textbook, used only to check the engine's answers."""
    swing = volatility * math.sqrt(years)
    d1 = (math.log(price / strike) + (rate - dividend + 0.5 * volatility**2) * years) / swing
    d2 = d1 - swing
    area = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    carry, discount = math.exp(-dividend * years), math.exp(-rate * years)
    sign = 1 if kind == "call" else -1
    theta = (-price * carry * bell_height(d1) * volatility / (2 * math.sqrt(years))
             - sign * rate * strike * discount * area(sign * d2) + sign * dividend * price * carry * area(sign * d1))
    return dict(delta=sign * carry * area(sign * d1), gamma=carry * bell_height(d1) / (price * swing),
                vega=price * carry * bell_height(d1) * math.sqrt(years) / 100, theta=theta / 365,
                rho=sign * strike * years * discount * area(sign * d2) / 100)


# ---------------------------------------------------------------------------
# A tree of up and down moves (Cox-Ross-Rubinstein)
# ---------------------------------------------------------------------------
def tree(price, strike, years, rate, volatility, dividend=0.0, kind="call", steps=500, american=False):
    """
    Cut the time into many short steps. In each one the share goes up a little or down a little.
    Work out the option's worth at every end point, then step back to today, averaging as you go.
    For an American option, check at every step whether using it now beats waiting.
    """
    dt = years / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1 / up
    chance_up = (math.exp((rate - dividend) * dt) - down) / (up - down)
    discount = math.exp(-rate * dt)
    sign = 1 if kind == "call" else -1
    ups = np.arange(steps + 1)
    worth = np.maximum(sign * (price * up**ups * down ** (steps - ups) - strike), 0.0)
    for step in range(steps - 1, -1, -1):
        worth = discount * (chance_up * worth[1:] + (1 - chance_up) * worth[:-1])
        if american:
            ups = np.arange(step + 1)
            worth = np.maximum(worth, sign * (price * up**ups * down ** (step - ups) - strike))
    return float(worth[0])


# ---------------------------------------------------------------------------
# Averaging over imagined futures (Monte Carlo)
# ---------------------------------------------------------------------------
def bell_curve_endings(n, years, volatility, seed=0):
    """End-of-period growth factors if daily moves followed a bell curve: the assumption inside Black-Scholes."""
    z = np.random.default_rng(seed).standard_normal(n)
    return np.exp(-0.5 * volatility**2 * years + volatility * math.sqrt(years) * z)


def from_paths(growth, price, strike, years, rate, dividend=0.0, kind="call"):
    """
    growth: for each imagined future, the share's end price divided by its start price.
    The futures are first re-centred so that, on average, the share earns exactly the safe
    interest rate. That is the rule for pricing (the official name is the risk-neutral measure):
    it removes any opinion about whether shares go up.
    Returns the price and how uncertain it is (one standard error).
    """
    forward = price * math.exp((rate - dividend) * years)
    endings = forward * growth / growth.mean()
    payoff = np.maximum(endings - strike, 0) if kind == "call" else np.maximum(strike - endings, 0)
    discount = math.exp(-rate * years)
    return float(discount * payoff.mean()), float(discount * payoff.std() / math.sqrt(len(payoff)))


# ---------------------------------------------------------------------------
# Implied volatility: the volatility that makes the formula match a price seen in the market
# ---------------------------------------------------------------------------
LOWEST, HIGHEST = 0.001, 5.0
FAINTEST = 1e-140  # a slope smaller than this is treated as "no slope, direction only"


def implied_volatility(target, price, strike, years, rate, dividend=0.0, kind="call"):
    """
    Returns (volatility, method). The method is "newton" or "adam".

    Newton's method: guess, see how far off the price is, divide by the slope (vega, from the
    engine), move by that much. Usually exact in about five steps. But when an option is far from
    the strike the slope is almost flat, and one step can throw the guess out of range.

    The fallback is Adam from project 1, shrinking the squared price error. Adam sets its
    step by the direction of the slope, not its size, so a nearly flat slope does not bother it.
    """
    floor = value(price, strike, years, rate, LOWEST, dividend, kind)
    if not floor + 1e-13 < target <= value(price, strike, years, rate, HIGHEST, dividend, kind):
        return float("nan"), "no answer"  # no volatility gives this price
    # "Close enough" is judged against the part of the price that volatility is responsible for.
    tolerance = max(1e-13, 1e-9 * (target - floor))

    def miss_and_slope(vol):
        v = Value(vol)
        option = black_scholes(price, strike, years, rate, v, dividend, kind)
        option.backward()
        return option.data - target, v.grad

    vol = 0.2
    for _ in range(50):
        miss, slope = miss_and_slope(vol)
        if abs(miss) < tolerance:
            return vol, "newton"
        if slope < 1e-10:
            break
        vol -= miss / slope
        if not LOWEST < vol < HIGHEST:
            break

    # Fall back on the optimizer. Work with log(volatility) so it can never go negative.
    log_vol = np.array([math.log(0.2)])
    optimizer = Adam(0.05, eps=1e-200)  # the slopes here are tiny, so Adam's usual safety margin would swamp them
    best, best_miss = 0.2, float("inf")
    for step in range(2000):
        vol = float(np.exp(log_vol[0]))
        miss, slope = miss_and_slope(vol)
        if abs(miss) < best_miss:
            best, best_miss = vol, abs(miss)
        if abs(miss) < tolerance:
            break
        if step in (500, 1000, 1500):
            optimizer.learning_rate /= 5  # smaller steps as it closes in
        push = 2 * miss * slope * vol  # slope of miss squared, against log(volatility)
        if abs(push) < FAINTEST:
            # The slope has faded to nothing, but the direction is still known: a price that is
            # too low needs more volatility. Adam only uses the direction, so hand it that.
            push = math.copysign(FAINTEST, miss)
        optimizer.step([log_vol], [np.array([push])])
        log_vol[0] = np.clip(log_vol[0], math.log(LOWEST), math.log(HIGHEST))
    # Polish with a few halving steps around Adam's answer, so the result is exact.
    low, high = max(best * 0.9, LOWEST), min(best * 1.1, HIGHEST)
    if (value(price, strike, years, rate, low, dividend, kind) - target) * (value(price, strike, years, rate, high, dividend, kind) - target) < 0:
        for _ in range(60):
            best = (low + high) / 2
            if value(price, strike, years, rate, best, dividend, kind) < target:
                low = best
            else:
                high = best
    return best, "adam"
