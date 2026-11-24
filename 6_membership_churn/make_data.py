"""
Makes a pretend customer list for a car wash with monthly memberships.

    python make_data.py

IMPORTANT: every row is invented by this script. No real customers, no real
business. That is deliberate: it means the project can be shared publicly,
and it means we know the true reasons people cancel (we wrote them below),
so we can check whether the model finds them.

Writes members.csv: one row per member, with what was known about them on
the 1st of the month and whether they cancelled during that month.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 20_000

# --- what we know about each member on the 1st of the month ---------------
months_as_member = np.minimum(rng.geometric(0.07, N), 72)
plan = rng.choice(["basic", "plus", "premium"], N, p=[0.45, 0.35, 0.20])
price = pd.Series(plan).map({"basic": 19.99, "plus": 29.99, "premium": 39.99}).to_numpy()
joined_through = rng.choice(["on site", "app", "website"], N, p=[0.6, 0.25, 0.15])

habit = rng.gamma(2.0, 1.6, N)  # how often this person likes to wash, per month
fading = rng.random(N) < 0.18  # some members are quietly losing interest
washes_previous_month = rng.poisson(habit)
washes_last_month = rng.poisson(habit * np.where(fading, 0.35, 1.0))
days_since_last_wash = np.where(
    washes_last_month > 0,
    rng.integers(0, 30, N) // np.maximum(washes_last_month, 1),
    30 + rng.integers(0, 45, N),
)
rainy_days_last_month = rng.binomial(30, 0.25, N)
card_declined_recently = rng.random(N) < 0.06
complained_recently = rng.random(N) < 0.04
used_family_plan = rng.random(N) < 0.15

# --- the hidden truth: what actually drives cancelling ---------------------
cost_per_wash = price / np.maximum(washes_last_month, 0.5)  # paying a lot per wash feels bad
score = (
    -4.0
    + 0.045 * np.minimum(cost_per_wash, 60)  # poor value for money
    + 0.9 * (months_as_member <= 3)  # new members leave most
    - 0.012 * np.minimum(months_as_member, 36)  # long-timers stay
    + 0.5 * np.maximum(washes_previous_month - washes_last_month, 0) / (washes_previous_month + 1)  # usage dropping
    + 1.3 * card_declined_recently
    + 1.0 * complained_recently
    - 0.6 * used_family_plan
    + 0.25 * (joined_through == "website")
)
chance_of_cancelling = 1 / (1 + np.exp(-score))
cancelled = rng.random(N) < chance_of_cancelling  # real life has luck in it too

members = pd.DataFrame(
    {
        "months_as_member": months_as_member,
        "plan": plan,
        "monthly_price": price,
        "joined_through": joined_through,
        "washes_last_month": washes_last_month,
        "washes_previous_month": washes_previous_month,
        "days_since_last_wash": days_since_last_wash,
        "rainy_days_last_month": rainy_days_last_month,  # has no effect at all: a trap for the model
        "card_declined_recently": card_declined_recently.astype(int),
        "complained_recently": complained_recently.astype(int),
        "used_family_plan": used_family_plan.astype(int),
        "cancelled": cancelled.astype(int),
    }
)
members.to_csv(Path(__file__).parent / "members.csv", index=False)
print(f"Wrote members.csv: {N:,} pretend members, {cancelled.mean():.1%} cancelled this month")
print(members.head(8).to_string(index=False))
