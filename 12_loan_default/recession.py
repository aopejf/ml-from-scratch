"""
An unsupervised recession detector.

"Unsupervised" means nobody tells the model which months were recessions. It is given
three monthly figures about the US economy and told only this: the economy is always in
one of three moods, each mood has its own typical figures, and moods tend to last.
It has to work out the moods, and which months belong to which, by itself. They come out
as growing, slowing and shrinking. "Shrinking" is its word for a recession.

The official name is a Markov regime-switching model (a three-state hidden Markov model,
fitted with the Baum-Welch algorithm). The official recession dates are used afterwards,
only to mark its homework.

The three figures, all things you can see while they are happening:
    unemployment_rise   change in unemployment (3-month average) over 6 months
    jobs_growth         % change in jobs over 6 months
    output_growth       % change in factory output over 6 months
"""

import numpy as np
import pandas as pd

FEATURES = ["unemployment_rise", "jobs_growth", "output_growth"]
MOODS = ["growing", "slowing", "shrinking"]
CLIP = 4.0  # a month more than 4 typical swings from normal is treated as 4 (April 2020 would otherwise drown out everything)


def features(economy):
    """Turn the raw monthly figures into the three the model reads."""
    e = economy
    smooth = e.unemployment.rolling(3).mean()
    table = pd.DataFrame({
        "unemployment_rise": smooth - smooth.shift(6),
        "jobs_growth": 100 * (e.jobs / e.jobs.shift(6) - 1),
        "output_growth": 100 * (e.factory_output / e.factory_output.shift(6) - 1),
    }, index=e.index).dropna()
    return table


def standardise(table):
    """Put each figure on the same scale: 0 is typical, 1 is one typical swing above."""
    z = (table - table.median()) / table.std()
    return z.clip(-CLIP, CLIP).to_numpy()


def bell(x, mean, spread):
    """How likely each month's figures are in each mood (a bell curve per figure). Returned as logs."""
    z = (x[:, None, :] - mean[None]) / spread[None]
    return (-0.5 * z**2 - np.log(spread[None]) - 0.5 * np.log(2 * np.pi)).sum(axis=2)


def forward_backward(log_like, stay, start):
    """
    Given the two moods, work out the chance each month was in each one, using the months
    before it (forward) and after it (backward). Rescaled at every step so the numbers never underflow.
    """
    n, k = log_like.shape
    like = np.exp(log_like - log_like.max(axis=1, keepdims=True))
    ahead, scale = np.zeros((n, k)), np.zeros(n)
    ahead[0] = start * like[0]
    scale[0] = ahead[0].sum()
    ahead[0] /= scale[0]
    for t in range(1, n):
        ahead[t] = (ahead[t - 1] @ stay) * like[t]
        scale[t] = ahead[t].sum()
        ahead[t] /= scale[t]
    back = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        back[t] = stay @ (like[t + 1] * back[t + 1]) / scale[t + 1]
    chance = ahead * back
    # The chance of each move from one mood to the next, month by month.
    moves = ahead[:-1, :, None] * stay[None] * (like[1:] * back[1:])[:, None, :] / scale[1:, None, None]
    total_log_like = np.log(scale).sum() + log_like.max(axis=1).sum()
    return chance, moves, ahead, total_log_like


def fit(x, rounds=300, tolerance=1e-8):
    """
    Baum-Welch: guess the moods, work out which months belong to which, re-estimate
    the moods from those months, and repeat until nothing changes.
    The starting guess is deliberately crude: a bit better than typical, typical, and well worse than typical.
    """
    k = len(MOODS)
    worse = np.sign(np.corrcoef(x.T)[0])  # which direction is "bad" for each figure, relative to rising unemployment
    mean = np.linspace(-0.5, 2.5, k)[:, None] * worse[None]
    spread = np.ones((k, x.shape[1]))
    stay = np.full((k, k), 0.05 / (k - 1))
    np.fill_diagonal(stay, 0.95)
    start = np.full(k, 1 / k)
    history = []
    for _ in range(rounds):
        chance, moves, _, log_like = forward_backward(bell(x, mean, spread), stay, start)
        history.append(log_like)
        weight = chance.sum(axis=0)
        mean = (chance.T @ x) / weight[:, None]
        spread = np.sqrt((chance.T @ x**2) / weight[:, None] - mean**2).clip(0.05)
        stay = moves.sum(axis=0)
        stay /= stay.sum(axis=1, keepdims=True)
        start = chance[0]
        if len(history) > 1 and history[-1] - history[-2] < tolerance:
            break
    return dict(mean=mean, spread=spread, stay=stay, start=start, log_likelihood=history)


def read(model, x):
    """
    Two readings for every month.
        hindsight   uses the months after it too. Best for studying history.
        live        uses only that month and earlier. What you would have known at the time.
    Both are the chance the month was in the last mood, "shrinking".
    """
    chance, _, ahead, _ = forward_backward(bell(x, model["mean"], model["spread"]), model["stay"], model["start"])
    return chance[:, -1], ahead[:, -1]


def run(economy):
    table = features(economy)
    x = standardise(table)
    model = fit(x)
    # Put the moods in order, best to worst, by how much unemployment rises in each.
    order = np.argsort(model["mean"][:, 0])
    model = dict(model, mean=model["mean"][order], spread=model["spread"][order],
                 stay=model["stay"][np.ix_(order, order)], start=model["start"][order])
    hindsight, live = read(model, x)
    months = table.assign(hindsight=hindsight, live=live, official=economy.official_recession.reindex(table.index))
    return model, months


def score(months, column="hindsight", cutoff=0.5):
    """Mark the model's homework against the official dates, which it never saw."""
    called, official = months[column] > cutoff, months.official == 1
    episodes = (official & ~official.shift(1, fill_value=False)).cumsum()[official]
    caught = sum(called[episodes[episodes == e].index].any() for e in episodes.unique())
    # A false alarm is a run of "shrinking" months with no official recession within six months of it.
    near = official.rolling(13, center=True, min_periods=1).max().astype(bool)
    runs = (called & ~called.shift(1, fill_value=False)).cumsum()[called]
    false_alarms = sum(not near[runs[runs == r].index].any() for r in runs.unique())
    return dict(
        months=int(len(months)), official_recession_months=int(official.sum()), official_recessions=int(episodes.nunique()),
        recessions_caught=int(caught), recession_months_caught=float((called & official).sum() / official.sum()),
        calm_months_called_stressed=float((called & ~official).sum() / (~official).sum()),
        months_agreeing=float((called == official).mean()), false_alarms=int(false_alarms),
    )


def losses_by_mood(months, losses):
    """How much card debt banks wrote off in calm quarters and in stressed ones."""
    quarterly = months.hindsight.copy()
    quarterly.index = pd.to_datetime(quarterly.index)
    stressed = quarterly.resample("QS").mean() > 0.5
    table = losses.copy()
    table.index = pd.to_datetime(table.index)
    table["stressed"] = stressed.reindex(table.index)
    table = table.dropna(subset=["written_off", "stressed"])
    # Card losses peak after the economy does: people run down savings first. Look up to a year past each stressed quarter.
    table["stressed_or_after"] = table.stressed.astype(bool).rolling(5, min_periods=1).max().astype(bool)
    calm = float(table.written_off[~table.stressed_or_after].mean())
    hard = float(table.written_off[table.stressed_or_after].mean())
    return dict(calm=calm, stressed=hard, multiplier=hard / calm, quarters=int(len(table)),
                stressed_quarters=int(table.stressed_or_after.sum())), table


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__).parent
    economy = pd.read_csv(here / "data" / "economy.csv", index_col="month")
    model, months = run(economy)
    print("Typical figures in each mood (0 = a typical month):")
    print(pd.DataFrame(model["mean"], index=MOODS, columns=FEATURES).round(2))
    print("Chance of staying in the same mood next month:", np.diag(model["stay"]).round(3))
    for column in ("hindsight", "live"):
        print(column, score(months, column))
    print(losses_by_mood(months, pd.read_csv(here / "data" / "card_losses.csv", index_col="quarter"))[0])
    called = months.hindsight > 0.5
    starts = months.index[called & ~called.shift(1, fill_value=False)]
    ends = months.index[called & ~called.shift(-1, fill_value=False)]
    print("Stressed spells:", ", ".join(f"{a[:7]} to {b[:7]}" for a, b in zip(starts, ends)))
    print("Latest month:", months.index[-1], f"live reading {months.live.iloc[-1]:.1%}")
