"""
Scoring a set of trades, and the tests that stop you believing the score.

Results are kept in R -- multiples of the risk taken -- so a 32-point S&P
stop and a 162-point Nasdaq stop are the same unit and can be pooled.

The two functions that matter most here are the ones that push back.
`bootstrap_ci` says how wide the uncertainty on the mean actually is, which
on a few hundred intraday trades is usually wide enough to contain zero.
`reality_check` asks the question a variant grid makes unavoidable: given
that N variants were tried, is the best one better than the best one the
same search would have found in data with no edge in it?

Both backtests run through this package so far failed the second test, and
both would have looked publishable without it.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def summary(trades, r_col="r"):
    """Headline numbers for one set of trades."""
    if trades is None or len(trades) == 0:
        return dict(n=0)
    r = np.asarray(trades[r_col], float)
    wins, losses = r[r > 0], r[r <= 0]
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    se = r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else float("nan")
    pf = float("inf")
    if len(losses) and losses.sum() != 0:
        pf = float(wins.sum() / abs(losses.sum()))
    return dict(
        n=int(len(r)),
        win=100.0 * len(wins) / len(r),
        avg_r=float(r.mean()),
        total_r=float(r.sum()),
        pf=pf,
        max_dd_r=dd,
        t=float(r.mean() / se) if se and math.isfinite(se) and se > 0 else float("nan"),
    )


def bootstrap_ci(r, n=20000, alpha=0.05, seed=0):
    """Percentile CI on mean R, resampling whole trades."""
    r = np.asarray(r, float)
    if len(r) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def equity(trades, r_col="r", day_col="day"):
    """Cumulative R indexed by date, for plotting."""
    t = trades.sort_values(day_col)
    return pd.Series(np.cumsum(np.asarray(t[r_col], float)),
                     index=pd.to_datetime(t[day_col]))


def reality_check(per_day, n_boot=10000, block=5, seed=0):
    """
    White's Reality Check across a whole variant search.

    `per_day` maps a variant name to a DataFrame indexed by session date
    with columns `sum` and `count` -- the summed R and number of trades that
    variant took that day. Sessions are the resampling unit because trades
    inside one session are not independent, but each resampled session
    contributes its trades rather than its average: someone taking every
    signal earns the per-trade mean, and averaging by day would quietly
    re-weight quiet days against whipsaw days.

    The null demeans every variant, so no variant has an edge by
    construction, then asks how high the best of N variants gets anyway.

    Returns (best_name, best_mean_r, p_value, null_max_distribution).
    """
    names = list(per_day)
    days = sorted(set().union(*[set(d.index) for d in per_day.values()]))
    di = {d: i for i, d in enumerate(days)}
    nd, nc = len(days), len(names)

    S = np.zeros((nd, nc))
    N = np.zeros((nd, nc))
    for c, name in enumerate(names):
        g = per_day[name]
        for d, row in g.iterrows():
            S[di[d], c] = row["sum"]
            N[di[d], c] = row["count"]

    tot = N.sum(axis=0)
    if (tot == 0).any():
        raise ValueError("a variant took no trades at all")
    obs = S.sum(axis=0) / tot
    best_i = int(np.argmax(obs))

    Sd = S - N * obs[None, :]          # demeaned per trade
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / block))
    maxes = np.empty(n_boot)
    span = np.arange(block)

    for b in range(n_boot):
        starts = rng.integers(0, nd, size=nb)
        idx = (starts[:, None] + span[None, :]).ravel()[:nd] % nd
        t = N[idx].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(t > 0, Sd[idx].sum(axis=0) / np.maximum(t, 1), -np.inf)
        maxes[b] = m.max()

    p = float((maxes >= obs[best_i]).mean())
    return names[best_i], float(obs[best_i]), p, maxes


def per_day_table(trades, r_col="r", day_col="day"):
    """Shape one variant's trades for `reality_check`."""
    t = trades.copy()
    t[day_col] = pd.to_datetime(t[day_col])
    return t.groupby(day_col)[r_col].agg(["sum", "count"])


def coin_flip(trades, cost_pts, r_col="r", risk_col="risk",
              n=20000, seed=0):
    """
    Would a coin have done as well from the same entries?

    Each trade is mirrored: gross R sign-flipped, cost re-charged. This
    isolates the direction call from the exit scheme and the drift, which
    is the only way to tell an edge from a rising market.
    """
    t = trades
    r = np.asarray(t[r_col], float)
    cost_r = cost_pts / np.asarray(t[risk_col], float)
    gross = r + cost_r
    flipped = -gross - cost_r
    rng = np.random.default_rng(seed)
    draws = np.where(rng.random((n, len(r))) < 0.5, r, flipped).mean(axis=1)
    real = float(r.mean())
    return dict(real=real, flip_mean=float(draws.mean()),
                flip_sd=float(draws.std()),
                pctile=float(100.0 * (draws < real).mean()),
                p=float((1 + (draws >= real).sum()) / (1 + len(draws))))
