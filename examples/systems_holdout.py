"""
Score the search, then open the holdout once.

The design period put "hold stocks above the 200-day average, bonds below"
well clear of everything else. Stressing it first, without touching
2019 onward, produced a much more specific picture:

  * Year by year, the edge is one year. 2008 beat buy-and-hold by 42.9
    points. In seven of the other ten years the difference was exactly
    zero, because the rule simply stayed in stocks all year.
  * Starting in 2010 instead, the rule returns 10.57% a year against
    buy-and-hold's 11.00%. From 2013, 8.75% against 9.30%. It gives up a
    little return in calm markets.
  * The lookback is not a magic number -- performance rises and falls
    smoothly from 50 to 300 days, peaking near 150 -- and the rule improves
    risk-adjusted return on every risky asset tried, not just the S&P.

So it is not a return edge. It is insurance: a small premium in ordinary
years for a large payout in a crash. 200 days was the value chosen before
any of this was run, and it stays, because 150 looking better afterwards is
exactly how a number gets fitted.

The holdout, 2019 to 2026, is a fair test of that claim, and a hard one. It
contains a crash too fast for a slow average to dodge (2020) and a slow
grinding bear it should handle well (2022).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import os
import sys

from bcbt import universe as U

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.systems import BENCHMARKS, COST_BPS, START, SYSTEMS  # noqa: E402

DESIGN_END = "2018-12-31"
HOLD_START = "2019-01-01"
CANDIDATE = "defensive (stocks or bonds)"


def reality_check_portfolios(daily: dict, n_boot=5000, block=21, seed=3):
    """
    Is the best system's risk-adjusted return better than the best of N
    under the null that none of them has an edge?

    Each series is demeaned, so no system has any edge by construction, and
    whole blocks of days are resampled together so that runs of good and bad
    weather survive the shuffle.
    """
    names = list(daily)
    M = pd.DataFrame(daily).dropna()
    X = (M - M.mean()).to_numpy()
    n, k = X.shape
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    span = np.arange(block)
    maxes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + span[None, :]).ravel()[:n] % n
        S = X[idx]
        mu, sd = S.mean(axis=0), S.std(axis=0, ddof=1)
        maxes[b] = np.nanmax(np.where(sd > 0, mu / sd, -np.inf))
    obs = (M.mean() / M.std(ddof=1))
    best = obs.idxmax()
    p = float((maxes >= obs.max()).mean())
    return best, float(obs.max()), p, maxes, names


def table(px, start, end, title):
    rows, daily = [], {}
    for name, (fn, freq) in {**SYSTEMS, **BENCHMARKS}.items():
        rec = U.run_portfolio(px, fn, freq=freq, cost_bps=COST_BPS,
                              start=start, end=end)
        s = U.summarise(rec, name)
        s["kind"] = "benchmark" if name in BENCHMARKS else "system"
        rows.append(s)
        r = rec["net"]
        daily[name] = r[r.index >= r.ne(0).idxmax()]
    d = pd.DataFrame(rows).set_index("label")
    cols = ["kind", "annual_pct", "vol_pct", "risk_adj", "worst_fall_pct",
            "calmar", "turnover_yr"]
    print(f"\n{title}")
    print(d[cols].sort_values("risk_adj", ascending=False).round(3).to_string())
    return d, daily


def main() -> int:
    pd.set_option("display.width", 220)
    px = U.load()

    dd, ddaily = table(px, START, DESIGN_END,
                       f"--- design {START} -> {DESIGN_END} ---")

    best, val, p, null, _ = reality_check_portfolios(ddaily)
    print(f"\n  scoring the search over {len(ddaily)} systems and benchmarks")
    print(f"    best on design: {best}")
    print(f"    its daily return per unit of wobble: {val:.4f}")
    print(f"    what luck alone produces: median {np.median(null):.4f}, "
          f"top 5% above {np.percentile(null, 95):.4f}")
    print(f"    p = {p:.4f}  ", end="")
    print("-> better than luck" if p < 0.05 else "-> NOT better than luck")

    print(f"\n  PRE-COMMITTED CANDIDATE: {CANDIDATE}")

    hd, _ = table(px, HOLD_START, None,
                  f"--- HOLDOUT {HOLD_START} onward, first and only look ---")

    print("\n" + "=" * 100)
    print("THE CANDIDATE, DESIGN vs HOLDOUT, against buying and holding")
    print("=" * 100)
    print(f"  {'':<26}{'a year':>10}{'wobble':>9}{'risk-adj':>10}"
          f"{'worst fall':>12}{'calmar':>9}")
    for label, frame in (("design", dd), ("HOLDOUT", hd)):
        for who in (CANDIDATE, "buy and hold SPY", "sixty forty"):
            r = frame.loc[who]
            print(f"  {label + ' ' + who:<26}{r['annual_pct']:>9.2f}%"
                  f"{r['vol_pct']:>8.1f}%{r['risk_adj']:>10.2f}"
                  f"{r['worst_fall_pct']:>11.1f}%{r['calmar']:>9.2f}")
        print()

    # What actually happened in the two declines the holdout contains.
    rec = U.run_portfolio(px, SYSTEMS[CANDIDATE][0], freq="M",
                          cost_bps=COST_BPS, start=HOLD_START)
    r = rec["net"]
    r = r[r.index >= r.ne(0).idxmax()]
    spy = px["SPY"].pct_change().reindex(r.index).fillna(0.0)
    print("  year by year on the holdout")
    print(f"    {'year':<6}{'defensive':>12}{'SPY':>10}{'difference':>13}")
    for y, g in r.groupby(r.index.year):
        a = (1 + g).prod() - 1
        b = (1 + spy.loc[g.index]).prod() - 1
        print(f"    {y:<6}{100 * a:>11.2f}%{100 * b:>9.2f}%"
              f"{100 * (a - b):>12.2f}%")

    for label, a, b in (("COVID crash", "2020-02-01", "2020-04-30"),
                        ("2022 bear", "2022-01-01", "2022-12-31")):
        g = r[(r.index >= a) & (r.index <= b)]
        s = spy[(spy.index >= a) & (spy.index <= b)]
        if len(g) < 5:
            continue
        print(f"    {label:<14} defensive {100 * ((1 + g).prod() - 1):+7.2f}%"
              f"   SPY {100 * ((1 + s).prod() - 1):+7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
