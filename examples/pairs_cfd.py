"""
Two correlated index CFDs drifting apart, then closing back up.

This is a different shape of bet from everything tested before it, and the
difference is the point. There is no view on whether markets go up. The
position is long one index and short another of similar size, so a crash
that takes both down together costs roughly nothing. What is being bet on is
that the GAP between two things that normally move together, having stretched
unusually wide, narrows again.

Why this suits a CFD account specifically:

  * Shorting is as easy as buying, so the trade is symmetric.
  * The financing charge on the long leg is largely offset by the credit on
    the short leg. What remains is the broker's markup, taken on both sides,
    which is why holding periods here are days and not months.
  * The right benchmark is not "did it beat owning the index", because there
    is no index exposure to compare against. The bar is simply whether the
    trade makes money after costs.

Costs charged, and they are the whole question at this size:

  spread      4 basis points for a round trip across both legs together
  financing   5% a year on the notional while the position is open, being
              the broker markup paid on the long leg and given up on the
              short. Over a five-day hold that is about 7 basis points --
              comparable to the spread, and the reason a slow version of
              this trade cannot work.

No peeking: the gap is judged against its own trailing average only, the
signal is read at one close and the trade is entered at the NEXT close, and
every threshold is fixed in advance rather than fitted to the sample.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANEL = "data/index_cfd_daily.parquet"
DESIGN_END = "2012-12-31"
HOLD_START = "2013-01-01"

SPREAD_BPS = 4.0            # both legs, in and out
FINANCE_PA = 0.05           # markup paid on both sides, per year

# Pairs a retail CFD account can actually hold, chosen for how tightly they
# move together rather than for how they performed.
PAIRS = [
    ("US500", "US30"), ("US500", "US100"), ("US500", "US2000"),
    ("US30", "US2000"), ("US100", "US2000"),
    ("GER40", "FRA40"), ("GER40", "UK100"), ("GER40", "NED25"),
    ("FRA40", "UK100"), ("FRA40", "NED25"), ("UK100", "NED25"),
    ("ESP35", "FRA40"), ("SWI20", "GER40"),
]


def spread_trades(px, a, b, lookback=60, entry_z=2.0, exit_z=0.5,
                  stop_z=4.0, max_hold=10, start=None, end=None):
    """
    Walk one pair and return every completed trade.

    The gap is the log of one index divided by the other. It is scored
    against its own trailing average and wobble, so "unusually wide" means
    wide compared with how this pair has behaved lately, not compared with
    the whole history.
    """
    d = px[[a, b]].dropna()
    if start:
        d = d[d.index >= start]
    if end:
        d = d[d.index <= end]
    if len(d) < lookback + 100:
        return pd.DataFrame()

    gap = np.log(d[a] / d[b])
    mu = gap.rolling(lookback).mean()
    sd = gap.rolling(lookback).std()
    z = ((gap - mu) / sd.replace(0, np.nan))

    zv = z.to_numpy()
    av = d[a].to_numpy()
    bv = d[b].to_numpy()
    idx = d.index
    n = len(d)

    rows = []
    i = lookback + 1
    while i < n - 1:
        if not np.isfinite(zv[i]) or abs(zv[i]) < entry_z:
            i += 1
            continue
        # Signal at close i, filled at close i+1.
        e = i + 1
        side = -1 if zv[i] > 0 else 1        # +1 means long a, short b
        exit_i, why = None, None
        for j in range(e + 1, min(e + max_hold + 1, n)):
            if not np.isfinite(zv[j]):
                continue
            if abs(zv[j]) >= stop_z and np.sign(zv[j]) == np.sign(zv[i]):
                exit_i, why = j, "stop"
                break
            if side * -np.sign(zv[i]) > 0 and abs(zv[j]) <= exit_z:
                exit_i, why = j, "closed"
                break
            if abs(zv[j]) <= exit_z:
                exit_i, why = j, "closed"
                break
        if exit_i is None:
            exit_i, why = min(e + max_hold, n - 1), "time"

        ra = av[exit_i] / av[e] - 1.0
        rb = bv[exit_i] / bv[e] - 1.0
        gross = side * (ra - rb)
        held = exit_i - e
        cost = SPREAD_BPS / 1e4 + FINANCE_PA * held / 365.0
        rows.append(dict(pair=f"{a}/{b}", entry=idx[e], exit=idx[exit_i],
                         side="long " + a if side > 0 else "long " + b,
                         entry_z=zv[i], held=held, gross_pct=100 * gross,
                         cost_pct=100 * cost, net_pct=100 * (gross - cost),
                         reason=why))
        i = exit_i + 1
    return pd.DataFrame(rows)


def score(t, label):
    if t is None or len(t) < 30:
        return None
    r = t["net_pct"].to_numpy()
    se = r.std(ddof=1) / np.sqrt(len(r))
    return dict(label=label, trades=len(r), win_pct=100 * (r > 0).mean(),
                avg_pct=r.mean(), t=r.mean() / se,
                total_pct=r.sum(), avg_held=t["held"].mean(),
                gross_pct=t["gross_pct"].mean(),
                cost_pct=t["cost_pct"].mean())


def run_all(px, start, end, **kw):
    parts = [spread_trades(px, a, b, start=start, end=end, **kw)
             for a, b in PAIRS]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts) if parts else pd.DataFrame()


def main() -> int:
    pd.set_option("display.width", 220)
    px = pd.read_parquet(PANEL)
    print(f"data {px.index[0].date()} -> {px.index[-1].date()}")
    print(f"pairs: {len(PAIRS)}   spread {SPREAD_BPS}bp, "
          f"financing {100 * FINANCE_PA:.0f}%/yr while open\n")

    print("=" * 96)
    print("DESIGN PERIOD (to 2012): does the gap actually close?")
    print("=" * 96)
    rows = []
    grid = list(itertools.product((20, 60, 120), (1.5, 2.0, 2.5),
                                  (5, 10, 20)))
    for lb, ez, mh in grid:
        t = run_all(px, None, DESIGN_END, lookback=lb, entry_z=ez,
                    max_hold=mh)
        s = score(t, f"look{lb}/z{ez}/hold{mh}")
        if s:
            rows.append(s)
    d = pd.DataFrame(rows).set_index("label").sort_values("avg_pct",
                                                          ascending=False)
    print(d.round(3).to_string())

    print("\n  gross vs net, best and worst cells:")
    for lab in (d.index[0], d.index[-1]):
        r = d.loc[lab]
        print(f"    {lab:<22} gross {r['gross_pct']:+.3f}%  "
              f"cost {r['cost_pct']:.3f}%  net {r['avg_pct']:+.3f}%  "
              f"(held {r['avg_held']:.1f} days)")

    print(f"\n  cells with positive net expectancy: "
          f"{int((d['avg_pct'] > 0).sum())} of {len(d)}")
    print(f"  cells positive BEFORE costs:         "
          f"{int((d['gross_pct'] > 0).sum())} of {len(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
