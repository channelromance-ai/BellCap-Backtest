"""
The same gap trade, inside a single session.

The daily version showed a clean dose-response -- the wider the gap between
two indices gets, the harder it snaps back -- but at gaps wide enough to pay
for the spread it fires about seven times a year, and it faded out of
sample.

Going intraday changes two things that were binding:

  * Nothing is held overnight, so the financing markup is zero rather than
    the largest cost in the trade. Only the spread is paid.
  * A gap that has to stretch and close within one day can do so many times
    a year rather than seven.

The instruments are the S&P and Nasdaq CFDs, 764 regular-hours sessions of
minute data. They move together about 0.84 on a daily basis, and far more
tightly minute to minute, which is what makes a gap between them mean
something.

Costs: 0.85 basis points on the S&P leg and 0.55 on the Nasdaq, measured
from real ask data earlier in this repo, so 1.4 basis points for a round
trip across both legs. No financing at all.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from bcbt import calendars, store

warnings.filterwarnings("ignore")

A, B = "SPX500", "NAS100"
COST_PCT = 0.014           # 1.4 basis points, both legs, round trip
SPLIT = "2025-06-01"       # design before, holdout after


def session_bars(sym, minutes=5):
    m1 = store.load_m1(sym)
    tbl = calendars.sessions()
    keys = store.day_keys(m1.index)
    good = {pd.Timestamp(d).tz_localize(None).to_datetime64() for d in tbl}
    mins = store.minutes(m1.index)
    keep = np.isin(keys, list(good)) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    s = m1.loc[keep, ["c"]]
    return s.resample(f"{minutes}min", closed="left",
                      label="left").last().dropna()


def build():
    a = session_bars(A).rename(columns={"c": A})
    b = session_bars(B).rename(columns={"c": B})
    d = a.join(b, how="inner").dropna()
    d["day"] = store.day_keys(d.index)
    return d


def trades(d, lookback=40, entry_z=2.0, exit_z=0.5, max_hold=12,
           start=None, end=None):
    """
    One pass. The gap is scored inside its own session only, so nothing
    carries across the overnight break, and every position is closed before
    the bell whatever it is doing.
    """
    if start:
        d = d[d.index >= start]
    if end:
        d = d[d.index < end]
    out = []
    for _day, g in d.groupby("day"):
        if len(g) < lookback + 10:
            continue
        gap = np.log(g[A] / g[B])
        mu = gap.rolling(lookback).mean()
        sd = gap.rolling(lookback).std()
        z = ((gap - mu) / sd.replace(0, np.nan)).to_numpy()
        av, bv = g[A].to_numpy(), g[B].to_numpy()
        idx = g.index
        n = len(g)

        i = lookback
        while i < n - 2:
            if not np.isfinite(z[i]) or abs(z[i]) < entry_z:
                i += 1
                continue
            e = i + 1                       # signal on a close, fill next bar
            side = -1 if z[i] > 0 else 1
            xi, why = None, None
            for j in range(e + 1, min(e + max_hold + 1, n)):
                if np.isfinite(z[j]) and abs(z[j]) <= exit_z:
                    xi, why = j, "closed"
                    break
            if xi is None:
                xi, why = min(e + max_hold, n - 1), "time"
            gross = side * ((av[xi] / av[e] - 1.0) - (bv[xi] / bv[e] - 1.0))
            out.append(dict(day=idx[e].date(), entry=idx[e], bars=xi - e,
                            gross_pct=100 * gross,
                            net_pct=100 * gross - COST_PCT, reason=why))
            i = xi + 1
    return pd.DataFrame(out)


def score(t, label):
    if t is None or len(t) < 30:
        return None
    r = t["net_pct"].to_numpy()
    se = r.std(ddof=1) / np.sqrt(len(r))
    return dict(label=label, trades=len(r), per_day=None,
                win_pct=100 * (r > 0).mean(), gross_pct=t["gross_pct"].mean(),
                net_pct=r.mean(), t=r.mean() / se, total_pct=r.sum(),
                bars=t["bars"].mean())


def main() -> int:
    pd.set_option("display.width", 210)
    d = build()
    sessions = d["day"].nunique()
    print(f"{len(d):,} five-minute bars over {sessions} sessions "
          f"({d.index[0].date()} -> {d.index[-1].date()})")
    print(f"cost {COST_PCT:.3f}% a round trip, no financing\n")

    print("=" * 96)
    print("DESIGN (before 2025-06)")
    print("=" * 96)
    rows = []
    for lb, ez, mh in itertools.product((20, 40, 60), (1.5, 2.0, 2.5, 3.0),
                                        (6, 12, 24)):
        t = trades(d, lb, ez, max_hold=mh, end=SPLIT)
        s = score(t, f"look{lb}/z{ez}/hold{mh}")
        if s:
            rows.append(s)
    tab = pd.DataFrame(rows).set_index("label").drop(columns=["per_day"])
    tab = tab.sort_values("net_pct", ascending=False)
    print(tab.round(4).to_string())
    print(f"\n  positive after costs: {int((tab['net_pct'] > 0).sum())} "
          f"of {len(tab)}   |   before costs: "
          f"{int((tab['gross_pct'] > 0).sum())} of {len(tab)}")

    print("\n  does a wider gap pay more? (look40, hold12)")
    print(f"  {'gap':>6}{'trades':>9}{'win%':>8}{'gross%':>9}{'net%':>9}"
          f"{'t':>7}")
    for ez in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        s = score(trades(d, 40, ez, max_hold=12, end=SPLIT), "")
        if s:
            print(f"  {ez:>6.1f}{s['trades']:>9}{s['win_pct']:>8.1f}"
                  f"{s['gross_pct']:>9.4f}{s['net_pct']:>9.4f}{s['t']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
