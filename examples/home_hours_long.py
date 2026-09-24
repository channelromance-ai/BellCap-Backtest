"""
Home-hours depreciation on 26 years of hourly data, at The5ers' real costs.

timing_effects.py tested the idea that a currency weakens while its own
market is open, on the 2.4 years of minute data available. At The5ers'
measured costs the Asian window came out slightly ahead in both halves --
right direction 52-54% of the time for all three currencies -- but far too
little data to tell that from luck.

This is the same test, with exactly the same windows and directions,
decided before this data was downloaded, run on hourly bars from The5ers'
own MT5 server back to 2000. Every window and pair from the original rules
is tested, not only the one that looked good: judging the survivor alone
would be picking the winner after the race.

    03:00-08:00 NY  Europe only   EUR, GBP, CHF weaken
    12:00-17:00 NY  US only       USD weakens
    19:00-02:00 NY  Asia only     JPY, AUD, NZD weaken

The server clock is New York + 7 hours all year: across 1,411 trading weeks
the market opens at server Monday 00:00 in every one except holiday weeks,
with no drift at the daylight-saving changes.

Nights are the unit of evidence, not trades: the three Asian pairs on the
same night are one bet on the dollar and are averaged before any statistic
is computed, so the sample is not inflated by counting one move three times.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PANEL = "data/the5ers_h1.parquet"
START = "2000-01-01"
NY_OFFSET = pd.Timedelta(hours=7)

WINDOWS = {
    "europe 03-08": (3, 8, {"EURUSD": -1, "GBPUSD": -1, "USDCHF": +1}),
    "us 12-17": (12, 17, {"EURUSD": +1, "GBPUSD": +1, "AUDUSD": +1,
                          "NZDUSD": +1, "USDJPY": -1, "USDCHF": -1}),
    "asia 19-02": (19, 2, {"USDJPY": +1, "AUDUSD": -1, "NZDUSD": -1}),
}
ERAS = [("2000-07", "2000", "2008"), ("2008-14", "2008", "2015"),
        ("2015-20", "2015", "2021"), ("2021-26", "2021", "2027")]


def load():
    a = pd.read_parquet(PANEL)
    a["ny"] = a["server"] - NY_OFFSET
    return a[a["ny"] >= START]


def windows(a, cost_bp):
    rows = []
    for sym, g in a.groupby("sym"):
        g = g.set_index("ny").sort_index()
        o, c = g["open"], g["close"]
        days = pd.DatetimeIndex(sorted(set(g.index.normalize())))
        for name, (h0, h1, pairs) in WINDOWS.items():
            if sym not in pairs:
                continue
            d = pairs[sym]
            for day in days:
                start = day + pd.Timedelta(hours=h0)
                last_bar = day + pd.Timedelta(hours=h1 - 1 + (24 if h1 < h0
                                                               else 0))
                if start.weekday() >= 5 or last_bar.weekday() >= 5:
                    continue
                if name.startswith("asia") and start.weekday() == 4:
                    continue
                if start not in o.index or last_bar not in c.index:
                    continue
                entry, exit_ = o.at[start], c.at[last_bar]
                rows.append(dict(window=name, sym=sym, day=day, d=d,
                                 gross_bp=1e4 * d * (exit_ / entry - 1),
                                 cost_bp=cost_bp[sym]))
    return pd.DataFrame(rows)


def by_night(t):
    """One row per window per night: the pairs averaged together."""
    return (t.groupby(["window", "day"])
            .agg(gross_bp=("gross_bp", "mean"), cost_bp=("cost_bp", "mean"),
                 pairs=("sym", "count")).reset_index())


def summ(g):
    x = g["gross_bp"].to_numpy()
    net = x - g["cost_bp"].to_numpy()
    n = len(x)
    if n < 10:
        return pd.Series(dict(n=n))
    return pd.Series(dict(
        nights=n, right_way_pct=100 * (x > 0).mean(), gross_bp=x.mean(),
        cost_bp=g["cost_bp"].mean(), net_bp=net.mean(),
        t_gross=x.mean() / (x.std(ddof=1) / np.sqrt(n)),
        t_net=net.mean() / (net.std(ddof=1) / np.sqrt(n))))


def era(day):
    y = str(day.year)
    for name, a, b in ERAS:
        if a <= y < b:
            return name
    return "?"


def main(cost_file=None) -> int:
    pd.set_option("display.width", 210)
    cost_bp = {"EURUSD": 0.392, "GBPUSD": 0.342, "USDCHF": 0.951,
               "USDJPY": 0.434, "AUDUSD": 0.642, "NZDUSD": 1.462}
    if cost_file:
        cost_bp.update(json.load(open(cost_file)))

    a = load()
    bars = a.groupby([a["sym"], a["ny"].dt.year]).size().unstack(0)
    print("hourly bars per year (a full year is about 6,200):")
    print(bars.loc[[2000, 2001, 2003, 2005, 2010, 2020, 2025]].to_string())

    t = windows(a, cost_bp)
    t["era"] = t["day"].map(era)
    n = by_night(t)
    n["era"] = n["day"].map(era)

    print("\n" + "=" * 100)
    print("EVERY WINDOW, 2000-2026, nights as the unit (pairs averaged)")
    print("=" * 100)
    print(n.groupby("window").apply(summ).round(3).to_string())

    print("\n  by era -- a real effect should hold in most of them")
    print(n.groupby(["window", "era"]).apply(summ).round(3).to_string())

    print("\n  each pair on its own, whole period")
    print(t.groupby(["window", "sym"]).apply(summ).round(3).to_string())

    asia = n[n["window"] == "asia 19-02"].copy()
    asia["year"] = asia["day"].dt.year
    yr = asia.groupby("year").apply(
        lambda g: pd.Series(dict(right=100 * (g["gross_bp"] > 0).mean(),
                                 net_bp=(g["gross_bp"] - g["cost_bp"]).mean())))
    print("\n  Asian window by year")
    print(yr.round(1).T.to_string())
    print(f"\n  years with a positive net result: "
          f"{int((yr['net_bp'] > 0).sum())} of {len(yr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
