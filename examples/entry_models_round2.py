"""
Round 2: the EUR/USD European-morning short, only on nights with a reason.

Round 1 (entry_models_2r.py) put the morning drift in a 2:1 bracket and got
a profit factor of 1.0-1.06 -- real, but far from the 1.25 target. The drift
is known to vary with conditions that exist BEFORE the entry:

  trend      The drift was larger in years the euro fell (correlation -0.51
             between the yearly drift and the euro's yearly move). Filter:
             EUR/USD's last 17:00 New York close below its 50-day average.
  reversal   The euro tends to rise in US hours and fall in European hours.
             Filter: the euro rose over the previous New York session
             (08:00 open to 16:00 close), the move Europe tends to undo.

Six tests, fixed before running: {trend, reversal, both} x {23:00 entry,
03:00 entry}. Everything else is round 1's M1/M1b: short, stop 0.25 x
ADR20 above, target 2x below, exit 08:00 New York, stop first on ties,
The5ers' spread at the entry hour plus $4/lot.

Passing needs PF >= 1.25 on 2013-26 AND >= 1.15 on 2000-12 AND >= 1.10 on
2021-26. Strict on purpose: filtering is how fake edges are made.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.entry_models_2r as E                   # noqa: E402

H = pd.Timedelta(hours=1)


def main() -> int:
    pd.set_option("display.width", 220)
    bars, spread = E.load()
    b = bars["EURUSD"]
    adr = E.adr20(b)
    idx = b.index
    # Daily 17:00 close = close of the 16:00 bar; US-session move = 08:00
    # open to 16:00 close (close of the 15:00 bar). Both finished before
    # 23:00 the same evening.
    close17 = b["close"][idx.hour == 16]
    close17.index = close17.index.normalize()
    sma50 = close17.rolling(50).mean()
    us_open = b["open"][idx.hour == 8]
    us_open.index = us_open.index.normalize()
    us_close = b["close"][idx.hour == 15]
    us_close.index = us_close.index.normalize()
    us_move = (us_close - us_open).dropna()

    rows = []
    for day in adr.dropna().index:
        prev = day - pd.Timedelta(days=1)
        if day.weekday() == 0:
            prev = day - pd.Timedelta(days=3)           # Monday: use Friday
        if prev not in close17.index or prev not in us_move.index:
            continue
        trend_ok = close17[prev] < sma50.get(prev, np.nan)
        rev_ok = us_move[prev] > 0
        for entry_name, t0 in (("23:00", day - H), ("03:00", day + 3 * H)):
            x = E.trade("x", "EURUSD", b, day, t0, day + 8 * H, -1,
                        0.25 * adr[day], spread)
            if not x:
                continue
            for filt, ok in (("none (round 1)", True), ("trend", trend_ok),
                             ("reversal", rev_ok),
                             ("both", trend_ok and rev_ok)):
                if ok:
                    rows.append(dict(entry=entry_name, filter=filt, day=day,
                                     r=x["r"]))
    t = pd.DataFrame(rows)

    def era_stats(g):
        out = {}
        for name, sel in (("2000-12", g[g["day"].dt.year < 2013]),
                          ("2013-26", g[g["day"].dt.year >= 2013]),
                          ("2021-26", g[g["day"].dt.year >= 2021])):
            r = sel["r"].to_numpy()
            out[f"pf_{name}"] = E.pf(r)
            out[f"n_{name}"] = len(r)
        r = g[g["day"].dt.year >= 2013]["r"].to_numpy()
        out["win_2013_26"] = 100 * (r > 0).mean()
        out["t_2013_26"] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        out["per_year"] = len(g) / g["day"].dt.year.nunique()
        return pd.Series(out)

    s = t.groupby(["entry", "filter"]).apply(era_stats)
    cols = ["per_year", "pf_2000-12", "pf_2013-26", "pf_2021-26",
            "win_2013_26", "t_2013_26", "n_2013-26"]
    print(s[cols].round(3).to_string())
    ok = s[(s["pf_2013-26"] >= 1.25) & (s["pf_2000-12"] >= 1.15)
           & (s["pf_2021-26"] >= 1.10)]
    print("\nPASSING ALL THREE BARS:")
    print(ok[cols].round(3).to_string() if len(ok) else "  none")
    t.to_csv("data/entry_models_round2_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
