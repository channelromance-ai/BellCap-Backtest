"""
Follow-ups from index_ev.py, on the long daily history (SPY 1993-, QQQ
1999-), with The5ers' costs. Halves: before 2010 / 2010-26.

  1  DAILY RE-ENTRY: be long each day but close before the 17:00 New York
     roll and reopen after it -- one round trip a day, NO financing. Close-
     to-close daily returns are used; they include the one hour the market
     is shut around the roll, so this is a slight approximation.
     Compared with simply holding (financing every night, x3 Fridays).
  2  BUY AFTER A BIG DOWN DAY: after a day more than 2 sigma down (sigma =
     previous 20 days), long the next day; with and without financing.
     Compared with an ordinary day, and with buying after a random day.
  3  NASDAQ WEEKLY REVERSAL: the mirror of index_ev's worst result
     (following the last 5 days lost consistently): fade it, hold 5 days.
     Shown both ways round and long-only (buy only after a down week).
     This is the mirror of a result already seen, so it is a hypothesis
     test of that one finding, not a fresh discovery.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.index_ev as I                          # noqa: E402
import examples.xau_ev as X                            # noqa: E402


def main() -> int:
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(11)
    for sym, cfg in I.INSTR.items():
        u = pd.read_parquet("data/universe_daily.parquet")[cfg["etf"]].dropna()
        u = u[u.index.weekday < 5]
        idx = u.index
        yrs = idx.year.to_numpy()
        rt, sw = cfg["rt"], cfg["swap"]
        X.SPLIT = cfg["split_daily"]
        r = 1e4 * (u / u.shift(1) - 1)
        nights = np.r_[np.nan, [3 if idx[k - 1].weekday() == 4 else 1
                                for k in range(1, len(u))]]
        hold_net = r - rt / 1 * 0 - nights * sw        # held: fee every night
        reentry = r - rt                               # out before the roll
        rows = [
            X.two_halves(hold_net, yrs, "hold long, pay financing every night",
                         dict(per_year_pct=np.nanmean(hold_net) * 252 / 100)),
            X.two_halves(reentry, yrs, "long by day, out before the roll "
                         "(no financing)",
                         dict(per_year_pct=np.nanmean(reentry) * 252 / 100)),
        ]
        vol = r.rolling(20).std().shift(1)
        big_dn = (r < -2 * vol).shift(1, fill_value=False).to_numpy()
        rows.append(X.two_halves(reentry[big_dn], yrs[big_dn],
                                 "day after a -2 sigma day, out before roll"))
        rows.append(X.two_halves((r - rt - nights * sw)[big_dn], yrs[big_dn],
                                 "day after a -2 sigma day, held overnight"))
        rows.append(X.two_halves(reentry[~big_dn], yrs[~big_dn],
                                 "all other days, out before roll"))
        # luck: the same number of random days, 200 times
        n_pick = int(big_dn.sum())
        valid = np.flatnonzero(np.isfinite(reentry.to_numpy()))
        sims = [np.nanmean(reentry.to_numpy()[rng.choice(valid, n_pick,
                                                         replace=False)])
                for _ in range(2000)]
        obs = np.nanmean(reentry.to_numpy()[big_dn])
        luck_p = np.mean(np.array(sims) >= obs)
        # weekly reversal
        cl = u.to_numpy()
        fade, longonly, ylist, ylong, uncond = [], [], [], [], []
        for i in range(5, len(cl) - 5, 5):
            sig = np.sign(cl[i] / cl[i - 5] - 1)
            if sig == 0:
                continue
            g = 1e4 * (cl[i + 5] / cl[i] - 1)
            n = sum(3 if idx[k].weekday() == 4 else 1 for k in range(i, i + 5))
            c = rt + n * sw
            fade.append(-sig * g - c)
            ylist.append(yrs[i])
            uncond.append(g - c)
            if sig < 0:
                longonly.append(g - c)
                ylong.append(yrs[i])
        rows.append(X.two_halves(fade, ylist, "FADE last 5 days, hold 5 "
                                 "(long or short)",
                                 dict(per_year_pct=np.mean(fade) * 252 / 5
                                      / 100)))
        rows.append(X.two_halves(longonly, ylong, "BUY only after a down "
                                 "week, hold 5"))
        rows.append(X.two_halves(uncond, ylist, "any week, hold 5 (baseline)"))
        print("\n" + "#" * 110)
        print(f"# {sym} via {cfg['etf']} {idx[0].year}-{idx[-1].year}  (halves: "
              f"before / from {cfg['split_daily']})")
        print("#" * 110)
        out = pd.DataFrame(rows).rename(columns={
            "bp_2005_15": "bp_1st_half", "t_2005_15": "t_1st",
            "bp_2016_26": "bp_2nd_half", "t_2016_26": "t_2nd"})
        print(out.round(2).to_string(index=False))
        print(f"\n  after a -2 sigma day: {obs:+.2f}bp vs random days; share of "
              f"2,000 random picks doing as well: {luck_p:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
