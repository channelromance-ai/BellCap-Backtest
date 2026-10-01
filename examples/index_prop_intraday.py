"""
Reality check for index_prop_friendly.py: does "0% failed" survive The5ers
watching equity LIVE?

index_prop_friendly.py checked the account only at each day's close. The5ers
counts open losses in real time, so a dip during the day that recovers by
the close still breaches the 3% daily or 6% overall limit. Here the same
rules run on The5ers' own NAS100 / SP500 hourly bars (mid-2020 to 2026):

  * one long position per server day, opened at the first bar after the
    17:00 New York roll and closed at the last bar before the next roll
  * signals from the CFD's own sessions: sigma = last 20 session-to-session
    returns; a "dip day" follows a session close more than 2 sigma down;
    turn of month = last trading day + first three
  * size = TV / sigma (capped at 3x), doubled on dip / turn-of-month days,
    times the share of the 6% allowance left (static floor)
  * breach checked on every hourly LOW (worst open loss), against the
    start-of-day balance for the daily limit and the start balance for 6%,
    and also, for comparison, only at the close
  * lots rounded down to 0.01 on a $5,000 account, as the platform would

Replayed from every month start in the period until pass, fail or the end
of the data.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

ACCOUNT = 5000.0
COST_BP = {"NAS100": 0.65, "SP500": 0.90}


def sessions(sym):
    a = pd.read_parquet("data/the5ers_idx_h1.parquet")
    g = a[a["sym"] == sym].copy()
    g["sday"] = g["server"].dt.normalize()
    per = g.groupby("sday").size()
    g = g[g["sday"].isin(per[per >= 18].index)]
    g = g[g["sday"].dt.weekday < 5]
    s = g.groupby("sday").agg(o=("open", "first"), c=("close", "last"),
                              lo=("low", "min"), bar_o=("open", list),
                              bar_l=("low", list))
    s["r"] = s["c"] / s["c"].shift(1) - 1
    sig = s["r"].rolling(20).std().shift(1)
    s["sig"] = sig
    s["dip"] = (s["r"] < -2 * sig).shift(1, fill_value=False)
    ym = pd.Series(s.index.year * 100 + s.index.month, index=s.index)
    s["tom"] = (ym.groupby(ym).cumcount(ascending=False) == 0) | \
        (ym.groupby(ym).cumcount() <= 2)
    return s.dropna(subset=["sig"])


def run(s, sym, tv, live, start, day_stop=None):
    """day_stop: close the day's position once its loss reaches this share
    of the ACCOUNT (1% = $50 on $5,000); filled at the stop price, or at an
    hourly bar's open if it opened beyond it."""
    eq, res = ACCOUNT, "open"
    floor = ACCOUNT * 0.94
    cost = COST_BP[sym] / 1e4
    rows = s.iloc[start:]
    for day in rows.itertuples():
        size = min(3.0, tv / day.sig) * (2.0 if (day.dip or day.tom) else 1.0)
        size *= max(0.0, min(1.0, (eq - floor) / (eq * 0.06)))
        lots = np.floor(eq * size / day.o / 0.01) * 0.01   # $1 per point
        if lots <= 0:
            continue
        notional = lots * day.o
        day_start = eq
        exit_px = day.c
        if day_stop is not None:
            stop_px = day.o * (1 - day_stop * ACCOUNT / notional)
            for bo, bl in zip(day.bar_o, day.bar_l, strict=True):
                if bl <= stop_px:
                    exit_px = min(stop_px, bo)
                    break
        if live:
            low_seen = max(day.lo, exit_px) if day_stop is not None else \
                day.lo
            worst = eq + notional * (low_seen / day.o - 1) - notional * cost
            if day_start - worst >= 0.03 * ACCOUNT or worst <= floor:
                return "fail"
        eq += notional * (exit_px / day.o - 1) - notional * cost
        if day_start - eq >= 0.03 * ACCOUNT or eq <= floor:
            return "fail"
        if eq >= ACCOUNT * 1.10:
            return "pass"
    return res


def main() -> int:
    pd.set_option("display.width", 200)
    rows = []
    for sym in ("NAS100", "SP500"):
        s = sessions(sym)
        starts = range(0, len(s) - 21, 21)
        for tv in (0.0025, 0.0035):
            for live, stop in ((False, None), (True, None), (True, 0.015),
                               (True, 0.01)):
                out = pd.Series([run(s, sym, tv, live, st, stop)
                                 for st in starts])
                rows.append(dict(market=sym, daily_swing_target=f"{tv:.2%}",
                                 checked="live (hourly lows)" if live else
                                 "at the close only",
                                 day_stop="none" if stop is None else
                                 f"{stop:.1%} of account", starts=len(out),
                                 pass_pct=100 * (out == "pass").mean(),
                                 fail_pct=100 * (out == "fail").mean(),
                                 still_open_pct=100 * (out == "open").mean()))
    print(f"The5ers hourly bars, {s.index[0].date()} to {s.index[-1].date()}; "
          f"$5,000 account, base+boost, size by volatility + shrink as you "
          f"lose, 6% static / 3% daily")
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
