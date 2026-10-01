"""
Can the EUR/USD morning short and the long-only S&P system share one
The5ers account? Both trade inside the same server day (17:00 New York
reset): the S&P position runs from just after the roll to just before the
next; the EUR short from 23:00 to 08:00 New York, entirely inside it. So
their losses share the 3% daily limit and the 6% overall limit.

On The5ers' own hourly bars (SP500 from Nov 2017; EURUSD throughout):
  EUR   as EurMorningShort runs it: 0.18 lots, short 23:00 NY, stop 40 pips,
        out 08:00 NY; $4/lot commission + measured spread
  SPX   0.25% volatility sizing, double on dip / turn-of-month days, size
        shrunk by the share of the 6% allowance left -- measured on TOTAL
        equity, so EUR losses shrink it too -- and a hard stop at 1% of the
        account each day
  LIVE  each day's worst moment is the EUR trade's worst point PLUS the
        S&P position's session low, as if both hit at once (pessimistic)

Replayed from every month start: alone and together, from a fresh $5,000
and from the user's current balance ($5,049.28), target $5,500, floor
$4,700, daily limit $150. Also: how their daily results move together.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.index_prop_intraday as IP              # noqa: E402

ACCOUNT, TARGET, FLOOR, DAILY = 5000.0, 5500.0, 4700.0, 150.0
EUR_LOTS, EUR_STOP_PIPS, PIP = 0.18, 40.0, 1e-4
EUR_COST_USD = EUR_LOTS * 4.0 + EUR_LOTS * 10 * 0.053


def eur_days():
    """Per server day: EUR short result ($) and its worst point ($)."""
    a = pd.read_parquet("data/the5ers_h1_all.parquet")
    e = a[a["sym"] == "EURUSD"].copy()
    e["ny"] = e["server"] - pd.Timedelta(hours=7)
    e = e.set_index("ny").sort_index()
    rows = {}
    for day in pd.bdate_range("2017-11-01", e.index[-1].normalize()):
        idx = pd.date_range(day - pd.Timedelta(hours=1), periods=9, freq="h")
        if not idx.isin(e.index).all():
            continue
        b = e.loc[idx]
        entry = b["open"].iloc[0]
        stop = entry + EUR_STOP_PIPS * PIP
        px, worst_px = b["close"].iloc[-1], b["high"].max()
        for o, h in zip(b["open"], b["high"], strict=True):
            if h >= stop:
                px = max(stop, o)
                worst_px = px
                break
        pnl = (entry - px) / PIP * EUR_LOTS * 10 - EUR_COST_USD
        worst = (entry - worst_px) / PIP * EUR_LOTS * 10 - EUR_COST_USD
        rows[day] = (pnl, min(worst, pnl))
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["eur_pnl", "eur_worst"])


def spx_day(day, eq, tv=0.0025, stop_share=0.01):
    floor = FLOOR
    size = min(3.0, tv / day.sig) * (2.0 if (day.dip or day.tom) else 1.0)
    size *= max(0.0, min(1.0, (eq - floor) / (eq * 0.06)))
    lots = np.floor(eq * size / day.o / 0.01) * 0.01
    if lots <= 0:
        return 0.0, 0.0
    notional = lots * day.o
    cost = notional * IP.COST_BP["SP500"] / 1e4
    stop_px = day.o * (1 - stop_share * ACCOUNT / notional)
    exit_px = day.c
    for bo, bl in zip(day.bar_o, day.bar_l, strict=True):
        if bl <= stop_px:
            exit_px = min(stop_px, bo)
            break
    pnl = notional * (exit_px / day.o - 1) - cost
    worst = notional * (max(day.lo, exit_px) / day.o - 1) - cost
    return pnl, worst


def replay(spx, eur, start_i, start_eq, use_eur, use_spx):
    eq = start_eq
    for day in spx.iloc[start_i:].itertuples():
        d0 = eq
        e_pnl = e_worst = 0.0
        if use_eur and day.Index in eur.index:
            e_pnl, e_worst = eur.loc[day.Index, ["eur_pnl", "eur_worst"]]
        s_pnl = s_worst = 0.0
        if use_spx:
            s_pnl, s_worst = spx_day(day, eq)
        worst = d0 + e_worst + s_worst
        if d0 - worst >= DAILY or worst <= FLOOR:
            return "fail"
        eq = d0 + e_pnl + s_pnl
        if eq <= FLOOR or d0 - eq >= DAILY:
            return "fail"
        if eq >= TARGET:
            return "pass"
    return "open"


def main() -> int:
    pd.set_option("display.width", 200)
    spx = IP.sessions("SP500")
    eur = eur_days()
    starts = range(0, len(spx) - 21, 21)
    rows = []
    for start_eq, label in ((ACCOUNT, "fresh $5,000"),
                            (5049.28, "from today's $5,049")):
        for name, ue, us in (("EUR short alone", True, False),
                             ("S&P system alone", False, True),
                             ("BOTH together", True, True)):
            out = pd.Series([replay(spx, eur, s, start_eq, ue, us)
                             for s in starts])
            rows.append(dict(start=label, setup=name, runs=len(out),
                             pass_pct=100 * (out == "pass").mean(),
                             fail_pct=100 * (out == "fail").mean(),
                             still_open_pct=100 * (out == "open").mean()))
    print(f"The5ers hourly bars {spx.index[0].date()} to "
          f"{spx.index[-1].date()}; target $5,500, floor $4,700, $150 a day")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    # How the two move together, day by day, at fixed full-account sizing.
    both = []
    for day in spx.itertuples():
        if day.Index in eur.index:
            s, _ = spx_day(day, ACCOUNT)
            both.append((eur.loc[day.Index, "eur_pnl"], s,
                         eur.loc[day.Index, "eur_worst"] + spx_day(day,
                                                                   ACCOUNT)[1]))
    b = pd.DataFrame(both, columns=["eur", "spx", "worst_combined"])
    print(f"\ndays with both trades: {len(b)}")
    print(f"correlation of their daily results: {b['eur'].corr(b['spx']):+.2f}")
    print(f"both lost on the same day: {100 * ((b['eur'] < 0) & (b['spx'] < 0)).mean():.0f}% "
          f"of days (if unrelated: {100 * (b['eur'] < 0).mean() * (b['spx'] < 0).mean():.0f}%)")
    print(f"worst combined day (both worst points at once): "
          f"${b['worst_combined'].min():.2f}  vs the $150 daily limit")
    print(f"daily swing: EUR ${b['eur'].std():.2f}, S&P ${b['spx'].std():.2f}, "
          f"together ${(b['eur'] + b['spx']).std():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
