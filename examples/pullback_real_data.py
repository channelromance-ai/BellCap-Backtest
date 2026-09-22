"""
Run the standalone pullback rule on real index-CFD bars, and then try to
break it.

The headline backtest is the easy part and the least informative. What
follows it is the part that decides whether the headline means anything:

  * Costs, at the measured spread and at multiples of it.
  * A randomised-entry control. The rule is long-only on two indices that
    rose 68% and 90% over the sample, so "makes money" is not evidence of
    anything. The control takes the same number of long trades, in the same
    entry window, with the same ATR stop and target, at random times. If the
    rule cannot beat that, the signal is decoration on a drift trade.
  * A Reality Check over the parameter grid, because trying 36 settings and
    reporting the best one is how noise gets published.
  * An out-of-sample split.

Run:  python examples/pullback_real_data.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

from bcbt import calendars, metrics, store

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "standalone"))
import intraday_pullback as ip  # noqa: E402

SYMBOLS = ("SPX500", "NAS100")

# Median spread measured from Dukascopy ask data, in index points.
SPREAD_PTS = {"SPX500": 0.51, "NAS100": 1.20}

RNG = np.random.default_rng(20260922)
N_CONTROL = 400


def rth_5min(sym: str) -> pd.DataFrame:
    """
    Regular-hours 5-minute bars, holidays and half-day stubs removed.

    The rule anchors VWAP to the session and squares off at 15:45, so it is a
    cash-session strategy. Feeding it the CFD's 23-hour tape would anchor the
    VWAP at midnight and make Layer 1 meaningless.
    """
    m1 = store.load_m1(sym)
    tbl = calendars.sessions()

    keys = store.day_keys(m1.index)
    # Calendar keys are tz-aware midnights; the per-bar day keys are naive.
    # Strip the zone rather than localising, which would throw.
    lut = {pd.Timestamp(d).tz_localize(None).to_datetime64(): (o, c)
           for d, (o, c) in tbl.items()}
    mins = store.minutes(m1.index)

    keep = np.zeros(len(m1), bool)
    o_arr = np.full(len(m1), -1, np.int32)
    c_arr = np.full(len(m1), -1, np.int32)
    uniq, first = np.unique(keys, return_index=True)
    ends = np.append(first[1:], len(keys))
    for k, a, b in zip(uniq, first, ends, strict=True):
        got = lut.get(k)
        if got is None:
            continue
        o_arr[a:b], c_arr[a:b] = got
    keep = (o_arr >= 0) & (mins >= o_arr) & (mins < c_arr)

    sub = m1.loc[keep, ["o", "h", "l", "c", "v"]]
    bars = sub.resample("5min", closed="left", label="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    ).dropna(subset=["o", "h", "l", "c"])
    bars.columns = ["open", "high", "low", "close", "volume"]
    return bars


def cost_bps_for(sym: str, bars: pd.DataFrame) -> float:
    """
    Convert the measured point spread into the script's one-way bps knob.

    `intraday_pullback` charges `2 * cost_bps`, and a CFD round trip crosses
    the spread once, so the one-way figure is half the spread in bps.
    """
    mid = float(bars["close"].mean())
    round_trip_bps = SPREAD_PTS[sym] / mid * 1e4
    return round_trip_bps / 2.0


def random_control(prepared, cfg, n_signals, n_iter=N_CONTROL):
    """
    The same trade count, window and risk model, at random moments.

    Entry timing is the only thing the rule contributes over this baseline,
    so the gap between them is the whole value of Layers 1-3.
    """
    eligible = np.flatnonzero(
        prepared["in_window"].to_numpy(bool)
        & np.isfinite(prepared["sl_dist"].to_numpy(float))
        & (prepared["sl_dist"].to_numpy(float) > 0))
    if eligible.size < n_signals or n_signals == 0:
        return np.array([])

    frame = prepared.copy()
    out = np.empty(n_iter)
    for i in range(n_iter):
        pick = RNG.choice(eligible, size=n_signals, replace=False)
        flag = np.zeros(len(frame), bool)
        flag[pick] = True
        frame["entry_signal"] = flag
        t = ip.run_backtest(frame, cfg)
        out[i] = t["return_pct"].mean() if len(t) else np.nan
    return out[np.isfinite(out)]


def main() -> int:
    pd.set_option("display.width", 200)
    base = {}

    for sym in SYMBOLS:
        bars = rth_5min(sym)
        cb = cost_bps_for(sym, bars)
        cfg = ip.Config(cost_bps=cb)
        prepared = ip.prepare_data(bars, cfg)
        trades = ip.run_backtest(prepared, cfg)
        base[sym] = (bars, prepared, trades, cfg)

        print("\n" + "=" * 72)
        print(f"{sym}   {len(bars):,} five-minute bars, "
              f"{bars.index.normalize().nunique()} sessions, "
              f"{bars.index[0].date()} -> {bars.index[-1].date()}")
        print(f"   zero-volume bars in RTH: "
              f"{100.0 * (bars['volume'] <= 0).mean():.2f}%")
        print(f"   spread {SPREAD_PTS[sym]} pts -> cost_bps={cb:.3f} "
              f"(round trip {2 * cb:.2f} bps)")
        print(f"   raw signals {int(prepared['raw_signal'].sum())}, "
              f"trades taken {len(trades)}")
        print("=" * 72)
        ip.performance_report(trades)

        if trades.empty:
            continue

        # ---- where the signals died, to see which layer is binding
        macro = prepared["macro_bull"] & prepared["cross_up"]
        print("\n  funnel (signal bars surviving each layer):")
        print(f"    EMA50 cross              : {int(prepared['cross_up'].sum()):>6}")
        print(f"    + macro (EMA200 & VWAP)  : {int(macro.sum()):>6}")
        print(f"    + entry window           : {int((macro & prepared['in_window']).sum()):>6}")
        print(f"    + RVOL > 1.3x            : {int((macro & prepared['in_window'] & prepared['rvol_ok']).sum()):>6}")
        print(f"    + ADR not exhausted      : {int(prepared['raw_signal'].sum()):>6}")

    # ---------------------------------------------------------------- costs
    print("\n" + "=" * 72)
    print("COST SENSITIVITY  (mean return per trade, %)")
    print("=" * 72)
    print(f"  {'symbol':10s}{'0x':>10s}{'1x measured':>14s}{'3x':>10s}{'6x':>10s}")
    for sym in SYMBOLS:
        bars, prepared, _, cfg = base[sym]
        row = []
        for mult in (0.0, 1.0, 3.0, 6.0):
            c = ip.Config(cost_bps=cost_bps_for(sym, bars) * mult)
            t = ip.run_backtest(prepared, c)
            row.append(t["return_pct"].mean() if len(t) else np.nan)
        print(f"  {sym:10s}{row[0]:>10.4f}{row[1]:>14.4f}"
              f"{row[2]:>10.4f}{row[3]:>10.4f}")

    # ------------------------------------------------------ random control
    print("\n" + "=" * 72)
    print(f"RANDOM-ENTRY CONTROL  ({N_CONTROL} draws: same trade count, same "
          f"window,\n                       same ATR stop/target, random timing)")
    print("=" * 72)
    for sym in SYMBOLS:
        _, prepared, trades, cfg = base[sym]
        if trades.empty:
            continue
        n_sig = int(prepared["entry_signal"].sum())
        draws = random_control(prepared, cfg, n_sig)
        if draws.size == 0:
            continue
        real = trades["return_pct"].mean()
        pct = 100.0 * (draws < real).mean()
        p = (1.0 + (draws >= real).sum()) / (1.0 + draws.size)
        print(f"  {sym}: rule {real:+.4f}%   random {draws.mean():+.4f}% "
              f"(sd {draws.std():.4f})   percentile {pct:.1f}%   p={p:.3f}")

    # ------------------------------------------------------ parameter grid
    print("\n" + "=" * 72)
    print("PARAMETER SEARCH + REALITY CHECK")
    print("=" * 72)
    grid = list(itertools.product(
        (1.0, 1.3, 1.8),          # rvol_mult
        (0.70, 0.85, 1.00),       # adr_exhaustion
        ((2.0, 4.0), (2.5, 5.0), (3.0, 6.0)),   # sl/tp ATR multiples
        (20, 50),                 # ema_fast
    ))
    for sym in SYMBOLS:
        bars, _, _, _ = base[sym]
        cb = cost_bps_for(sym, bars)
        rows, per_day = [], {}
        for rv, adr, (sl, tp), fast in grid:
            cfg = ip.Config(cost_bps=cb, rvol_mult=rv, adr_exhaustion=adr,
                            atr_sl_mult=sl, atr_tp_mult=tp, ema_fast=fast)
            prep = ip.prepare_data(bars, cfg)
            t = ip.run_backtest(prep, cfg)
            if len(t) < 30:
                continue
            tag = f"rv{rv}/adr{adr}/{sl}-{tp}/ema{fast}"
            rows.append(dict(cell=tag, n=len(t),
                             mean_pct=t["return_pct"].mean(),
                             total_pct=ip.performance_report(
                                 t, verbose=False)["total_return_pct"],
                             win=100.0 * (t["return_pct"] > 0).mean()))
            d = t.copy()
            d["day"] = pd.DatetimeIndex(d["entry_time"]).normalize()
            per_day[tag] = d.groupby("day")["return_pct"].agg(["sum", "count"])

        if len(per_day) < 2:
            print(f"  {sym}: too few populated cells")
            continue
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        print(f"\n  --- {sym}: top and bottom cells of {len(df)} ---")
        print(df.head(5).round(4).to_string(index=False))
        print(df.tail(3).round(4).to_string(index=False))

        name, val, p, null = metrics.reality_check(per_day, n_boot=4000)
        print(f"\n  Reality Check: best {name}  mean {val:+.4f}%")
        print(f"    null best median {np.median(null):+.4f}%, "
              f"95th {np.percentile(null, 95):+.4f}%")
        print(f"    p = {p:.3f}  ", end="")
        print("survives the search" if p < 0.05
              else "-> indistinguishable from search noise")

    # -------------------------------------------------------- out of sample
    print("\n" + "=" * 72)
    print("OUT OF SAMPLE  (default settings, split at 2025-06-01)")
    print("=" * 72)
    for sym in SYMBOLS:
        _, _, trades, _ = base[sym]
        if trades.empty:
            continue
        t = trades.copy()
        t["entry_time"] = pd.DatetimeIndex(t["entry_time"])
        a = t[t["entry_time"] < "2025-06-01"]["return_pct"]
        b = t[t["entry_time"] >= "2025-06-01"]["return_pct"]
        for label, s in (("in ", a), ("out", b)):
            if len(s) < 5:
                print(f"  {sym} {label}: too few trades ({len(s)})")
                continue
            tstat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
            print(f"  {sym} {label}: n={len(s):>4}  mean {s.mean():+.4f}%  "
                  f"total {s.sum():+.2f}%  t={tstat:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
