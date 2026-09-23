"""
The pullback rule, with the three things the original could not express.

The first version was long-only, always aimed at a target twice the size of
its stop, and always waited for either that target or the closing bell. Its
own trade log said that was wrong: on the design period the target was
reached 5 times in 29, most trades died at the bell, and the best 10% of
trades only ever ran about 1.7 to 2.0 times the risk taken. Aiming at 2 and
usually getting stopped or timed out is a bad way to arrange a bet.

So this version can:

  * trade short as well as long, mirroring every rule, which doubles the
    sample and tests whether the logic is about markets or about a market
    that happened to rise;
  * aim at any multiple of risk, including less than 1, rather than only 2;
  * follow price with a trailing stop instead of, or as well as, a fixed
    target.

Everything about how signals are timed is unchanged and still lagged: the
rule is read on a bar's close and filled at the next bar's open, and the
risk distance comes from the bar that produced the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import calendars, store
from ..fills import OPEN, REASONS, resolve

COLUMNS = ["sym", "day", "entry_time", "exit_time", "dir", "entry", "stop",
           "risk", "exit", "reason", "r_multiple", "return_pct", "ambiguous"]


def prepare(bars, ema_fast=50, ema_slow=200, atr_period=14,
            rvol_days=20, adr_days=14, cal="NYSE"):
    """Indicators, session anchors and both directions' triggers."""
    df = bars.copy()
    day = store.day_keys(df.index)
    mins = store.minutes(df.index)
    df["day_key"] = day
    df["mins"] = mins

    df["ema_fast"] = store.ema(df["close"], ema_fast)
    df["ema_slow"] = store.ema(df["close"], ema_slow)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    df["vwap"] = pv / vv.where(vv > 0)

    df["bull"] = (df["close"] > df["ema_slow"]) & (df["close"] > df["vwap"])
    df["bear"] = (df["close"] < df["ema_slow"]) & (df["close"] < df["vwap"])

    prev_close, prev_fast = df["close"].shift(1), df["ema_fast"].shift(1)
    df["cross_up"] = (df["close"] > df["ema_fast"]) & (prev_close <= prev_fast)
    df["cross_dn"] = (df["close"] < df["ema_fast"]) & (prev_close >= prev_fast)

    base = df["volume"].groupby(mins, sort=False).transform(
        lambda s: s.shift(1).rolling(rvol_days, min_periods=rvol_days).mean())
    df["rvol"] = df["volume"] / base

    daily = df.groupby(day).agg(hi=("high", "max"), lo=("low", "min"))
    adr = (daily["hi"] - daily["lo"]).rolling(
        adr_days, min_periods=adr_days).mean().shift(1)
    df["adr"] = pd.Series(day, index=df.index).map(adr)
    df["session_range"] = (df["high"].groupby(day).cummax()
                           - df["low"].groupby(day).cummin())

    # Wilder's ATR, computed here rather than via store.atr because this
    # module speaks open/high/low/close and that helper speaks o/h/l/c.
    prev = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev).abs(),
         (df["low"] - prev).abs()], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1.0 / atr_period, adjust=False,
                               min_periods=atr_period).mean()

    tbl = calendars.sessions(cal=cal)
    lut = {pd.Timestamp(d).tz_localize(None).to_datetime64(): c
           for d, (_o, c) in tbl.items()}
    df["close_min"] = pd.Series(day, index=df.index).map(lut)
    return df.dropna(subset=["close_min"])


def run(sym, df, cost_bps=0.5, direction="long", rvol_mult=1.3,
        adr_exhaustion=0.85, atr_sl=2.5, rr=2.0, trail=0.0, be_mult=0.0,
        entry_start=10 * 60, entry_end=15 * 60 + 30, flat_before_close=15):
    """
    Walk the bars once.

    `rr` is the target as a multiple of the risk taken; 0 means no target,
    which only makes sense alongside a trail. `trail` follows price that many
    multiples of risk behind its best level. `flat_before_close` is how many
    minutes before the real closing bell the position is squared off.
    """
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    mins = df["mins"].to_numpy(np.int32)
    day = df["day_key"].to_numpy()
    close_min = df["close_min"].to_numpy(np.int32)

    atr = df["atr"].to_numpy(np.float64)
    rvol = df["rvol"].to_numpy(np.float64)
    adr = df["adr"].to_numpy(np.float64)
    srange = df["session_range"].to_numpy(np.float64)
    bull = df["bull"].to_numpy(bool)
    bear = df["bear"].to_numpy(bool)
    up = df["cross_up"].to_numpy(bool)
    dn = df["cross_dn"].to_numpy(bool)
    stamps = df.index
    n = len(df)

    want_long = direction in ("long", "both")
    want_short = direction in ("short", "both")

    ok_rvol = rvol > rvol_mult if rvol_mult > 0 else np.ones(n, bool)
    ok_adr = (srange <= adr_exhaustion * adr if adr_exhaustion < 90
              else np.ones(n, bool))
    in_win = (mins >= entry_start) & (mins <= entry_end)
    common = in_win & ok_rvol & ok_adr & np.isfinite(atr) & (atr > 0)

    sig = np.zeros(n, np.int8)
    if want_long:
        sig[common & bull & up] = 1
    if want_short:
        sig[common & bear & dn] = -1

    # The signal is read at a bar's close and filled at the next bar's open.
    fill_sig = np.zeros(n, np.int8)
    fill_sig[1:] = sig[:-1]
    atr_sig = np.empty(n)
    atr_sig[0] = np.nan
    atr_sig[1:] = atr[:-1]

    # Last bar of each session, and the bar at which to be flat.
    idx = np.arange(n)
    dser = pd.Series(day)
    last_idx = pd.Series(idx).groupby(dser).transform("max").to_numpy()
    flat_at = close_min - flat_before_close
    elig = np.where(mins >= flat_at, idx, np.iinfo(np.int64).max)
    flat_idx = pd.Series(elig).groupby(dser).transform("min").to_numpy()
    has_flat = flat_idx != np.iinfo(np.int64).max
    flat_idx = np.where(has_flat, flat_idx, last_idx)

    cand = np.flatnonzero((fill_sig != 0) & np.isfinite(atr_sig)
                          & (atr_sig > 0))
    cost = cost_bps / 10_000.0
    rows = []
    ptr = 0

    while ptr < cand.size:
        i = int(cand[ptr])
        if has_flat[i] and i >= flat_idx[i]:
            ptr += 1
            continue
        d = int(fill_sig[i])
        entry = float(o[i])
        risk = atr_sl * float(atr_sig[i])
        stop = entry - d * risk
        target = entry + d * rr * risk if rr else 0.0

        if has_flat[i]:
            scan_end = int(flat_idx[i]) - 1
            fb_i, fb_px, fb_why = int(flat_idx[i]), float(o[int(flat_idx[i])]), "eod"
        else:
            scan_end = int(flat_idx[i])
            fb_i, fb_px, fb_why = scan_end, float(c[scan_end]), "session_end"

        xi, px, why, amb = fb_i, fb_px, fb_why, False
        if scan_end >= i:
            j, p, code, a, _, _, _ = resolve(
                o, h, lo, i, scan_end, d, entry, stop, risk, target,
                be_mult, False, trail, 0.0)
            if code != OPEN:
                xi, px, why, amb = j, p, REASONS[code], a

        gross = d * (px - entry) / entry
        net = gross - 2.0 * cost
        rows.append((sym, pd.Timestamp(day[i]).date(), stamps[i], stamps[xi],
                     "long" if d > 0 else "short", entry, stop, risk,
                     float(px), why, d * (px - entry) / risk,
                     100.0 * net, bool(amb)))
        ptr = int(np.searchsorted(cand, xi, side="right"))

    return pd.DataFrame(rows, columns=COLUMNS)
