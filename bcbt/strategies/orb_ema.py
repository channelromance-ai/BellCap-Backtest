"""
Opening-range breakout confirmed by an EMA close, on 5-minute bars.

Take the first N minutes of the cash session as a range. Go long on the
first 5-minute bar that closes above the range high *and* above the EMA;
short on the mirror. Stop at the opposite side of the range, at the signal
bar's extreme, or at an ATR multiple. Exit at a fixed R multiple, or when a
bar closes back through the EMA, or at the bell.

From the backtest of this rule:

  * The EMA is the weak half. Strip the range test out and trade the EMA
    close alone and it loses 0.17 to 0.40R per trade -- a reliable way to
    lose money. All the structure is in the opening range.
  * The opposite side of the range is the only stop that is not
    consistently negative. Signal-bar and ATR stops sit inside the noise
    and get taken about 65% of the time.
  * A fixed 1.5R target beat the EMA exit nearly everywhere, because a
    12-EMA close-flip on a 5-minute chart fires constantly -- it accounted
    for 91 to 98% of exits and cut winners early.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .. import calendars, store
from ..fills import resolve, OPEN, REASONS

COLUMNS = ["sym", "day", "entry_ts", "exit_ts", "dir", "entry", "stop",
           "risk", "exit", "reason", "r", "ambiguous"]


def prepare(m1, ema_span=12, atr_span=14, cal="NYSE"):
    """Per-session numpy bundles: 5-minute signal bars over 1-minute fills."""
    m5 = store.resample(m1, "5min")
    m5["ema"] = store.ema(m5["c"], ema_span)
    m5["atr"] = store.atr(m5, atr_span)
    m5["mins"] = store.minutes(m5.index)
    m5["day"] = m5.index.normalize()

    a = store.arrays(m1)
    tbl = calendars.sessions(cal=cal)
    days = {}

    # The minute series is sorted, so each day is one contiguous block.
    # Finding those blocks once beats re-scanning 1.1M rows per session.
    day_arr = a["day"]
    uniq, starts = np.unique(day_arr, return_index=True)
    ends = np.append(starts[1:], len(day_arr))
    block = {d: (int(s), int(e)) for d, s, e in zip(uniq, starts, ends, strict=True)}

    for day, g in m5.groupby("day"):
        got = tbl.get(day)
        if got is None:                       # holiday: the feed still quotes
            continue
        o_min, c_min = got
        s = g[(g["mins"] >= o_min) & (g["mins"] < c_min)]
        if len(s) < 20:                       # shortened or broken session
            continue
        span = block.get(store.day_key(day))
        if span is None:
            continue
        b0, b1 = span
        sub = a["mins"][b0:b1]
        lo_off = int(np.searchsorted(sub, o_min, "left"))
        hi_off = int(np.searchsorted(sub, c_min, "right"))
        if hi_off <= lo_off:
            continue
        idx = np.array([b0 + lo_off, b0 + hi_off - 1])
        days[day] = dict(
            open_min=o_min, close_min=c_min,
            i0=int(idx[0]), i1=int(idx[-1]),
            t5=s["mins"].to_numpy(np.int32),
            h5=s["h"].to_numpy(float), l5=s["l"].to_numpy(float),
            c5=s["c"].to_numpy(float), e5=s["ema"].to_numpy(float),
            a5=s["atr"].to_numpy(float))
    return days, a


def run(sym, days, a, cost_pts, or_min=15, stop_mode="or", rr=1.5,
        ema_exit=False, atr_k=1.0, buffer_frac=0.0, entry_mode="orb",
        cutoff_min=15 * 60, be_mult=0.0):
    """Walk every session once. Returns a trade table."""
    o, hi, lo = a["o"], a["h"], a["l"]
    mins = a["mins"]
    n_or = max(1, or_min // 5)
    rows = []

    for day in sorted(days):
        s = days[day]
        t5, c5, e5, h5, l5 = s["t5"], s["c5"], s["e5"], s["h5"], s["l5"]
        if len(t5) <= n_or:
            continue
        or_hi, or_lo = h5[:n_or].max(), l5[:n_or].min()
        cutoff = min(cutoff_min, s["close_min"] - 60)

        d = 0
        k = -1
        for i in range(n_or, len(t5)):
            if t5[i] > cutoff:
                break
            if entry_mode == "orb":
                if c5[i] > or_hi and c5[i] > e5[i]:
                    d, k = 1, i
                    break
                if c5[i] < or_lo and c5[i] < e5[i]:
                    d, k = -1, i
                    break
            else:
                if c5[i] > e5[i]:
                    d, k = 1, i
                    break
                if c5[i] < e5[i]:
                    d, k = -1, i
                    break
        if d == 0:
            continue

        # Fill at the first minute open after the signal bar closed.
        want = t5[k] + 5
        seg = mins[s["i0"]:s["i1"] + 1]
        off = int(np.searchsorted(seg, want, "left"))
        j = s["i0"] + off
        if off >= seg.size or mins[j] >= s["close_min"]:
            continue
        entry = float(o[j])

        if stop_mode == "or":
            stop = or_lo if d > 0 else or_hi
        elif stop_mode == "bar":
            pad = buffer_frac * (h5[k] - l5[k])
            stop = (l5[k] - pad) if d > 0 else (h5[k] + pad)
        else:
            stop = entry - d * atr_k * float(s["a5"][k])

        risk = abs(entry - stop)
        if risk <= 0 or not math.isfinite(risk):
            continue
        # A gap past the level leaves the stop on the wrong side; there is
        # no trade to take there.
        if (d > 0 and stop >= entry) or (d < 0 and stop <= entry):
            continue

        target = entry + d * rr * risk if rr else 0.0
        last = s["i1"]

        if ema_exit:
            cursor = j
            cur_stop, armed = stop, False
            px = None
            for i in range(k, len(t5)):
                nxt = int(np.searchsorted(seg, t5[i] + 5, "left"))
                seg_end = min(s["i0"] + nxt - 1, last)
                if seg_end >= cursor:
                    xi, p, code, amb, cur_stop, armed = resolve(
                        o, hi, lo, cursor, seg_end, d, entry, cur_stop,
                        risk, target, be_mult, armed)
                    if code != OPEN:
                        px, mi, why = p, xi, REASONS[code]
                        break
                flipped = (c5[i] < e5[i]) if d > 0 else (c5[i] > e5[i])
                if flipped and i > k:
                    if seg_end + 1 <= last:
                        px, mi, why, amb = float(o[seg_end + 1]), seg_end + 1, "ema", False
                    break
                cursor = seg_end + 1
            if px is None:
                px, mi, why, amb = float(a["c"][last]), last, "eod", False
        else:
            xi, p, code, amb, _, _ = resolve(
                o, hi, lo, j, last, d, entry, stop, risk, target, be_mult)
            if code == OPEN:
                px, mi, why = float(a["c"][last]), last, "eod"
            else:
                px, mi, why = p, xi, REASONS[code]

        rows.append((sym, day.date(), a["idx"][j], a["idx"][mi],
                     "long" if d > 0 else "short", entry, stop, risk,
                     float(px), why,
                     (d * (float(px) - entry) - cost_pts) / risk, bool(amb)))

    return pd.DataFrame(rows, columns=COLUMNS)
