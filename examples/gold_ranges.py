"""
Gold ranges, version 4: calibrated to the user's own marked examples.

The user drew four 15-minute ranges on gold (22, 24, 28 and 29 Sep 2026).
They are nothing like the 8-bar tight boxes tested before: 6-8 hours long,
several candles tall, edges where price repeatedly turned (wicks may poke
through), and usually formed after a strong move. This module encodes that
and is first checked by eye against those four examples (calibrate.py
below draws its boxes on the same days). Only the DEFINITION of a range is
calibrated here; nothing about profitability is looked at until it is fixed.

A range ending at bar e (all from bars at or before e):
  * a window of N bars, N between MIN_BARS and MAX_BARS
  * edges from candle BODIES: top = 90th percentile of body tops,
    bottom = 10th percentile of body bottoms, so stray wicks do not set them
  * at least CONTAIN of closes inside the edges
  * height between MIN_H and MAX_H x the window's median candle range
  * at least TOUCHES separate visits to each edge (a visit = a bar reaching
    within 20% of the height of that edge; visits at least 3 bars apart)
  * sideways: a line fitted through the closes moves less than half the
    height across the window
  * at least 80% of candles (wicks included) within a quarter-height of
    the edges
Thresholds were set by MEASURING the user's four marked boxes (heights
1.8-2.5 median candles, 82-94% of closes inside, 84-96% of candles near
the edges, 2-3 touches a side, slope 3-45% of height). The first guesses,
made by eye from screenshots, were too tall and too strict.
"""
from __future__ import annotations

import numpy as np
from numba import njit
import pandas as pd

MIN_BARS, MAX_BARS = 20, 48
CONTAIN = 0.80
MIN_H, MAX_H = 1.5, 4.0
TOUCHES = 2


def visits(mask, gap=3):
    """Number of separate runs of True, runs closer than `gap` merged."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    return 1 + int((np.diff(idx) > gap).sum())


def qualifies(o, h, l, c, s, e):
    bt = np.maximum(o[s:e + 1], c[s:e + 1])
    bb = np.minimum(o[s:e + 1], c[s:e + 1])
    top, bot = np.quantile(bt, 0.9), np.quantile(bb, 0.1)
    height = top - bot
    med = np.median(h[s:e + 1] - l[s:e + 1])
    if height <= 0 or med <= 0:
        return None
    if not (MIN_H * med <= height <= MAX_H * med):
        return None
    cc = c[s:e + 1]
    if ((cc >= bot) & (cc <= top)).mean() < CONTAIN:
        return None
    # Truly sideways: the fitted line across the window moves less than
    # half the height. A slow drift is a trend, not a range.
    x = np.arange(len(cc))
    slope = np.polyfit(x, cc, 1)[0]
    if abs(slope * len(cc)) > 0.5 * height:
        return None
    # Spikes break a range rather than belong to it: 80% of candles must
    # stay within a quarter-height of the edges, wicks included.
    hh, ll = h[s:e + 1], l[s:e + 1]
    if ((hh <= top + 0.25 * height) & (ll >= bot - 0.25 * height)).mean() \
            < 0.8:
        return None
    near_top = h[s:e + 1] >= top - 0.2 * height
    near_bot = l[s:e + 1] <= bot + 0.2 * height
    if visits(near_top) < TOUCHES or visits(near_bot) < TOUCHES:
        return None
    return top, bot


def live_ranges(df, step=2):
    """
    The range in force at each bar, using only bars up to that bar.

    For every bar e, every window ending at e (lengths MIN_BARS..MAX_BARS)
    is tested and the TIGHTEST qualifying one kept (height measured in
    median candles, with a small preference for longer windows). Choosing
    among windows that end later would need candles not yet printed, which
    a trader watching the chart does not have -- so the choice is made bar
    by bar, live. Returns arrays (start, top, bot), NaN where no range.
    """
    o, h, l, c = (df[x].to_numpy(np.float64) for x in ("open", "high",
                                                        "low", "close"))
    n = len(c)
    start = np.full(n, -1)
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    for e in range(MIN_BARS - 1, n):
        best, best_score = None, np.inf
        for length in range(MIN_BARS, MAX_BARS + 1, step):
            s = e - length + 1
            if s < 0:
                break
            q = qualifies(o, h, l, c, s, e)
            if q is None:
                continue
            med = np.median(h[s:e + 1] - l[s:e + 1])
            score = (q[0] - q[1]) / med - 0.03 * length
            if score < best_score:
                best, best_score = (s, *q), score
        if best:
            start[e], top[e], bot[e] = best
    return start, top, bot


@njit(cache=True)
def _qual_fast(o, h, l, c, s, e, min_h, max_h, contain, touches):
    m = e - s + 1
    bt = np.empty(m)
    bb = np.empty(m)
    rg = np.empty(m)
    for i in range(m):
        bt[i] = max(o[s + i], c[s + i])
        bb[i] = min(o[s + i], c[s + i])
        rg[i] = h[s + i] - l[s + i]
    top = np.percentile(bt, 90.0)
    bot = np.percentile(bb, 10.0)
    height = top - bot
    med = np.median(rg)
    if height <= 0 or med <= 0:
        return False, 0.0, 0.0, 0.0
    if height < min_h * med or height > max_h * med:
        return False, 0.0, 0.0, 0.0
    inside = 0
    near = 0
    sx = sy = sxx = sxy = 0.0
    for i in range(m):
        ci = c[s + i]
        if bot <= ci <= top:
            inside += 1
        if h[s + i] <= top + 0.25 * height and l[s + i] >= bot - 0.25 * height:
            near += 1
        sx += i
        sy += ci
        sxx += i * i
        sxy += i * ci
    if inside < contain * m or near < 0.8 * m:
        return False, 0.0, 0.0, 0.0
    slope = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    if abs(slope * m) > 0.5 * height:
        return False, 0.0, 0.0, 0.0
    vt = vb = 0
    last_t = last_b = -100
    for i in range(m):
        if h[s + i] >= top - 0.2 * height:
            if i - last_t > 3:
                vt += 1
            last_t = i
        if l[s + i] <= bot + 0.2 * height:
            if i - last_b > 3:
                vb += 1
            last_b = i
    if vt < touches or vb < touches:
        return False, 0.0, 0.0, 0.0
    return True, top, bot, med


@njit(cache=True)
def _live_fast(o, h, l, c, min_bars, max_bars, step, min_h, max_h, contain,
               touches):
    n = len(c)
    start = np.full(n, -1)
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    for e in range(min_bars - 1, n):
        best_score = 1e18
        for length in range(min_bars, max_bars + 1, step):
            s = e - length + 1
            if s < 0:
                break
            ok, t, b, med = _qual_fast(o, h, l, c, s, e, min_h, max_h,
                                       contain, touches)
            if not ok:
                continue
            score = (t - b) / med - 0.03 * length
            if score < best_score:
                best_score = score
                start[e] = s
                top[e] = t
                bot[e] = b
    return start, top, bot


def live_ranges_fast(df, step=2):
    """Same as live_ranges, compiled; checked equal on the calibration days."""
    o, h, l, c = (df[x].to_numpy(np.float64) for x in ("open", "high",
                                                        "low", "close"))
    return _live_fast(o, h, l, c, MIN_BARS, MAX_BARS, step, MIN_H, MAX_H,
                      CONTAIN, TOUCHES)


def breakouts(df, start, top, bot, buffer=0.1):
    """
    A breakout: a close beyond the range in force at the PREVIOUS bar, by
    `buffer` x its height. One per range: after a breakout the next one
    needs a range that began after it.
    """
    c = df["close"].to_numpy(np.float64)
    out = []
    last_bo = -1
    for k in range(1, len(c)):
        s, t, b = start[k - 1], top[k - 1], bot[k - 1]
        if s < 0 or s <= last_bo:
            continue
        hgt = t - b
        if c[k] > t + buffer * hgt:
            out.append((k, 1, s, k - 1, t, b))
            last_bo = k
        elif c[k] < b - buffer * hgt:
            out.append((k, -1, s, k - 1, t, b))
            last_bo = k
    return pd.DataFrame(out, columns=["k", "d", "start", "end", "top", "bot"])


def find_ranges(df):
    """All non-overlapping ranges: (start_i, end_i, top, bot)."""
    o, h, l, c = (df[x].to_numpy(np.float64) for x in ("open", "high",
                                                        "low", "close"))
    n = len(c)
    out = []
    e = MIN_BARS
    last_end = -1
    while e < n:
        best = None
        for length in range(MAX_BARS, MIN_BARS - 1, -1):
            s = e - length + 1
            if s <= last_end:          # never overlap the previous range
                continue
            q = qualifies(o, h, l, c, s, e)
            if q:
                best = (s, e, *q)
                break
        if best is None:
            e += 1
            continue
        # Extend the end while it keeps qualifying, then record it.
        s = best[0]
        while e + 1 < n and e + 1 - s < MAX_BARS:
            q = qualifies(o, h, l, c, s, e + 1)
            if not q:
                break
            e += 1
            best = (s, e, *q)
        out.append(best)
        last_end = best[1]
        e = best[1] + MIN_BARS
    return pd.DataFrame(out, columns=["start", "end", "top", "bot"])
