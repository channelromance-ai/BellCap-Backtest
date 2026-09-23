"""
9/21 EMA crossover on the hourly chart, London and New York only.

Long when the 9 EMA closes above the 21, out when it closes back below,
mirrored for shorts. The stop sits at the most recent confirmed swing and
optionally moves to breakeven once the trade is 1R onside.

What the backtest of this rule found, kept here because it is the reason
the knobs exist:

  * Under "sessions only", the EMA cross is barely the exit. 62% of trades
    are closed by the session bell, 17% by the stop, and only 11% by the
    cross the rule is named after. The two instructions are in tension --
    hold a trend, but be flat by 4pm -- and the bell wins.
  * The breakeven move costs money. It helped in 6 of 24 configurations,
    average effect -0.016R per trade, because it converts trades that would
    have recovered into scratches at the price of the spread.
  * Sessions that fire one signal average +0.31R; sessions that fire three
    or more average -0.46R. That split is hindsight -- at the first signal
    you do not know how many follow -- and the tradable version, taking
    only the day's first signal, is worth about zero.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .. import calendars, store
from ..fills import resolve, OPEN, REASONS

COLUMNS = ["sym", "day", "entry_ts", "exit_ts", "dir", "entry", "stop",
           "risk", "exit", "reason", "r", "ambiguous"]


def prepare(m1, k=2, ema_basis="continuous", fast=9, slow=21,
            window="london", cal="NYSE"):
    """Hourly bars with EMAs, confirmed pivots, session flags and cursors."""
    h = store.resample(m1, "1h")
    h["mins"] = store.minutes(h.index)
    h["close_min"] = h["mins"] + 60
    h["day"] = h.index.normalize()

    tbl = calendars.sessions(cal=cal)
    bounds = h["day"].map(lambda d: tbl.get(d))
    starts = np.where(
        window == "london", calendars.LONDON_OPEN_ET,
        bounds.map(lambda b: b[0] if b else np.nan).to_numpy(float))
    ends = bounds.map(lambda b: b[1] if b else np.nan).to_numpy(float)
    h["sess_end"] = ends

    # A bar closing exactly at the bell has nowhere to go, so it is not an
    # entry bar.
    h["tradable"] = (~np.isnan(ends)) & (h["mins"] >= starts) & \
                    (h["close_min"] < ends)

    if ema_basis == "continuous":
        src = h["c"]
    else:
        src = h["c"].where(h["tradable"]).dropna()
    e_f = store.ema(src, fast)
    e_s = store.ema(src, slow)
    h["fast"] = e_f.reindex(h.index).ffill()
    h["slow"] = e_s.reindex(h.index).ffill()

    hi, lo = store.pivots(h["h"].to_numpy(), h["l"].to_numpy(), k)
    h["piv_hi"], h["piv_lo"] = hi, lo

    closes = (h.index + pd.Timedelta(hours=1)).to_numpy()
    h["m_idx"] = np.searchsorted(m1.index.to_numpy(), closes, "left")
    return h.dropna(subset=["fast", "slow"])


def run(sym, m1, h, cost_pts, use_be=True, be_mult=1.0,
        hold_overnight=False, first_signal_only=False):
    """Walk the hourly bars once. Returns a trade table."""
    a = store.arrays(m1)
    o, hi, lo = a["o"], a["h"], a["l"]
    mins, dayv = a["mins"], a["day"]
    nm = len(o)

    fast = h["fast"].to_numpy(float)
    slow = h["slow"].to_numpy(float)
    trad = h["tradable"].to_numpy(bool)
    p_hi = h["piv_hi"].to_numpy(float)
    p_lo = h["piv_lo"].to_numpy(float)
    midx = h["m_idx"].to_numpy(int)
    s_end = h["sess_end"].to_numpy(float)
    # Naive datetime64, to match the per-minute day keys exactly.
    hdays = store.day_keys(h.index)
    above = fast > slow
    n = len(h)

    rows = []
    pos = 0
    entry = stop = risk = 0.0
    cur_stop = 0.0
    armed = False
    e_day = None
    e_end = calendars.NY_CLOSE_ET
    e_mi = 0
    taken_today = None

    def close_out(px, mi, reason, amb):
        rows.append((sym, pd.Timestamp(e_day).date(), a["idx"][e_mi],
                     a["idx"][mi], "long" if pos > 0 else "short",
                     entry, stop, risk, float(px), reason,
                     (pos * (float(px) - entry) - cost_pts) / risk, amb))

    for i in range(1, n):
        j0 = midx[i]
        if j0 >= nm:
            break
        crossed = above[i] != above[i - 1]

        # 1. the bar that just closed may end the position
        if pos != 0 and crossed:
            close_out(o[j0], j0, "cross", False)
            pos = 0

        # 2. and may start one
        if pos == 0 and crossed and trad[i]:
            d = 1 if above[i] else -1
            if first_signal_only and taken_today == hdays[i]:
                pass
            else:
                sw = p_lo[i] if d > 0 else p_hi[i]
                px0 = float(o[j0])
                ok = math.isfinite(sw) and (sw < px0 if d > 0 else sw > px0)
                if ok:
                    pos, entry, stop = d, px0, sw
                    risk = abs(px0 - sw)
                    cur_stop, armed = sw, False
                    e_day, e_end, e_mi = hdays[i], s_end[i], j0
                    taken_today = hdays[i]

        # 3. carry it through the hour that follows
        if pos != 0:
            j1 = (midx[i + 1] - 1) if i + 1 < n else nm - 1
            j1 = min(j1, nm - 1)
            forced = -1
            if not hold_overnight and j1 >= j0:
                bad = (dayv[j0:j1 + 1] != e_day) | (mins[j0:j1 + 1] >= e_end)
                if bad.any():
                    forced = j0 + int(np.argmax(bad))
                    j1 = forced - 1

            if j1 >= j0:
                # The breakeven latch has to survive from one hour to the
                # next, or the position re-earns its protection every bar.
                xi, px, code, amb, cur_stop, armed, _ = resolve(
                    o, hi, lo, j0, j1, pos, entry, cur_stop, risk,
                    0.0, be_mult if use_be else 0.0, armed)
            else:
                xi, px, code, amb = -1, 0.0, OPEN, False

            if code != OPEN:
                close_out(px, xi, REASONS[code], amb)
                pos = 0
            elif forced >= 0:
                close_out(o[forced], forced, "session", False)
                pos = 0

    return pd.DataFrame(rows, columns=COLUMNS)
