"""
NAS100 opening range, 12 EMA direction, 1R breakeven-then-trail.

The rules as written:

  * Opening range: high and low of the five-minute bars from 09:30 up to
    10:00 ET (six bars).
  * At the close of the bar that ends at 10:00, compare the close with the
    12-period EMA of five-minute closes. Above: long. Below: short. Equal:
    no trade. One trade a day, no re-entry.
  * Enter at that close -- here the open of the first minute after it.
  * Stop just beyond the other side of the range. 1R = entry to stop.
  * Once the trade is 1R onside the stop moves to entry, then trails 1R
    behind the best price since entry, never loosening.
  * Out on the stop, or flat at the session close.

Everything a reading of those rules leaves open is a parameter, so the
alternative readings can be tested rather than argued about:

  delay     which bar is "the 10:00 bar". 0 is the bar that ends at 10:00
            (the last bar of the range); 5 is the bar that opens at 10:00
            and closes at 10:05, which is what a chart labelled by open
            time calls "the 10:00 bar".
  ema       "all12" is the EMA over every five-minute bar the CFD prints,
            overnight included, which is what a 24-hour NAS100 chart shows.
            "rth12" is the EMA over cash-session bars only, which is what
            an NDX cash-index chart shows.
  buf       how far beyond the range "just below" is, in index points.

The trade table keeps gross and net R, the excursions, and the range
statistics the filters need, so the study can slice it without re-running.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .. import calendars, store
from ..fills import resolve_trail, OPEN, REASONS

COLUMNS = ["sym", "day", "entry_ts", "exit_ts", "dir", "entry", "stop",
           "risk", "exit", "reason", "r", "r_gross", "bp", "ambiguous",
           "armed", "mfe_r", "mae_r", "mins_held", "or_hi", "or_lo",
           "or_rng", "or_ratio", "sig_close", "sig_ema"]

SPANS = (9, 12, 20, 26)


def prepare(m1, spans=SPANS, cal="NYSE"):
    """
    Per-session numpy bundles: five-minute signal bars over one-minute fills.

    Both EMA families are computed on the full history before it is cut
    into sessions, so each session's EMA carries the previous days in it
    exactly as a chart would.
    """
    first = m1.index[0].date().isoformat()
    last = m1.index[-1].date().isoformat()
    tbl = calendars.sessions(start=first, end=last, cal=cal)

    m5 = store.resample(m1, "5min")
    mins5 = store.minutes(m5.index)
    day5 = m5.index.normalize()

    # A bar belongs to the cash session if its whole five minutes fall
    # inside that day's real hours, half-days included.
    o_map = pd.Series({k: v[0] for k, v in tbl.items()})
    c_map = pd.Series({k: v[1] for k, v in tbl.items()})
    o_min = o_map.reindex(day5).to_numpy()
    c_min = c_map.reindex(day5).to_numpy()
    rth = (mins5 >= o_min) & (mins5 + 5 <= c_min)

    for sp in spans:
        m5[f"all{sp}"] = store.ema(m5["c"], sp)
        e = store.ema(m5.loc[rth, "c"], sp)
        m5[f"rth{sp}"] = e.reindex(m5.index)

    a = store.arrays(m1)
    day_arr = a["day"]
    uniq, starts = np.unique(day_arr, return_index=True)
    ends = np.append(starts[1:], len(day_arr))
    block = {d: (int(s), int(e)) for d, s, e in zip(uniq, starts, ends, strict=True)}

    ema_cols = [c for c in m5.columns if c[:3] in ("all", "rth")]
    m5["mins"] = mins5
    days = {}
    for day, g in m5.groupby(day5):
        got = tbl.get(day)
        if got is None:                       # holiday: the feed still quotes
            continue
        om, cm = got
        s = g[(g["mins"] >= om) & (g["mins"] < cm)]
        if len(s) < 12:
            continue
        span = block.get(store.day_key(day))
        if span is None:
            continue
        b0, b1 = span
        sub = a["mins"][b0:b1]
        lo_off = int(np.searchsorted(sub, om, "left"))
        hi_off = int(np.searchsorted(sub, cm, "left"))   # last bar 15:59
        if hi_off <= lo_off:
            continue
        days[day] = dict(
            open_min=om, close_min=cm,
            i0=b0 + lo_off, i1=b0 + hi_off - 1,
            t5=s["mins"].to_numpy(np.int32),
            h5=s["h"].to_numpy(float), l5=s["l"].to_numpy(float),
            c5=s["c"].to_numpy(float),
            ema={c: s[c].to_numpy(float) for c in ema_cols})
    return days, a


def opening_ranges(days, or_min=30):
    """(or_hi, or_lo) per session, or None where a range bar is missing."""
    out = {}
    n = or_min // 5
    for day, s in days.items():
        want = s["open_min"] + 5 * np.arange(n)
        t5 = s["t5"]
        if len(t5) < n or not np.array_equal(t5[:n], want):
            out[day] = None
            continue
        out[day] = (float(s["h5"][:n].max()), float(s["l5"][:n].min()))
    return out


def range_ratio(ranges, lookback=20):
    """
    Today's range over the median of the previous `lookback` ranges.

    Strictly prior days only: a filter that includes today in its own
    average is reading the thing it is meant to judge.
    """
    keys = sorted(k for k, v in ranges.items() if v is not None)
    rng = pd.Series([ranges[k][0] - ranges[k][1] for k in keys], index=keys)
    med = rng.shift(1).rolling(lookback, min_periods=lookback).median()
    return (rng / med).to_dict()


def run(sym, days, a, cost_pts=1.5, or_min=30, delay=0, ema="all12",
        buf=0.0, act_r=1.0, trail_r=1.0, rr=0.0, force=0, stop_first=True,
        or_band=None, or_lookback=20, min_risk=0.0, ranges=None,
        ratios=None):
    """
    Walk every session once and return the trade table.

    force      +1 or -1 takes that side every day regardless of the EMA, and
               -2 takes the side the EMA rejects. Used for the direction
               nulls; 0 is the rule.
    or_band    (lo, hi): trade only when today's range over the trailing
               median is inside it.
    min_risk   skip days whose stop is closer than this many points.
    rr         fixed target in R; 0 for none.
    """
    o, hi, lo, cl = a["o"], a["h"], a["l"], a["c"]
    mins = a["mins"]
    if ranges is None:
        ranges = opening_ranges(days, or_min)
    if or_band is not None and ratios is None:
        ratios = range_ratio(ranges, or_lookback)
    rows = []

    for day in sorted(days):
        s = days[day]
        rg = ranges.get(day)
        if rg is None:
            continue
        or_hi, or_lo = rg
        or_end = s["open_min"] + or_min
        sig_t = or_end + delay - 5            # label of the signal bar
        t5 = s["t5"]
        k = int(np.searchsorted(t5, sig_t))
        if k >= len(t5) or t5[k] != sig_t:
            continue
        c_sig = float(s["c5"][k])
        e_sig = float(s["ema"][ema][k])
        if not math.isfinite(e_sig):
            continue

        ratio = float("nan")
        if ratios is not None:
            ratio = ratios.get(day, float("nan"))
        if or_band is not None:
            if not math.isfinite(ratio) or not (or_band[0] <= ratio <= or_band[1]):
                continue

        if c_sig > e_sig:
            d = 1
        elif c_sig < e_sig:
            d = -1
        else:
            continue
        if force in (1, -1):
            d = force
        elif force == -2:
            d = -d

        # Fill at the first minute that opens after the signal bar closed.
        want = sig_t + 5
        seg = mins[s["i0"]:s["i1"] + 1]
        off = int(np.searchsorted(seg, want, "left"))
        if off >= seg.size:
            continue
        j = s["i0"] + off
        if mins[j] - want > 5:                # feed gap at the entry
            continue
        entry = float(o[j])

        stop = (or_lo - buf) if d > 0 else (or_hi + buf)
        risk = abs(entry - stop)
        # Price already through the far side: no stop to put there.
        if (d > 0 and stop >= entry) or (d < 0 and stop <= entry):
            continue
        if risk <= 0 or not math.isfinite(risk) or risk < min_risk:
            continue

        target = entry + d * rr * risk if rr else 0.0
        last = s["i1"]
        xi, px, code, amb, armed, mfe, mae = resolve_trail(
            o, hi, lo, j, last, d, entry, stop, risk, act_r, trail_r,
            target, stop_first)
        if code == OPEN:
            px, xi, why = float(cl[last]), last, "eod"
        else:
            why = REASONS[code]
            if why == "trail" and px == entry:
                why = "be"
        gross = d * (float(px) - entry)
        rows.append((
            sym, day.date(), a["idx"][j], a["idx"][xi],
            "long" if d > 0 else "short", entry, stop, risk, float(px), why,
            (gross - cost_pts) / risk, gross / risk,
            1e4 * (gross - cost_pts) / entry, bool(amb), bool(armed),
            mfe / risk, mae / risk, int(mins[xi] - mins[j]) + 1,
            or_hi, or_lo, or_hi - or_lo, ratio, c_sig, e_sig))

    return pd.DataFrame(rows, columns=COLUMNS)
