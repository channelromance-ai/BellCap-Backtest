"""
Three timeframes: daily trend, 4-hour swing levels, 15-minute entries, with
the stop on the nearest minor 15-minute swing.

The hourly swing test (swing_levels.py) found the first effect in this repo
that held on both halves of the data: trading with the trend beat trading
against it by a few points of win rate. It was worth about 0.02-0.05R a
trade and the spread cost 0.1-0.2R, because the stops were tight. This moves
everything up a timeframe to see whether the trend effect grows faster than
the cost share shrinks.

Rules, fixed before any result was looked at:

  Trend     Daily bars (New York days). Up when the last completed day
            closed above its 50-day EMA with the 20 above the 50; down is
            the mirror; anything else is none.

  Level     A 4-hour swing high is a 4-hour bar whose high is the highest of
            the three bars either side; a swing low likewise. Known once
            those three following bars have closed; watched for 30 days.

  cont      Break and retest. A 4-hour close beyond the level by a quarter of
            a 4-hour ATR breaks it. Within 30 days, a 15-minute close back
            across the level then a close on the breakout side again is the
            signal. Trade in the breakout direction.

  rev       Sweep and reject. The level is unbroken on the 4-hour chart. A
            15-minute close beyond it then a close back is the signal. Trade
            away from the level.

  Stop      The most recent CONFIRMED 15-minute swing (two bars either side,
            both closed before the signal) beyond the entry, within the last
            five days, less a tenth of a 15-minute ATR. No such swing, no
            trade. A 15-minute close beyond the level by a 4-hour ATR voids
            the level.

  Trade     Next 15-minute bar's open, 03:00-15:59 New York only. One trade
            per level per setup. Run to stop or target (1:1, 2:1, 3:1),
            capped at ten days.

The headline test is `with`-trend trades whose spread is at most 10% of the
risk, on the later 40% of each market's history. `against` is the control.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import examples.zone_retest as ZR                      # noqa: E402
from bcbt.audit import future_poison                   # noqa: E402

warnings.filterwarnings("ignore")

K4, K15 = 3, 2
LIFE = pd.Timedelta(days=30)
LOOKBACK_15 = 5 * 96                    # five days of 15-minute bars
BREAK_ATR, VOID_ATR, BUF_ATR = 0.25, 1.0, 0.1
HOURS = (3, 15)
COLS = ["sym", "i", "entry_ts", "d", "entry", "stop", "risk", "setup",
        "trend", "cost_share"]


def ohlc(m1, rule):
    b = m1.resample(rule, closed="left", label="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    prev = b["c"].shift(1)
    tr = pd.concat([b["h"] - b["l"], (b["h"] - prev).abs(),
                    (b["l"] - prev).abs()], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return b


def pivot_flags(hi, lo, k):
    n = len(hi)
    is_hi = np.zeros(n, bool)
    is_lo = np.zeros(n, bool)
    for j in range(k, n - k):
        if hi[j - k:j + k + 1].argmax() == k:
            is_hi[j] = True
        if lo[j - k:j + k + 1].argmin() == k:
            is_lo[j] = True
    return is_hi, is_lo


def daily_trend(m1):
    d = ohlc(m1, "1D")
    e20 = d["c"].ewm(span=20, adjust=False, min_periods=50).mean()
    e50 = d["c"].ewm(span=50, adjust=False, min_periods=50).mean()
    tr = np.where((d["c"] > e50) & (e20 > e50), 1,
                  np.where((d["c"] < e50) & (e20 < e50), -1, 0))
    return tr, d.index + pd.Timedelta(days=1)


def _first_cross(cl, ok, a, b, level, d, void):
    """First bar in [a, b) closing back on side d having been across it."""
    if b - a < 2:
        return -1
    seg = cl[a:b]
    hit = ((seg[1:] - level) * d > 0) & ((seg[:-1] - level) * d < 0) \
        & ok[a + 1:b]
    dead = (seg - void) * d < 0
    q = int(hit.argmax()) + 1 if hit.any() else -1
    v = int(dead.argmax()) if dead.any() else -1
    if q < 0 or (0 <= v <= q):
        return -1
    return a + q


def signals(sym, m1, setups=("cont", "rev")):
    h4 = ohlc(m1, "4h")
    hi4, lo4, c4 = (h4[x].to_numpy() for x in ("h", "l", "c"))
    atr4 = h4["atr"].to_numpy()
    idx4 = h4.index
    four = pd.Timedelta(hours=4)
    tr, tr_known = daily_trend(m1)

    m15 = ohlc(m1, "15min")
    ts = m15.index
    cl = m15["c"].to_numpy()
    hi15, lo15 = m15["h"].to_numpy(), m15["l"].to_numpy()
    atr15 = m15["atr"].to_numpy()
    ok = (ts.hour >= HOURS[0]) & (ts.hour <= HOURS[1])
    ph, pl = pivot_flags(hi15, lo15, K15)
    piv_hi, piv_lo = np.flatnonzero(ph), np.flatnonzero(pl)

    o1 = m1["o"].to_numpy(np.float64)
    m_idx = m1.index
    quarter = pd.Timedelta(minutes=15)
    cost = ZR.COST_PTS[sym]
    rows = []

    def minor_stop(q, d, entry):
        # Pivots at j are confirmed by the close of bar j + K15, which must
        # be no later than the signal bar q.
        piv = piv_lo if d > 0 else piv_hi
        top = int(np.searchsorted(piv, q - K15, side="right"))
        for p in piv[:top][::-1]:
            if p < q - LOOKBACK_15:
                break
            lvl = lo15[p] if d > 0 else hi15[p]
            if (entry - lvl) * d > 0:
                a = atr15[q] if np.isfinite(atr15[q]) else 0.0
                return lvl - d * BUF_ATR * a
        return None

    def emit(q, d, setup):
        close_at = ts[q] + quarter
        j = int(m_idx.searchsorted(close_at))
        if j >= len(o1) - 2:
            return
        entry = float(o1[j])
        stop = minor_stop(q, d, entry)
        if stop is None:
            return
        risk = (entry - stop) * d
        if risk <= 0:
            return
        p = int(tr_known.searchsorted(close_at, side="right")) - 1
        t = tr[p] if p >= 0 else 0
        rows.append(dict(sym=sym, i=j, entry_ts=m_idx[j], d=d, entry=entry,
                         stop=stop, risk=risk, setup=setup,
                         trend="none" if t == 0 else
                         ("with" if t == d else "against"),
                         cost_share=cost / risk))

    is_hi, is_lo = pivot_flags(hi4, lo4, K4)
    levels = [(j, 1, hi4[j]) for j in np.flatnonzero(is_hi)] + \
             [(j, -1, lo4[j]) for j in np.flatnonzero(is_lo)]
    for j, kind, level in levels:
        c = j + K4
        a4 = atr4[c]
        if not np.isfinite(a4) or a4 <= 0:
            continue
        known = idx4[c] + four
        a15 = int(ts.searchsorted(known))
        b15 = int(ts.searchsorted(known + LIFE, side="right"))

        if "rev" in setups:
            d = -kind
            q = _first_cross(cl, ok, a15, b15, level, d,
                             level - d * VOID_ATR * a4)
            if q >= 0:
                emit(q, d, "rev")

        if "cont" in setups:
            end = int(idx4.searchsorted(known + LIFE - four, side="right"))
            beyond = (c4[c + 1:end] - level) * kind > BREAK_ATR * a4
            if not beyond.any():
                continue
            b = c + 1 + int(beyond.argmax())
            broke_known = idx4[b] + four
            d = kind
            s15 = int(ts.searchsorted(broke_known))
            e15 = int(ts.searchsorted(broke_known + LIFE, side="right"))
            q = _first_cross(cl, ok, s15, e15, level, d,
                             level - d * VOID_ATR * atr4[b])
            if q >= 0:
                emit(q, d, "cont")
    # Two nearby levels can fire on the same bar. That is one trade, not
    # two, and counting it twice only makes the sample look bigger.
    return (pd.DataFrame(rows, columns=COLS)
            .drop_duplicates(["i", "d", "setup"])
            .sort_values("i").reset_index(drop=True))


def summ(g):
    r = g["r"].to_numpy()
    if len(r) < 2:
        return pd.Series(dict(n=len(r)))
    return pd.Series(dict(n=len(r), win=100 * (r > 0).mean(),
                          avg_r=r.mean(),
                          t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                          total=r.sum()))


def main() -> int:
    pd.set_option("display.width", 210)
    t0 = time.time()
    out = []
    for s in ZR.INDICES + ZR.FX:
        m1 = ZR.load(s)
        sg = signals(s, m1)
        cut = ZR.split_date(m1)
        for rr in (1.0, 2.0, 3.0):
            t = ZR.evaluate(s, m1, sg, rr=rr, max_hold_min=14400)
            if t.empty:
                continue
            t["rr"] = rr
            for col in ("setup", "trend", "cost_share"):
                t[col] = sg[col].values
            t["half"] = np.where(sg["entry_ts"] < cut, "design", "holdout")
            t["cls"] = "FX" if s in ZR.FX else "IDX"
            out.append(t)
        print(f"  {s}: {len(sg)} signals from {m1.index[0].date()}, "
              f"median spread share {sg['cost_share'].median():.1%}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    a = pd.concat(out, ignore_index=True)
    a["cheap"] = a["cost_share"] <= 0.10

    print("\n" + "=" * 96)
    print("coin-toss win rate: 50% at 1:1, 33.3% at 2:1, 25% at 3:1")
    print("=" * 96)
    print("\nHEADLINE: spread <= 10% of risk")
    print(a[a["cheap"]].groupby(["rr", "setup", "trend", "half"])
          .apply(summ).round(3).unstack("half").to_string())
    print("\nALL TRADES")
    print(a.groupby(["rr", "setup", "trend", "half"]).apply(summ).round(3)
          .unstack("half").to_string())
    print("\n  holdout, spread <= 10%, with trend, by asset class")
    print(a[a["cheap"] & (a["half"] == "holdout") & (a["trend"] == "with")]
          .groupby(["rr", "setup", "cls"]).apply(summ).round(3).to_string())
    print("\n  holdout, spread <= 10%, with trend, 2:1, by market")
    print(a[a["cheap"] & (a["half"] == "holdout") & (a["trend"] == "with")
            & (a["rr"] == 2.0)].groupby(["setup", "sym"]).apply(summ)
          .round(3).to_string())
    print("\n  share of trades passing the spread filter:",
          f"{a[a['rr'] == 1.0]['cheap'].mean():.1%}")

    print("\n" + "=" * 96)
    print("LOOK-AHEAD CHECK")
    print("=" * 96)
    spx = ZR.load("SPX500")
    spx = spx[(spx.index >= "2023-06-01") & (spx.index < "2024-09-01")]
    bad = False
    for st in ("cont", "rev"):
        r = future_poison(lambda df, st=st: signals("SPX500", df, (st,))[
            ["entry_ts", "entry", "stop", "trend"]], spx, probes=10)
        bad |= bool(r["leak"])
        print(f"  {st:<5} {'LEAK' if r['leak'] else 'clean'}   probes "
              f"{r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}   "
              f"decisions {r.get('n_clean')}")
    print(f"\ndone {time.time() - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
