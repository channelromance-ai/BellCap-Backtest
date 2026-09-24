"""
Hourly swing highs and lows, traded on a 5-minute reaction, with and against
the hourly trend.

Consolidation zones won at exactly coin-toss rates whichever way they were
traded. This swaps the level for the other one most traders draw -- the last
swing high or low -- and adds the filter most of them use: only trade in
the direction of the higher-timeframe trend.

Rules, fixed before any result was looked at:

  Level     A swing high is an hourly bar whose high is the highest of the
            five bars either side; a swing low likewise. It is only known
            once those five following bars have CLOSED, and is watched for
            14 calendar days from then.

  Trend     Hourly close above its 200-bar EMA with the 50 above the 200 is
            an uptrend; the mirror is a downtrend; anything else is none.
            Read from the last hourly bar that had closed before the signal.

  cont      Break and retest. An hourly close beyond the level by a quarter
            of an hourly ATR breaks it. Within 14 days of that break, a
            5-minute close back across the level followed by a close on the
            breakout side again is the signal. Trade in the breakout
            direction.

  rev       Sweep and reject. The level has not been broken. A 5-minute
            close beyond it followed by a close back on the original side is
            the signal. Trade away from the level.

  Risk      Stop half an hourly ATR beyond the level. A close beyond the
            stop before the signal voids the level. Entry the next bar's
            open, 03:00-15:59 New York only. One trade per level per setup.
            Run to stop or target (1:1 and 2:1), capped at ten days.

Every trade is tagged `with`, `against` or `none` relative to the trend.
The claim under test is that `with` beats its coin-toss win rate on the
later 40% of each market's history; `against` is the control that shows
whether the trend filter is doing anything at all.
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
from bcbt import zones as Z                            # noqa: E402
from bcbt.audit import future_poison                   # noqa: E402

warnings.filterwarnings("ignore")

K = 5
LIFE = pd.Timedelta(days=14)
BREAK_ATR = 0.25
STOP_ATR = 0.5
HOURS = (3, 15)
HOUR = pd.Timedelta(hours=1)
COLS = ["sym", "i", "entry_ts", "d", "entry", "stop", "risk", "setup",
        "trend"]


def swings(h):
    """Every confirmed swing: (bar, +1 high / -1 low, level, known time)."""
    hi, lo = h["h"].to_numpy(), h["l"].to_numpy()
    out = []
    for j in range(K, len(h) - K):
        if hi[j - K:j + K + 1].argmax() == K:
            out.append((j, 1, hi[j]))
        if lo[j - K:j + K + 1].argmin() == K:
            out.append((j, -1, lo[j]))
    return out


def trend_series(h):
    """+1 / -1 / 0 per hourly bar, and the moment each became known."""
    e50 = h["c"].ewm(span=50, adjust=False, min_periods=200).mean()
    e200 = h["c"].ewm(span=200, adjust=False, min_periods=200).mean()
    tr = np.where((h["c"] > e200) & (e50 > e200), 1,
                  np.where((h["c"] < e200) & (e50 < e200), -1, 0))
    return tr, h.index + HOUR


def _first_cross(cl, ok, a, b, level, d, void):
    """
    First 5-minute bar in [a, b) that closes back on side `d` of `level`
    having closed on the other side the bar before, and is inside trading
    hours, provided no close went beyond `void` first. Returns index or -1.
    """
    if b - a < 2:
        return -1
    seg = cl[a:b]
    back = (seg[1:] - level) * d > 0
    was_across = (seg[:-1] - level) * d < 0
    hit = back & was_across & ok[a + 1:b]
    dead = (seg - void) * d < 0            # closed beyond the stop
    q = int(hit.argmax()) + 1 if hit.any() else -1
    v = int(dead.argmax()) if dead.any() else -1
    if q < 0 or (0 <= v <= q):
        return -1
    return a + q


def signals(sym, m1, setups=("cont", "rev")):
    h = Z.hourly(m1)
    hc = h["c"].to_numpy()
    atr = h["atr"].to_numpy()
    hidx = h.index
    tr, tr_known = trend_series(h)

    lo_tf = ZR.bars(m1, 5)
    ts = lo_tf.index
    cl = lo_tf["c"].to_numpy(np.float64)
    ok = (ts.hour >= HOURS[0]) & (ts.hour <= HOURS[1])
    o1 = m1["o"].to_numpy(np.float64)
    m_idx = m1.index
    five = pd.Timedelta(minutes=5)

    rows = []

    def emit(q, d, stop, setup):
        signal_close = ts[q] + five
        j = int(m_idx.searchsorted(signal_close))
        if j >= len(o1) - 2:
            return
        entry = float(o1[j])
        risk = (entry - stop) * d
        if risk <= 0:
            return
        p = int(tr_known.searchsorted(signal_close, side="right")) - 1
        t = tr[p] if p >= 0 else 0
        rows.append(dict(sym=sym, i=j, entry_ts=m_idx[j], d=d, entry=entry,
                         stop=stop, risk=risk, setup=setup,
                         trend="none" if t == 0 else
                         ("with" if t == d else "against")))

    for j, kind, level in swings(h):
        c = j + K                           # bar whose close confirms it
        a_ = atr[c]
        if not np.isfinite(a_) or a_ <= 0:
            continue
        known = hidx[c] + HOUR
        a5 = int(ts.searchsorted(known))
        b5 = int(ts.searchsorted(known + LIFE, side="right"))

        if "rev" in setups:
            # Trade away from the level: short a swing high, long a low.
            d = -kind
            void = level - d * STOP_ATR * a_
            q = _first_cross(cl, ok, a5, b5, level, d, void)
            if q >= 0:
                emit(q, d, void, "rev")

        if "cont" in setups:
            # The break: first hourly close clearly beyond, after it was
            # known and within its life.
            end = int(hidx.searchsorted(known + LIFE - HOUR, side="right"))
            beyond = (hc[c + 1:end] - level) * kind > BREAK_ATR * a_
            if not beyond.any():
                continue
            b = c + 1 + int(beyond.argmax())
            ab = atr[b]
            broke_known = hidx[b] + HOUR
            d = kind                        # the breakout direction
            void = level - d * STOP_ATR * ab
            s5 = int(ts.searchsorted(broke_known))
            e5 = int(ts.searchsorted(broke_known + LIFE, side="right"))
            q = _first_cross(cl, ok, s5, e5, level, d, void)
            if q >= 0:
                emit(q, d, void, "cont")
    # Two nearby levels can fire on the same bar. That is one trade, not
    # two; 7.3% of the first run's trades were such duplicates.
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
        for rr in (1.0, 2.0):
            t = ZR.evaluate(s, m1, sg, rr=rr, max_hold_min=14400)
            t["rr"] = rr
            t["setup"], t["trend"] = sg["setup"].values, sg["trend"].values
            t["half"] = np.where(sg["entry_ts"] < cut, "design", "holdout")
            t["cls"] = "FX" if s in ZR.FX else "IDX"
            out.append(t)
        print(f"  {s}: {sg.groupby(['setup', 'trend']).size().to_dict()}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    a = pd.concat(out, ignore_index=True)

    print("\n" + "=" * 96)
    print("SWING LEVELS, run to stop or target")
    print("coin-toss win rate: 50% at 1:1, 33.3% at 2:1")
    print("=" * 96)
    print(a.groupby(["rr", "setup", "trend", "half"]).apply(summ).round(3)
          .unstack("half").to_string())
    print("\n  holdout, by asset class")
    print(a[a["half"] == "holdout"].groupby(["rr", "setup", "trend", "cls"])
          .apply(summ).round(3).to_string())
    print("\n  holdout, by market, 1:1, with the trend")
    print(a[(a["half"] == "holdout") & (a["rr"] == 1.0)
            & (a["trend"] == "with")].groupby(["setup", "sym"])
          .apply(summ).round(3).to_string())
    print("\n  exit reasons, 1:1:",
          a[a["rr"] == 1.0]["reason"].value_counts().to_dict())

    print("\n" + "=" * 96)
    print("LOOK-AHEAD CHECK")
    print("=" * 96)
    spx = ZR.load("SPX500")
    spx = spx[(spx.index >= "2023-06-01") & (spx.index < "2024-06-01")]
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
