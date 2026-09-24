"""
Zones that have already held: trade a level only after it has bounced price
two or three times.

The first zone test found hourly consolidations, waited for a breakout and
traded the first return. Run to stop or target, it won exactly as often as a
coin toss would: 50% at 1:1, 33% at 2:1, 25% at 3:1. The zones marked where
price had paused, not where it would turn.

The idea here is that a level which has already turned price back more than
once is one other people are watching and defending, so it should hold more
often than a fresh one. That is directly testable: tag every entry with how
many times its zone had already held, and see whether the win rate climbs.

Rules, fixed before any result was looked at:

  * Zones are unchanged from examples/zone_retest.py.
  * A zone is watched for 28 calendar days after its breakout hour closes,
    instead of roughly five, so it has time to be revisited.
  * Entry signal is unchanged: on the 5-minute chart a candle closes inside
    the zone, then a later candle closes back outside on the breakout side.
    Entry is the next bar's open, stop beyond the far side plus 10% of the
    zone's width.
  * A signal only counts as a BOUNCE once price has then travelled at least
    one zone-width away from the zone. A close back inside before that is a
    failed bounce and is not counted. This is what stops a choppy hour at
    the edge being scored as five separate bounces.
  * A close through the far side ends the zone.
  * Every signal is tagged with `prior`, the number of confirmed bounces the
    zone had before it. The claim under test is that prior >= 2 does better
    than prior == 0.

Everything a signal depends on is a completed bar before its entry, which
examples/audit_all.py-style poisoning checks at the bottom of `main`.
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

WATCH = pd.Timedelta(days=28)
COLS = ["sym", "i", "entry_ts", "d", "entry", "stop", "risk", "prior",
        "zone_id"]


def signals(sym, m1, entry_tf=5, zone_bars=4, width=1.5, buffer_frac=0.1,
            away=1.0):
    """Every entry, tagged with how many times its zone had already held."""
    h = Z.hourly(m1)
    z = Z.find_zones(h, bars=zone_bars, max_width_atr=width)
    if z.empty:
        return pd.DataFrame(columns=COLS)
    lo_tf = ZR.bars(m1, entry_tf)
    ts = lo_tf.index
    cl = lo_tf["c"].to_numpy(np.float64)
    hi = lo_tf["h"].to_numpy(np.float64)
    lw = lo_tf["l"].to_numpy(np.float64)
    o1 = m1["o"].to_numpy(np.float64)
    m_idx = m1.index

    rows = []
    for zid, zn in enumerate(z.itertuples(index=False)):
        a = int(ts.searchsorted(zn.broke_known))
        b = int(ts.searchsorted(zn.broke_known + WATCH, side="right"))
        if b - a < 3:
            continue
        up = zn.side == Z.UP
        top, bot, wid = zn.top, zn.bot, zn.width
        # How far price must travel from the zone for a bounce to count.
        far = (top + away * wid) if up else (bot - away * wid)

        bounces = 0
        been_inside = False
        pending = False
        for k in range(a, b - 1):
            c = cl[k]
            if pending and (hi[k] >= far if up else lw[k] <= far):
                bounces += 1
                pending = False
            if (up and c < bot) or (not up and c > top):
                break                       # closed through: level failed
            if bot <= c <= top:
                been_inside = True
                pending = False             # came back before it got away
                continue
            if not been_inside:
                continue
            if (up and c > top) or (not up and c < bot):
                fill_ts = ts[k] + pd.Timedelta(minutes=entry_tf)
                j = int(m_idx.searchsorted(fill_ts))
                been_inside = False
                pending = True
                if j >= len(o1) - 2:
                    continue
                d = 1 if up else -1
                entry = float(o1[j])
                pad = buffer_frac * wid
                stop = (bot - pad) if up else (top + pad)
                risk = abs(entry - stop)
                if risk <= 0 or (d > 0 and stop >= entry) or \
                        (d < 0 and stop <= entry):
                    continue
                rows.append(dict(sym=sym, i=j, entry_ts=m_idx[j], d=d,
                                 entry=entry, stop=stop, risk=risk,
                                 prior=bounces, zone_id=zid))
    return pd.DataFrame(rows, columns=COLS)


def summ(g):
    r = g["r"].to_numpy()
    if len(r) < 2:
        return pd.Series(dict(n=len(r), win=np.nan, avg_r=np.nan, t=np.nan,
                              total=np.nan))
    return pd.Series(dict(n=len(r), win=100 * (r > 0).mean(),
                          avg_r=r.mean(),
                          t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                          total=r.sum()))


def bucket(p):
    return np.where(p >= 3, "3+", p.astype(str))


def main() -> int:
    pd.set_option("display.width", 210)
    t0 = time.time()
    trades = []
    for s in ZR.INDICES + ZR.FX:
        m1 = ZR.load(s)
        sg = signals(s, m1)
        cut = ZR.split_date(m1)
        for rr in (1.0, 2.0):
            t = ZR.evaluate(s, m1, sg, rr=rr, max_hold_min=14400)
            if t.empty:
                continue
            t["rr"] = rr
            t["prior"] = sg["prior"].to_numpy()
            t["half"] = np.where(sg["entry_ts"] < cut, "design", "holdout")
            trades.append(t)
        counts = sg["prior"].clip(upper=3).value_counts().sort_index()
        print(f"  {s}: {len(sg)} signals, by prior bounces "
              f"{counts.to_dict()}  ({time.time() - t0:.0f}s)", flush=True)
    a = pd.concat(trades, ignore_index=True)
    a["bounces_before"] = bucket(a["prior"].to_numpy())
    a["cls"] = np.where(a["sym"].isin(ZR.FX), "FX", "IDX")

    print("\n" + "=" * 90)
    print("WIN RATE BY HOW MANY TIMES THE ZONE HAD ALREADY HELD")
    print("coin-toss win rate: 50% at 1:1, 33% at 2:1")
    print("=" * 90)
    print(a.groupby(["rr", "bounces_before", "half"]).apply(summ)
          .round(3).unstack("half").to_string())

    print("\n" + "=" * 90)
    print("THE PRE-COMMITTED TEST: 2+ earlier bounces vs none, holdout only")
    print("=" * 90)
    ho = a[a["half"] == "holdout"].copy()
    ho["group"] = np.where(ho["prior"] >= 2, "2+ bounces",
                           np.where(ho["prior"] == 0, "fresh", "1 bounce"))
    print(ho.groupby(["rr", "group"]).apply(summ).round(3).to_string())

    print("\n  by asset class, 2+ bounces, both halves")
    print(a[a["prior"] >= 2].groupby(["rr", "cls", "half"]).apply(summ)
          .round(3).unstack("half").to_string())
    print("\n  by market, 2+ bounces, 1:1, both halves")
    print(a[(a["prior"] >= 2) & (a["rr"] == 1.0)]
          .groupby(["sym", "half"]).apply(summ)[["n", "win", "avg_r"]]
          .round(3).unstack("half").to_string())

    print("\n" + "=" * 90)
    print("LOOK-AHEAD CHECK")
    print("=" * 90)
    spx = ZR.load("SPX500")
    spx = spx[(spx.index >= "2023-06-01") & (spx.index < "2024-06-01")]
    r = future_poison(lambda df: signals("SPX500", df)[
        ["entry_ts", "entry", "stop", "prior"]], spx, probes=10)
    print(f"  {'LEAK' if r['leak'] else 'clean'}   probes "
          f"{r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}   "
          f"decisions {r.get('n_clean')}")
    if r["leak"]:
        print(f"  {r['detail']}")
    print(f"\ndone {time.time() - t0:.0f}s")
    return 1 if r["leak"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
