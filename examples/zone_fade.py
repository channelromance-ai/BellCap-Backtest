"""
The zone trade turned around: bet that the breakout fails.

Everything in zone_retest.py and zone_respected.py is a continuation trade --
price breaks out of an hourly range, comes back, and the range is expected
to hold as a floor or ceiling. Run to stop or target, that won at exactly
coin-toss rates. This file asks the opposite question: does price that
breaks out of a range tend to fall back into it?

Three versions, fixed before any result was looked at:

  flip     The continuation signals from zone_retest.py, traded the other
           way with the same distances. Included mainly to show that
           reversing a coin toss is still a coin toss.

  failed   After the hourly breakout closes, wait for the first 5-minute
           close back INSIDE the zone -- the breakout has visibly failed.
           Enter against the breakout at the next bar's open. Stop beyond
           the furthest point the breakout reached, plus 10% of the zone
           width.

  fade     Bet against the breakout as soon as the breakout hour closes.
           Stop beyond that hour's extreme plus 10% of the zone width.

`failed` and `fade` are each scored with three targets: 1:1, the middle of
the zone, and the far side of the zone. Trades run until stop or target,
capped at ten days. Judged on the later 40% of each market's history.
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
from bcbt.fills import OPEN, REASONS, resolve          # noqa: E402

warnings.filterwarnings("ignore")

COLS = ["sym", "i", "entry_ts", "d", "entry", "stop", "risk", "mid", "far"]
MAX_HOLD = 14400


def _row(sym, m1, j, d, stop, zn):
    entry = float(m1["o"].iat[j])
    risk = (stop - entry) * -d
    if risk <= 0:
        return None
    mid = (zn.top + zn.bot) / 2
    far = zn.bot if d < 0 else zn.top
    return dict(sym=sym, i=j, entry_ts=m1.index[j], d=d, entry=entry,
                stop=stop, risk=risk, mid=mid, far=far)


def signals(sym, m1, kind, buffer_frac=0.1):
    """Entries for `failed` or `fade`. Both trade against the breakout."""
    z = Z.find_zones(Z.hourly(m1), bars=4, max_width_atr=1.5)
    if z.empty:
        return pd.DataFrame(columns=COLS)
    idx = m1.index
    h1 = m1["h"].to_numpy(np.float64)
    l1 = m1["l"].to_numpy(np.float64)
    lo_tf = ZR.bars(m1, 5)
    ts, cl = lo_tf.index, lo_tf["c"].to_numpy(np.float64)

    rows = []
    for zn in z.itertuples(index=False):
        up = zn.side == Z.UP
        d = -1 if up else 1                 # against the breakout
        pad = buffer_frac * zn.width
        # First minute of the breakout hour, and the first minute after it.
        b0 = int(idx.searchsorted(zn.broke))
        b1 = int(idx.searchsorted(zn.broke_known))
        if b1 >= len(idx) - 2 or b1 <= b0:
            continue

        if kind == "fade":
            ext = h1[b0:b1].max() if up else l1[b0:b1].min()
            stop = ext + pad if up else ext - pad
            r = _row(sym, m1, b1, d, stop, zn)
            if r:
                rows.append(r)
            continue

        # failed: first 5-minute close back inside, within the zone's life.
        a = int(ts.searchsorted(zn.broke_known))
        e = int(ts.searchsorted(zn.expires, side="right"))
        for k in range(a, e - 1):
            if zn.bot <= cl[k] <= zn.top:
                fill_ts = ts[k] + pd.Timedelta(minutes=5)
                j = int(idx.searchsorted(fill_ts))
                if j >= len(idx) - 2:
                    break
                # Furthest the breakout got, up to the signal bar's close.
                ext = h1[b0:j].max() if up else l1[b0:j].min()
                stop = ext + pad if up else ext - pad
                r = _row(sym, m1, j, d, stop, zn)
                if r:
                    rows.append(r)
                break
    return pd.DataFrame(rows, columns=COLS)


def flip_signals(sym, m1):
    sg = ZR.signals(sym, m1, entry_tf=5)
    if sg.empty:
        return pd.DataFrame(columns=COLS)
    sg = sg.copy()
    sg["d"] = -sg["d"]
    sg["stop"] = sg["entry"] - sg["d"] * sg["risk"]
    sg["mid"] = np.nan
    sg["far"] = np.nan
    return sg[COLS]


def evaluate(sym, m1, sg, target):
    """`target` is 'rr1', 'mid' or 'far'."""
    if sg.empty:
        return pd.DataFrame()
    o1 = m1["o"].to_numpy(np.float64)
    h1 = m1["h"].to_numpy(np.float64)
    l1 = m1["l"].to_numpy(np.float64)
    c1 = m1["c"].to_numpy(np.float64)
    n, cost = len(o1), ZR.COST_PTS[sym]
    rows = []
    for s in sg.itertuples(index=False):
        tp = s.entry + s.d * s.risk if target == "rr1" else getattr(s, target)
        # A target already behind the entry is not a trade.
        if (tp - s.entry) * s.d <= 0:
            continue
        last = min(s.i + MAX_HOLD, n - 1)
        _, px, code, _, _, _, _ = resolve(o1, h1, l1, s.i, last, s.d,
                                           s.entry, s.stop, s.risk, tp, 0.0)
        if code == OPEN:
            px, why = float(c1[last]), "time"
        else:
            why = REASONS[code]
        rows.append(dict(sym=sym, entry_ts=s.entry_ts, reason=why,
                         reward_risk=(tp - s.entry) * s.d / s.risk,
                         r=(s.d * (float(px) - s.entry) - cost) / s.risk))
    return pd.DataFrame(rows)


def summ(g):
    r = g["r"].to_numpy()
    rr = g["reward_risk"].to_numpy()
    # The win rate a coin toss would give for these reward:risk ratios.
    coin = 100 * np.mean(1 / (1 + rr))
    return pd.Series(dict(
        n=len(r), win=100 * (r > 0).mean(), coin_win=coin,
        avg_r=r.mean(), t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
        total=r.sum()))


def main() -> int:
    pd.set_option("display.width", 210)
    t0 = time.time()
    out = []
    for s in ZR.INDICES + ZR.FX:
        m1 = ZR.load(s)
        cut = ZR.split_date(m1)
        sets = {"flip": flip_signals(s, m1),
                "failed": signals(s, m1, "failed"),
                "fade": signals(s, m1, "fade")}
        for kind, sg in sets.items():
            for tgt in (("rr1",) if kind == "flip" else ("rr1", "mid", "far")):
                t = evaluate(s, m1, sg, tgt)
                if t.empty:
                    continue
                t["kind"], t["target"] = kind, tgt
                t["half"] = np.where(t["entry_ts"] < cut, "design",
                                     "holdout")
                t["cls"] = "FX" if s in ZR.FX else "IDX"
                out.append(t)
        print(f"  {s}: flip {len(sets['flip'])}  failed "
              f"{len(sets['failed'])}  fade {len(sets['fade'])}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    a = pd.concat(out, ignore_index=True)

    print("\n" + "=" * 96)
    print("AGAINST THE BREAKOUT, run to stop or target")
    print("coin_win = what a coin toss would win at these reward:risk ratios")
    print("=" * 96)
    print(a.groupby(["kind", "target", "half"]).apply(summ).round(3)
          .unstack("half").to_string())
    print("\n  by asset class, holdout")
    print(a[a["half"] == "holdout"].groupby(["kind", "target", "cls"])
          .apply(summ).round(3).to_string())
    print("\n  by market, holdout, failed breakout to zone middle")
    print(a[(a["half"] == "holdout") & (a["kind"] == "failed")
            & (a["target"] == "mid")].groupby("sym").apply(summ)
          .round(3).to_string())

    print("\n" + "=" * 96)
    print("LOOK-AHEAD CHECK")
    print("=" * 96)
    spx = ZR.load("SPX500")
    spx = spx[(spx.index >= "2023-06-01") & (spx.index < "2024-06-01")]
    bad = False
    for kind in ("failed", "fade"):
        r = future_poison(lambda df, k=kind: signals("SPX500", df, k)[
            ["entry_ts", "entry", "stop"]], spx, probes=10)
        bad |= bool(r["leak"])
        print(f"  {kind:<7} {'LEAK' if r['leak'] else 'clean'}   probes "
              f"{r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}   "
              f"decisions {r.get('n_clean')}")
    print(f"\ndone {time.time() - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
