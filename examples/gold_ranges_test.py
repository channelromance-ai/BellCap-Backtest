"""
The user's gold ranges, detected live (gold_ranges.py, calibrated to their
four marked examples), tested on The5ers' 15-minute gold history 2005-2026.

Fixed before running:

  Reactions (signal on an M15 close, entry at the next M15 open)
    breakout        the first close outside the live range: go with it
    retest-and-go   after a breakout, within 8 hours: a close back inside,
                    then a close beyond the same edge again: go with it
    failed-break    after a breakout, within 2 hours: a close back inside:
                    trade back toward the other side
  Filter            none, or the H4 50-EMA must agree with the trade
  Scoring A         no stops/targets/caps: the move after cost at 1 hour,
                    4 hours, and to the 17:00 New York roll; trade-weighted,
                    t-statistic over daily P&L sums
  Scoring B         the user's own style: stop at the range's far edge
                    (a failed-break's stop beyond its breakout extreme),
                    target 2R, out by the roll, stop first on ties
  Periods           develop 2005-2015, ONE look at 2016-2026
  Luck              the same timestamps with random directions
  Pass              development mean > 0 with t >= 3; then 2016-26 > 0

Costs: each bar's own spread, but never less than $0.30 (older data on
this server quotes implausibly tight), plus $8.22 a lot commission.
The four calibration examples are from September 2026; only the shape of a
range was tuned on them, never any outcome.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_ranges as R                       # noqa: E402
import examples.gold_zones as G                        # noqa: E402

DEV_END = np.datetime64("2016-01-01")
SPREAD_FLOOR = 0.30
HORIZONS = {"1h": 4, "4h": 16, "to_roll": None}


def load():
    a = pd.read_parquet("data/xau_M15.parquet")
    b = pd.read_parquet("data/xau_M15_recent.parquet")
    m = pd.concat([a, b]).drop_duplicates("time").sort_values("time")
    m = m[m["server"] >= "2005-02-01"].reset_index(drop=True)
    h4 = pd.read_parquet("data/xau_H4.parquet")
    return m, h4


def events(m, bo):
    c = m["close"].to_numpy(np.float64)
    h, l = m["high"].to_numpy(np.float64), m["low"].to_numpy(np.float64)
    n = len(c)
    rows = []
    for r in bo.itertuples(index=False):
        k, d, top, bot = r.k, r.d, r.top, r.bot
        # A FALSE breakout: a close back inside the range within 8 bars
        # (2 hours). Known only afterwards -- used to explain losses, never
        # as an entry filter.
        false_bo = bool(any(bot <= c[j] <= top
                            for j in range(k + 1, min(k + 9, n))))
        rows.append((k, d, "breakout", bot if d > 0 else top, false_bo))
        inside = False
        failed_logged = False
        ext = h[k] if d > 0 else l[k]
        for j in range(k + 1, min(k + 33, n)):
            ext = max(ext, h[j]) if d > 0 else min(ext, l[j])
            if (d > 0 and c[j] < bot) or (d < 0 and c[j] > top):
                break                                   # broke the other way
            if bot <= c[j] <= top:
                inside = True
                if j <= k + 8 and not failed_logged:
                    # failed-break: the FIRST close back inside, within 2h,
                    # traded back toward the other side.
                    rows.append((j, -d, "failed-break",
                                 ext + (0.1 * (top - bot)) * d, False))
                    failed_logged = True
                continue
            if inside and ((d > 0 and c[j] > top) or (d < 0 and c[j] < bot)):
                rows.append((j, d, "retest-and-go", bot if d > 0 else top,
                             False))
                break
    ev = pd.DataFrame(rows, columns=["i", "d", "kind", "stop", "false_bo"])
    return ev.sort_values("i", kind="stable").reset_index(drop=True)


def forward(m):
    o = m["open"].to_numpy(np.float64)
    c = m["close"].to_numpy(np.float64)
    sp = np.maximum(m["spread"].to_numpy(np.float64) * G.POINT, SPREAD_FLOOR)
    day = m["server"].dt.normalize().to_numpy()
    n = len(c)
    last = pd.Series(np.arange(n)).groupby(day).transform("max").to_numpy()
    ei = np.clip(np.arange(n) + 1, 0, n - 1)
    out = {}
    for name, hb in HORIZONS.items():
        xi = last if hb is None else np.minimum(ei + hb - 1, last)
        ok = (np.arange(n) + 1 < n) & (last[ei] == last) & (xi >= ei)
        gross = 1e4 * (c[xi] / o[ei] - 1)
        cost = 1e4 * (sp[ei] + G.COMMISSION) / o[ei]
        out[name] = (np.where(ok, gross - cost, np.nan),
                     np.where(ok, -gross - cost, np.nan))
    return out, sp


def day_stats(v, dayidx):
    s = pd.Series(v).groupby(dayidx).sum()
    if len(s) < 3:
        return np.nan, np.nan
    return np.nansum(v) / np.isfinite(v).sum(), s.mean() / (s.std(ddof=1)
                                                          / np.sqrt(len(s)))


def style_b(m, ev, sp, rr=2.0):
    """Stop at the given level, 2R target, out by the roll; R after cost."""
    o, h, l, c = (m[x].to_numpy(np.float64) for x in ("open", "high", "low",
                                                      "close"))
    day = m["server"].dt.normalize().to_numpy()
    n = len(c)
    rs = np.full(len(ev), np.nan)
    for q, e in enumerate(ev.itertuples(index=False)):
        j = e.i + 1
        if j >= n:
            continue
        entry = o[j]
        risk = (entry - e.stop) * e.d
        cost = sp[j] + G.COMMISSION
        if risk <= 2 * cost:
            continue
        tgt = entry + e.d * rr * risk
        k, px = j, np.nan
        while k < n and day[k] == day[j]:
            if (e.d > 0 and l[k] <= e.stop) or (e.d < 0 and h[k] >= e.stop):
                px = min(e.stop, o[k]) if e.d > 0 else max(e.stop, o[k])
                break
            if (e.d > 0 and h[k] >= tgt) or (e.d < 0 and l[k] <= tgt):
                px = max(tgt, o[k]) if e.d > 0 else min(tgt, o[k])
                break
            k += 1
        if np.isnan(px):
            px = c[min(k, n) - 1]
        rs[q] = (e.d * (px - entry) - cost) / risk
    return rs


def pf(x):
    x = x[np.isfinite(x)]
    w, lo = x[x > 0].sum(), -x[x <= 0].sum()
    return w / lo if lo > 0 else np.inf


def main() -> int:
    pd.set_option("display.width", 230)
    t0 = time.time()
    m, h4 = load()
    print(f"gold M15: {len(m):,} bars {m['server'].iloc[0]:%Y-%m-%d} -> "
          f"{m['server'].iloc[-1]:%Y-%m-%d}")
    st, tp, bt = R.live_ranges_fast(m)
    bo = R.breakouts(m, st, tp, bt)
    ev = events(m, bo)
    print(f"ranges with breakouts: {len(bo):,}; events: "
          f"{ev['kind'].value_counts().to_dict()}  [{time.time() - t0:.0f}s]")
    t = m["server"].to_numpy()
    close_t = t[ev["i"].to_numpy()] + np.timedelta64(15, "m")
    ev["bias"] = G.h4_bias(h4, close_t)
    ev["dev"] = t[ev["i"].to_numpy()] < DEV_END
    ev["year"] = m["server"].dt.year.to_numpy()[ev["i"].to_numpy()]
    dayidx = pd.factorize(m["server"].dt.normalize())[0][ev["i"].to_numpy()]
    fwd, sp = forward(m)
    ev["rB"] = style_b(m, ev, sp)
    rng = np.random.default_rng(3)

    rows, luck = [], []
    for kind in ("breakout", "retest-and-go", "failed-break"):
        for filt in ("none", "H4 agrees"):
            sel = (ev["kind"] == kind).to_numpy().copy()
            if filt != "none":
                sel &= (ev["bias"] == ev["d"]).to_numpy()
            idx = np.flatnonzero(sel)
            d = ev["d"].to_numpy()[idx]
            ii = ev["i"].to_numpy()[idx]
            dev = ev["dev"].to_numpy()[idx]
            for hz, (L, S) in fwd.items():
                for _label, dd, store in (("real", d, rows),
                                         ("random", np.where(rng.random(len(d)) < 0.5, 1, -1), luck)):
                    v = np.where(dd > 0, L[ii], S[ii])
                    res = dict(kind=kind, filter=filt, horizon=hz)
                    for part, pm in (("dev", dev), ("val", ~dev)):
                        mean, tt = day_stats(v[pm], dayidx[idx][pm])
                        res[f"{part}_n"] = int(pm.sum())
                        res[f"{part}_bp"] = mean
                        res[f"{part}_t"] = tt
                    store.append(res)
            rb = ev["rB"].to_numpy()[idx]
            rows.append(dict(kind=kind, filter=filt, horizon="stop/2R (R)",
                             dev_n=int(np.isfinite(rb[dev]).sum()),
                             dev_bp=np.nanmean(rb[dev]), dev_t=pf(rb[dev]),
                             val_n=int(np.isfinite(rb[~dev]).sum()),
                             val_bp=np.nanmean(rb[~dev]), val_t=pf(rb[~dev])))
    real, rnd = pd.DataFrame(rows), pd.DataFrame(luck)
    print("\n" + "=" * 110)
    print("RESULTS  (bp = average per trade after cost; t over daily P&L. For "
          "the stop/2R rows: avg R, and the t columns show PROFIT FACTOR)")
    print("=" * 110)
    print(real.round(2).to_string(index=False))
    a = real[real["horizon"] != "stop/2R (R)"]
    passed = a[(a["dev_bp"] > 0) & (a["dev_t"] >= 3)]
    print(f"\npassed development (t >= 3): real {len(passed)} of {len(a)};  "
          f"random directions {((rnd['dev_bp'] > 0) & (rnd['dev_t'] >= 3)).sum()} of {len(rnd)}")
    print(passed.round(2).to_string(index=False) if len(passed) else "  none")
    print("\n" + "=" * 110)
    print("FALSE BREAKOUTS: how much of the breakout losses do they explain?")
    print("  (false = closed back inside the range within 2 hours; known only "
          "after the fact)")
    print("=" * 110)
    bo_ev = ev[ev["kind"] == "breakout"].copy()
    L, S = fwd["to_roll"]
    bi = bo_ev["i"].to_numpy()
    bo_ev["to_roll_bp"] = np.where(bo_ev["d"].to_numpy() > 0, L[bi], S[bi])
    for label, col in (("your style, stop at far edge / 2R (R)", "rB"),
                       ("no stops, held to the roll (bp)", "to_roll_bp")):
        x = bo_ev[np.isfinite(bo_ev[col])]
        lose = x[col] <= 0
        print(f"\n  {label}: {len(x):,} breakout trades")
        print(f"    false breakouts: {x['false_bo'].mean():.0%} of all "
              f"breakouts")
        print(f"    of the LOSING trades, false breakouts were "
              f"{x.loc[lose, 'false_bo'].mean():.0%}")
        for name, g in (("false breakouts", x[x["false_bo"]]),
                        ("breakouts that held", x[~x["false_bo"]])):
            print(f"    {name:<20} {len(g):>6,} trades  win "
                  f"{100 * (g[col] > 0).mean():5.1f}%  average "
                  f"{g[col].mean():+7.2f}  profit factor "
                  f"{pf(g[col].to_numpy()):.2f}")

    print("\nyear by year, breakout (no filter), bp per trade to the roll:")
    sel = (ev["kind"] == "breakout").to_numpy()
    L, S = fwd["to_roll"]
    v = np.where(ev["d"].to_numpy() > 0, L[ev["i"].to_numpy()],
                 S[ev["i"].to_numpy()])
    yr = pd.Series(v[sel]).groupby(ev["year"].to_numpy()[sel]).agg(
        ["count", "mean"])
    print(yr.round(1).T.to_string())
    ev.to_csv("data/gold_ranges_events.csv", index=False)
    print(f"done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
