"""
Gold lab: try to develop profitability from the zone work, without fooling
ourselves.

Design, fixed before any result:

  development  Aug 2019 - Dec 2023   all choosing happens here
  validation   Jan 2024 - Sep 2026   looked at ONCE, for every finalist
               (not pristine: earlier versions' 2023-26 results were seen)

  grid (192)   setup    break | bounce | rejection | poi
               session  asia (19-02 NY) | london (02-08) | ny_am (08-12) | all
               stop     edge | edge + 0.5 x ATR15 buffer
               target   1.5R | 2R | 3R
               htf      H4 50-EMA must agree | no filter

  finalist     development PF >= 1.2 on >= 100 trades
  luck level   the same grid with every signal's direction randomised; the
               number of random "finalists" is what luck alone produces

Signals (M5 closes, entry next M5 open; zones from gold_zones.find_zones;
POIs from gold_poi.pois; a zone stops trading after its first stop-out):
  break      close from inside the zone to outside it
  bounce     after a close outside, a bar dips back to the edge and closes
             outside again
  rejection  close back inside after a close beyond the far side; stop at
             the extreme of that excursion
  poi        at a zone left by an impulse, a close inside then a close back
             out in the impulse direction; stop at the far edge
Direction for break/bounce/rejection is the direction of the move (a break
up is long); for poi it is the impulse. One position at a time, 3 a day, out
by the 17:00 New York roll, stop first on ties, each bar's spread plus $8.22
a lot commission.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_poi as P                          # noqa: E402
import examples.gold_zones as G                        # noqa: E402

DEV_END = np.datetime64("2024-01-01")
SESSIONS = {"asia": set(range(2, 9)), "london": set(range(9, 15)),
            "ny_am": set(range(15, 19)), "all": set(range(24))}


def cached(name, fn):
    path = f"data/gold_lab_{name}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path)
    return df


def zone_signals(m5, zones):
    """break / bounce / rejection events, any time, either direction."""
    t = m5["server"].to_numpy()
    h, l, c = (m5[x].tolist() for x in ("high", "low", "close"))
    rows = []
    for zid, z in enumerate(zones.itertuples(index=False)):
        a = int(np.searchsorted(t, z.known))
        b = int(np.searchsorted(t, z.known + np.timedelta64(24, "h")))
        top, bot = z.top, z.bot
        been_above = been_below = False
        lo_since, hi_since = 1e18, -1e18
        for k in range(max(a, 1), min(b, len(c) - 1)):
            ck, cp = c[k], c[k - 1]
            if cp >= bot > ck:
                lo_since = l[k]
            else:
                lo_since = min(lo_since, l[k])
            if cp <= top < ck:
                hi_since = h[k]
            else:
                hi_since = max(hi_since, h[k])
            if cp <= top < ck:
                rows.append((k + 1, 1, "break", bot, zid))
            elif cp >= bot > ck:
                rows.append((k + 1, -1, "break", top, zid))
            elif been_above and cp > top and l[k] <= top < ck:
                rows.append((k + 1, 1, "bounce", bot, zid))
            elif been_below and cp < bot and h[k] >= bot > ck:
                rows.append((k + 1, -1, "bounce", top, zid))
            elif cp < bot <= ck:
                rows.append((k + 1, 1, "rejection", lo_since, zid))
            elif cp > top >= ck:
                rows.append((k + 1, -1, "rejection", hi_since, zid))
            been_above |= ck > top
            been_below |= ck < bot
    return pd.DataFrame(rows, columns=["i", "d", "kind", "stop", "zone"])


def poi_signals(m5, m15, poi, zone_offset):
    t = m5["server"].to_numpy()
    c = m5["close"].tolist()
    t15, c15 = m15["server"].to_numpy(), m15["close"].to_numpy()
    rows = []
    for zid, p in enumerate(poi.itertuples(index=False)):
        a = int(np.searchsorted(t, p.known))
        end_t = p.known + np.timedelta64(P.LIFE_H, "h")
        a15, b15 = np.searchsorted(t15, p.known), np.searchsorted(t15, end_t)
        far = np.flatnonzero(c15[a15:b15] < p.bot) if p.d > 0 else \
            np.flatnonzero(c15[a15:b15] > p.top)
        if len(far):
            end_t = min(end_t, t15[a15 + far[0]] + np.timedelta64(15, "m"))
        b = int(np.searchsorted(t, end_t))
        inside = False
        for k in range(a, min(b, len(c) - 1)):
            ck = c[k]
            if p.bot <= ck <= p.top:
                inside = True
                continue
            if inside and ((p.d > 0 and ck > p.top) or (p.d < 0 and ck < p.bot)):
                rows.append((k + 1, p.d, "poi", p.bot if p.d > 0 else p.top,
                             zone_offset + zid))
            inside = False
    return pd.DataFrame(rows, columns=["i", "d", "kind", "stop", "zone"])


@njit(cache=True)
def simulate(si, sd, sstop, szone, o, h, l, c, spread, day, rr, commission,
             n_zones):
    n = len(o)
    out_i = np.full(len(si), -1)
    out_r = np.zeros(len(si))
    out_usd = np.zeros(len(si))
    dead = np.full(n_zones, -1)
    busy = -1
    cur_day = -1
    count = 0
    m = 0
    for q in range(len(si)):
        j = si[q]
        d = sd[q]
        stop = sstop[q]
        z = szone[q]
        if j >= n - 1 or j <= busy:
            continue
        if dead[z] >= 0 and j > dead[z]:
            continue
        if day[j] != cur_day:
            cur_day = day[j]
            count = 0
        if count >= 3:
            continue
        entry = o[j]
        risk = (entry - stop) * d
        cost = spread[j] + commission
        if risk <= 2 * cost:
            continue
        tgt = entry + d * rr * risk
        k = j
        px = np.nan
        stopped = False
        while k < n and day[k] == day[j]:
            if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
                px = min(stop, o[k]) if d > 0 else max(stop, o[k])
                stopped = True
                break
            if (d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt):
                px = max(tgt, o[k]) if d > 0 else min(tgt, o[k])
                break
            k += 1
        if np.isnan(px):
            k = min(k, n) - 1
            px = c[k]
        if stopped:
            dead[z] = k
        pnl = d * (px - entry) - cost
        out_i[m] = j
        out_r[m] = pnl / risk
        out_usd[m] = pnl * 3.0           # 0.03 lots x 100 oz
        m += 1
        busy = k
        count += 1
    return out_i[:m], out_r[:m], out_usd[:m]


def pf(x):
    w, lo = x[x > 0].sum(), -x[x <= 0].sum()
    return w / lo if lo > 0 else np.inf


def run_grid(sig, m5, arr, n_zones, label):
    rows = []
    for kind in ("break", "bounce", "rejection", "poi"):
        base = sig[sig["kind"] == kind]
        for sess, hours in SESSIONS.items():
            s1 = base[base["hour"].isin(hours)]
            for stopmode in ("edge", "buffer"):
                stop = s1["stop"] - s1["d"] * (0.5 * s1["atr15"]
                                               if stopmode == "buffer" else 0)
                for htf in ("agree", "none"):
                    keep = (s1["bias"] == s1["d"]) if htf == "agree" else \
                        np.ones(len(s1), bool)
                    ss = s1[keep]
                    st = stop[keep].to_numpy()
                    for rr in (1.5, 2.0, 3.0):
                        ti, r, usd = simulate(
                            ss["i"].to_numpy(), ss["d"].to_numpy(), st,
                            ss["zone"].to_numpy(), *arr, rr, G.COMMISSION,
                            n_zones)
                        when = arr_time[ti]
                        dev, val = r[when < DEV_END], r[when >= DEV_END]
                        rows.append(dict(
                            run=label, setup=kind, session=sess, stop=stopmode,
                            htf=htf, rr=rr, dev_n=len(dev), dev_pf=pf(dev),
                            dev_avg_r=dev.mean() if len(dev) else np.nan,
                            val_n=len(val), val_pf=pf(val),
                            val_avg_r=val.mean() if len(val) else np.nan,
                            usd_pf_all=pf(usd)))
    return pd.DataFrame(rows)


def main() -> int:
    global arr_time
    pd.set_option("display.width", 240)
    t0 = time.time()
    m5, m15, h4 = G.load()
    zones = cached("zones", lambda: G.find_zones(m15))
    poi = cached("poi", lambda: P.pois(m15))
    print(f"zones {len(zones)}, POIs {len(poi)}  [{time.time() - t0:.0f}s]")
    sig = cached("signals", lambda: pd.concat([
        zone_signals(m5, zones), poi_signals(m5, m15, poi, len(zones))],
        ignore_index=True))
    print(f"signals {len(sig)}: {sig['kind'].value_counts().to_dict()}  "
          f"[{time.time() - t0:.0f}s]")

    t = m5["server"].to_numpy()
    arr_time = t
    sig = sig.sort_values("i").reset_index(drop=True)
    sig = sig[sig["i"] < len(m5) - 1]
    close_t = t[sig["i"].to_numpy() - 1] + np.timedelta64(5, "m")
    sig["bias"] = G.h4_bias(h4, close_t)
    sig["hour"] = m5["server"].dt.hour.to_numpy()[sig["i"].to_numpy()]
    atr15 = G.atr(m15).to_numpy()
    t15_close = m15["server"].to_numpy() + np.timedelta64(15, "m")
    pos = np.searchsorted(t15_close, close_t, side="right") - 1
    sig["atr15"] = atr15[np.clip(pos, 0, None)]
    sig = sig.dropna(subset=["atr15"])

    o, h, l, c = (m5[x].to_numpy(np.float64) for x in ("open", "high", "low",
                                                      "close"))
    spread = (m5["spread"] * G.POINT).to_numpy(np.float64)
    day = m5["server"].dt.normalize().to_numpy().astype("int64")
    arr = (o, h, l, c, spread, day)
    n_zones = int(sig["zone"].max()) + 1

    real = run_grid(sig, m5, arr, n_zones, "real")
    rng = np.random.default_rng(42)
    shuf = sig.copy()
    flip = rng.random(len(shuf)) < 0.5
    # Random direction: mirror the stop to the other side at the same
    # distance, so risk sizes are unchanged and only the direction is luck.
    entry_px = o[shuf["i"].to_numpy()]
    shuf.loc[flip, "stop"] = 2 * entry_px[flip] - shuf.loc[flip, "stop"]
    shuf.loc[flip, "d"] = -shuf.loc[flip, "d"]
    luck = run_grid(shuf, m5, arr, n_zones, "random")
    print(f"grid done  [{time.time() - t0:.0f}s]")

    for name, df in (("REAL SIGNALS", real), ("RANDOM DIRECTIONS", luck)):
        fin = df[(df["dev_pf"] >= 1.2) & (df["dev_n"] >= 100)]
        print(f"\n{name}: {len(fin)} of {len(df)} configurations are "
              f"development finalists (PF >= 1.2 on >= 100 trades)")
    fin = real[(real["dev_pf"] >= 1.2) & (real["dev_n"] >= 100)] \
        .sort_values("dev_pf", ascending=False)
    print("\nEVERY REAL FINALIST, with its one look at 2024-26:")
    print(fin.drop(columns="run").round(3).to_string(index=False)
          if len(fin) else "  none")
    if len(fin):
        print(f"\n  finalists still >= 1.25 on 2024-26: "
              f"{(fin['val_pf'] >= 1.25).sum()} of {len(fin)};  >= 1.0: "
              f"{(fin['val_pf'] >= 1.0).sum()}")
    print("\nfor context, the 10 best development results with ANY trade "
          "count:")
    print(real.sort_values("dev_pf", ascending=False).head(10)
          .drop(columns="run").round(3).to_string(index=False))
    print("\nhow development PF relates to validation PF across all 192: "
          f"correlation {real['dev_pf'].replace(np.inf, np.nan).corr(real['val_pf'].replace(np.inf, np.nan)):+.2f}")
    real.to_csv("data/gold_lab_grid.csv", index=False)
    luck.to_csv("data/gold_lab_grid_random.csv", index=False)
    print(f"done [{time.time() - t0:.0f}s]")
    return 0


arr_time = None

if __name__ == "__main__":
    raise SystemExit(main())
