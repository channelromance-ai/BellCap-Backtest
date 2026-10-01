"""
Gold points of interest: base, impulse, return, reaction.

The user's clarified idea of a valid zone: a consolidation that price later
left DECISIVELY in one direction. That zone becomes a point of interest, and
when price comes back to it the trade is the reaction in the direction of
the original breakout -- if the 4-hour chart agrees. Fixed before running:

  Base      8 M15 bars, range <= 2.0 x ATR14(M15) (as gold_zones.py).
  Impulse   The first M15 close outside the base sets the direction. Within
            12 M15 bars (3 hours) of that close, price must reach at least
            2.0 x the base's height beyond its edge. No impulse, no POI.
  Bias      H4 close vs its 50-EMA must agree with the impulse direction,
            read at each M5 signal.
  Entry     Asia only (entry bar 19:00-01:59 New York). After the impulse, an
            M5 close back INSIDE the zone, then a later M5 close back out on
            the impulse side. Entry at the next M5 open.
  Stop      The zone's far edge. Target 2R. Exit by the 17:00 New York roll.
  Life      72 hours from the impulse. A stop-out kills the zone, and so
            does an M15 close through the far side. After a winner the zone
            stays live and may trade again.
  Book      One position at a time, 3 a day, each bar's spread plus $8.22 a
            lot commission, stop first on ties.

Robustness only, never used to choose: impulse multiples 1.5 and 3.0.
The scan works on plain Python lists rather than numpy scalars: the v2
zone scan spent hours paying numpy's per-element overhead.
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

import examples.gold_zones as G                        # noqa: E402

ASIA = set(range(2, 9))          # server hours = 19:00-01:59 New York
RR = 2.0
LIFE_H = 72


def pois(m15, impulse_mult=2.0):
    """Bases that were left with an impulse: (known_time, top, bot, d)."""
    zones = G.find_zones(m15)
    t = m15["server"].to_numpy()
    hi, lo, cl = (m15[x].to_numpy() for x in ("high", "low", "close"))
    out = []
    for z in zones.itertuples(index=False):
        a = np.searchsorted(t, z.known)       # first bar after the base
        height = z.top - z.bot
        seg_c = cl[a:a + 400]
        out_up = np.flatnonzero(seg_c > z.top)
        out_dn = np.flatnonzero(seg_c < z.bot)
        first = min(out_up[0] if len(out_up) else 10**9,
                    out_dn[0] if len(out_dn) else 10**9)
        if first >= 10**9:
            continue
        d = 1 if (len(out_up) and out_up[0] == first) else -1
        b = a + first
        win = slice(b, b + 12)
        if d > 0:
            reached = hi[win].max() - z.top
        else:
            reached = z.bot - lo[win].min()
        if reached < impulse_mult * height:
            continue
        # The impulse is only known once its 12-bar window has closed, or
        # earlier if the threshold was met earlier: use the bar it was met.
        need = (z.top + impulse_mult * height) if d > 0 else \
            (z.bot - impulse_mult * height)
        hit = np.flatnonzero(hi[win] >= need) if d > 0 else \
            np.flatnonzero(lo[win] <= need)
        known = t[b + hit[0]] + np.timedelta64(15, "m")
        out.append((known, z.top, z.bot, d))
    return pd.DataFrame(out, columns=["known", "top", "bot", "d"])


def candidates(m5, m15, h4, poi):
    """Every inside-then-back-out reaction at a live POI, in time order."""
    t = m5["server"].to_numpy()
    c = m5["close"].tolist()
    hour = m5["server"].dt.hour.tolist()
    bias = G.h4_bias(h4, t + np.timedelta64(5, "m")).tolist()
    t15 = m15["server"].to_numpy()
    c15 = m15["close"].to_numpy()
    rows = []
    for zid, p in enumerate(poi.itertuples(index=False)):
        a = int(np.searchsorted(t, p.known))
        end_t = p.known + np.timedelta64(LIFE_H, "h")
        # An M15 close through the far side ends the zone.
        a15 = np.searchsorted(t15, p.known)
        b15 = np.searchsorted(t15, end_t)
        far = np.flatnonzero(c15[a15:b15] < p.bot) if p.d > 0 else \
            np.flatnonzero(c15[a15:b15] > p.top)
        if len(far):
            end_t = min(end_t, t15[a15 + far[0]] + np.timedelta64(15, "m"))
        b = int(np.searchsorted(t, end_t))
        top, bot, d = p.top, p.bot, p.d
        inside = False
        for k in range(a, min(b, len(c) - 1)):
            ck = c[k]
            if bot <= ck <= top:
                inside = True
                continue
            if inside and ((d > 0 and ck > top) or (d < 0 and ck < bot)):
                inside = False
                if hour[k + 1] in ASIA and bias[k] == d:
                    rows.append((k + 1, d, bot if d > 0 else top, zid))
            elif not (bot <= ck <= top):
                inside = False
    return pd.DataFrame(rows, columns=["i", "d", "stop", "zone"]) \
        .sort_values("i").reset_index(drop=True)


def simulate(m5, sig):
    t = m5["server"].to_numpy()
    o, h, l, c = (m5[x].tolist() for x in ("open", "high", "low", "close"))
    spread = (m5["spread"] * G.POINT).tolist()
    day = m5["server"].dt.normalize().to_numpy()
    out, busy, per_day, dead, skipped = [], -1, {}, {}, 0
    n = len(o)
    for s in sig.itertuples(index=False):
        j = s.i
        if j >= n - 1 or j <= busy or (s.zone in dead and j > dead[s.zone]):
            continue
        dk = day[j]
        if per_day.get(dk, 0) >= 3:
            continue
        entry = o[j]
        risk = (entry - s.stop) * s.d
        cost = spread[j] + G.COMMISSION
        if risk <= 2 * cost:
            skipped += 1
            continue
        tgt = entry + s.d * RR * risk
        k, px, why = j, None, "roll"
        while k < n and day[k] == dk:
            if (s.d > 0 and l[k] <= s.stop) or (s.d < 0 and h[k] >= s.stop):
                px = min(s.stop, o[k]) if s.d > 0 else max(s.stop, o[k])
                why = "stop"
                break
            if (s.d > 0 and h[k] >= tgt) or (s.d < 0 and l[k] <= tgt):
                px = max(tgt, o[k]) if s.d > 0 else min(tgt, o[k])
                why = "target"
                break
            k += 1
        if px is None:
            k = min(k, n) - 1
            px = c[k]
        if why == "stop":
            dead[s.zone] = k
        pnl = s.d * (px - entry) - cost
        out.append(dict(entry_time=t[j], d=s.d, risk=risk, r=pnl / risk,
                        usd=pnl * G.OZ_PER_LOT * G.LOTS, why=why))
        busy = k
        per_day[dk] = per_day.get(dk, 0) + 1
    return pd.DataFrame(out), skipped


def table(tr):
    yr = pd.to_datetime(tr["entry_time"]).dt.year
    rows = []
    for name, g in (("ALL", tr), ("2019-22", tr[yr < 2023]),
                    ("2023-26", tr[yr >= 2023]), ("2025-26", tr[yr >= 2025]),
                    ("longs", tr[tr["d"] > 0]), ("shorts", tr[tr["d"] < 0])):
        r, u = g["r"].to_numpy(), g["usd"].to_numpy()
        if len(r) < 2:
            continue
        rows.append(dict(slice=name, trades=len(r),
                         win_pct=100 * (r > 0).mean(), avg_r=r.mean(),
                         pf_r=G.pf(r), pf_usd_003=G.pf(u),
                         net_usd_003=u.sum(),
                         median_stop_usd=np.median(g["risk"])))
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 220)
    t0 = time.time()
    m5, m15, h4 = G.load()
    print("2:1 needs 38.5% wins for PF 1.25; coin toss 33.3%")
    for mult in (2.0, 1.5, 3.0):
        p = pois(m15, mult)
        sig = candidates(m5, m15, h4, p)
        tr, skipped = simulate(m5, sig)
        label = "PRIMARY" if mult == 2.0 else "robustness only"
        print(f"\n{'=' * 100}\nIMPULSE >= {mult} x ZONE HEIGHT ({label}): "
              f"{len(p)} points of interest, {len(sig)} reactions, "
              f"skipped (stop inside cost) {skipped}   [{time.time() - t0:.0f}s]"
              f"\n{'=' * 100}")
        print(table(tr).round(3).to_string(index=False))
        if mult == 2.0:
            tr.to_csv("data/gold_poi_trades.csv", index=False)
            primary_sig = sig
    # Does it fire where the user traded?
    mine = pd.read_csv("data/my_trades.csv", parse_dates=["t_open"])
    mine = mine[(mine["sym"] == "XAUUSD")
                & mine["source"].str.startswith("manual")]
    st = m5["server"].to_numpy()[primary_sig["i"].clip(upper=len(m5) - 1)]

    def cover(times, sides):
        hit = 0
        for tm, sd in zip(times, sides, strict=True):
            d = 1 if sd == "buy" else -1
            lo_ = np.searchsorted(st, tm - pd.Timedelta(minutes=30))
            hi_ = np.searchsorted(st, tm + pd.Timedelta(minutes=30),
                                  side="right")
            hit += bool((primary_sig["d"].iloc[lo_:hi_] == d).any())
        return 100 * hit / max(1, len(times))
    real = cover(mine["t_open"], mine["side"])
    base = np.mean([cover(mine["t_open"] + pd.Timedelta(days=dd),
                          mine["side"]) for dd in (-14, -7, 7, 14)])
    print(f"\nyour 59 manual gold trades with a same-direction POI reaction "
          f"within 30 min: {real:.0f}%   (chance level {base:.0f}%)")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
