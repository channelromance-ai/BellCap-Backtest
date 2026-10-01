"""
Gold zones, version 2: the method as the user clarified it.

Changes from gold_zones.py, all from the user's own description:
  * Asia only: entries from 19:00 to 02:00 New York (18:00-01:00 Winnipeg),
    which is when the user's manual gold trades cluster.
  * A zone keeps trading until it stops you out. Breaks, bounces and
    rejections may repeat on the same zone; the first stop-out kills it.
  * Stop at the zone's own edge (bottom for longs, top for shorts), no pad.
  * Fixed 2:1 target.

Unchanged: M15 zones (8 bars, range <= 2.0 x ATR14), H4 50-EMA bias, M5
signals with entry at the next M5 open, one position at a time, 3 a day,
exit by the 17:00 New York roll, stop first on ties, each bar's own spread
plus $8.22 a lot commission.

A stop at the zone edge sits almost on the entry for a rejection, which
re-enters right at that edge. Trades whose stop is within 2x the cost are
skipped as untradeable, and a variant puts rejection stops at the extreme
of the failed break instead. Both are reported; neither is chosen from.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_zones as G                        # noqa: E402

ASIA_SERVER_HOURS = set(range(2, 9))     # 19:00-01:59 New York = server 02-08
RR = 2.0


def events(m5, zones, h4, rejection_stop="edge"):
    t = m5["server"].to_numpy()
    h, l, c = (m5[x].to_numpy() for x in ("high", "low", "close"))
    bias = G.h4_bias(h4, t + np.timedelta64(5, "m"))
    hour = m5["server"].dt.hour.to_numpy()
    rows = []
    for zid, z in enumerate(zones.itertuples(index=False)):
        a = np.searchsorted(t, z.known)
        b = np.searchsorted(t, z.known + np.timedelta64(24, "h"))
        if b - a < 3:
            continue
        top, bot = z.top, z.bot
        been_above = been_below = False
        lo_since, hi_since = np.inf, -np.inf
        for k in range(max(a, 1), b - 1):
            ck, cp = c[k], c[k - 1]
            been_above |= ck > top
            been_below |= ck < bot
            lo_since = l[k] if cp >= bot and ck < bot else min(lo_since, l[k])
            hi_since = h[k] if cp <= top and ck > top else max(hi_since, h[k])
            if hour[k + 1] not in ASIA_SERVER_HOURS:
                continue
            bb, sig = bias[k], None
            if bb > 0:
                if cp <= top < ck:
                    sig, stop = "break", bot
                elif been_above and cp > top and l[k] <= top < ck:
                    sig, stop = "bounce", bot
                elif cp < bot <= ck:
                    sig = "rejection"
                    stop = bot if rejection_stop == "edge" else lo_since
                d = 1
            elif bb < 0:
                if cp >= bot > ck:
                    sig, stop = "break", top
                elif been_below and cp < bot and h[k] >= bot > ck:
                    sig, stop = "bounce", top
                elif cp > top >= ck:
                    sig = "rejection"
                    stop = top if rejection_stop == "edge" else hi_since
                d = -1
            if sig:
                rows.append(dict(i=k + 1, d=d, kind=sig, stop=stop, zone=zid))
    s = pd.DataFrame(rows)
    return s.sort_values("i").reset_index(drop=True) if len(s) else s


def simulate(m5, sig):
    t = m5["server"].to_numpy()
    o, h, l, c = (m5[x].to_numpy() for x in ("open", "high", "low",
                                               "close"))
    spread = m5["spread"].to_numpy() * G.POINT
    day = m5["server"].dt.normalize().to_numpy()
    out, busy_until, per_day, dead, skipped = [], -1, {}, {}, 0
    for s in sig.itertuples(index=False):
        j = s.i
        if j >= len(o) - 1 or j <= busy_until:
            continue
        if s.zone in dead and j > dead[s.zone]:
            continue                       # zone already stopped you out
        dkey = day[j]
        if per_day.get(dkey, 0) >= 3:
            continue
        entry = o[j]
        risk = (entry - s.stop) * s.d
        cost = spread[j] + G.COMMISSION
        if risk <= 2 * cost:
            skipped += 1
            continue
        target = entry + s.d * RR * risk
        k, px, why = j, None, "roll"
        while k < len(o) and day[k] == dkey:
            if (s.d > 0 and l[k] <= s.stop) or (s.d < 0 and h[k] >= s.stop):
                px = min(s.stop, o[k]) if s.d > 0 else max(s.stop, o[k])
                why = "stop"
                break
            if (s.d > 0 and h[k] >= target) or (s.d < 0 and l[k] <= target):
                px = max(target, o[k]) if s.d > 0 else min(target, o[k])
                why = "target"
                break
            k += 1
        if px is None:
            k = min(k, len(o)) - 1
            px = c[k]
        if why == "stop":
            dead[s.zone] = k
        pnl = s.d * (px - entry) - cost
        out.append(dict(entry_time=t[j], kind=s.kind, d=s.d, risk=risk,
                        r=pnl / risk, usd=pnl * G.OZ_PER_LOT * G.LOTS,
                        why=why, zone=s.zone))
        busy_until = k
        per_day[dkey] = per_day.get(dkey, 0) + 1
    return pd.DataFrame(out), skipped


def table(tr):
    yr = pd.to_datetime(tr["entry_time"]).dt.year
    rows = []
    for name, g in (("ALL", tr), ("2019-22", tr[yr < 2023]),
                    ("2023-26", tr[yr >= 2023]), ("2025-26", tr[yr >= 2025]),
                    ("bounce", tr[tr["kind"] == "bounce"]),
                    ("break", tr[tr["kind"] == "break"]),
                    ("rejection", tr[tr["kind"] == "rejection"])):
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
    m5, m15, h4 = G.load()
    zones = G.find_zones(m15)
    print("2:1 needs 38.5% wins for PF 1.25; coin toss 33.3%")
    keep = {}
    for rs in ("edge", "extreme"):
        sig = events(m5, zones, h4, rs)
        tr, skipped = simulate(m5, sig)
        keep[rs] = (sig, tr)
        print(f"\n{'=' * 100}\nREJECTION STOP AT THE {rs.upper()}   "
              f"(signals {len(sig)}, skipped as stop inside cost {skipped})"
              f"\n{'=' * 100}")
        print(table(tr).round(3).to_string(index=False))

    print(f"\n{'=' * 100}\nROBUSTNESS (zone edge stops), not used to choose"
          f"\n{'=' * 100}")
    rows = []
    for n in (6, 8, 12):
        for k in (1.5, 2.0, 2.5):
            tr, _ = simulate(m5, events(m5, G.find_zones(m15, n, k), h4))
            yr = pd.to_datetime(tr["entry_time"]).dt.year
            rows.append(dict(bars=n, width=k, trades=len(tr),
                             pf_r=G.pf(tr["r"].to_numpy()),
                             pf_2019_22=G.pf(tr.loc[yr < 2023, "r"].to_numpy()),
                             pf_2023_26=G.pf(tr.loc[yr >= 2023, "r"]
                                             .to_numpy()),
                             pf_usd=G.pf(tr["usd"].to_numpy())))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print(f"\n{'=' * 100}\nYOUR TRADES vs THIS VERSION\n{'=' * 100}")
    mine = pd.read_csv("data/my_trades.csv", parse_dates=["t_open"])
    mine = mine[(mine["sym"] == "XAUUSD")
                & mine["source"].str.startswith("manual")].copy()
    mine["stop_usd_per_oz"] = (mine["entry"] - mine["sl0"]).abs()
    print(f"  your stops (where recorded, {mine['sl0'].notna().sum()} of "
          f"{len(mine)}): median ${mine['stop_usd_per_oz'].median():.2f}/oz;"
          f" model's median stop ${np.median(keep['edge'][1]['risk']):.2f}/oz")
    sig = keep["edge"][0]
    st = m5["server"].to_numpy()[sig["i"].clip(upper=len(m5) - 1)]

    def cover(times, sides):
        hit = 0
        for tm, sd in zip(times, sides, strict=True):
            d = 1 if sd == "buy" else -1
            lo = np.searchsorted(st, tm - pd.Timedelta(minutes=30))
            hi = np.searchsorted(st, tm + pd.Timedelta(minutes=30),
                                 side="right")
            hit += bool((sig["d"].iloc[lo:hi] == d).any())
        return 100 * hit / max(1, len(times))
    asia = mine[mine["t_open"].dt.hour.isin(ASIA_SERVER_HOURS)]
    print(f"  your manual gold trades in the Asia window: {len(asia)} of "
          f"{len(mine)}")
    print(f"  of those, with a same-direction signal within 30 min: "
          f"{cover(asia['t_open'], asia['side']):.0f}%   chance level: "
          f"{np.mean([cover(asia['t_open'] + pd.Timedelta(days=dd), asia['side']) for dd in (-14, -7, 7, 14)]):.0f}%")
    keep["edge"][1].to_csv("data/gold_zones_v2_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
