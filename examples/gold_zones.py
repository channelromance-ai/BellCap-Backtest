"""
The user's discretionary gold method, written down as rules and tested.

As described: consolidation zones on the 15-minute chart, direction from a
higher timeframe, entries on the 5-minute chart, trading bounces, breaks and
rejections when the higher timeframe allows. Traded at a flat 0.03 lots.
Fixed before running:

  Zone      8 consecutive M15 bars whose whole range is at most 2.0 x
            ATR14(M15) at the last of them. Known when that bar closes.
            Alive for 24 hours. Windows do not overlap.
  Bias      Last completed H4 bar closed above its 50-EMA = long only;
            below = short only. Read at each M5 signal's close.
  Signals   On M5 closes, for a long bias (short is the mirror):
              BREAK      first M5 close above the zone top
              BOUNCE     after a break up, an M5 bar dips to the top and
                         closes back above it
              REJECTION  an M5 close below the zone bottom, then a close
                         back at or above the bottom (a failed breakdown)
            Each type at most once per zone. Entry at the next M5 open.
  Stop      Break/bounce: zone bottom less 10% of the zone's height.
            Rejection: the lowest point of the failed break, less the same.
  Exit      Target 1.5R (2R also reported), stop, or the 17:00 New York roll.
            Stop assumed first when one M5 bar holds both. Gaps fill at the
            open.
  Book      One position at a time, at most 3 trades per server day.
  Cost      That M5 bar's own spread plus $8.22 a lot round-trip commission
            (The5ers' $4.11 a side, measured from the account).

Reported in R (risk-based sizing) and in dollars at a flat 0.03 lots, the
way the user actually traded. Nearby zone settings are shown for
robustness, never to pick a winner. Finally: does this rule set fire where
the user's own 59 manual gold trades were?
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

COMMISSION = 0.0822          # $ per ounce round trip ($8.22 a 100oz lot)
POINT = 0.01
LOTS = 0.03
OZ_PER_LOT = 100


def load():
    m5 = pd.read_parquet("data/xau_M5.parquet")
    m15 = pd.read_parquet("data/xau_M15.parquet")
    h4 = pd.read_parquet("data/xau_H4.parquet")
    m15 = m15[m15["server"] >= m5["server"].iloc[0] - pd.Timedelta(days=5)]
    return m5.reset_index(drop=True), m15.reset_index(drop=True), h4


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev).abs(),
                               (df["low"] - prev).abs()))
    return tr.rolling(n).mean()


def find_zones(m15, n_bars=8, k=2.0):
    a = atr(m15).to_numpy()
    hi, lo = m15["high"].to_numpy(), m15["low"].to_numpy()
    t = m15["server"].to_numpy()
    zones = []
    i = n_bars
    while i < len(m15):
        top, bot = hi[i - n_bars + 1:i + 1].max(), lo[i - n_bars + 1:i + 1].min()
        if np.isfinite(a[i]) and a[i] > 0 and top - bot <= k * a[i]:
            known = t[i] + np.timedelta64(15, "m")
            zones.append((known, top, bot))
            i += n_bars                     # no overlapping windows
        else:
            i += 1
    return pd.DataFrame(zones, columns=["known", "top", "bot"])


def h4_bias(h4, times):
    """+1/-1 from the last H4 bar that had CLOSED at each time."""
    ema = h4["close"].ewm(span=50, adjust=False).mean()
    sign = np.sign(h4["close"] - ema).to_numpy()
    close_time = (h4["server"] + pd.Timedelta(hours=4)).to_numpy()
    pos = np.searchsorted(close_time, times, side="right") - 1
    out = np.where(pos >= 0, sign[np.clip(pos, 0, None)], 0)
    return out


def signals(m5, zones, h4):
    t = m5["server"].to_numpy()
    o, h, l, c = (m5[x].to_numpy() for x in ("open", "high", "low",
                                               "close"))
    close_time = t + np.timedelta64(5, "m")
    bias = h4_bias(h4, close_time)
    rows = []
    for z in zones.itertuples(index=False):
        a = np.searchsorted(t, z.known)
        b = np.searchsorted(t, z.known + np.timedelta64(24, "h"))
        if b - a < 3:
            continue
        top, bot = z.top, z.bot
        pad = 0.1 * (top - bot)
        state = dict(up=False, down=False, bounce_up=False, bounce_dn=False,
                     brk_up=False, brk_dn=False, rej_up=False, rej_dn=False,
                     below_lo=np.inf, above_hi=-np.inf)
        for k in range(a, b - 1):
            bb = bias[k]
            ck = c[k]
            # track failed breaks for rejections
            if ck < bot:
                state["down"] = True
            if ck > top:
                state["up"] = True
            if state["down"]:
                state["below_lo"] = min(state["below_lo"], l[k])
            if state["up"]:
                state["above_hi"] = max(state["above_hi"], h[k])
            sig = None
            if bb > 0:
                if not state["brk_up"] and ck > top and c[k - 1] <= top:
                    sig, stop = "break", bot - pad
                    state["brk_up"] = True
                elif state["brk_up"] and not state["bounce_up"] and \
                        l[k] <= top and ck > top and k > a:
                    sig, stop = "bounce", bot - pad
                    state["bounce_up"] = True
                elif state["down"] and not state["rej_up"] and ck >= bot \
                        and c[k - 1] < bot:
                    sig, stop = "rejection", state["below_lo"] - pad
                    state["rej_up"] = True
                d = 1
            elif bb < 0:
                if not state["brk_dn"] and ck < bot and c[k - 1] >= bot:
                    sig, stop = "break", top + pad
                    state["brk_dn"] = True
                elif state["brk_dn"] and not state["bounce_dn"] and \
                        h[k] >= bot and ck < bot and k > a:
                    sig, stop = "bounce", top + pad
                    state["bounce_dn"] = True
                elif state["up"] and not state["rej_dn"] and ck <= top \
                        and c[k - 1] > top:
                    sig, stop = "rejection", state["above_hi"] + pad
                    state["rej_dn"] = True
                d = -1
            if sig:
                rows.append(dict(i=k + 1, d=d, kind=sig, stop=stop,
                                 zone_h=top - bot))
    s = pd.DataFrame(rows)
    return s.sort_values("i").reset_index(drop=True) if len(s) else s


def simulate(m5, sig, rr=1.5):
    """One position at a time, 3 a day, exit by the 17:00 NY roll."""
    t = m5["server"].to_numpy()
    o, h, l, c = (m5[x].to_numpy() for x in ("open", "high", "low",
                                               "close"))
    spread = m5["spread"].to_numpy() * POINT
    day = m5["server"].dt.normalize().to_numpy()   # server day = NY roll
    out, busy_until, per_day = [], -1, {}
    for s in sig.itertuples(index=False):
        j = s.i
        if j >= len(o) - 1 or j <= busy_until:
            continue
        dkey = day[j]
        if per_day.get(dkey, 0) >= 3:
            continue
        entry = o[j]
        risk = (entry - s.stop) * s.d
        cost = spread[j] + COMMISSION
        if risk <= 2 * cost:
            continue                       # stop inside the cost: untradeable
        target = entry + s.d * rr * risk
        k = j
        px, why = None, "roll"
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
        pnl = s.d * (px - entry) - cost
        out.append(dict(entry_time=t[j], kind=s.kind, d=s.d, risk=risk,
                        r=pnl / risk, usd=pnl * OZ_PER_LOT * LOTS, why=why))
        busy_until = k
        per_day[dkey] = per_day.get(dkey, 0) + 1
    return pd.DataFrame(out)


def pf(x):
    w, lo = x[x > 0].sum(), -x[x <= 0].sum()
    return w / lo if lo > 0 else np.inf


def summary(tr):
    def one(g):
        r, u = g["r"].to_numpy(), g["usd"].to_numpy()
        return pd.Series(dict(trades=len(r),
                              per_year=len(r) / 7.1,
                              win_pct=100 * (r > 0).mean(), avg_r=r.mean(),
                              pf_r=pf(r), pf_usd_003=pf(u),
                              net_usd_003=u.sum()))
    tr = tr.assign(period=np.where(pd.to_datetime(tr["entry_time"]).dt.year
                                   < 2023, "2019-22", "2023-26"))
    rows = [one(tr).rename(("ALL", "all"))]
    for (p,), g in tr.groupby(["period"]):
        rows.append(one(g).rename((p, "all")))
    for k, g in tr.groupby("kind"):
        rows.append(one(g).rename(("ALL", k)))
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 220)
    m5, m15, h4 = load()
    zones = find_zones(m15)
    sig = signals(m5, zones, h4)
    print(f"zones found: {len(zones)}; raw signals: {len(sig)} "
          f"({sig['kind'].value_counts().to_dict()})")
    for rr in (1.5, 2.0):
        tr = simulate(m5, sig, rr)
        print(f"\n{'=' * 100}\nPRIMARY RULES, {rr}:1  (PF 1.25 needs "
              f"{100 * 1.25 / (1.25 + rr):.1f}% wins; coin toss "
              f"{100 / (1 + rr):.1f}%)\n{'=' * 100}")
        print(summary(tr).round(3).to_string())
        if rr == 1.5:
            primary = tr

    print(f"\n{'=' * 100}\nROBUSTNESS: nearby zone settings, 1.5:1, all "
          f"signal types (not used to choose)\n{'=' * 100}")
    rows = []
    for n in (6, 8, 12):
        for k in (1.5, 2.0, 2.5):
            s2 = signals(m5, find_zones(m15, n, k), h4)
            t2 = simulate(m5, s2, 1.5)
            yr = pd.to_datetime(t2["entry_time"]).dt.year
            rows.append(dict(bars=n, width_x_atr=k, trades=len(t2),
                             pf_r=pf(t2["r"].to_numpy()),
                             pf_2019_22=pf(t2.loc[yr < 2023, "r"].to_numpy()),
                             pf_2023_26=pf(t2.loc[yr >= 2023, "r"].to_numpy()),
                             pf_usd_003=pf(t2["usd"].to_numpy())))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print(f"\n{'=' * 100}\nDOES THE RULE SET FIRE WHERE YOU TRADED?\n"
          f"{'=' * 100}")
    mine = pd.read_csv("data/my_trades.csv", parse_dates=["t_open"])
    mine = mine[(mine["sym"] == "XAUUSD")
                & mine["source"].str.startswith("manual")]
    all_sig = sig.assign(t=m5["server"].to_numpy()[sig["i"].clip(
        upper=len(m5) - 1)])

    def coverage(times, sides, window_min=30):
        hit = 0
        st = all_sig["t"].to_numpy()
        for tm, sd in zip(times, sides, strict=True):
            d = 1 if sd == "buy" else -1
            lo_i = np.searchsorted(st, tm - pd.Timedelta(minutes=window_min))
            hi_i = np.searchsorted(st, tm + pd.Timedelta(minutes=window_min),
                                   side="right")
            if (all_sig["d"].iloc[lo_i:hi_i] == d).any():
                hit += 1
        return 100 * hit / max(1, len(times))
    real = coverage(mine["t_open"], mine["side"])
    # The same test on your trades shifted by a week: what chance alone gives.
    shifted = np.mean([coverage(mine["t_open"] + pd.Timedelta(days=dd),
                                mine["side"]) for dd in (-14, -7, 7, 14)])
    print(f"  your manual gold trades: {len(mine)}")
    print(f"  with a same-direction model signal within 30 minutes: "
          f"{real:.0f}%   (same test on your trade times shifted a week "
          f"or two, i.e. chance: {shifted:.0f}%)")

    print(f"\n{'=' * 100}\nFILL CHECK: 2025-26 trades re-resolved on 1-minute"
          f" bars\n{'=' * 100}")
    m1 = pd.read_parquet("data/xau_M1.parquet")
    p = primary.copy()
    p["t"] = pd.to_datetime(p["entry_time"])
    p = p[p["t"] >= m1["server"].iloc[0] + pd.Timedelta(days=1)]
    mt = m1["server"].to_numpy()
    mo, mh, ml = (m1[x].to_numpy() for x in ("open", "high", "low"))
    mday = m1["server"].dt.normalize().to_numpy()
    agree = 0
    for r in p.itertuples(index=False):
        a = np.searchsorted(mt, np.datetime64(r.t))
        if a >= len(m1):
            continue
        # Same stop and target as the M5 run, walked minute by minute until
        # the same 17:00 New York roll (server midnight).
        e = mo[a]
        stop, tgt = e - r.d * r.risk, e + r.d * 1.5 * r.risk
        why, k = "roll", a
        while k < len(m1) and mday[k] == mday[a]:
            if (r.d > 0 and ml[k] <= stop) or (r.d < 0 and mh[k] >= stop):
                why = "stop"
                break
            if (r.d > 0 and mh[k] >= tgt) or (r.d < 0 and ml[k] <= tgt):
                why = "target"
                break
            k += 1
        agree += why == r.why
    print(f"  trades checked: {len(p)}; same outcome on 1-minute bars: "
          f"{100 * agree / max(1, len(p)):.1f}%")
    primary.to_csv("data/gold_zones_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
