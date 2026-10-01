"""
What produces a positive expected value on XAUUSD? A fresh, first-principles
look -- no zones, no entry models, nothing carried over.

A positive EV on a single market can only come from a few places, so each is
measured directly on The5ers' gold hourly bars, 2005-2026, after The5ers'
real costs (spread by hour + $8.22 a lot commission + $89.10 a lot per
night financing on BOTH sides, tripled on Friday nights):

  A  DRIFT        simply being long (or short) for an hour, a session, a
                  day, a week, a month
  B  CALENDAR     hour of day, weekday, month of year, turn of the month
  C  MOMENTUM     does the direction of the last 1/5/20/60/120/250 days
                  predict the next 1/5/20 days?
  D  REVERSAL     does a very large day predict the next session?

Every finding is shown for 2005-15 and 2016-26 separately. With ~90 tests,
a few will clear t = 2 by luck in either half; only effects with the same
sign and t >= 2 in BOTH halves are called consistent, and C is compared
with random signals.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PRICE_NOW = 4176.0
COMM_BP = 1e4 * 8.22 / (100 * PRICE_NOW)            # ~0.20bp round trip
SWAP_BP = 1e4 * 89.10 / (100 * PRICE_NOW)           # ~2.13bp per night
SPLIT = 2016


def load():
    a = pd.read_parquet("data/the5ers_h1_all.parquet")
    g = a[a["sym"] == "XAUUSD"].copy()
    g["ny"] = g["server"] - pd.Timedelta(hours=7)
    g = g[g["ny"] >= "2005-01-01"].set_index("ny").sort_index()
    per_day = g.groupby(g.index.normalize()).size()
    g = g[g.index.normalize().isin(per_day[per_day >= 18].index)]
    sp = pd.read_csv("data/the5ers_spread_by_hour.csv")
    sp = sp[sp["sym"] == "XAUUSD"].set_index("ny_hour")["spread_bp"]
    return g, sp


def sessions(g):
    """One row per server day (17:00 NY to 17:00 NY): open, close, weekday."""
    sday = (g["server"].dt.normalize())
    d = g.groupby(sday.to_numpy()).agg(open=("open", "first"),
                                       close=("close", "last"))
    d.index = pd.DatetimeIndex(d.index)
    d["wd"] = d.index.weekday                    # server weekday: Mon=0..Fri=4
    d["yr"] = d.index.year
    d["month"] = d.index.month
    return d[d["wd"] < 5]


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(n)), n


def two_halves(x, years, label, extra=None):
    x = np.asarray(x, float)
    years = np.asarray(years)
    a = stat(x[years < SPLIT])
    b = stat(x[years >= SPLIT])
    allx = stat(x)
    consistent = (np.sign(a[0]) == np.sign(b[0])) and a[1] * np.sign(a[0]) \
        >= 2 and b[1] * np.sign(b[0]) >= 2
    row = dict(test=label, bp_2005_15=a[0], t_2005_15=a[1], bp_2016_26=b[0],
               t_2016_26=b[1], bp_all=allx[0], n=allx[2],
               consistent="YES" if consistent else "")
    if extra:
        row.update(extra)
    return row


def nights_between(d0, d1):
    """Financing nights charged holding from session d0's open to session
    d1's close (rollovers crossed: one per session boundary, Friday x3)."""
    n = 0
    for day in pd.bdate_range(d0, d1)[:-1]:
        n += 3 if day.weekday() == 4 else 1
    return n


def main() -> int:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    g, sp = load()
    d = sessions(g)
    rt = sp.median() + COMM_BP                     # typical round-trip cost
    print(f"gold: {d.index[0].date()} -> {d.index[-1].date()}, "
          f"{len(d):,} sessions; price {d['close'].iloc[0]:.0f} -> "
          f"{d['close'].iloc[-1]:.0f}")
    print(f"costs: round trip ~{rt:.2f}bp; financing {SWAP_BP:.2f}bp a night "
          f"(x3 Fridays) = {SWAP_BP * 365:.0f}bp a year")

    # ------------------------------------------------------------- A drift
    rows = []
    hr = g.index.hour
    yrs_h = g.index.year
    hret = 1e4 * (g["close"] / g["open"] - 1)
    rows.append(two_halves(hret - rt, yrs_h, "LONG any 1 hour"))
    sess = 1e4 * (d["close"] / d["open"] - 1)
    rows.append(two_halves(sess - rt, d["yr"], "LONG one session, out "
                           "before the roll (no fee)"))
    rows.append(two_halves(-sess - rt, d["yr"], "SHORT one session, out "
                           "before the roll"))
    cl = d["close"].to_numpy()
    for hold, name in ((1, "1 day"), (5, "1 week"), (20, "1 month"),
                       (60, "3 months")):
        gross, cost, yrs = [], [], []
        for i in range(0, len(d) - hold, hold):
            gross.append(1e4 * (cl[i + hold] / cl[i] - 1))
            nights = nights_between(d.index[i], d.index[i + hold])
            cost.append(rt + nights * SWAP_BP)   # both sides pay financing
            yrs.append(d["yr"].iat[i])
        gross, cost = np.array(gross), np.array(cost)
        for side, sgn in (("LONG", 1), ("SHORT", -1)):
            r = sgn * gross - cost
            rows.append(two_halves(r, yrs, f"{side} {name}, with financing",
                                   dict(per_year_pct=r.mean() / 100
                                        * 252 / hold)))
    print("\n" + "=" * 120)
    print("A. DRIFT: being long or short, after all costs (bp per trade; "
          "per_year_pct where held for days)")
    print("=" * 120)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ----------------------------------------------------------- B calendar
    rows = []
    for h in range(24):
        sel = hr == h
        rows.append(two_halves(hret[sel] - (sp.get(h, rt) + COMM_BP),
                               yrs_h[sel], f"long the {h:02d}:00 NY hour"))
    for wd, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        s = d["wd"] == wd
        rows.append(two_halves(sess[s] - rt, d["yr"][s],
                               f"long the {nm} session"))
    for mo in range(1, 13):
        s = d["month"] == mo
        rows.append(two_halves(sess[s] - rt, d["yr"][s],
                               f"long sessions in month {mo:02d}"))
    pos_in_month = d.groupby([d["yr"], d["month"]]).cumcount()
    from_end = d.groupby([d["yr"], d["month"]]).cumcount(ascending=False)
    tom = (pos_in_month < 3) | (from_end < 2)
    rows.append(two_halves(sess[tom] - rt, d["yr"][tom],
                           "long turn-of-month sessions (last 2 + first 3)"))
    rows.append(two_halves(sess[~tom] - rt, d["yr"][~tom],
                           "long all other sessions"))
    b = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print(f"B. CALENDAR: {len(b)} tests (luck alone would put ~{0.05 * len(b):.0f} "
          f"past |t| 2 in each half)")
    print("=" * 120)
    print(b.round(2).to_string(index=False))

    # ----------------------------------------------------------- C momentum
    rows, rnd = [], []
    rng = np.random.default_rng(9)
    for look in (1, 5, 20, 60, 120, 250):
        for hold in (1, 5, 20):
            r, yrs, longr = [], [], []
            for i in range(look, len(d) - hold, hold):
                sig = np.sign(cl[i] / cl[i - look] - 1)
                if sig == 0:
                    continue
                gross = 1e4 * (cl[i + hold] / cl[i] - 1)
                nights = nights_between(d.index[i], d.index[i + hold])
                cost = rt + nights * SWAP_BP
                r.append(sig * gross - cost)
                longr.append(gross - cost)
                rnd.append((look, hold, rng.choice([-1, 1]) * gross - cost,
                            d["yr"].iat[i]))
                yrs.append(d["yr"].iat[i])
            rows.append(two_halves(r, yrs, f"follow {look}d direction, hold "
                                   f"{hold}d", dict(long_only_bp=np.mean(longr),
                                                    per_year_pct=np.mean(r)
                                                    / 100 * 252 / hold)))
    c = pd.DataFrame(rows)
    rr = pd.DataFrame(rnd, columns=["look", "hold", "r", "yr"])
    print("\n" + "=" * 120)
    print("C. MOMENTUM: trade the sign of the last N days, after costs and "
          "financing (long_only_bp = just being long over the same trades)")
    print("=" * 120)
    print(c.round(2).to_string(index=False))
    rc = rr.groupby(["look", "hold"]).apply(
        lambda x: pd.Series(two_halves(x["r"], x["yr"], "random"))).reset_index()
    print(f"\n  random-sign versions passing both halves: "
          f"{(rc['consistent'] == 'YES').sum()} of {len(rc)}")

    # ----------------------------------------------------------- D reversal
    rows = []
    ret = 1e4 * (d["close"] / d["close"].shift(1) - 1)
    vol = ret.rolling(20).std().shift(1)
    nxt = sess.shift(-1)
    for k in (1.5, 2.0, 2.5):
        up, dn = ret > k * vol, ret < -k * vol
        rows.append(two_halves(-nxt[up] - rt, d["yr"][up],
                               f"after a +{k} sigma day: SHORT next session"))
        rows.append(two_halves(nxt[dn] - rt, d["yr"][dn],
                               f"after a -{k} sigma day: LONG next session"))
        rows.append(two_halves(nxt[up] - rt, d["yr"][up],
                               f"after a +{k} sigma day: LONG next session"))
        rows.append(two_halves(-nxt[dn] - rt, d["yr"][dn],
                               f"after a -{k} sigma day: SHORT next session"))
    print("\n" + "=" * 120)
    print("D. AFTER BIG DAYS (next session, out before the roll)")
    print("=" * 120)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    allrows = pd.concat([b, c, pd.DataFrame(rows)])
    print("\n" + "=" * 120)
    print("CONSISTENT IN BOTH HALVES (same sign, t >= 2 in each):")
    print("=" * 120)
    cons = allrows[allrows["consistent"] == "YES"]
    print(cons[["test", "bp_2005_15", "t_2005_15", "bp_2016_26", "t_2016_26",
                "n"]].round(2).to_string(index=False) if len(cons) else "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
