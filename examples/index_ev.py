"""
What produces a positive expected value on NAS100 / SP500? The same
first-principles study as xau_ev.py, for The5ers' index CFDs.

Two data sources, because The5ers' index history is only genuinely hourly
from mid-2020:
  intraday   The5ers NAS100 / SP500 hourly bars, Jun 2020 - Sep 2026
             (halves: 2020-22 and 2023-26)
  daily      SPY (1993-) and QQQ (1999-) closes, which include dividends
             (halves: to 2009 and 2010-26). A CFD long is usually credited
             dividends too; if The5ers does not, multi-day longs earn about
             1.3% (SPY) / 0.6% (QQQ) a year less than shown.

Costs measured from The5ers: round trip SP500 ~0.9bp, NAS100 ~0.65bp, no
commission; financing on BOTH sides every night, x3 on Fridays: SP500
1.88bp a night (~6.9% a year), NAS100 1.17bp (~4.3% a year).

  A  DRIFT     long / short for an hour, a session, a day, week, month,
               quarter
  B  CALENDAR  hour of day, weekday, month of year, turn of the month
               (buy the close two days before month end, sell the close of
               the 3rd trading day)
  C  MOMENTUM  follow the last 1/5/20/60/120/250 days' direction for
               1/5/20 days
  D  BIG DAYS  the next session / day after a 2-sigma day, both ways

Consistent = same sign and t >= 2 in both halves.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.xau_ev as X                            # noqa: E402

INSTR = {
    "SP500": dict(etf="SPY", rt=0.90, swap=1e4 * 1.44 / 7668.6,
                  split_daily=2010),
    "NAS100": dict(etf="QQQ", rt=0.65, swap=1e4 * 3.575 / 30508.2,
                   split_daily=2010),
}
INTRADAY_SPLIT = 2023


def halves(x, years, split, label, extra=None):
    X.SPLIT = split
    return X.two_halves(x, years, label, extra)


def intraday(sym, cfg):
    a = pd.read_parquet("data/the5ers_idx_h1.parquet")
    g = a[a["sym"] == sym].copy()
    g["ny"] = g["server"] - pd.Timedelta(hours=7)
    g = g.set_index("ny").sort_index()
    per_day = g.groupby(g.index.normalize()).size()
    g = g[g.index.normalize().isin(per_day[per_day >= 18].index)]
    rt = cfg["rt"]
    rows = []
    hret = 1e4 * (g["close"] / g["open"] - 1)
    y = g.index.year
    rows.append(halves(hret - rt, y, INTRADAY_SPLIT, "LONG any 1 hour"))
    d = X.sessions(g)
    sess = 1e4 * (d["close"] / d["open"] - 1)
    rows.append(halves(sess - rt, d["yr"], INTRADAY_SPLIT,
                       "LONG one session (out before the roll)"))
    rows.append(halves(-sess - rt, d["yr"], INTRADAY_SPLIT,
                       "SHORT one session"))
    for h in range(24):
        s = g.index.hour == h
        if s.sum() < 100:
            continue
        rows.append(halves(hret[s] - rt, y[s], INTRADAY_SPLIT,
                           f"long the {h:02d}:00 NY hour"))
    for wd, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        s = d["wd"] == wd
        rows.append(halves(sess[s] - rt, d["yr"][s], INTRADAY_SPLIT,
                           f"long the {nm} session"))
    ret = 1e4 * (d["close"] / d["close"].shift(1) - 1)
    vol = ret.rolling(20).std().shift(1)
    nxt = sess.shift(-1)
    up, dn = ret > 2 * vol, ret < -2 * vol
    for lab, x, s in (("after +2 sigma day: LONG next session", nxt, up),
                      ("after +2 sigma day: SHORT next session", -nxt, up),
                      ("after -2 sigma day: LONG next session", nxt, dn),
                      ("after -2 sigma day: SHORT next session", -nxt, dn)):
        rows.append(halves(x[s] - rt, d["yr"][s], INTRADAY_SPLIT, lab))
    return pd.DataFrame(rows), d


def daily(cfg):
    u = pd.read_parquet("data/universe_daily.parquet")[cfg["etf"]].dropna()
    u = u[u.index.weekday < 5]
    cl = u.to_numpy()
    idx = u.index
    yrs = idx.year.to_numpy()
    rt, sw, split = cfg["rt"], cfg["swap"], cfg["split_daily"]

    def nights(i0, i1):
        return sum(3 if idx[k].weekday() == 4 else 1 for k in range(i0, i1))

    rows = []
    for hold, name in ((1, "1 day"), (5, "1 week"), (20, "1 month"),
                       (60, "3 months")):
        g, c, y = [], [], []
        for i in range(0, len(cl) - hold, hold):
            g.append(1e4 * (cl[i + hold] / cl[i] - 1))
            c.append(rt + nights(i, i + hold) * sw)
            y.append(yrs[i])
        g, c = np.array(g), np.array(c)
        for side, sgn in (("LONG", 1), ("SHORT", -1)):
            r = sgn * g - c
            rows.append(halves(r, y, split, f"{side} {name}, with financing",
                               dict(per_year_pct=r.mean() / 100 * 252 / hold)))
    # calendar on daily closes (each day = one close-to-close, one night)
    dret = 1e4 * (u / u.shift(1) - 1)
    one = np.array([rt + (3 if idx[k - 1].weekday() == 4 else 1) * sw
                    if k else np.nan for k in range(len(u))])
    for mo in range(1, 13):
        s = (idx.month == mo)
        rows.append(halves((dret - one)[s], yrs[s], split,
                           f"long days in month {mo:02d}"))
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    pos = ym.groupby(ym).cumcount().to_numpy()
    rev = ym.groupby(ym).cumcount(ascending=False).to_numpy()
    tom_r, tom_y = [], []
    starts = np.flatnonzero(rev == 1)                 # 2nd-to-last day close
    for i in starts:
        # i = close 2 days before month end; i+1 = last day; i+2..i+4 =
        # the first three days of the next month, so i+4 is the 3rd day.
        j = i + 4
        if j < len(cl) and pos[j] == 2:
            tom_r.append(1e4 * (cl[j] / cl[i] - 1) - rt - nights(i, j) * sw)
            tom_y.append(yrs[i])
    rows.append(halves(tom_r, tom_y, split, "TURN OF MONTH: buy 2 days before "
                       "month end, sell 3rd day", dict(per_year_pct=np.mean(
                           tom_r) / 100 * 12)))
    inmask = np.zeros(len(u), bool)
    for i in starts:
        inmask[i + 1:i + 5] = True                    # last day + first 3
    rows.append(halves((dret - one)[~inmask], yrs[~inmask], split,
                       "long a day OUTSIDE the turn of month"))
    rows.append(halves((dret - one)[inmask], yrs[inmask], split,
                       "long a day INSIDE the turn of month"))
    # momentum
    rng = np.random.default_rng(4)
    rnd = []
    for look in (1, 5, 20, 60, 120, 250):
        for hold in (1, 5, 20):
            r, y, lo = [], [], []
            for i in range(look, len(cl) - hold, hold):
                sig = np.sign(cl[i] / cl[i - look] - 1)
                if sig == 0:
                    continue
                g = 1e4 * (cl[i + hold] / cl[i] - 1)
                c = rt + nights(i, i + hold) * sw
                r.append(sig * g - c)
                lo.append(g - c)
                y.append(yrs[i])
                rnd.append((look, hold, rng.choice([-1, 1]) * g - c, yrs[i]))
            rows.append(halves(r, y, split, f"follow {look}d direction, hold "
                               f"{hold}d", dict(long_only_bp=np.mean(lo),
                                                per_year_pct=np.mean(r) / 100
                                                * 252 / hold)))
    rr = pd.DataFrame(rnd, columns=["look", "hold", "r", "yr"])
    luck = sum(halves(g["r"], g["yr"], split, "")["consistent"] == "YES"
               for _, g in rr.groupby(["look", "hold"]))
    # big days, next close-to-close (one night)
    vol = dret.rolling(20).std().shift(1)
    nxt = (dret - one).shift(-1)
    nxt_s = (-dret - one).shift(-1)
    up, dn = dret > 2 * vol, dret < -2 * vol
    for lab, x, s in (("after +2 sigma day: LONG next day", nxt, up),
                      ("after +2 sigma day: SHORT next day", nxt_s, up),
                      ("after -2 sigma day: LONG next day", nxt, dn),
                      ("after -2 sigma day: SHORT next day", nxt_s, dn)):
        rows.append(halves(x[s], yrs[s], split, lab))
    return pd.DataFrame(rows), luck, u


def main() -> int:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    cols = ["test", "bp_2005_15", "t_2005_15", "bp_2016_26", "t_2016_26",
            "bp_all", "n", "consistent", "per_year_pct", "long_only_bp"]
    for sym, cfg in INSTR.items():
        print("\n" + "#" * 120)
        print(f"# {sym}   (round trip {cfg['rt']:.2f}bp, financing "
              f"{cfg['swap']:.2f}bp a night = {cfg['swap'] * 365 / 100:.1f}% a "
              f"year)")
        print("#" * 120)
        dd, luck, u = daily(cfg)
        print(f"\nDAILY ({cfg['etf']} {u.index[0].year}-{u.index[-1].year}); "
              f"halves: first = before {cfg['split_daily']}, second = from it")
        print(dd.reindex(columns=[c for c in cols if c in dd.columns])
              .round(2).to_string(index=False))
        print(f"\n  random-sign momentum versions consistent in both halves: "
              f"{luck} of 18")
        di, _ = intraday(sym, cfg)
        print(f"\nINTRADAY (The5ers {sym} hourly, 2020-2026); halves: first = "
              f"2020-22, second = 2023-26")
        print(di.reindex(columns=[c for c in cols if c in di.columns])
              .round(2).to_string(index=False))
        cons = pd.concat([dd, di])
        cons = cons[cons["consistent"] == "YES"]
        print(f"\n  CONSISTENT in both halves for {sym}:")
        print(cons[["test", "bp_2005_15", "t_2005_15", "bp_2016_26",
                    "t_2016_26"]].round(2).to_string(index=False)
              if len(cons) else "  none")
    print("\n(column names say 2005_15 / 2016_26 for the first / second half "
          "of each section's own split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
