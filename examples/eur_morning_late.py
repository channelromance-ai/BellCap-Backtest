"""
The European-morning EUR/USD short, entered the night before.

The user can place a trade no later than about 22:00 Winnipeg, which is 23:00
New York -- four hours before the 03:00 window the tested trade uses. So the
position has to be opened early and held through the quiet Asian hours, with
a wider stop and less risk. Fixed before running:

  Entry   23:00 New York (22:00 Winnipeg), Sunday to Thursday nights.
  Exit    08:00 New York (07:00 Winnipeg). Variant: 09:00 New York (08:00
          Winnipeg), for closing it on waking -- the all-hours placebo in
          eur_morning.py says EUR/USD tends to RISE once New York opens, so
          this is expected to cost something.
  Stop    30, 40 or 50 pips above entry.
  Risk    0.75%, 1% or 1.5% of the $5,000 starting balance.

Against: the original 03:00 short with a 20-pip stop at 1.5%, and the same
late trade with a random direction each night. Replayed on real sequences
for The5ers one-step from +4.0% and +4.5%, exactly as in eur_morning.py.
The5ers' day resets at 17:00 New York, so a 23:00-08:00 trade sits inside
one trading day and the daily limit applies to it alone.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.eur_morning as EM                      # noqa: E402

PIP = EM.PIP


def load_all_hours():
    """Hourly EUR/USD in New York time, Sunday evenings included."""
    a = pd.read_parquet("data/the5ers_h1.parquet")
    e = a[a["sym"] == "EURUSD"].copy()
    e["ny"] = e["server"] - pd.Timedelta(hours=7)
    e = e[e["ny"] >= "2000-01-01"]
    return e.set_index("ny").sort_index()[["open", "high", "low", "close"]]


def overnight_trades(h, entry_h=23, exit_h=8, stop_pips=40.0,
                     direction=None, rng=None):
    """
    One trade per weekday morning D: in at entry_h on D-1 (or on D if
    entry_h < exit_h), out at the open of exit_h on D. Stop checked first
    inside every hour.
    """
    rows = []
    mornings = pd.bdate_range(h.index[0].normalize() + pd.Timedelta(days=1),
                              h.index[-1].normalize())
    for day in mornings:
        start = day + pd.Timedelta(hours=entry_h) - (
            pd.Timedelta(days=1) if entry_h > exit_h else pd.Timedelta(0))
        idx = pd.date_range(start, day + pd.Timedelta(hours=exit_h - 1),
                            freq="h")
        # Friday-night entries do not exist: the week ends at 17:00 Friday.
        if start.weekday() == 5 or not idx.isin(h.index).all():
            continue
        bars = h.loc[idx]
        d = -1
        if direction == "random":
            d = 1 if rng.random() < 0.5 else -1
        entry = bars["open"].iloc[0]
        stop = entry - d * stop_pips * PIP
        px, why = bars["close"].iloc[-1], "time"
        for o, hi, lo in zip(bars["open"], bars["high"], bars["low"],
                             strict=True):
            if (d < 0 and hi >= stop) or (d > 0 and lo <= stop):
                px = max(stop, o) if d < 0 else min(stop, o)
                why = "stop"
                break
        rows.append(dict(day=day, pips=d * (px - entry) / PIP, why=why))
    return pd.DataFrame(rows)


def describe(t, stop, label):
    c = EM.cost_pips()
    t = t.assign(r=(t["pips"] - c) / stop,
                 era=pd.cut(t["day"].dt.year, [1999, 2007, 2014, 2020, 2026],
                            labels=["2000-07", "2008-14", "2015-20",
                                    "2021-26"]))
    rows = []
    for era, g in list(t.groupby("era")) + [("ALL", t)]:
        r = g["r"].to_numpy()
        rows.append(dict(system=label, era=era, n=len(r),
                         win_pct=100 * (r > 0).mean(),
                         avg_pips=g["pips"].mean() - c, avg_r=r.mean(),
                         t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                         stopped_pct=100 * (g["why"] == "stop").mean()))
    return rows


def replay_table(seqsets, risks, stops):
    rows = []
    for start_pct in (4.0, 4.5):
        start_bal = EM.ACCOUNT * (1 + start_pct / 100)
        for label, seqs in seqsets.items():
            stop = stops[label]
            for risk in risks:
                for era_name, y0 in (("2000-26", 2000), ("2021-26", 2021)):
                    res = []
                    for s in seqs:
                        s = s.reset_index(drop=True)
                        starts = s.index[(s["day"].dt.year >= y0)
                                         & (s.index < len(s) - EM.DAYS_LEFT)]
                        for i0 in starts[::5]:
                            seq = s["pips"].iloc[i0:i0 + EM.DAYS_LEFT]
                            res.append(EM.replay(seq.to_numpy(), start_bal,
                                                 risk, stop)[0])
                    res = pd.Series(res)
                    rows.append(dict(start=f"+{start_pct}%", system=label,
                                     risk_pct=risk,
                                     lots=EM.lots_for(EM.ACCOUNT * risk / 100,
                                                      stop),
                                     starts=era_name,
                                     pass_pct=100 * (res == "pass").mean(),
                                     fail_pct=100 * (res == "fail").mean(),
                                     open_pct=100 * (res == "open").mean()))
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 230)
    h = load_all_hours()
    rng = np.random.default_rng(5)

    print("=" * 100)
    print("THE TRADE ON ITS OWN (after 0.45 pips of cost per trade)")
    print("=" * 100)
    sets, stops, rows = {}, {}, []
    for stop in (30.0, 40.0, 50.0):
        for exit_h in (8, 9):
            label = f"22:00-{exit_h - 1:02d}:00 Wpg, {stop:.0f} pip stop"
            t = overnight_trades(h, 23, exit_h, stop)
            rows += describe(t, stop, label)
            if exit_h == 8:
                sets[label], stops[label] = [t], stop
    ref = overnight_trades(h, 3, 8, 20.0)
    rows += describe(ref, 20.0, "02:00-07:00 Wpg, 20 pip stop (original)")
    d = pd.DataFrame(rows)
    print(d.round(3).to_string(index=False))

    print("\n  where the overnight hold makes and loses its money "
          "(short, no stop, pips per hour, New York hour):")
    x = h.copy()
    x["hr"] = x.index.hour
    x["short_pips"] = (x["open"] - x["close"]) / PIP
    x["yr"] = x.index.year
    hours = [23, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    prof = pd.DataFrame({
        "2000-26": x[x["hr"].isin(hours)].groupby("hr")["short_pips"].mean(),
        "2021-26": x[x["hr"].isin(hours) & (x["yr"] >= 2021)]
        .groupby("hr")["short_pips"].mean()}).loc[hours]
    prof.index = [f"{(hh - 1) % 24:02d}:00 Wpg" for hh in hours]
    print(prof.round(2).T.to_string())

    print("\n" + "=" * 100)
    print("THE5ERS ONE-STEP, $5,000, replayed on real mornings")
    print("=" * 100)
    for label in list(sets):
        sets[f"{label} RANDOM"] = [
            overnight_trades(h, 23, 8, stops[label], "random", rng)
            for _ in range(6)]
        stops[f"{label} RANDOM"] = stops[label]
    sets["02:00-07:00 Wpg, 20 pip (original)"] = [ref]
    stops["02:00-07:00 Wpg, 20 pip (original)"] = 20.0
    tab = replay_table(sets, (0.75, 1.0, 1.5), stops)
    print(tab.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
