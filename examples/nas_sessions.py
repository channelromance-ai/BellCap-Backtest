"""
Is there a Nasdaq version of the European-morning EUR/USD drift?

The EUR/USD trade exists because the euro tends to weaken while Europe is at
work. The nearest documented effect for US stock indices is the opposite
arrangement: over long samples most of their gain has come OVERNIGHT, while
the New York cash session has contributed roughly nothing.

Fixed before running, on The5ers' own NAS100 hourly bars (genuinely hourly
only from mid-2020; before that the server holds one bar a day):

  A  New York session: long 10:00 -> 16:00 New York.
  B  Overnight: long 23:00 New York (22:00 Winnipeg) -> 09:00 New York,
     before the cash open.
  C  Overnight including the open: 23:00 -> 10:00 New York.

B and C sit inside one The5ers trading day (it resets at 17:00 New York), so
no index swap is charged. Cost is The5ers' measured 2.0 points a round trip
and no commission. Every hour of the day is profiled first, as the placebo,
and Dukascopy's independent minute data checks the same days from 2023.
If one survives, stops of 0.5% and 1.0% are tried and the evaluation is
replayed exactly as for EUR/USD.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

COST_PTS = 2.0


def load():
    a = pd.read_parquet("data/the5ers_idx_h1.parquet")
    n = a[a["sym"] == "NAS100"].copy()
    n["ny"] = n["server"] - pd.Timedelta(hours=7)
    n = n.set_index("ny").sort_index()[["open", "high", "low", "close"]]
    # Keep only the stretch where the bars are really hourly.
    per_day = n.groupby(n.index.normalize()).size()
    first = per_day[per_day >= 18].index[0]
    return n[n.index >= first]


def window(n, entry_h, exit_h, stop_pct=None):
    """Long from the open of entry_h to the open of exit_h, weekday mornings."""
    rows = []
    days = pd.bdate_range(n.index[0].normalize() + pd.Timedelta(days=1),
                          n.index[-1].normalize())
    for day in days:
        start = day + pd.Timedelta(hours=entry_h) - (
            pd.Timedelta(days=1) if entry_h > exit_h else pd.Timedelta(0))
        idx = pd.date_range(start, day + pd.Timedelta(hours=exit_h - 1),
                            freq="h")
        if not idx.isin(n.index).all():
            continue
        b = n.loc[idx]
        entry = b["open"].iloc[0]
        px, why = b["close"].iloc[-1], "time"
        if stop_pct:
            stop = entry * (1 - stop_pct / 100)
            for o, lo in zip(b["open"], b["low"], strict=True):
                if lo <= stop:
                    px, why = min(stop, o), "stop"
                    break
        rows.append(dict(day=day, entry=entry, pts=px - entry - COST_PTS,
                         bp=1e4 * (px - entry - COST_PTS) / entry, why=why))
    return pd.DataFrame(rows)


def summ(t, col="bp"):
    x = t[col].to_numpy()
    return dict(n=len(x), up_pct=100 * (x > 0).mean(), avg_bp=x.mean(),
                t=x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> int:
    pd.set_option("display.width", 220)
    n = load()
    print(f"hourly NAS100 from {n.index[0].date()} to {n.index[-1].date()}, "
          f"price now {n['close'].iloc[-1]:,.0f}")

    print("\n" + "=" * 100)
    print("EVERY HOUR: average move of one hourly bar, bp (New York hour ->"
          " Winnipeg is one hour earlier)")
    print("=" * 100)
    x = n.copy()
    x["hr"] = x.index.hour
    x["bp"] = 1e4 * (x["close"] / x["open"] - 1)
    x = x[x.index.weekday < 5]
    prof = x.groupby("hr")["bp"].agg(["count", "mean", "std"])
    prof["t"] = prof["mean"] / (prof["std"] / np.sqrt(prof["count"]))
    prof["share_up"] = x.groupby("hr")["bp"].apply(lambda s: 100 * (s > 0)
                                                   .mean())
    print(prof[["count", "mean", "t", "share_up"]].round(2).T.to_string())
    cash = prof.loc[10:15, "mean"].sum()
    night = prof.loc[[h for h in prof.index if h >= 18 or h <= 8],
                     "mean"].sum()
    print(f"\n  sum of average hourly moves: New York session 10:00-16:00 "
          f"{cash:+.1f}bp, 18:00-09:00 {night:+.1f}bp")

    print("\n" + "=" * 100)
    print("THE THREE PRE-SET TRADES, no stop, after 2 points of cost")
    print("=" * 100)
    rows = []
    res = {}
    for name, eh, xh in (("A  NY session 10:00-16:00", 10, 16),
                         ("B  overnight 23:00-09:00", 23, 9),
                         ("C  overnight 23:00-10:00", 23, 10)):
        t = window(n, eh, xh)
        res[name] = t
        for yr, g in list(t.groupby(t["day"].dt.year)) + [("ALL", t)]:
            rows.append(dict(trade=name, year=yr, **summ(g),
                             avg_pts=g["pts"].mean()))
    d = pd.DataFrame(rows)
    print(d.round(2).to_string(index=False))

    print("\n  Dukascopy minutes, same mornings (independent source):")
    try:
        import examples.zone_retest as ZR
        m1 = ZR.load("NAS100").tz_localize(None)
        for name, (eh, xh) in (("B", (23, 9)), ("C", (23, 10)),
                               ("A", (10, 16))):
            out = []
            for day in pd.bdate_range(m1.index[0].normalize()
                                      + pd.Timedelta(days=1),
                                      m1.index[-1].normalize()):
                s = day + pd.Timedelta(hours=eh) - (
                    pd.Timedelta(days=1) if eh > xh else pd.Timedelta(0))
                e = day + pd.Timedelta(hours=xh)
                seg = m1.loc[s:e - pd.Timedelta(minutes=1)]
                if len(seg) < 30 or seg.index[0] - s > pd.Timedelta(
                        minutes=5):
                    continue
                out.append((day, 1e4 * (seg["c"].iloc[-1] - seg["o"].iloc[0]
                                        - COST_PTS) / seg["o"].iloc[0]))
            dk = pd.DataFrame(out, columns=["day", "duka"]).set_index("day")
            key = [k for k in res if k.startswith(name)][0]
            t5 = res[key].set_index("day")["bp"]
            both = dk.join(t5, how="inner")
            print(f"    {key}: {len(both)} days, correlation "
                  f"{both['duka'].corr(both['bp']):.3f}, average Dukascopy "
                  f"{both['duka'].mean():+.2f}bp vs The5ers "
                  f"{both['bp'].mean():+.2f}bp")
    except Exception as exc:                            # noqa: BLE001
        print(f"    skipped: {exc}")

    print("\n" + "=" * 100)
    print("WITH A STOP (overnight versions)")
    print("=" * 100)
    rows = []
    for name, eh, xh in (("B  23:00-09:00", 23, 9), ("C  23:00-10:00", 23, 10)):
        for sp in (0.5, 1.0):
            t = window(n, eh, xh, sp)
            s = summ(t)
            r = t["bp"] / (sp * 100)
            rows.append(dict(trade=name, stop_pct=sp, **s,
                             avg_r=r.mean(),
                             stopped_pct=100 * (t["why"] == "stop").mean(),
                             yrs_positive=f"{int((t.groupby(t['day'].dt.year)['bp'].mean() > 0).sum())}"
                                          f"/{t['day'].dt.year.nunique()}"))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
