"""
Gold time-of-day flows: is there a gold version of the EUR/USD morning?

Zones are dropped (gold_lab.py). The one confirmed edge in this repo is a
forced flow at a predictable time, so the replacement candidates are gold's
own forced flows. Fixed before running, with the mechanism for each:

  H1 Asia buying     LONG 20:00 -> 02:00 New York. China and India are the
                     largest physical buyers; the Shanghai Gold Exchange
                     opens around then, and Asian demand is widely said to
                     lift gold in those hours.
  H2 PM fix          SHORT 13:00 -> 15:00 London. The 15:00 London fix set
                     the world benchmark; prices were documented drifting
                     DOWN into it, and banks were fined for manipulating it.
                     The fix became an electronic auction in March 2015, so
                     a real effect should be strong before 2015 and weaker
                     after -- a built-in test.
  H3 AM fix          SHORT 09:00 -> 11:00 London, same reasoning for the
                     10:30 fix.

Eras: 2005-14, 2015-20, 2021-26. Every hour of the day is profiled first as
the placebo, in London time. Costs: The5ers' measured gold spread at the
hour plus $8.22 a lot. A survivor must be positive after cost in every era,
clearly beyond luck over the whole period, and stand out from the other
hours of the day.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

LDN, NY = "Europe/London", "America/New_York"
COMM_BP = 0.19                     # $8.22 a lot on ~$4,400 x 100 oz, in bp


def load():
    a = pd.read_parquet("data/the5ers_h1_all.parquet")
    g = a[a["sym"] == "XAUUSD"].copy()
    ny = (g["server"] - pd.Timedelta(hours=7)).dt.tz_localize(
        NY, ambiguous="NaT", nonexistent="NaT")
    g["utc"] = ny.dt.tz_convert("UTC")
    g = g.dropna(subset=["utc"])
    g = g[g["utc"] >= pd.Timestamp("2005-01-01", tz="UTC")]
    g = g.set_index("utc").sort_index()[["open", "high", "low", "close"]]
    # Genuinely hourly days only.
    per_day = g.groupby(g.index.normalize()).size()
    good = per_day[per_day >= 18].index
    return g[g.index.normalize().isin(good)]


def spread_bp_by_ny_hour():
    sp = pd.read_csv("data/the5ers_spread_by_hour.csv")
    sp = sp[sp["sym"] == "XAUUSD"]
    return dict(zip(sp["ny_hour"], sp["spread_bp"], strict=True))


def era(ts):
    y = ts.year
    return "2005-14" if y < 2015 else ("2015-20" if y < 2021 else "2021-26")


def window(g, tz, start_h, end_h, d, cost_by_ny_hour):
    """One trade per weekday: open at start_h (local tz), close at end_h."""
    loc = g.index.tz_convert(tz)
    o = pd.Series(g["open"].to_numpy(), index=loc)
    days = pd.DatetimeIndex(sorted(set(loc.normalize())))
    rows = []
    for day in days:
        if day.weekday() >= 5:
            continue
        s = day + pd.Timedelta(hours=start_h)
        e = day + pd.Timedelta(hours=end_h + (24 if end_h <= start_h else 0))
        if s not in o.index or e not in o.index or e.weekday() >= 5:
            continue
        if start_h > end_h and day.weekday() == 4:
            continue                       # Friday evening runs into Saturday
        gross = d * 1e4 * (o[e] / o[s] - 1)
        ny_h = s.tz_convert(NY).hour
        cost = cost_by_ny_hour.get(ny_h, 1.2) + COMM_BP
        rows.append(dict(day=day.tz_localize(None), gross=gross,
                         net=gross - cost))
    return pd.DataFrame(rows)


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    return dict(n=n, right_pct=100 * (x > 0).mean(), avg_bp=x.mean(),
                t=x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan)


def main() -> int:
    pd.set_option("display.width", 220)
    g = load()
    cost = spread_bp_by_ny_hour()
    print(f"gold hourly bars: {len(g):,}, {g.index[0].date()} to "
          f"{g.index[-1].date()}")

    print("\n" + "=" * 100)
    print("PLACEBO: average move of each London hour (bp, long), by era")
    print("=" * 100)
    x = g.copy()
    x["ldn_hour"] = x.index.tz_convert(LDN).hour
    x["bp"] = 1e4 * (x["close"] / x["open"] - 1)
    x["era"] = [era(t) for t in x.index]
    x = x[x.index.tz_convert(LDN).weekday < 5]
    prof = x.pivot_table(index="era", columns="ldn_hour", values="bp",
                         aggfunc="mean")
    print(prof.round(2).to_string())
    allh = x.groupby("ldn_hour")["bp"].agg(["mean", "std", "count"])
    allh["t"] = allh["mean"] / (allh["std"] / np.sqrt(allh["count"]))
    print("\n  all years, t by London hour:")
    print(allh["t"].round(2).to_frame().T.to_string())
    print(f"\n  gold's average hourly drift over all hours: "
          f"{x['bp'].mean():+.3f}bp (the trend every window inherits)")

    print("\n" + "=" * 100)
    print("THE THREE PRE-SET WINDOWS, after cost")
    print("=" * 100)
    rows = []
    for name, tz, s, e, d in (
            ("H1 Asia buy 20:00-02:00 NY", NY, 20, 2, 1),
            ("H2 PM-fix sell 13:00-15:00 London", LDN, 13, 15, -1),
            ("H3 AM-fix sell 09:00-11:00 London", LDN, 9, 11, -1)):
        t = window(g, tz, s, e, d, cost)
        t["era"] = [era(dd) for dd in t["day"]]
        for er, sub in list(t.groupby("era")) + [("ALL", t)]:
            st = stats(sub["net"])
            rows.append(dict(window=name, era=er, **st,
                             gross_bp=sub["gross"].mean(),
                             yrs_pos=f"{int((sub.groupby(sub['day'].dt.year)['net'].mean() > 0).sum())}"
                                     f"/{sub['day'].dt.year.nunique()}"))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
