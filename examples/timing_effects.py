"""
WHEN to trade rather than WHERE: three documented time-of-day effects.

Every level-based idea in this repo -- consolidation zones, swing highs and
lows, with and against the trend, on one timeframe and three -- won at
coin-toss rates to its stop or target. So this tests effects that published
research says come from the structure of the market itself, measured raw
(no stops) and against cost first, before any trade design is built on them.

Rules, fixed before any result was looked at:

A  Intraday momentum (Gao, Han, Li & Zhou, 2018). The return from the
   previous session's close to 10:00 New York predicts the last half hour.
   Enter at the open of the half hour before the real close, in the
   direction of that morning return; exit at the close. Half-days use their
   real 13:00 close. Secondary, also fixed now: only days where the morning
   move is larger than its 67th percentile over the previous 60 sessions.
   S&P and Nasdaq CFDs; SPY and QQQ as a cross-check.

B  Home-hours depreciation (Ranaldo, 2009; Breedon & Ranaldo, 2013). A
   currency tends to weaken while its own market is open. Windows in New
   York time, chosen so only one home market is open in each:
       03:00-08:00  Europe only   EUR, GBP, CHF weaken
       12:00-17:00  US only       USD weakens
       19:00-02:00  Asia only     JPY, AUD, NZD weaken
   Hold the predicted direction for the whole window. USDCAD is excluded
   because both its currencies share home hours.

C  Pre-FOMC drift (Lucca & Moench, 2015). S&P from 14:00 the day before an
   announcement to 13:55 on the day, against the same window on every other
   day. Only 25 announcements are in this data, so this is descriptive.

Each is judged on the later 40% of each instrument's history.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import examples.zone_retest as ZR                      # noqa: E402
from bcbt import calendars                             # noqa: E402

warnings.filterwarnings("ignore")
ET = "America/New_York"

# FOMC statement days, from federalreserve.gov, within the data.
FOMC = ["2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20",
        "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28",
        "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]

# One spread per round trip, in each instrument's own units. ETFs: a cent.
ETF_COST = 0.01


def load_etf(sym):
    e = pd.read_parquet("data/equity_5min.parquet",
                        filters=[("sym", "==", sym)])
    e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.tz_convert(ET)
    return e.drop(columns=["sym"]).set_index("ts").sort_index()


def price_at(bars, when, field, step):
    """The bar starting exactly at `when`, or NaN if the feed has none."""
    try:
        return float(bars.at[when, field])
    except KeyError:
        return np.nan


def intraday_momentum(sym, bars, step, cost):
    """One row per real NYSE session: morning signal, last-half-hour move."""
    tbl = calendars.sessions()
    days = sorted(d for d in tbl if bars.index[0] <= d <= bars.index[-1])
    rows = []
    prev_close = np.nan
    for day in days:
        o_min, c_min = tbl[day]
        t = lambda m, day=day: day + pd.Timedelta(minutes=m)  # noqa: E731
        close_now = price_at(bars, t(c_min) - step, "c", step)
        at_10 = price_at(bars, t(600) - step, "c", step)
        entry = price_at(bars, t(c_min - 30), "o", step)
        if np.isfinite(prev_close) and np.isfinite(at_10) and \
                np.isfinite(entry) and np.isfinite(close_now):
            sig = at_10 / prev_close - 1
            if sig != 0:
                d = np.sign(sig)
                rows.append(dict(sym=sym, day=day, signal=sig, d=d,
                                 gross_bp=1e4 * d * (close_now / entry - 1),
                                 cost_bp=1e4 * cost / entry))
        if np.isfinite(close_now):
            prev_close = close_now
    df = pd.DataFrame(rows)
    # Secondary: morning move large relative to the PREVIOUS 60 sessions.
    thr = df["signal"].abs().shift(1).rolling(60, min_periods=40).quantile(
        0.67)
    df["big"] = df["signal"].abs() > thr
    return df


WINDOWS = {
    # name: (start hour, end hour, {pair: predicted direction})
    "europe 03-08": (3, 8, {"EURUSD": -1, "GBPUSD": -1, "USDCHF": +1}),
    "us 12-17": (12, 17, {"EURUSD": +1, "GBPUSD": +1, "AUDUSD": +1,
                          "NZDUSD": +1, "USDJPY": -1, "USDCHF": -1}),
    "asia 19-02": (19, 2, {"USDJPY": +1, "AUDUSD": -1, "NZDUSD": -1}),
}


def home_hours(sym, m1):
    rows = []
    cost = ZR.COST_PTS[sym]
    days = pd.DatetimeIndex(sorted(set(m1.index.normalize())))
    for name, (h0, h1, pairs) in WINDOWS.items():
        if sym not in pairs:
            continue
        d = pairs[sym]
        for day in days:
            start = day + pd.Timedelta(hours=h0)
            end = day + pd.Timedelta(hours=h1 + (24 if h1 < h0 else 0))
            # Weekday sessions only: no window that starts or ends in the
            # weekend gap.
            if start.weekday() >= 5 or end.weekday() >= 5:
                continue
            if name.startswith("asia") and start.weekday() == 4:
                continue                       # Friday night runs into Sat
            i0 = m1.index.searchsorted(start)
            i1 = m1.index.searchsorted(end) - 1
            if i1 <= i0 or m1.index[i0] - start > pd.Timedelta(minutes=5) \
                    or end - m1.index[i1] > pd.Timedelta(minutes=6):
                continue
            entry = float(m1["o"].iat[i0])
            exit_ = float(m1["c"].iat[i1])
            rows.append(dict(sym=sym, window=name, day=day, d=d,
                             gross_bp=1e4 * d * (exit_ / entry - 1),
                             cost_bp=1e4 * cost / entry))
    return pd.DataFrame(rows)


def summ(g):
    g_ = g["gross_bp"].to_numpy()
    net = g_ - g["cost_bp"].to_numpy()
    n = len(net)
    if n < 3:
        return pd.Series(dict(n=n))
    return pd.Series(dict(
        n=n, right_way_pct=100 * (g_ > 0).mean(),
        gross_bp=g_.mean(), cost_bp=g["cost_bp"].mean(),
        net_bp=net.mean(), t_net=net.mean() / (net.std(ddof=1) / np.sqrt(n)),
        edge_over_cost=g_.mean() / g["cost_bp"].mean()))


def add_half(df, key="day"):
    out = []
    for _, g in df.groupby("sym"):
        days = np.sort(g[key].unique())
        cut = days[int(len(days) * 0.6)]
        out.append(g.assign(half=np.where(g[key] < cut, "design",
                                          "holdout")))
    return pd.concat(out)


def main() -> int:
    pd.set_option("display.width", 210)

    print("=" * 96)
    print("A. FIRST HALF HOUR -> LAST HALF HOUR   (bp = hundredths of a %)")
    print("=" * 96)
    parts = []
    for s in ZR.INDICES:
        m1 = ZR.load(s)
        parts.append(intraday_momentum(s, m1, pd.Timedelta(minutes=1),
                                       ZR.COST_PTS[s]))
    for s in ("SPY", "QQQ"):
        parts.append(intraday_momentum(s, load_etf(s),
                                       pd.Timedelta(minutes=5), ETF_COST))
    a = add_half(pd.concat(parts, ignore_index=True))
    print(a.groupby(["sym", "half"]).apply(summ).round(3).to_string())
    print("\n  large morning moves only")
    print(a[a["big"]].groupby(["sym", "half"]).apply(summ).round(3)
          .to_string())

    print("\n" + "=" * 96)
    print("B. CURRENCIES WEAKEN IN THEIR OWN HOURS")
    print("=" * 96)
    b = add_half(pd.concat([home_hours(s, ZR.load(s)) for s in ZR.FX
                            if s != "USDCAD"], ignore_index=True))
    print(b.groupby(["window", "half"]).apply(summ).round(3)
          .unstack("half").to_string())
    print("\n  by pair and window, holdout")
    print(b[b["half"] == "holdout"].groupby(["window", "sym"]).apply(summ)
          .round(3).to_string())

    print("\n" + "=" * 96)
    print("C. S&P, 14:00 the day before to 13:55 on the day  (descriptive)")
    print("=" * 96)
    m1 = ZR.load("SPX500")
    tbl = calendars.sessions()
    days = sorted(d for d in tbl if m1.index[0] <= d <= m1.index[-1])
    fomc = {pd.Timestamp(x).tz_localize(ET) for x in FOMC}
    rows = []
    for prev, day in zip(days[:-1], days[1:], strict=True):
        p0 = price_at(m1, prev + pd.Timedelta(hours=14), "o", None)
        p1 = price_at(m1, day + pd.Timedelta(hours=13, minutes=55), "c",
                      None)
        if np.isfinite(p0) and np.isfinite(p1):
            rows.append(dict(sym="SPX500", day=day, d=1,
                             is_fomc=day in fomc,
                             gross_bp=1e4 * (p1 / p0 - 1),
                             cost_bp=1e4 * ZR.COST_PTS["SPX500"] / p0))
    c = pd.DataFrame(rows)
    print(c.groupby("is_fomc").apply(summ).round(3).to_string())
    print(f"  FOMC days found in the data: {int(c['is_fomc'].sum())} "
          f"of {len(FOMC)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
