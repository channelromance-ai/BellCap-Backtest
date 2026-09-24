"""
Forced flows: money that has to move at a known time, whatever the chart says.

Everything before this looked for patterns in price itself and found coin
tosses. These three start from the other end -- a participant who MUST trade
at a predictable moment for reasons that have nothing to do with price -- and
ask whether that pressure leaves a mark. Each rule was written down before
any of it was run, on The5ers' own hourly history back to 2000.

1  Gotobi (USD/JPY). Japanese firms settle on the 5th, 10th, 15th, 20th,
   25th and last day of the month (the previous business day when that falls
   on a weekend). Importers buy dollars into the Tokyo fix at 09:55 JST.
   Long USD/JPY from 08:00 to 10:00 JST on those days; every other weekday
   in the same window is the comparison.

2  Month-end hedge rebalancing. Foreign holders of US stocks hedge the dollar
   exposure. If US stocks beat foreign stocks over the month, the hedge is too
   small and they sell dollars at the month-end London 4pm fix; if US stocks
   lagged, they buy. Signal: SPY minus EFA from the last month-end close to
   the close two trading days before month-end (known well in advance).
   Trade the three hours before the 16:00 London fix on the last business
   day, against the dollar or with it, on all six pairs. Pairs are averaged
   each month so one month counts once.

3  Weekend gap. The market reopens on Sunday in Wellington with almost
   nobody trading. At 19:00 New York Sunday, once spreads are normal, if
   price is still 5bp or more from Friday's close, trade back toward that
   close. Exit on touching it, otherwise at 02:00 New York.

Passing requires, fixed in advance: positive after The5ers' measured costs
over 2000-2026 with t above 2, positive in at least three of the four eras,
and positive in 2021-26.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PANEL = "data/the5ers_h1.parquet"
NY, UTC, LDN, TKY = ("America/New_York", "UTC", "Europe/London",
                     "Asia/Tokyo")
COST_BP = {"EURUSD": 0.392, "GBPUSD": 0.342, "USDCHF": 0.951,
           "USDJPY": 0.434, "AUDUSD": 0.642, "NZDUSD": 1.462}
# +1 when the pair rises as the dollar falls.
USD_WEAK = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1,
            "USDJPY": -1, "USDCHF": -1}
ERAS = [("2000-07", 2000, 2008), ("2008-14", 2008, 2015),
        ("2015-20", 2015, 2021), ("2021-26", 2021, 2027)]


def load():
    """Hourly bars indexed by the UTC moment each bar starts."""
    a = pd.read_parquet(PANEL)
    ny = (a["server"] - pd.Timedelta(hours=7)).dt.tz_localize(
        NY, ambiguous="NaT", nonexistent="NaT")
    a["utc"] = ny.dt.tz_convert(UTC)
    a = a.dropna(subset=["utc"])
    a = a[a["utc"] >= pd.Timestamp("2000-01-01", tz=UTC)]
    return {s: g.set_index("utc").sort_index()[["open", "high", "low",
                                                 "close"]]
            for s, g in a.groupby("sym")}


def era(ts):
    y = ts.year
    return next(n for n, a, b in ERAS if a <= y < b)


def window_ret(bars, start_utc, end_utc):
    """Open of the bar starting at start, close of the bar ending at end."""
    last = end_utc - pd.Timedelta(hours=1)
    if start_utc not in bars.index or last not in bars.index:
        return np.nan
    return bars.at[last, "close"] / bars.at[start_utc, "open"] - 1


# ---------------------------------------------------------------- gotobi

def gotobi_days(first, last):
    days = pd.bdate_range(first, last)
    marks = set()
    for m in pd.period_range(first, last, freq="M"):
        month_days = days[(days.year == m.year) & (days.month == m.month)]
        if len(month_days) == 0:
            continue
        targets = [5, 10, 15, 20, 25, m.days_in_month]
        for t in targets:
            d = pd.Timestamp(m.year, m.month, min(t, m.days_in_month))
            # A weekend settlement moves to the business day before it.
            prior = month_days[month_days <= d]
            if len(prior):
                marks.add(prior[-1])
    return marks


def test_gotobi(bars):
    b = bars["USDJPY"]
    first, last = b.index[0].tz_convert(TKY).date(), \
        b.index[-1].tz_convert(TKY).date()
    marks = gotobi_days(first, last)
    rows = []
    for d in pd.bdate_range(first, last):
        start = pd.Timestamp(d.date()).tz_localize(TKY) + pd.Timedelta(
            hours=8)
        r = window_ret(b, start.tz_convert(UTC),
                       (start + pd.Timedelta(hours=2)).tz_convert(UTC))
        if np.isfinite(r):
            rows.append(dict(day=d, gotobi=d in marks,
                             gross_bp=1e4 * r, cost_bp=COST_BP["USDJPY"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- month end

def test_month_end(bars):
    u = pd.read_parquet("data/universe_daily.parquet")[["SPY", "EFA"]]
    u = u.dropna()
    rows = []
    for _, g in u.groupby(u.index.to_period("M")):
        if len(g) < 5:
            continue
        prev = u[u.index < g.index[0]]
        if prev.empty:
            continue
        base = prev.iloc[-1]
        known = g.iloc[-3]                   # two trading days before the end
        us_minus_foreign = (known["SPY"] / base["SPY"]) - \
            (known["EFA"] / base["EFA"])
        last_day = g.index[-1]
        fix = pd.Timestamp(last_day.date()).tz_localize(LDN) + \
            pd.Timedelta(hours=16)
        end, start = fix.tz_convert(UTC), (fix - pd.Timedelta(hours=3)) \
            .tz_convert(UTC)
        # Stocks outperformed -> hedgers sell dollars -> dollar weakens.
        want = 1 if us_minus_foreign > 0 else -1
        per_pair = []
        for s, sign in USD_WEAK.items():
            r = window_ret(bars[s], start, end)
            if np.isfinite(r):
                per_pair.append((1e4 * want * sign * r, COST_BP[s]))
        if per_pair:
            x = np.array(per_pair)
            rows.append(dict(day=last_day, signal_pct=100 * us_minus_foreign,
                             gross_bp=x[:, 0].mean(), cost_bp=x[:, 1].mean(),
                             pairs=len(x)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- weekend gap

def test_weekend_gap(bars, min_gap_bp=5.0):
    rows = []
    for s, b in bars.items():
        ny_idx = b.index.tz_convert(NY)
        fri_close_bars = b[(ny_idx.weekday == 4) & (ny_idx.hour == 16)]
        for t, fr in fri_close_bars.iterrows():
            ref = fr["close"]
            # Wall-clock times, built naive then localised: the clocks
            # change on a Sunday morning, so adding hours to a zone-aware
            # midnight would land an hour off twice a year.
            sun = pd.Timestamp(t.tz_convert(NY).date()) + pd.Timedelta(days=2)
            entry_t = (sun + pd.Timedelta(hours=19)).tz_localize(NY) \
                .tz_convert(UTC)
            exit_t = (sun + pd.Timedelta(hours=26)).tz_localize(NY) \
                .tz_convert(UTC)
            if entry_t not in b.index:
                continue
            entry = b.at[entry_t, "open"]
            gap_bp = 1e4 * (entry / ref - 1)
            if abs(gap_bp) < min_gap_bp:
                continue
            d = -np.sign(gap_bp)                 # back toward Friday
            path = b.loc[entry_t:exit_t - pd.Timedelta(hours=1)]
            if len(path) < 3:
                continue
            # A long is waiting for price to RISE back to Friday's close, a
            # short for it to FALL back. The first version of this line had
            # the two swapped, which is true on every bar -- price is
            # already on that side -- and scored 4,202 trades out of 4,202
            # as winners.
            hit = (path["high"] >= ref) if d > 0 else (path["low"] <= ref)
            exit_px = ref if hit.any() else path["close"].iloc[-1]
            rows.append(dict(sym=s, day=sun, gap_bp=gap_bp, filled=hit.any(),
                             gross_bp=1e4 * d * (exit_px / entry - 1),
                             cost_bp=COST_BP[s]))
    return pd.DataFrame(rows)


def summ(g):
    x = g["gross_bp"].to_numpy()
    net = x - g["cost_bp"].to_numpy()
    n = len(x)
    if n < 10:
        return pd.Series(dict(n=n))
    return pd.Series(dict(
        n=n, right_way_pct=100 * (x > 0).mean(), gross_bp=x.mean(),
        cost_bp=g["cost_bp"].mean(), net_bp=net.mean(),
        t_net=net.mean() / (net.std(ddof=1) / np.sqrt(n))))


def verdict(name, g):
    full = summ(g)
    eras = g.groupby(g["day"].map(era)).apply(summ)
    pos_eras = int((eras["net_bp"] > 0).sum())
    recent = eras.loc["2021-26", "net_bp"] if "2021-26" in eras.index \
        else np.nan
    ok = full["t_net"] > 2 and pos_eras >= 3 and recent > 0
    print(f"\n  VERDICT {name}: t={full['t_net']:+.2f}, positive eras "
          f"{pos_eras}/4, 2021-26 net {recent:+.2f}bp -> "
          f"{'PASSES' if ok else 'fails'}")
    return ok


def main() -> int:
    pd.set_option("display.width", 210)
    bars = load()

    print("=" * 96)
    print("1. GOTOBI: long USD/JPY 08:00-10:00 Tokyo (bp; net = after cost)")
    print("=" * 96)
    g = test_gotobi(bars)
    g["era"] = g["day"].map(era)
    print(g.groupby("gotobi").apply(summ).round(3).to_string())
    print(g.groupby(["gotobi", "era"]).apply(summ).round(3).unstack(0)
          .to_string())
    verdict("gotobi", g[g["gotobi"]])

    print("\n" + "=" * 96)
    print("2. MONTH-END HEDGE REBALANCING, 3h into the London 4pm fix")
    print("=" * 96)
    m = test_month_end(bars)
    m["era"] = m["day"].map(era)
    print(summ(m).round(3).to_string())
    print(m.groupby("era").apply(summ).round(3).to_string())
    big = m[m["signal_pct"].abs() > m["signal_pct"].abs().median()]
    print("\n  months with a larger-than-usual US vs foreign gap:")
    print(summ(big).round(3).to_string())
    verdict("month-end", m)

    print("\n" + "=" * 96)
    print("3. WEEKEND GAP, back toward Friday's close from 19:00 Sunday NY")
    print("=" * 96)
    w = test_weekend_gap(bars)
    w["era"] = w["day"].map(era)
    print(summ(w).round(3).to_string())
    print(f"  gaps reaching Friday's close by 02:00: {w['filled'].mean():.1%}")
    print(w.groupby("era").apply(summ).round(3).to_string())
    print(w.groupby("sym").apply(summ).round(3).to_string())
    verdict("weekend gap", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
