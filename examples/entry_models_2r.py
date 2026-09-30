"""
Entry models under a fixed risk system: 0.5% a trade, 2:1, 3 trades a day,
3% maximum drawdown. Target: profit factor 1.25 or better.

The arithmetic first. At 2:1 a profit factor of 1.25 needs a 38.5% win rate
before costs; a coin toss wins 33.3%. And 3% at 0.5% a trade is six full
losses from the high-water mark -- a run that a 38.5% system meets often.

Four models, fixed before running. Every one: a 2:1 bracket, the stop set by
the model, time exit if neither side is hit, the stop taken first when one
hourly bar holds both, The5ers' spread at the entry hour plus $4/lot. Hours
are New York (Winnipeg is one earlier). ADR20 = average high-low range of the
previous 20 New York days, known before the entry.

  M1  EUR/USD European-morning short. Short at 03:00, stop 0.25 x ADR20,
      exit 08:00. Variant M1b enters 23:00 the evening before.
  M2  Asian-range breakout, 8 majors. Range = 19:00-02:00 high/low. From
      02:00 to 08:00, the first hourly close outside it is the signal;
      entry next open, stop at the range midpoint, exit 12:00.
  M3  20-day trend, 8 majors. At 03:00, trade the direction of the last 20
      days, stop 0.3 x ADR20, exit 16:00.
  M4  EUR/USD New York-morning long. Long 08:00, stop 0.25 x ADR20, exit
      12:00. SEEN BEFORE: this window was visible in eur_morning.py's
      all-hours placebo, so its result carries less weight.

Direction and settings are never tuned here; the split is only for honesty:
2000-2012 and 2013-2026 are reported separately, and 2021-26 on its own.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD",
          "EURJPY", "GBPJPY"]
COMMISSION_BP = 0.4
RR = 2.0
MAX_PER_DAY = 3
RISK_PCT = 0.5
MAX_DD_PCT = 3.0


def load():
    a = pd.read_parquet("data/the5ers_h1_all.parquet")
    a = a[a["sym"].isin(set(MAJORS))].copy()
    a["ny"] = a["server"] - pd.Timedelta(hours=7)
    a = a[(a["ny"] >= "2000-01-01") & (a["ny"].dt.weekday < 5)]
    sp = pd.read_csv("data/the5ers_spread_by_hour.csv")
    spread = {(r.sym, r.ny_hour): r.spread_bp for r in sp.itertuples()}
    bars = {s: g.set_index("ny").sort_index()[["open", "high", "low",
                                                 "close"]]
            for s, g in a.groupby("sym")}
    return bars, spread


def adr20(b):
    """Average NY-day high-low range of the previous 20 days, by day."""
    day = b.index.normalize()
    rng = b.groupby(day).agg(h=("high", "max"), l=("low", "min"))
    return (rng["h"] - rng["l"]).rolling(20).mean().shift(1)


def resolve(bars, d, entry, stop, target):
    """Walk hourly bars; stop first when a bar holds both. R before cost."""
    risk = abs(entry - stop)
    for o, hi, lo in zip(bars["open"], bars["high"], bars["low"],
                         strict=True):
        if (d > 0 and lo <= stop) or (d < 0 and hi >= stop):
            px = min(stop, o) if d > 0 else max(stop, o)
            return d * (px - entry) / risk, "stop"
        if (d > 0 and hi >= target) or (d < 0 and lo <= target):
            px = max(target, o) if d > 0 else min(target, o)
            return d * (px - entry) / risk, "target"
    return d * (bars["close"].iloc[-1] - entry) / risk, "time"


def trade(model, sym, b, day, t_entry, t_exit, d, stop_dist, spread):
    seg = b.loc[t_entry:t_exit - pd.Timedelta(hours=1)]
    if len(seg) < 1 or seg.index[0] != t_entry:
        return None
    entry = seg["open"].iloc[0]
    if not np.isfinite(stop_dist) or stop_dist <= 0:
        return None
    stop = entry - d * stop_dist
    target = entry + d * RR * stop_dist
    r, why = resolve(seg, d, entry, stop, target)
    stop_bp = 1e4 * stop_dist / entry
    cost_bp = spread.get((sym, t_entry.hour), 1.0) + COMMISSION_BP
    return dict(model=model, sym=sym, day=day, entry_ts=t_entry, d=d,
                stop_bp=stop_bp, r=r - cost_bp / stop_bp, why=why)


def run_models(bars, spread):
    rows = []
    H = pd.Timedelta(hours=1)
    for sym, b in bars.items():
        adr = adr20(b)
        days = adr.dropna().index
        close_by_day = b["close"].groupby(b.index.normalize()).last()
        for day in days:
            a = adr.get(day, np.nan)
            # ---- M1 / M1b / M4: EUR/USD only
            if sym == "EURUSD":
                for model, t0, t1, d in (
                        ("M1 EUR short 03-08", day + 3 * H, day + 8 * H, -1),
                        ("M1b EUR short 23-08", day - H, day + 8 * H, -1),
                        ("M4 EUR long 08-12 (seen)", day + 8 * H,
                         day + 12 * H, 1)):
                    x = trade(model, sym, b, day, t0, t1, d, 0.25 * a,
                              spread)
                    if x:
                        rows.append(x)
            # ---- M3: 20-day trend at 03:00
            prev = close_by_day[close_by_day.index < day]
            if len(prev) > 21:
                d = 1 if prev.iloc[-1] > prev.iloc[-21] else -1
                x = trade("M3 20d trend 03-16", sym, b, day, day + 3 * H,
                          day + 16 * H, d, 0.3 * a, spread)
                if x:
                    rows.append(x)
            # ---- M2: Asian range breakout
            asia = b.loc[day - 5 * H:day + 1 * H]      # 19:00 prev .. 01:00
            if len(asia) >= 6:
                hi, lo = asia["high"].max(), asia["low"].min()
                mid = (hi + lo) / 2
                london = b.loc[day + 2 * H:day + 7 * H]
                for t, c in london["close"].items():
                    d = 1 if c > hi else (-1 if c < lo else 0)
                    if d == 0:
                        continue
                    t_entry = t + H
                    if t_entry not in b.index:
                        break
                    dist = abs(b.at[t_entry, "open"] - mid)
                    # A stop inside the spread cannot be traded.
                    if 1e4 * dist / b.at[t_entry, "open"] < 5:
                        break
                    if (d > 0 and b.at[t_entry, "open"] <= mid) or \
                            (d < 0 and b.at[t_entry, "open"] >= mid):
                        break
                    x = trade("M2 Asian breakout", sym, b, day, t_entry,
                              day + 12 * H, d, dist, spread)
                    if x:
                        rows.append(x)
                    break
    return pd.DataFrame(rows)


def cap_per_day(t):
    """At most three trades a day across the model: the earliest three."""
    return (t.sort_values("entry_ts").groupby(["model", "day"])
            .head(MAX_PER_DAY))


def pf(r):
    w, lo = r[r > 0].sum(), -r[r <= 0].sum()
    return w / lo if lo > 0 else np.inf


def summ(g):
    r = g["r"].to_numpy()
    n = len(r)
    return pd.Series(dict(
        trades=n, per_year=n / max(1, g["day"].dt.year.nunique()),
        win_pct=100 * (r > 0).mean(), avg_r=r.mean(), pf=pf(r),
        t=r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan,
        cost_r=np.nan))


def drawdown_odds(r, n_trades=60, sims=20000, seed=1):
    """Chance of a 3% drawdown from the high within n_trades, at 0.5% a
    trade, resampling this model's own trade results in blocks of 5."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r)
    hits = 0
    for _ in range(sims):
        starts = rng.integers(0, len(r) - 5, size=n_trades // 5 + 1)
        seq = np.concatenate([r[s:s + 5] for s in starts])[:n_trades]
        eq = np.cumsum(RISK_PCT * seq)
        dd = np.maximum.accumulate(np.r_[0, eq])[1:] - eq
        hits += dd.max() >= MAX_DD_PCT
    return 100 * hits / sims


def main() -> int:
    pd.set_option("display.width", 230)
    bars, spread = load()
    t = cap_per_day(run_models(bars, spread))
    t["era"] = np.where(t["day"].dt.year < 2013, "2000-12", "2013-26")
    print("=" * 110)
    print("PROFIT FACTOR BY MODEL (after cost; 2:1; stop first on ties)")
    print("=" * 110)
    parts = []
    for era, g in list(t.groupby("era")) + [("2021-26",
                                             t[t["day"].dt.year >= 2021])]:
        s = g.groupby("model").apply(summ)
        s["era"] = era
        parts.append(s)
    out = pd.concat(parts).reset_index().set_index(["model", "era"])
    print(out.drop(columns="cost_r").sort_index().round(3).to_string())

    print("\n  win rate a 2:1 needs for PF 1.25: 38.5% (before costs); "
          "coin toss 33.3%")
    print("\n" + "=" * 110)
    print("3% DRAWDOWN: chance of hitting it within 60 trades, at 0.5% a trade,"
          " using each model's own 2013-26 trades")
    print("=" * 110)
    hold = t[t["era"] == "2013-26"].sort_values("entry_ts")
    for m, g in hold.groupby("model"):
        print(f"  {m:<28} PF {pf(g['r'].to_numpy()):.2f}   chance of a 3% "
              f"drawdown in 60 trades: {drawdown_odds(g['r']):.0f}%")
    # A model that meets PF 1.25 exactly, for reference.
    rng = np.random.default_rng(2)
    ref = np.where(rng.random(20000) < 0.385, 2.0, -1.0)
    print(f"  {'reference: exactly PF 1.25':<28} PF {pf(ref):.2f}   chance of "
          f"a 3% drawdown in 60 trades: {drawdown_odds(ref):.0f}%")
    t.to_csv("data/entry_models_2r_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
