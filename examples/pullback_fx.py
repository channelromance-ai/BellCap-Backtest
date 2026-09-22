"""
The pullback strategy on the seven major currency pairs.

Run with the busy-ness filter and without it, on every pair, then pooled.

The reason for running it both ways: on the stock indexes the entire result
came from the busy-ness filter, and that filter turned out to be measuring a
number that did not track real trading. The currency tick count tracks real
trading better -- but only about half the bars it calls busy really are. So
if the filter "creates" an edge here too, out of an input that is half noise,
that is evidence the pattern is an artifact of filtering on a noisy measure
rather than anything about markets.

Currencies trade around the clock, so the strategy's New York clock has to be
imposed. Bars are cut to 09:30-16:00 New York, which anchors the day's
average price at the same place the stock version does and leaves the
strategy's own 10:00-15:30 entry window sitting inside it.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

from bcbt import calendars, metrics, store

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "standalone"))
sys.path.insert(0, _ROOT)

import intraday_pullback as ip  # noqa: E402

FX_PARQUET = "data/fx_m1.parquet"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# About one pip round trip on a major, expressed as a share of price. The
# script charges this twice, so it is the one-way figure.
COST_BPS = 0.4

RNG = np.random.default_rng(99)
N_DRAWS = 300


def ny_5min(pair: str) -> pd.DataFrame:
    m1 = store.load_m1(pair, src=FX_PARQUET)
    tbl = calendars.sessions()
    keys = store.day_keys(m1.index)
    good = {pd.Timestamp(d).tz_localize(None).to_datetime64()
            for d in tbl}
    mins = store.minutes(m1.index)
    keep = np.isin(keys, list(good)) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    sub = m1.loc[keep, ["o", "h", "l", "c", "v"]]
    bars = sub.resample("5min", closed="left", label="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    ).dropna(subset=["o", "h", "l", "c"])
    bars.columns = ["open", "high", "low", "close", "volume"]
    return bars


def summarise(tag, trades):
    if trades is None or trades.empty or len(trades) < 5:
        print(f"  {tag:>9}: {0 if trades is None else len(trades)} trades")
        return None
    r = trades["return_pct"].to_numpy()
    lo, hi = metrics.bootstrap_ci(r / 100.0, seed=2)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    print(f"  {tag:>9}: n={len(r):>4}  mean {r.mean():+.4f}%  "
          f"total {r.sum():+6.2f}%  win {100 * (r > 0).mean():4.1f}%  "
          f"t={t:+5.2f}  CI [{100 * lo:+.4f}, {100 * hi:+.4f}]")
    return r


def main() -> int:
    pd.set_option("display.width", 220)
    on_cfg = ip.Config(cost_bps=COST_BPS)
    off_cfg = ip.Config(cost_bps=COST_BPS, rvol_mult=0.0, adr_exhaustion=99.0)

    bars = {}
    on_all, off_all = [], []

    print("=" * 96)
    print("WITH the busy-ness filter (the rule as written)")
    print("=" * 96)
    for p in PAIRS:
        bars[p] = ny_5min(p)
        t = ip.run_backtest(ip.prepare_data(bars[p], on_cfg), on_cfg)
        if len(t):
            on_all.append(t)
        summarise(p, t)

    print("\n" + "=" * 96)
    print("WITHOUT it (every crossover that passes trend and time)")
    print("=" * 96)
    for p in PAIRS:
        t = ip.run_backtest(ip.prepare_data(bars[p], off_cfg), off_cfg)
        if len(t):
            off_all.append(t)
        summarise(p, t)

    print("\n" + "=" * 96)
    print("POOLED across the seven pairs")
    print("=" * 96)
    on = pd.concat(on_all) if on_all else pd.DataFrame()
    off = pd.concat(off_all) if off_all else pd.DataFrame()
    r_on = summarise("filter on", on)
    summarise("filter off", off)

    # Does the filtered set beat the same number of trades taken at random?
    if r_on is not None and len(on) >= 20:
        print("\n  random-timing control, pooled")
        draws = np.empty(N_DRAWS)
        prepared = {p: ip.prepare_data(bars[p], on_cfg) for p in PAIRS}
        counts = {}
        for p in PAIRS:
            counts[p] = int(prepared[p]["entry_signal"].sum())
        for i in range(N_DRAWS):
            picked = []
            for p in PAIRS:
                fr = prepared[p]
                sl = fr["sl_dist"].to_numpy(float)
                pool = np.flatnonzero(fr["in_window"].to_numpy(bool)
                                      & np.isfinite(sl) & (sl > 0))
                if pool.size < counts[p] or counts[p] == 0:
                    continue
                flag = np.zeros(len(fr), bool)
                flag[RNG.choice(pool, size=counts[p], replace=False)] = True
                g = fr.copy()
                g["entry_signal"] = flag
                tt = ip.run_backtest(g, on_cfg)
                if len(tt):
                    picked.append(tt["return_pct"].to_numpy())
            draws[i] = np.concatenate(picked).mean() if picked else np.nan
        draws = draws[np.isfinite(draws)]
        real = r_on.mean()
        p_val = (1 + (draws >= real).sum()) / (1 + draws.size)
        print(f"    rule {real:+.4f}%   random {draws.mean():+.4f}% "
              f"(sd {draws.std():.4f})   p={p_val:.3f}")

    # Parameter search, scored as a search.
    print("\n" + "=" * 96)
    print("PARAMETER SEARCH, pooled, scored against what searching alone finds")
    print("=" * 96)
    grid = list(itertools.product(
        (1.0, 1.3, 1.8), (0.70, 0.85, 1.00),
        ((2.0, 4.0), (2.5, 5.0), (3.0, 6.0)), (20, 50)))
    rows, per_day = [], {}
    for rv, adr, (sl, tp), fast in grid:
        cfg = ip.Config(cost_bps=COST_BPS, rvol_mult=rv, adr_exhaustion=adr,
                        atr_sl_mult=sl, atr_tp_mult=tp, ema_fast=fast)
        parts = []
        for p in PAIRS:
            t = ip.run_backtest(ip.prepare_data(bars[p], cfg), cfg)
            if len(t):
                parts.append(t)
        if not parts:
            continue
        allt = pd.concat(parts)
        if len(allt) < 40:
            continue
        tag = f"rv{rv}/adr{adr}/{sl}-{tp}/ema{fast}"
        rows.append(dict(cell=tag, n=len(allt),
                         mean_pct=allt["return_pct"].mean(),
                         win=100.0 * (allt["return_pct"] > 0).mean()))
        d = allt.copy()
        d["day"] = pd.DatetimeIndex(d["entry_time"]).normalize()
        per_day[tag] = d.groupby("day")["return_pct"].agg(["sum", "count"])

    if len(per_day) >= 2:
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        print(f"  {len(df)} settings tried, "
              f"{int((df['mean_pct'] > 0).sum())} made money")
        print(df.head(5).round(4).to_string(index=False))
        print(df.tail(3).round(4).to_string(index=False))
        name, val, pv, null = metrics.reality_check(per_day, n_boot=4000)
        print(f"\n  best setting {name}: {val:+.4f}%")
        print(f"  what pure luck produces across {len(per_day)} settings: "
              f"median {np.median(null):+.4f}%, "
              f"top 5% above {np.percentile(null, 95):+.4f}%")
        print(f"  p = {pv:.3f}  ", end="")
        print("-> better than luck" if pv < 0.05
              else "-> no better than trying lots of settings and "
                   "keeping the best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
