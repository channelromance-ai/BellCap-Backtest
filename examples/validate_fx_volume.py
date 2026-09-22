"""
Is the currency "volume" we have any good?

There is no such thing as published volume for spot currency trading. No
single exchange handles it, so nobody counts it. What data providers supply
instead is a count of how many times the price ticked, on the reasonable
theory that a busy market updates its price more often.

The pullback strategy only buys when activity is unusually high, so that
count is the strategy. If it does not track real trading, nothing built on
top of it means anything -- which is exactly how the stock-index version of
this strategy produced a convincing result that turned out to be nothing.

Currency futures on the Chicago exchange settle the question. They are a
separate venue with genuinely published volume, and neither feed is derived
from the other, so agreement between them would be real evidence rather than
one market being measured twice. Their detailed history is too short to
backtest on, but there is enough at hourly resolution to score the tick
count against.

The bar was set before the numbers were seen: the tick count is trustworthy
if the busy-ness measure correlates comfortably above 0.5 with the real one,
and if most bars it calls busy are also busy on real volume.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bcbt import store

warnings.filterwarnings("ignore")

# Spot pair -> the Chicago futures contract on the same currency. Volume does
# not care which way round the quote is, so USD/JPY against the yen contract
# is a fair comparison.
PAIRS = {
    "EURUSD": "6E=F",
    "GBPUSD": "6B=F",
    "USDJPY": "6J=F",
    "AUDUSD": "6A=F",
    "USDCAD": "6C=F",
    "USDCHF": "6S=F",
    "NZDUSD": "6N=F",
}

FX_PARQUET = "data/fx_m1.parquet"
RVOL_DAYS = 20
RVOL_MULT = 1.3
WINDOW = (10 * 60, 15 * 60 + 30)      # the hours the strategy actually trades


def rvol(series: pd.Series, slot: np.ndarray, days: int = RVOL_DAYS):
    """
    How busy is this bar compared with the same time of day recently?

    The comparison is against the previous `days` sessions at the same clock
    time, and the current bar is excluded from its own benchmark.
    """
    grouped = series.groupby(slot, sort=False)
    base = grouped.transform(
        lambda s: s.shift(1).rolling(days, min_periods=days).mean())
    return series / base


def main() -> int:
    import yfinance as yf

    rows = []
    for pair, fut in PAIRS.items():
        # Tick counts, rolled up to the hour so they line up with futures.
        m1 = store.load_m1(pair, src=FX_PARQUET)
        tick = m1["v"].resample("1h", closed="left", label="left").sum()

        real = yf.download(fut, period="730d", interval="1h",
                           progress=False, auto_adjust=False)
        if isinstance(real.columns, pd.MultiIndex):
            real.columns = real.columns.get_level_values(0)
        real = real.tz_convert(store.ET)["Volume"]

        j = pd.DataFrame({"tick": tick, "real": real}).dropna()
        j = j[(j["tick"] > 0) & (j["real"] > 0)]
        if len(j) < 500:
            print(f"{pair}: only {len(j)} overlapping hours, skipped")
            continue

        slot = (j.index.hour * 60 + j.index.minute).to_numpy()
        j["tick_rvol"] = rvol(j["tick"], slot)
        j["real_rvol"] = rvol(j["real"], slot)
        k = j.dropna()

        lvl = np.corrcoef(np.log(k["tick"]), np.log(k["real"]))[0, 1]
        rv = np.corrcoef(k["tick_rvol"], k["real_rvol"])[0, 1]
        a = k["tick_rvol"] > RVOL_MULT
        b = k["real_rvol"] > RVOL_MULT

        # And the same thing restricted to the strategy's own trading hours.
        mins = k.index.hour * 60 + k.index.minute
        w = (mins >= WINDOW[0]) & (mins <= WINDOW[1])
        rv_w = (np.corrcoef(k.loc[w, "tick_rvol"], k.loc[w, "real_rvol"])[0, 1]
                if w.sum() > 200 else np.nan)

        rows.append(dict(
            pair=pair, futures=fut, hours=len(k),
            log_vol_corr=round(lvl, 3),
            rvol_corr=round(rv, 3),
            rvol_corr_window=round(rv_w, 3),
            tick_flags_pct=round(100 * a.mean(), 1),
            real_flags_pct=round(100 * b.mean(), 1),
            agree_pct=round(100 * (a == b).mean(), 1),
            tick_flagged_also_real=round(100 * b[a].mean(), 1),
        ))
        print(f"  {pair} vs {fut}: {len(k):,} hours", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("TICK COUNT vs REAL FUTURES VOLUME")
    print("=" * 96)
    print(df.to_string(index=False))

    print("\n  averages across the seven pairs")
    print(f"    raw volume, correlation      : "
          f"{df['log_vol_corr'].mean():.3f}")
    print(f"    busy-ness measure            : {df['rvol_corr'].mean():.3f}")
    print(f"    busy-ness, trading hours only: "
          f"{df['rvol_corr_window'].mean():.3f}")
    print(f"    bars tick calls busy that really are: "
          f"{df['tick_flagged_also_real'].mean():.1f}%")
    print(f"    agreement on busy / not busy : {df['agree_pct'].mean():.1f}%")

    print("\n  for comparison, the same test on the stock indexes")
    print("    index CFD tick proxy vs SPY : busy-ness 0.107, "
          "42% of flagged bars real  -> strategy died")

    ok = (df["rvol_corr"].mean() > 0.5
          and df["tick_flagged_also_real"].mean() > 65.0)
    print("\n  VERDICT: " + (
        "tick counts track real trading well enough to build on."
        if ok else
        "tick counts do NOT track real trading well enough to trust."))
    df.to_csv("fx_volume_validation.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
