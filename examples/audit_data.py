"""
Check the bars themselves before trusting anything built on them.

Every stored panel is examined for the faults that do not raise an error but
do change results: bars out of order, the same timestamp twice, a high below
its own low, a price of zero, a return no market ever made. A single
duplicated minute silently double-counts a trade; one bad print becomes an
enormous fake move that a stop-loss test will happily "capture".

This is separate from the look-ahead audit. That one asks whether the
strategy cheated. This one asks whether the data was ever worth using.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from bcbt import store

warnings.filterwarnings("ignore")

PANELS = [
    ("index CFD minutes", "data/m1.parquet", ["SPX500", "NAS100"]),
    ("FX minutes", "data/fx_m1.parquet",
     ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]),
    ("ETF 5-minute", "data/equity_5min.parquet", ["SPY", "QQQ"]),
]


def audit_one(sym, df):
    o, h, low, c = df["o"], df["h"], df["l"], df["c"]
    ret = c.pct_change()
    idx = df.index
    return dict(
        symbol=sym,
        bars=len(df),
        out_of_order=int((~idx.to_series().diff().dropna().gt(
            pd.Timedelta(0))).sum()),
        duplicate_stamps=int(idx.duplicated().sum()),
        nan_prices=int(df[["o", "h", "l", "c"]].isna().sum().sum()),
        non_positive=int((df[["o", "h", "l", "c"]] <= 0).sum().sum()),
        high_below_low=int((h < low).sum()),
        close_outside_range=int(((c > h) | (c < low)).sum()),
        open_outside_range=int(((o > h) | (o < low)).sum()),
        # A single bar moving more than 10% is almost always a bad print at
        # this resolution, and it is exactly what a stop test will treat as
        # a windfall.
        moves_over_10pct=int((ret.abs() > 0.10).sum()),
        max_move_pct=round(100 * float(ret.abs().max()), 2),
        frozen_runs=int((ret == 0).rolling(60).sum().ge(60).sum()),
    )


def main() -> int:
    pd.set_option("display.width", 220)
    all_rows = []
    for label, path, syms in PANELS:
        if not os.path.exists(path):
            print(f"\n{label}: {path} not present, skipped")
            continue
        print(f"\n{'=' * 100}\n{label}  ({path})\n{'=' * 100}")
        rows = []
        for s in syms:
            try:
                df = store.load_m1(s, src=path)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {s}: {exc}")
                continue
            rows.append(audit_one(s, df))
        if rows:
            d = pd.DataFrame(rows).set_index("symbol")
            print(d.to_string())
            all_rows.extend(rows)

    if not all_rows:
        print("\nno panels found")
        return 1

    d = pd.DataFrame(all_rows)
    fatal = ["out_of_order", "duplicate_stamps", "nan_prices", "non_positive",
             "high_below_low", "close_outside_range", "open_outside_range"]
    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    bad = {c: int(d[c].sum()) for c in fatal if d[c].sum() > 0}
    if bad:
        print(f"  structural faults found: {bad}")
    else:
        print("  structure: clean on every panel "
              "(ordered, unique, positive, high >= low, close inside range)")
    susp = int(d["moves_over_10pct"].sum())
    print(f"  single bars moving more than 10%: {susp}"
          f"   largest single move: {d['max_move_pct'].max():.2f}%")
    frozen = int(d["frozen_runs"].sum())
    print(f"  stretches of 60+ unchanged bars: {frozen}"
          f"   (illiquid or stale quoting, not necessarily wrong)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
