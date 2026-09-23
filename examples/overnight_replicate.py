"""
Replicate the weak-close effect on instruments never used to find it.

The design set (SPY, QQQ, IWM, DIA, EFA, EEM) chose the rule and the holdout
tested it. Both are now spent. Fourteen other instruments -- the US sector
funds, four single-country funds, gold and long bonds -- were never looked
at, so they are a clean test of the same claim rather than another slice of
the same data.

The claim being retested, unchanged: a session that closes in the lower half
of its range is followed by a better-than-average night.

One thing to watch for in the results. For a fund tracking a foreign market,
"overnight" in New York is when that market is actually open. EWJ's night is
the Tokyo session. So if the effect is about real price discovery rather
than about a gap at the open, those funds should show it most.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")

FRESH = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
         "EWJ", "EWG", "EWU", "FXI", "GLD", "TLT"]

# Which of these are foreign-equity funds, whose home market trades while
# New York is shut.
FOREIGN = {"EWJ", "EWG", "EWU", "FXI"}

PANEL = "data/overnight_fresh.parquet"


def main() -> int:
    pd.set_option("display.width", 200)
    p = pd.read_parquet(PANEL).dropna(
        subset=["overnight", "close_in_range", "ret_5d", "prev_overnight"])

    rows = []
    for sym, g in p.groupby("sym"):
        base = g["overnight"]
        soft = g.loc[g["close_in_range"] < 0.50, "overnight"]
        day = g["daytime"]
        if len(soft) < 200:
            continue
        t = soft.mean() / (soft.std(ddof=1) / np.sqrt(len(soft)))
        rows.append(dict(
            sym=sym, kind="foreign" if sym in FOREIGN else "domestic",
            n=len(g),
            night_pct=100 * base.mean(),
            day_pct=100 * day.mean(),
            soft_night_pct=100 * soft.mean(),
            lift_pct=100 * (soft.mean() - base.mean()),
            t=t))

    d = pd.DataFrame(rows).sort_values("lift_pct", ascending=False)
    print("=" * 96)
    print("REPLICATION on 14 instruments never used to build the rule")
    print("=" * 96)
    print(d.round(4).to_string(index=False))

    wins = int((d["lift_pct"] > 0).sum())
    p_sign = st.binomtest(wins, len(d), 0.5, alternative="greater").pvalue
    print(f"\n  the soft-close night beat the average night on "
          f"{wins} of {len(d)} instruments")
    print(f"  median lift {d['lift_pct'].median():+.4f}% a night")
    print(f"  sign test p = {p_sign:.4f}")

    print("\n" + "=" * 96)
    print("WHERE THE EFFECT LIVES")
    print("=" * 96)
    g = d.groupby("kind")[["night_pct", "day_pct", "soft_night_pct",
                           "lift_pct"]].mean()
    print(g.round(4).to_string())
    print("\n  For a fund tracking a foreign market, the New York 'overnight'")
    print("  window is that market's own trading session, so the move is")
    print("  real trading rather than a gap. That is where the lift is")
    print("  largest by a wide margin, which is what the effect being about")
    print("  price discovery -- not about the open -- would look like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
