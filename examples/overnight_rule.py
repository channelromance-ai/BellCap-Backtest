"""
Hold overnight only after a weak session. Design, then one look at 2016-2026.

The findings this rests on, all established on data before 2016:

  * Overnight returns carry nearly all of the index's gain, and the daytime
    session contributes roughly nothing. True on five of six instruments.
  * Nights are not alike. After a day that closed in the lower half of its
    range, or followed a night that gapped down, or ended a weak week, the
    next overnight is roughly twice as good as an average one.
  * That is not a measurement artifact. Bid-ask bounce would manufacture
    exactly this pattern, but bounce should have collapsed when the tick
    went from 24 basis points to under 1 after decimalisation, and the
    effect did not -- measured against the baseline it grew.

THE PRE-COMMITTED CANDIDATE is "2 or more of 3, price only": hold the coming
night when at least two of these were true at today's close.

    the close sat in the lower half of the day's range
    last night gapped down
    the last five days were down

It is not the best cell on the design period. It was chosen before the
holdout was opened, on two grounds that have nothing to do with performance:
it uses no VIX, which prints 15 minutes after the close a trade would fill
at, and it needs two of three signals rather than betting on any single one.
The whole table is printed on the holdout as well, so the family can be seen
to hold up or not -- but the candidate was fixed first, and picking a
different winner afterwards would make the holdout worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bcbt import metrics

PANEL = "data/overnight_panel.parquet"
SPLIT = "2016-01-01"
COST = 2e-4                 # round trip on a liquid ETF
TRADING_DAYS = 252.0

CANDIDATE = "2+ of 3 (price only)"


def flags(g: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=g.index)
    f["weak_close"] = g["close_in_range"] < 0.25
    f["soft_close"] = g["close_in_range"] < 0.50
    f["gapped_down"] = g["prev_overnight"] < 0
    f["week_down"] = g["ret_5d"] < 0
    f["vix_up"] = g["vix_chg"] > 0
    f["score"] = (f["soft_close"].astype(int) + f["gapped_down"].astype(int)
                  + f["week_down"].astype(int) + f["vix_up"].astype(int))
    f["score_price"] = (f["soft_close"].astype(int)
                        + f["gapped_down"].astype(int)
                        + f["week_down"].astype(int))
    return f


RULES = {
    "always": lambda f: pd.Series(True, index=f.index),
    "weak close (bottom 1/4)": lambda f: f["weak_close"],
    "soft close (bottom 1/2)": lambda f: f["soft_close"],
    "gapped down yesterday": lambda f: f["gapped_down"],
    "down over 5 days": lambda f: f["week_down"],
    "VIX rose today*": lambda f: f["vix_up"],
    "soft close + week down": lambda f: f["soft_close"] & f["week_down"],
    "weak close + VIX up*": lambda f: f["weak_close"] & f["vix_up"],
    "2+ of 4*": lambda f: f["score"] >= 2,
    "3+ of 4*": lambda f: f["score"] >= 3,
    "2+ of 3 (price only)": lambda f: f["score_price"] >= 2,
    "gap down + soft close": lambda f: f["gapped_down"] & f["soft_close"],
}


def portfolio(d: pd.DataFrame, rule, cost=COST) -> pd.Series:
    """
    Equally-weighted basket: each instrument gets a sixth of the money and
    sits in cash, earning nothing, on nights its own signal is quiet. Cash
    really earns interest, so this understates the result.
    """
    cols = []
    for sym, g in d.groupby("sym"):
        held = rule(flags(g)).reindex(g.index).fillna(False)
        cols.append(pd.Series(
            np.where(held, g["overnight"] - cost, 0.0),
            index=g.index, name=sym))
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def stats(ret: pd.Series) -> dict:
    eq = np.cumprod(1.0 + ret.to_numpy())
    peak = np.maximum.accumulate(eq)
    years = len(ret) / TRADING_DAYS
    ann = eq[-1] ** (1.0 / years) - 1.0
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    nz = ret[ret != 0]
    t = (nz.mean() / (nz.std(ddof=1) / np.sqrt(len(nz)))
         if len(nz) > 30 else np.nan)
    return dict(pct_nights=100.0 * len(nz) / len(ret),
                mean_pct=100.0 * nz.mean(), t=t,
                annual_pct=100.0 * ann,
                worst_fall_pct=100.0 * (eq / peak - 1.0).min(),
                risk_adj=ann / vol if vol > 0 else np.nan,
                total_pct=100.0 * (eq[-1] - 1.0))


def buy_and_hold(d: pd.DataFrame) -> dict:
    cols = [g["all_day"].rename(s) for s, g in d.groupby("sym")]
    ret = pd.concat(cols, axis=1).mean(axis=1).dropna()
    return stats(ret)


def table(d: pd.DataFrame, title: str) -> pd.DataFrame:
    rows = []
    for name, fn in RULES.items():
        s = stats(portfolio(d, fn))
        s["rule"] = name
        rows.append(s)
    t = pd.DataFrame(rows).set_index("rule")
    print(f"\n{title}")
    print(t.round(4).to_string())
    return t


def main() -> int:
    pd.set_option("display.width", 220)
    p = pd.read_parquet(PANEL)
    need = ["overnight", "close_in_range", "ret_5d", "prev_overnight",
            "vix_chg", "all_day"]
    p = p.dropna(subset=need)
    design = p[p.index < SPLIT]
    hold = p[p.index >= SPLIT]

    print("=" * 104)
    print(f"DESIGN {design.index.min().date()} -> {design.index.max().date()}"
          f"   ({len(design):,} rows)")
    print(f"HOLDOUT {hold.index.min().date()} -> {hold.index.max().date()}"
          f"   ({len(hold):,} rows)")
    print(f"cost charged: {1e4 * COST:.0f} basis points per round trip")
    print(f"PRE-COMMITTED CANDIDATE: {CANDIDATE}")
    print("=" * 104)

    td = table(design, "--- design period ---")
    bh = buy_and_hold(design)
    print(f"  buy and hold: {bh['annual_pct']:+.2f}%/yr  worst fall "
          f"{bh['worst_fall_pct']:.1f}%  risk-adjusted {bh['risk_adj']:.2f}")

    th = table(hold, "--- HOLDOUT, first and only look ---")
    bhh = buy_and_hold(hold)
    print(f"  buy and hold: {bhh['annual_pct']:+.2f}%/yr  worst fall "
          f"{bhh['worst_fall_pct']:.1f}%  risk-adjusted {bhh['risk_adj']:.2f}")

    print("\n" + "=" * 104)
    print("THE CANDIDATE, DESIGN vs HOLDOUT")
    print("=" * 104)
    a, b = td.loc[CANDIDATE], th.loc[CANDIDATE]
    print(f"  {'':<10}{'nights':>9}{'mean %':>10}{'t':>8}{'a year':>10}"
          f"{'worst':>9}{'risk-adj':>10}")
    for label, row in (("design", a), ("HOLDOUT", b)):
        print(f"  {label:<10}{row['pct_nights']:>8.1f}%{row['mean_pct']:>10.4f}"
              f"{row['t']:>8.2f}{row['annual_pct']:>9.2f}%"
              f"{row['worst_fall_pct']:>8.1f}%{row['risk_adj']:>10.2f}")
    print(f"  {'always':<10}{th.loc['always','pct_nights']:>8.1f}%"
          f"{th.loc['always','mean_pct']:>10.4f}"
          f"{th.loc['always','t']:>8.2f}"
          f"{th.loc['always','annual_pct']:>9.2f}%"
          f"{th.loc['always','worst_fall_pct']:>8.1f}%"
          f"{th.loc['always','risk_adj']:>10.2f}   (holdout)")
    print(f"  {'buy&hold':<10}{'100.0%':>9}{'':>10}{'':>8}"
          f"{bhh['annual_pct']:>9.2f}%{bhh['worst_fall_pct']:>8.1f}%"
          f"{bhh['risk_adj']:>10.2f}   (holdout)")

    r = portfolio(hold, RULES[CANDIDATE])
    nz = r[r != 0]
    lo, hi = metrics.bootstrap_ci(nz.to_numpy(), seed=5)
    print(f"\n  95% range for the candidate's true nightly average on the "
          f"holdout: [{100 * lo:+.4f}%, {100 * hi:+.4f}%]")

    print("\n  by year on the holdout:")
    for y, g in nz.groupby(nz.index.year):
        print(f"    {y}: {len(g):>4} nights  {100 * g.mean():+.4f}%/night  "
              f"{100 * g.sum():+6.2f}% total")

    print("\n  cost sensitivity on the holdout (candidate):")
    for c in (1e-4, 2e-4, 3e-4, 5e-4):
        s = stats(portfolio(hold, RULES[CANDIDATE], cost=c))
        print(f"    {1e4 * c:.0f}bp round trip: {s['annual_pct']:+.2f}%/yr, "
              f"risk-adjusted {s['risk_adj']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
