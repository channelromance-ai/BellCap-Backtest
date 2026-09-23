"""
The Asia-Pacific overnight trade, built and costed properly.

The clock test passed: funds whose home market trades during the New York
night show the weak-close effect, the Americas control group does not, and
the difference is significant. That earns a proper look, not a victory lap.

Two decisions here that change the answer, both made deliberately.

The signal is the S&P's close, not the fund's own. The fund's own close is
the stronger signal in the raw numbers, and it should not be trusted: these
funds are thin, a closing print that lands on the bid makes the close look
weak AND the next morning look strong, and that manufactures precisely this
result. The S&P is liquid and is a different instrument from the one being
traded, so it cannot contaminate the thing it is predicting.

Only funds with real volume are included. An effect of a tenth of a percent
against a fund quoting two tenths wide is not an effect, it is a fee.

And the sample is split in half by time. Everything so far has been measured
over the whole history of these funds, which is one long in-sample fit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bcbt import metrics, overnight_data as od

warnings.filterwarnings("ignore")

# Asia-Pacific funds liquid enough to trade. Median dollar volume in
# millions, measured in the previous study, in brackets.
LIQUID = ["EWY", "INDA", "EWT", "AAXJ", "EWH", "EWA"]   # 23-115m a day
THIN = ["EPP", "EWS", "EWM"]                            # 5-12m, excluded

# Round-trip costs to test, as a share of the trade. A fund turning over
# tens of millions a day typically quotes 3-5 basis points wide; 10 is
# pessimistic and 20 is what a thin one would cost.
COSTS = [0.0003, 0.0005, 0.0010, 0.0020]

PANEL = "data/overnight_foreign.parquet"
SPLIT = "2013-01-01"


def main() -> int:
    pd.set_option("display.width", 210)
    p = pd.read_parquet(PANEL)
    spy = od.build("SPY")[["close_in_range"]].rename(
        columns={"close_in_range": "spy_cir"})
    p = p.join(spy, how="left").dropna(subset=["overnight", "spy_cir"])
    p = p[p["sym"].isin(LIQUID)]

    print("=" * 100)
    print("ASIA-PACIFIC BASKET, triggered by a weak S&P close")
    print("=" * 100)
    print(f"  funds: {', '.join(LIQUID)}")
    print(f"  excluded as too thin: {', '.join(THIN)}")

    # Equally-weighted basket: each fund gets a share of the money and sits
    # in cash when the signal is quiet.
    held = p["spy_cir"] < 0.50
    p = p.assign(held=held)

    def basket(frame, cost):
        cols = []
        for sym, g in frame.groupby("sym"):
            r = np.where(g["held"], g["overnight"] - cost, 0.0)
            cols.append(pd.Series(r, index=g.index, name=sym))
        return pd.concat(cols, axis=1).mean(axis=1).dropna()

    def stats(ret):
        eq = np.cumprod(1.0 + ret.to_numpy())
        peak = np.maximum.accumulate(eq)
        years = len(ret) / 252.0
        ann = eq[-1] ** (1.0 / years) - 1.0
        vol = ret.std(ddof=1) * np.sqrt(252)
        nz = ret[ret != 0]
        return dict(nights=len(nz),
                    pct_nights=100.0 * len(nz) / len(ret),
                    mean_pct=100.0 * nz.mean(),
                    t=nz.mean() / (nz.std(ddof=1) / np.sqrt(len(nz))),
                    annual_pct=100.0 * ann,
                    worst_fall_pct=100.0 * (eq / peak - 1.0).min(),
                    risk_adj=ann / vol if vol > 0 else np.nan)

    print("\n  --- whole sample, by dealing cost ---")
    print(f"  {'round trip':>12}{'nights':>9}{'per night':>12}{'t':>7}"
          f"{'a year':>9}{'worst':>9}{'risk-adj':>10}")
    for c in COSTS:
        s = stats(basket(p, c))
        print(f"  {1e4 * c:>10.0f}bp{s['nights']:>9}{s['mean_pct']:>11.4f}%"
              f"{s['t']:>7.2f}{s['annual_pct']:>8.2f}%"
              f"{s['worst_fall_pct']:>8.1f}%{s['risk_adj']:>10.2f}")

    print(f"\n  --- split in half at {SPLIT} (5bp round trip) ---")
    early, late = p[p.index < SPLIT], p[p.index >= SPLIT]
    for label, frame in (("first half ", early), ("second half", late)):
        if len(frame) < 500:
            continue
        s = stats(basket(frame, 0.0005))
        lo, hi = metrics.bootstrap_ci(
            basket(frame, 0.0005).replace(0, np.nan).dropna().to_numpy(),
            seed=9)
        print(f"  {label}  {frame.index.min().date()} -> "
              f"{frame.index.max().date()}  n={s['nights']:>5}  "
              f"{s['mean_pct']:+.4f}%/night  t={s['t']:+.2f}  "
              f"{s['annual_pct']:+.2f}%/yr  risk-adj {s['risk_adj']:.2f}")
        print(f"               95% range for the nightly average: "
              f"[{100 * lo:+.4f}%, {100 * hi:+.4f}%]")

    print("\n  --- per fund, second half only, 5bp ---")
    for sym, g in late.groupby("sym"):
        r = g.loc[g["held"], "overnight"] - 0.0005
        if len(r) < 100:
            continue
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        print(f"    {sym:<6} n={len(r):>5}  {100 * r.mean():+.4f}%/night  "
              f"t={t:+.2f}")

    print("\n  --- by year, 5bp round trip ---")
    ret = basket(p, 0.0005)
    nz = ret[ret != 0]
    for y, g in nz.groupby(nz.index.year):
        bar = "+" * int(max(0, 100 * g.sum())) or ("-" * int(max(0, -100 * g.sum())))
        print(f"    {y}: {len(g):>4} nights {100 * g.mean():+.4f}%/night "
              f"{100 * g.sum():+7.2f}%  {bar[:40]}")

    print("\n  --- what it is competing with ---")
    for sym, g in p.groupby("sym"):
        r = g["all_day"].dropna()
        eq = np.cumprod(1 + r.to_numpy())
        years = len(r) / 252.0
        ann = eq[-1] ** (1 / years) - 1
        vol = r.std(ddof=1) * np.sqrt(252)
        peak = np.maximum.accumulate(eq)
        print(f"    own {sym:<6} outright: {100 * ann:+6.2f}%/yr  "
              f"worst fall {100 * (eq / peak - 1).min():6.1f}%  "
              f"risk-adj {ann / vol:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
