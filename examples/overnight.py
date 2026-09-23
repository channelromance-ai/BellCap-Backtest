"""
Where does the index return actually happen -- daytime or overnight?

The controls run earlier said something worth chasing. Entering at random
moments during the day and closing before the bell returned roughly nothing,
yet these indices rose 68% and 90% over the same three years. If the gain is
not in the day session, it is in the gap between one close and the next open.

This measures that directly, and then answers the question that decides
whether it is usable: holding overnight normally costs financing, so how much
can it pay per night before the whole thing stops being worth doing?

Three holdings are compared over identical days:

    overnight   buy the close, sell the next open
    daytime     buy the open, sell the same day's close
    buy & hold  own it the whole time

All three are unleveraged and long only, so the comparison is about when the
money is made, not about gearing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bcbt import twelvedata

ETFS = ["SPY", "QQQ"]

# Overnight financing on a leveraged product, as a share of position value
# per night. A CFD or margin long is charged roughly the cash rate plus a
# broker markup; at recent rates that is about 7% a year.
FINANCING_PER_NIGHT = 0.07 / 365.0


def daily_open_close(etf: str) -> pd.DataFrame:
    """First open and last close of each session."""
    bars = twelvedata.load(etf)
    day = bars.index.normalize()
    g = bars.groupby(day)
    d = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last()})
    d.index = pd.DatetimeIndex(d.index)
    return d.sort_index()


def curve_stats(r: np.ndarray, periods_per_year: float) -> dict:
    """Compounded growth, worst fall from a peak, and return per unit risk."""
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    ann = eq[-1] ** (periods_per_year / len(r)) - 1.0
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    return dict(total_pct=100.0 * (eq[-1] - 1.0),
                annual_pct=100.0 * ann,
                worst_fall_pct=100.0 * dd.min(),
                per_unit_risk=(ann / vol) if vol > 0 else np.nan,
                avg_per_day_pct=100.0 * r.mean())


def main() -> int:
    pd.set_option("display.width", 200)
    print("=" * 84)
    print("WHEN THE MONEY IS MADE")
    print("=" * 84)

    store = {}
    for etf in ETFS:
        d = daily_open_close(etf)
        # Overnight: last night's close to this morning's open.
        overnight = (d["open"] / d["close"].shift(1) - 1.0).dropna()
        daytime = (d["close"] / d["open"] - 1.0).loc[overnight.index]
        both = (d["close"] / d["close"].shift(1) - 1.0).loc[overnight.index]
        store[etf] = pd.DataFrame(
            {"overnight": overnight, "daytime": daytime, "hold": both})

        n = len(overnight)
        ppy = 252.0
        print(f"\n  {etf}: {n} nights, "
              f"{overnight.index[0].date()} -> {overnight.index[-1].date()}")
        rows = []
        for name in ("overnight", "daytime", "hold"):
            s = curve_stats(store[etf][name].to_numpy(), ppy)
            s["holding"] = name
            rows.append(s)
        t = pd.DataFrame(rows).set_index("holding")
        print(t.round(3).to_string())

    print("\n" + "=" * 84)
    print("WHAT FINANCING WOULD DO TO IT")
    print("=" * 84)
    print(f"  assumed charge: {100 * FINANCING_PER_NIGHT:.4f}% per night "
          f"(about 7% a year)")
    for etf in ETFS:
        r = store[etf]["overnight"].to_numpy()
        gross = r.mean()
        net = gross - FINANCING_PER_NIGHT
        breakeven_annual = gross * 365.0
        print(f"\n  {etf}")
        print(f"    average gain per night, before costs : "
              f"{100 * gross:+.4f}%")
        print(f"    after {100 * FINANCING_PER_NIGHT:.4f}% financing      : "
              f"{100 * net:+.4f}%")
        print(f"    share of the gain eaten by financing : "
              f"{100 * FINANCING_PER_NIGHT / gross:.0f}%")
        print(f"    it stops being worth doing above     : "
              f"{100 * breakeven_annual:.2f}% a year financing")
        after = curve_stats(r - FINANCING_PER_NIGHT, 252.0)
        print(f"    net of financing: {after['annual_pct']:+.2f}% a year, "
              f"worst fall {after['worst_fall_pct']:.1f}%, "
              f"return per unit risk {after['per_unit_risk']:.2f}")

    print("\n" + "=" * 84)
    print("IS IT STEADY, OR A FEW GOOD NIGHTS?")
    print("=" * 84)
    for etf in ETFS:
        s = store[etf]["overnight"]
        by_year = s.groupby(s.index.year).agg(["count", "mean", "sum"])
        print(f"\n  {etf} by year: " + "  ".join(
            f"{y}: {int(v['count'])} nights, {100 * v['mean']:+.4f}%/night, "
            f"{100 * v['sum']:+.1f}% total"
            for y, v in by_year.iterrows()))
        arr = np.sort(s.to_numpy())[::-1]
        print(f"    best 10 nights are {100 * arr[:10].sum():+.1f}% of the "
              f"{100 * arr.sum():+.1f}% total")
        print(f"    share of nights that were up: "
              f"{100 * (s > 0).mean():.1f}%")

    print("\n" + "=" * 84)
    print("THE CATCH: IT TRADES EVERY SINGLE DAY")
    print("=" * 84)
    print("  Owning shares outright has no nightly financing -- the charge")
    print("  exists only because a CFD or margin position is borrowed. But")
    print("  capturing only the night means buying every close and selling")
    print("  every open: roughly 252 round trips a year, against buy and")
    print("  hold's one. Dealing costs are charged per round trip below.")
    print("\n  a round trip on a liquid ETF is the penny spread (about")
    print("  1.7 basis points on a $600 share) plus commission, so 2-4 is")
    print("  the realistic range; the CFD column adds 7%/yr financing on")
    print("  top of its tighter 0.9 basis point spread.")

    for etf in ETFS:
        r = store[etf]["overnight"].to_numpy()
        hold = store[etf]["hold"].to_numpy()
        h = curve_stats(hold, 252.0)
        print(f"\n  {etf}   (buy and hold: {h['annual_pct']:+.2f}%/yr, "
              f"worst fall {h['worst_fall_pct']:.1f}%, "
              f"per unit risk {h['per_unit_risk']:.2f})")
        print(f"    {'route':<26}{'a year':>9}{'worst fall':>13}"
              f"{'per unit risk':>15}")
        rows = [
            ("shares, 1bp round trip", 1e-4, 0.0),
            ("shares, 2bp round trip", 2e-4, 0.0),
            ("shares, 3bp round trip", 3e-4, 0.0),
            ("shares, 5bp round trip", 5e-4, 0.0),
            ("CFD, 0.9bp + financing", 0.9e-4, FINANCING_PER_NIGHT),
        ]
        for label, trade_cost, nightly in rows:
            net = r - trade_cost - nightly
            s = curve_stats(net, 252.0)
            print(f"    {label:<26}{s['annual_pct']:>8.2f}%"
                  f"{s['worst_fall_pct']:>12.1f}%"
                  f"{s['per_unit_risk']:>15.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
