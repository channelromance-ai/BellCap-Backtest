"""
Six different systems, tested the same way, on a design period only.

Nothing here is an intraday price pattern, and nothing trades more than
monthly except where noted. That is the point: the previous six attempts all
traded two hundred times a year on one liquid index, which is both the most
competed idea in markets and the most expensive way to express it.

The families, chosen to be genuinely different from each other rather than
variations on one theme:

  trend        own each asset only while its own year has been positive,
               otherwise sit in cash. A bet on persistence, asset by asset.
  cross-mom    own whichever third of the universe has risen most over the
               last year. A bet on relative strength, not on direction.
  dual-mom     the two combined: the strongest assets, but only those also
               positive outright.
  inverse-vol  own everything, weighted so each contributes similar risk.
               No forecast of return at all.
  defensive    stocks while the market's trend is up, bonds when it is not.
  reversal     own last week's worst performers. Weekly, so it is also the
               test of whether turnover really is what kills things.

Benchmarks are honest ones: not zero, but buying and holding the same
assets, and the classic sixty-forty.
"""
from __future__ import annotations

import pandas as pd

from bcbt import universe as U

BROAD = list(U.UNIVERSE)
SECTORS = U.SECTORS
CASH = U.CASH

START = "2007-06-01"
DESIGN_END = "2018-12-31"
COST_BPS = 5.0


# ----------------------------------------------------------------- helpers

def _ret(hist, days, skip=0):
    """Total return over `days`, ending `skip` days ago."""
    if len(hist) < days + skip + 1:
        return pd.Series(dtype=float)
    end = hist.iloc[-1 - skip]
    start = hist.iloc[-1 - skip - days]
    return (end / start - 1.0).dropna()


def _tradeable(hist, assets, need=260):
    ok = [a for a in assets
          if a in hist.columns and hist[a].iloc[-need:].notna().all()]
    return ok


def _eq(names):
    if not names:
        return {CASH: 1.0}
    w = 1.0 / len(names)
    return {n: w for n in names}


# ----------------------------------------------------------------- systems

def trend(hist, assets=BROAD, lookback=252):
    ok = _tradeable(hist, assets)
    if not ok:
        return {CASH: 1.0}
    r = _ret(hist[ok], lookback)
    winners = [a for a in ok if r.get(a, -1) > 0]
    if not winners:
        return {CASH: 1.0}
    # Equal weight across everything held; the rest of the money is cash,
    # so a thin month is genuinely defensive rather than concentrated.
    w = {a: 1.0 / len(ok) for a in winners}
    w[CASH] = 1.0 - sum(w.values())
    return w


def cross_mom(hist, assets=BROAD, lookback=252, skip=21, frac=3):
    ok = _tradeable(hist, assets)
    if len(ok) < frac:
        return {CASH: 1.0}
    r = _ret(hist[ok], lookback, skip)
    if r.empty:
        return {CASH: 1.0}
    n = max(1, len(ok) // frac)
    return _eq(list(r.sort_values(ascending=False).head(n).index))


def dual_mom(hist, assets=BROAD, lookback=252, skip=21, frac=3):
    ok = _tradeable(hist, assets)
    if len(ok) < frac:
        return {CASH: 1.0}
    r = _ret(hist[ok], lookback, skip)
    abs_r = _ret(hist[ok], lookback)
    n = max(1, len(ok) // frac)
    top = list(r.sort_values(ascending=False).head(n).index)
    keep = [a for a in top if abs_r.get(a, -1) > 0]
    w = {a: 1.0 / n for a in keep}
    w[CASH] = 1.0 - sum(w.values())
    return w


def inverse_vol(hist, assets=BROAD, lookback=60):
    ok = _tradeable(hist, assets)
    if not ok:
        return {CASH: 1.0}
    v = hist[ok].pct_change().iloc[-lookback:].std()
    v = v[v > 0]
    if v.empty:
        return {CASH: 1.0}
    inv = 1.0 / v
    return (inv / inv.sum()).to_dict()


def defensive(hist, risk="SPY", safe="IEF", lookback=200):
    if risk not in hist.columns or len(hist) < lookback + 1:
        return {CASH: 1.0}
    px = hist[risk].dropna()
    if len(px) < lookback + 1:
        return {CASH: 1.0}
    above = px.iloc[-1] > px.iloc[-lookback:].mean()
    if above:
        return {risk: 1.0}
    return {safe: 1.0} if safe in hist.columns else {CASH: 1.0}


def reversal(hist, assets=None, lookback=5, frac=3):
    assets = assets or (BROAD + SECTORS)
    ok = _tradeable(hist, assets)
    if len(ok) < frac:
        return {CASH: 1.0}
    r = _ret(hist[ok], lookback)
    if r.empty:
        return {CASH: 1.0}
    n = max(1, len(ok) // frac)
    return _eq(list(r.sort_values().head(n).index))


def sector_mom(hist, lookback=252, skip=21, frac=3):
    return cross_mom(hist, SECTORS, lookback, skip, frac)


# -------------------------------------------------------------- benchmarks

def equal_weight(hist, assets=BROAD):
    ok = _tradeable(hist, assets)
    return _eq(ok) if ok else {CASH: 1.0}


def sixty_forty(hist):
    if "SPY" in hist.columns and "IEF" in hist.columns:
        return {"SPY": 0.6, "IEF": 0.4}
    return {CASH: 1.0}


def buy_spy(hist):
    return {"SPY": 1.0}


SYSTEMS = {
    "trend (own what is rising)": (trend, "M"),
    "cross-mom (own the strongest third)": (cross_mom, "M"),
    "dual-mom (strongest AND rising)": (dual_mom, "M"),
    "inverse-vol (equal risk)": (inverse_vol, "M"),
    "defensive (stocks or bonds)": (defensive, "M"),
    "sector-mom (strongest sectors)": (sector_mom, "M"),
    "reversal (own last week's losers)": (reversal, "W"),
}

BENCHMARKS = {
    "buy and hold everything": (equal_weight, "M"),
    "sixty forty": (sixty_forty, "M"),
    "buy and hold SPY": (buy_spy, "M"),
}


def main() -> int:
    pd.set_option("display.width", 220)
    px = U.load()

    print("=" * 108)
    print(f"DESIGN PERIOD {START} -> {DESIGN_END}   "
          f"cost {COST_BPS:.0f}bp on turnover")
    print("=" * 108)

    rows = []
    for name, (fn, freq) in {**SYSTEMS, **BENCHMARKS}.items():
        rec = U.run_portfolio(px, fn, freq=freq, cost_bps=COST_BPS,
                              start=START, end=DESIGN_END)
        s = U.summarise(rec, name)
        s["kind"] = "benchmark" if name in BENCHMARKS else "system"
        rows.append(s)

    d = pd.DataFrame(rows).set_index("label")
    cols = ["kind", "annual_pct", "vol_pct", "risk_adj", "sortino",
            "worst_fall_pct", "calmar", "turnover_yr", "cost_pct_yr"]
    print(d[cols].sort_values("risk_adj", ascending=False).round(3).to_string())

    print("\n  risk_adj = return per unit of wobble;  calmar = return per "
          "unit of worst fall")
    print("  turnover_yr = how much of the portfolio is bought and sold "
          "each year (1.0 = all of it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
