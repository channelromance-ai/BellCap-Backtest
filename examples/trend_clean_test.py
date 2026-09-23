"""
A clean test of the trend rule: new markets, new eras, honest cash and
dividends.

The rule -- own an asset only while its own past year has been positive,
otherwise hold cash, rebalanced monthly -- beat buying and holding on
return-per-unit-of-risk in all three periods it was measured on. But those
three periods used the same twelve US funds. They differed in WHEN, not in
WHAT. So the rule has never actually been tried on anything new.

Two tests that fix that, both on data the rule has never touched:

  A  Thirteen foreign share indices -- Germany, Britain, France, Japan,
     Hong Kong, Australia, Korea, Brazil, Mexico, Canada, Switzerland,
     the Netherlands, Spain. Different markets, different currencies,
     different crises.

  B  The S&P itself from 1928 to 1992, which is before the fund data used
     everywhere else in this repo begins. Sixty-five years containing the
     1929 crash, the 1930s, the 1970s inflation and 1987.

Two things have to be handled or the comparison is rigged.

Cash is not free money, it earns interest, and in 1981 it earned 15%. A
rule that sits in cash a quarter of the time looks far worse than it is if
cash pays nothing. Treasury bill rates are used from 1960 onward.

These are price indices: they exclude dividends. Buying and holding collects
every dividend; the rule collects only those paid while it is invested. So a
price-only comparison flatters the rule, and both are reported -- price only,
and with dividends added back.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bcbt import universe as U

warnings.filterwarnings("ignore")

FOREIGN = {
    "^GDAXI": "Germany", "^FTSE": "Britain", "^FCHI": "France",
    "^N225": "Japan", "^HSI": "Hong Kong", "^AXJO": "Australia",
    "^KS11": "Korea", "^BVSP": "Brazil", "^MXX": "Mexico",
    "^GSPTSE": "Canada", "^SSMI": "Switzerland", "^AEX": "Netherlands",
    "^IBEX": "Spain",
}
CASHCOL = "CASH"
COST_BPS = 5.0

# Typical dividend yields for share indices, used to turn a price index
# into an approximate total-return one. Deliberately generous to the
# buy-and-hold side rather than the rule's.
DIV_YIELD = 0.03


def fetch(tickers, start="1920-01-01") -> pd.DataFrame:
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, start=start, interval="1d", progress=False,
                        auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if len(d) > 300:
            out[t] = d["Close"]
    px = pd.DataFrame(out).sort_index()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    # Markets keep different holidays, so on a shared calendar each one is
    # missing 2-5% of days. Carrying the last price across a national
    # holiday is correct -- a shut market did not move -- and without it a
    # "last 260 days all present" test fails on virtually every day, which
    # silently parks the whole portfolio in cash. Leading gaps, before a
    # market existed, stay missing on purpose.
    return px.ffill()


def cash_index(index: pd.DatetimeIndex) -> pd.Series:
    """
    A money-market balance that actually earns the Treasury bill rate.

    Before 1960 there is no series here, so cash earns nothing, which
    penalises the rule rather than helping it.
    """
    import yfinance as yf
    r = yf.download("^IRX", period="max", interval="1d", progress=False,
                    auto_adjust=False)
    if isinstance(r.columns, pd.MultiIndex):
        r.columns = r.columns.get_level_values(0)
    rate = r["Close"] / 100.0
    rate.index = pd.DatetimeIndex(rate.index).tz_localize(None)
    daily = (rate.reindex(index).ffill().fillna(0.0) / 252.0).clip(lower=0)
    return (1.0 + daily).cumprod()


def add_dividends(px: pd.DataFrame, cols, yield_pa=DIV_YIELD) -> pd.DataFrame:
    """Turn price indices into rough total-return indices."""
    out = px.copy()
    n = np.arange(len(px))
    factor = pd.Series(np.exp(yield_pa * n / 252.0), index=px.index)
    for c in cols:
        out[c] = out[c] * factor
    return out


def trend_rule(assets, lookback=252):
    def fn(hist):
        ok = [a for a in assets
              if a in hist.columns and hist[a].iloc[-260:].notna().all()]
        if not ok:
            return {CASHCOL: 1.0}
        end, start = hist[ok].iloc[-1], hist[ok].iloc[-1 - lookback]
        r = (end / start - 1.0).dropna()
        winners = [a for a in ok if r.get(a, -1) > 0]
        if not winners:
            return {CASHCOL: 1.0}
        w = {a: 1.0 / len(ok) for a in winners}
        w[CASHCOL] = 1.0 - sum(w.values())
        return w
    return fn


def hold_rule(assets):
    def fn(hist):
        ok = [a for a in assets
              if a in hist.columns and hist[a].iloc[-260:].notna().all()]
        if not ok:
            return {CASHCOL: 1.0}
        return {a: 1.0 / len(ok) for a in ok}
    return fn


def compare(px, assets, label, start=None, end=None, show_invested=False):
    rows, invested = [], None
    for name, fn in (("trend", trend_rule(assets)),
                     ("buy and hold", hold_rule(assets))):
        rec = U.run_portfolio(px, fn, freq="M", cost_bps=COST_BPS,
                              start=start, end=end)
        s = U.summarise(rec, name)
        rows.append(s)
        if name == "trend":
            # A portfolio that is always in cash is the failure mode this
            # test had first time round, so it is reported, not assumed away.
            invested = _invested_share(px, fn, start, end)
    if any(r.get("years", 0) < 3 for r in rows):
        return None
    t, b = rows
    tag = f" [in market {100 * invested:.0f}%]" if show_invested else ""
    print(f"  {label:<26}"
          f"{t['annual_pct']:>8.2f}%{t['risk_adj']:>8.2f}"
          f"{t['worst_fall_pct']:>9.1f}%"
          f"   | {b['annual_pct']:>7.2f}%{b['risk_adj']:>8.2f}"
          f"{b['worst_fall_pct']:>9.1f}%"
          f"   |{t['risk_adj'] - b['risk_adj']:>+7.2f}{tag}")
    return t["risk_adj"] - b["risk_adj"]


def _invested_share(px, fn, start, end):
    """Average share of the portfolio not sitting in cash."""
    sub = px
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]
    dates = U.rebalance_dates(sub.index, "M")
    vals = []
    for day in dates:
        hist = px.loc[:day]
        if len(hist) < 300:
            continue
        w = fn(hist)
        vals.append(1.0 - float(w.get(CASHCOL, 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def header():
    print(f"  {'':<26}{'--------- trend ---------':>25}"
          f"   | {'------ buy & hold ------':>24}   | {'diff':>6}")
    print(f"  {'':<26}{'a year':>8}{'risk-adj':>8}{'worst':>10}"
          f"   | {'a year':>7}{'risk-adj':>8}{'worst':>10}   |")


def main() -> int:
    pd.set_option("display.width", 220)

    print("=" * 104)
    print("TEST A: thirteen foreign share indices, never used before")
    print("=" * 104)
    fx = fetch(list(FOREIGN))
    fx[CASHCOL] = cash_index(fx.index)
    fx_td = add_dividends(fx, list(FOREIGN))
    assets = list(FOREIGN)

    print("\n  price only (no dividends -- flatters the rule)")
    header()
    diffs = []
    for lab, a, b in (("whole history", None, None),
                      ("1995-2005", "1995-01-01", "2005-12-31"),
                      ("2006-2015", "2006-01-01", "2015-12-31"),
                      ("2016-2026", "2016-01-01", None)):
        d = compare(fx, assets, lab, a, b, show_invested=True)
        if d is not None:
            diffs.append(d)

    print("\n  with dividends added back (3% a year -- the fair comparison)")
    header()
    fair = []
    for lab, a, b in (("whole history", None, None),
                      ("1995-2005", "1995-01-01", "2005-12-31"),
                      ("2006-2015", "2006-01-01", "2015-12-31"),
                      ("2016-2026", "2016-01-01", None)):
        d = compare(fx_td, assets, lab, a, b, show_invested=True)
        if d is not None:
            fair.append(d)

    print(f"\n  trend beat buy-and-hold in {sum(x > 0 for x in fair)} of "
          f"{len(fair)} windows once dividends are counted")

    print("\n  --- each market on its own, whole history, with dividends ---")
    header()
    wins = 0
    total = 0
    for t, country in FOREIGN.items():
        d = compare(fx_td, [t], f"{country} ({t})")
        if d is not None:
            total += 1
            wins += d > 0
    print(f"\n  trend improved risk-adjusted return on {wins} of {total} "
          f"markets")

    print("\n" + "=" * 104)
    print("TEST B: the S&P 1928-1992, before any data used elsewhere here")
    print("=" * 104)
    us = fetch(["^GSPC"])
    us[CASHCOL] = cash_index(us.index)
    us_td = add_dividends(us, ["^GSPC"], 0.04)   # dividends were richer then

    print("\n  price only")
    header()
    for lab, a, b in (("1928-1992 all", None, "1992-12-31"),
                      ("1928-1949", None, "1949-12-31"),
                      ("1950-1969", "1950-01-01", "1969-12-31"),
                      ("1970-1992", "1970-01-01", "1992-12-31")):
        compare(us, ["^GSPC"], lab, a, b)

    print("\n  with dividends added back (4% a year)")
    header()
    outcomes = []
    for lab, a, b in (("1928-1992 all", None, "1992-12-31"),
                      ("1928-1949", None, "1949-12-31"),
                      ("1950-1969", "1950-01-01", "1969-12-31"),
                      ("1970-1992", "1970-01-01", "1992-12-31")):
        d = compare(us_td, ["^GSPC"], lab, a, b)
        if d is not None:
            outcomes.append((lab, d))
    print(f"\n  trend beat buy-and-hold in "
          f"{sum(d > 0 for _, d in outcomes)} of {len(outcomes)} eras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
