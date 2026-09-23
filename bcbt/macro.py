"""
Signals from other markets, for timing shares.

Everything tested in this repo so far used the price of the thing being
traded to predict itself. That is the most examined data series in the
world, and none of it worked.

This is a different kind of bet. The corporate bond market, the government
yield curve and the options market all price the same economy as the share
market, and the usual claim is that bond investors notice deterioration
first: they are lending, not owning, so they care about survival rather
than growth. If that is true then a widening gap between what risky
companies pay to borrow and what the government pays should show up before
share prices fall.

The headline series is BAA10Y -- what a middling-quality company pays to
borrow for ten years, minus what the US government pays. It is daily, goes
back to 1986, and needs no key. The better-known high-yield spreads are
licensed and FRED only serves three years of those publicly, which is not
enough to test anything.

Two rules govern everything here.

Nothing may use a number before it was published. Market-priced series
(spreads, yields, the VIX) are known the same evening but are lagged one
business day anyway. The Chicago Fed's financial conditions index is
weekly and comes out the following Wednesday, so it is lagged a full
fortnight -- generous, and the point is to be certain rather than clever.

Nothing may use the whole history to judge the present. "A wide spread" has
to mean wide compared with what had already happened, so every z-score is
computed on a trailing window only.
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# series id -> (readable name, business days to lag before it may be used)
SERIES = {
    "BAA10Y": ("credit_spread", 1),      # Baa corporate minus 10y Treasury
    "AAA10Y": ("safe_spread", 1),        # Aaa minus 10y, a quality gauge
    "T10Y3M": ("yield_curve", 1),        # 10y minus 3m
    "T10Y2Y": ("curve_2s10s", 1),
    "VIXCLS": ("vix", 1),
    "DGS10": ("yield_10y", 1),
    "DFF": ("fed_funds", 1),
    "NFCI": ("fin_conditions", 10),      # weekly, published the next week
}

PANEL = "data/macro_panel.parquet"


def fred(series_id: str) -> pd.Series:
    r = requests.get(FRED, params={"id": series_id, "cosd": "1900-01-01"},
                     timeout=60)
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().set_index("date")["value"].sort_index()


def equity(ticker="^GSPC", div_yield=0.025) -> pd.DataFrame:
    """
    The share index, with dividends approximated back in.

    These are price indices. A buy-and-hold investor receives dividends, so
    comparing a timing rule against a price-only index hands the rule a free
    advantage. Adding a flat yield back is rough but errs the right way.
    """
    import yfinance as yf
    d = yf.download(ticker, period="max", interval="1d", progress=False,
                    auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    px = d["Close"].dropna()
    total = px * np.exp(div_yield * np.arange(len(px)) / 252.0)
    return pd.DataFrame({"price": px, "total": total})


def roll_z(s: pd.Series, window: int = 750, minp: int = 250) -> pd.Series:
    """How unusual is today, judged only against what came before it."""
    mu = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std()
    return (s - mu) / sd.replace(0, np.nan)


def build(path=PANEL) -> pd.DataFrame:
    eq = equity()
    idx = eq.index

    cols = {}
    for sid, (name, lag) in SERIES.items():
        try:
            s = fred(sid)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {sid}: {exc}")
            continue
        # Put it on the share calendar, carry it forward, then delay it by
        # the number of days it takes to become public.
        cols[name] = s.reindex(idx).ffill().shift(lag)

    df = pd.DataFrame(cols, index=idx)
    df["price"] = eq["price"]
    df["total"] = eq["total"]

    # --- what we are trying to predict: the next month, and the next day
    df["fwd_1m"] = df["total"].shift(-21) / df["total"] - 1.0
    df["fwd_1d"] = df["total"].shift(-1) / df["total"] - 1.0

    # --- derived signals, all backward-looking
    df["credit_chg_1m"] = df["credit_spread"] - df["credit_spread"].shift(21)
    df["credit_chg_3m"] = df["credit_spread"] - df["credit_spread"].shift(63)
    df["credit_z"] = roll_z(df["credit_spread"])
    df["credit_chg_z"] = roll_z(df["credit_chg_1m"])
    df["quality_spread"] = df["credit_spread"] - df["safe_spread"]
    df["quality_z"] = roll_z(df["quality_spread"])

    df["curve_z"] = roll_z(df["yield_curve"])
    df["vix_z"] = roll_z(df["vix"])
    realised = df["price"].pct_change().rolling(21).std() * np.sqrt(252) * 100
    df["realised_vol"] = realised
    # What the options market charges above what actually happened. A
    # positive number is the premium paid for insurance.
    df["vol_premium"] = df["vix"] - realised
    df["vol_premium_z"] = roll_z(df["vol_premium"])
    df["nfci_chg"] = df["fin_conditions"] - df["fin_conditions"].shift(21)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return df


def load(path=PANEL) -> pd.DataFrame:
    if not os.path.exists(path):
        return build(path)
    return pd.read_parquet(path)
