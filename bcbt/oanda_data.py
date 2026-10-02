"""
OANDA CFD history from the public FutureSharks/financial-data repository.

1-minute OANDA bars, 2005 - mid 2020, for 28 CFDs: equity indices, FX,
gold, oil, gas, grains and government-bond CFDs. These are a retail CFD
broker's own quotes (midpoint, with tick volume), which is the right thing
to test retail CFD models on. Used here because it is reachable over plain
GitHub when Dukascopy and Yahoo are not.

Bars are stored resampled to 15 minutes and 1 hour, UTC, in
data/oanda_m15.parquet and data/oanda_h1.parquet; daily VIX goes to
data/vix_daily.csv.

    python -m bcbt.oanda_data          # ~190 MB, a few minutes
"""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

RAW = ("https://raw.githubusercontent.com/FutureSharks/financial-data/master/"
       "pyfinancialdata/data/currencies/oanda/{s}/{y}/oanda-{s}-{y}-{m}.csv")

SYMBOLS = ["SPX500_USD", "NAS100_USD", "US2000_USD", "UK100_GBP", "FR40_EUR",
           "JP225_USD", "AU200_AUD", "NL25_EUR",
           "EUR_USD", "GBP_USD", "AUD_USD", "USD_CAD", "EUR_JPY", "AUD_JPY",
           "XAU_USD", "WTICO_USD", "NATGAS_USD", "CORN_USD", "WHEAT_USD",
           "SOYBN_USD", "SUGAR_USD",
           "USB02Y_USD", "USB10Y_USD", "DE10YB_EUR", "UK10YB_GBP"]

VIX_URL = ("https://raw.githubusercontent.com/datasets/finance-vix/main/data/"
           "vix-daily.csv")
VIX = "data/vix_daily.csv"
M15 = "data/oanda_m15.parquet"
H1 = "data/oanda_h1.parquet"
AGG = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}


def _one(sym, y, m, session):
    for _ in range(4):
        try:
            r = session.get(RAW.format(s=sym, y=y, m=m), timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text), parse_dates=["time"])
            d = d.rename(columns=dict(open="o", high="h", low="l", close="c",
                                      volume="v")).set_index("time")
            return d[["o", "h", "l", "c", "v"]]
        except Exception:
            continue
    print("failed", sym, y, m, file=sys.stderr)
    return None


def fetch_symbol(sym, years=range(2005, 2021)):
    s = requests.Session()
    jobs = [(y, m) for y in years for m in range(1, 13)]
    with ThreadPoolExecutor(12) as ex:
        parts = list(ex.map(lambda ym: _one(sym, ym[0], ym[1], s), jobs))
    m1 = pd.concat([p for p in parts if p is not None]).sort_index()
    m1 = m1[~m1.index.duplicated()]
    out = {}
    for rule in ("15min", "1h"):
        b = m1.resample(rule, closed="left", label="left").agg(AGG).dropna()
        b["sym"] = sym
        out[rule] = b
    return out


def fetch_vix(path=VIX):
    """CBOE VIX daily OHLC (datasets/finance-vix on GitHub)."""
    r = requests.get(VIX_URL, timeout=60)
    r.raise_for_status()
    with open(path, "w") as f:
        f.write(r.text)


def build(symbols=SYMBOLS):
    os.makedirs("data", exist_ok=True)
    fetch_vix()
    m15, h1 = [], []
    for s in symbols:
        o = fetch_symbol(s)
        m15.append(o["15min"])
        h1.append(o["1h"])
        print(s, len(o["1h"]), o["1h"].index[0], o["1h"].index[-1], flush=True)
    pd.concat(m15).to_parquet(M15)
    pd.concat(h1).to_parquet(H1)


def load(rule="1h", sym=None):
    path = H1 if rule == "1h" else M15
    filters = [("sym", "==", sym)] if sym else None
    d = pd.read_parquet(path, filters=filters)
    d.index = pd.DatetimeIndex(d.index).tz_localize("UTC")
    return d


if __name__ == "__main__":
    build(sys.argv[1:] or SYMBOLS)
