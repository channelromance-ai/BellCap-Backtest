"""
Daily data for overnight research, with every predictor lagged.

The intraday work was starved of data: 763 nights is enough to measure a
large effect and nothing else. Overnight returns need only each day's open
and close, and that is available free for decades, so the same question can
be asked with eleven times the sample.

Prices are dividend-adjusted on purpose. An overnight holder who buys the
close before an ex-dividend date owns the shares on the ex-date and receives
the dividend, so the raw price gap on that morning overstates the loss.
Adjusted prices put the dividend back where it belongs.

Every column here is knowable at the closing bell of day t. The thing being
predicted, `overnight`, is the move from that close to the next morning's
open. Nothing in a row may come from the night it is trying to predict.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM"]


def _download(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ticker, period="max", interval="1d",
                    progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)
    return d[["open", "high", "low", "close", "volume"]].dropna()


def build(ticker: str, vix: pd.Series | None = None) -> pd.DataFrame:
    """One instrument's daily frame with lagged predictors."""
    d = _download(ticker)
    out = pd.DataFrame(index=d.index)
    out["sym"] = ticker
    out["open"] = d["open"]
    out["close"] = d["close"]

    # --- the two things to be explained
    # Held from tonight's close to tomorrow's open.
    out["overnight"] = d["open"].shift(-1) / d["close"] - 1.0
    # Held through tomorrow's session, for comparison.
    out["daytime"] = d["close"] / d["open"] - 1.0
    out["all_day"] = d["close"].pct_change()

    # --- predictors, all as of today's close
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    out["day_ret"] = d["close"] / d["open"] - 1.0
    out["close_in_range"] = (d["close"] - d["low"]) / rng
    out["ret_5d"] = d["close"].pct_change(5)
    out["ret_1d"] = d["close"].pct_change()
    out["vol_20d"] = d["close"].pct_change().rolling(20).std()
    out["vol_ratio"] = out["vol_20d"] / out["vol_20d"].rolling(100).mean()
    out["vol_vs_avg"] = d["volume"] / d["volume"].rolling(20).mean()
    out["prev_overnight"] = out["overnight"].shift(1)
    out["gap_streak"] = (
        np.sign(out["overnight"].shift(1))
        .groupby((np.sign(out["overnight"].shift(1))
                  != np.sign(out["overnight"].shift(2))).cumsum())
        .cumcount() + 1)

    out["dow"] = out.index.dayofweek
    out["dom"] = out.index.day
    # Trading days from the end of the calendar month, counted backwards.
    grp = out.groupby([out.index.year, out.index.month])
    out["td_from_end"] = grp.cumcount(ascending=False)
    out["td_from_start"] = grp.cumcount()
    out["turn_of_month"] = ((out["td_from_end"] <= 2)
                            | (out["td_from_start"] <= 2))

    if vix is not None:
        v = vix.reindex(out.index).ffill()
        out["vix"] = v
        out["vix_chg"] = v.pct_change()
        out["vix_z"] = (v - v.rolling(250).mean()) / v.rolling(250).std()

    # The last row has no next open, so nothing to predict.
    return out.dropna(subset=["overnight"])


def vix_series() -> pd.Series:
    import yfinance as yf
    v = yf.download("^VIX", period="max", interval="1d",
                    progress=False, auto_adjust=False)
    if isinstance(v.columns, pd.MultiIndex):
        v.columns = v.columns.get_level_values(0)
    return v["Close"]


def panel(tickers=TICKERS, with_vix=True) -> pd.DataFrame:
    """Every instrument stacked into one frame."""
    vix = vix_series() if with_vix else None
    frames = []
    for t in tickers:
        try:
            frames.append(build(t, vix))
        except Exception as exc:                       # noqa: BLE001
            print(f"  {t}: {exc}")
    out = pd.concat(frames).sort_index()
    out.index.name = "date"
    return out
