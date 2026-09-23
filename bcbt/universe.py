"""
A multi-asset daily price panel, and a portfolio engine that charges costs.

Every system tested in this repo so far shared two properties, and both are
deliberately abandoned here.

They traded constantly. A hundred to two hundred and fifty round trips a
year, against a spread, is a two to twelve percent annual hurdle before the
idea gets a chance. A monthly rebalance is twelve trades a year and a hurdle
near zero, so an effect one tenth the size is still worth having.

And they were all directional bets on one liquid index. That is the most
competed corner of the market. Owning the strongest of twenty things and
avoiding the weakest is a different question, and one where being wrong about
the market's direction does not automatically cost money.

The universe is built from long-lived, liquid, tradeable funds spanning
equities, bonds, commodities and property. Choosing today's survivors is a
mild bias -- the funds that closed are not here -- but these are all major
products that existed throughout, so it is small and it is stated.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Broad asset classes, each a liquid fund with a long record.
UNIVERSE = {
    "SPY": "US large cap",
    "IWM": "US small cap",
    "QQQ": "US tech",
    "EFA": "developed intl",
    "EEM": "emerging mkts",
    "VNQ": "US property",
    "GLD": "gold",
    "DBC": "commodities",
    "TLT": "long treasuries",
    "IEF": "7-10y treasuries",
    "LQD": "corporate bonds",
    "HYG": "high yield",
}

# US sectors, for cross-sectional work inside one asset class.
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]

CASH = "BIL"        # T-bill fund, the do-nothing option

PANEL = "data/universe_daily.parquet"


def fetch(tickers, start="1990-01-01") -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker."""
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, start=start, interval="1d",
                        progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if len(d) > 250:
            out[t] = d["Close"]
    px = pd.DataFrame(out).sort_index()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px


def build(path=PANEL) -> pd.DataFrame:
    tickers = list(UNIVERSE) + SECTORS + [CASH]
    px = fetch(tickers)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_parquet(path)
    return px


def load(path=PANEL) -> pd.DataFrame:
    if not os.path.exists(path):
        return build(path)
    return pd.read_parquet(path)


# ------------------------------------------------------------------ engine

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each month, week, or every day."""
    if freq == "D":
        return index
    key = (index.to_period("M") if freq == "M" else index.to_period("W"))
    return index[~pd.Series(key, index=index).duplicated(keep="last")]


def run_portfolio(px: pd.DataFrame, weight_fn, freq="M", cost_bps=5.0,
                  warmup=252, start=None, end=None) -> pd.DataFrame:
    """
    Walk a rebalanced portfolio and return its daily record.

    `weight_fn(history) -> dict` is handed only prices up to and including
    the rebalance date and must return target weights. It never sees the
    future; the weights it returns are applied from the NEXT day onward,
    so a decision made on a close is not also earning that close's move.

    Costs are charged on turnover at each rebalance: moving from 0% to 100%
    of a holding costs the full spread, trimming 10% costs a tenth of it.
    """
    px = px.sort_index()
    if start:
        px = px[px.index >= start]
    if end:
        px = px[px.index <= end]

    rets = px.pct_change()
    dates = px.index
    rebals = set(rebalance_dates(dates, freq))

    cost = cost_bps / 10_000.0
    w = pd.Series(0.0, index=px.columns)
    rows = []

    for i, day in enumerate(dates):
        if i < warmup:
            rows.append((day, 0.0, 0.0, 0.0))
            continue

        # Today's return is earned on the weights set before today.
        r = float((w * rets.loc[day].fillna(0.0)).sum())

        turn = 0.0
        if day in rebals:
            hist = px.iloc[: i + 1]
            target = weight_fn(hist)
            if target is not None:
                tgt = pd.Series(target, dtype=float).reindex(
                    px.columns).fillna(0.0)
                turn = float((tgt - w).abs().sum())
                w = tgt
        rows.append((day, r, turn, turn * cost))

    out = pd.DataFrame(rows, columns=["date", "gross", "turnover",
                                      "cost"]).set_index("date")
    out["net"] = out["gross"] - out["cost"]
    return out


def summarise(rec: pd.DataFrame, label="", cash_ret: pd.Series | None = None
              ) -> dict:
    """
    Score a portfolio's record.

    `cash_ret` is the daily return on cash, and passing it matters more than
    it looks. Return per unit of wobble computed on RAW returns rewards any
    rule that sits in cash, because cash pays interest and barely moves. Run
    that way a cash-only portfolio scores 14 and wins everything, which is
    how this was found. Worse, it quietly flattered every rule that spends
    time out of the market: the trend rule's apparent advantage over buying
    and holding across thirteen foreign markets was 13 of 13 on raw returns
    and 6 of 13 -- a coin toss -- once cash was subtracted.

    Pass the cash series whenever the comparison is between rules that hold
    different amounts of cash. Leaving it out keeps the old behaviour, which
    is only safe when everything being compared is always fully invested.
    """
    r = rec["net"].fillna(0.0)
    r = r[r.index >= r.ne(0).idxmax()]          # drop the warm-up zeros
    if len(r) < 250:
        return dict(label=label, years=0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    years = len(r) / 252.0
    ann = eq.iloc[-1] ** (1.0 / years) - 1.0
    dd = (eq / peak - 1.0).min()
    downside = r[r < 0].std(ddof=1) * np.sqrt(252)

    # The reward for taking risk is measured above cash, not above zero.
    ex = r if cash_ret is None else r - cash_ret.reindex(r.index).fillna(0.0)
    vol = ex.std(ddof=1) * np.sqrt(252)
    reward = (ann if cash_ret is None else ex.mean() * 252)
    return dict(
        label=label,
        years=round(years, 1),
        annual_pct=100 * ann,
        vol_pct=100 * r.std(ddof=1) * np.sqrt(252),
        risk_adj=reward / vol if vol > 0 else np.nan,
        sortino=ann / downside if downside > 0 else np.nan,
        worst_fall_pct=100 * dd,
        calmar=ann / abs(dd) if dd < 0 else np.nan,
        turnover_yr=rec["turnover"].sum() / years,
        cost_pct_yr=100 * rec["cost"].sum() / years,
    )
