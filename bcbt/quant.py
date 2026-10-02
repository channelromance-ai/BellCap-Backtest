"""
A vectorised engine for testing quant signals across a CFD universe.

Every model in the quant battery reduces to the same thing: a target
position in each instrument, decided from information up to a bar's close
and held to the next close. Keeping that one shape means the costs, the
financing, the sizing and the scoring are written once and are the same for
every model, so the only thing that differs between two rows of the results
is the idea being tested.

Conventions
-----------
* A signal frame `s` (index = bar times, columns = symbols) is the desired
  direction or weight, known at that bar's close. It is shifted one bar
  inside `run` -- a model never has to remember to do that itself.
* Each instrument is scaled to the same risk (10% a year, from an EWMA of
  its own past returns) unless the model asks for raw weights, so a 2-year
  bond CFD and natural gas contribute alike and can be averaged.
* Costs are charged on every change of position at half the round-trip
  spread plus slippage. Financing is charged on the notional held over each
  17:00 New York roll, three nights over a weekend, at a flat retail markup
  on both sides (the benchmark rate itself is left out: it was close to zero
  for USD, EUR and JPY over most of this sample, and it is a carry the model
  did not choose).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

# Round-trip cost in basis points of price: a typical retail CFD spread for
# the instrument plus one basis point of slippage. Deliberately not the
# tightest quotes a broker advertises.
SPREAD_BP = {
    "SPX500_USD": 1.0, "NAS100_USD": 1.5, "US2000_USD": 3.0,
    "UK100_GBP": 1.5, "FR40_EUR": 2.0, "JP225_USD": 2.0, "AU200_AUD": 2.0,
    "NL25_EUR": 3.0,
    "EUR_USD": 1.2, "GBP_USD": 1.6, "AUD_USD": 2.0, "USD_CAD": 2.0,
    "EUR_JPY": 2.0, "AUD_JPY": 2.5,
    "XAU_USD": 3.0, "WTICO_USD": 5.0, "NATGAS_USD": 12.0,
    "CORN_USD": 15.0, "WHEAT_USD": 15.0, "SOYBN_USD": 10.0,
    "SUGAR_USD": 20.0,
    "USB02Y_USD": 2.0, "USB10Y_USD": 3.0, "DE10YB_EUR": 3.0,
    "UK10YB_GBP": 4.0,
}
SLIP_BP = 1.0
FIN_YR = {"fx": 0.01, "other": 0.025}     # retail markup, both directions

GROUPS = {
    "index": ["SPX500_USD", "NAS100_USD", "US2000_USD", "UK100_GBP",
              "FR40_EUR", "JP225_USD", "AU200_AUD", "NL25_EUR"],
    "fx": ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CAD", "EUR_JPY", "AUD_JPY"],
    "commodity": ["XAU_USD", "WTICO_USD", "NATGAS_USD", "CORN_USD",
                  "WHEAT_USD", "SOYBN_USD", "SUGAR_USD"],
    "bond": ["USB02Y_USD", "USB10Y_USD", "DE10YB_EUR", "UK10YB_GBP"],
}
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}
ALL = [s for ss in GROUPS.values() for s in ss]

DEV_END = pd.Timestamp("2013-01-01")      # development / hold-out split
TARGET_VOL = 0.10
MAX_LEV = 3.0      # no retail account levers a 2-year bond CFD 8x


# ------------------------------------------------------------------ panels

def to_ny_day(idx):
    """Trading day of each UTC bar: the day that ends at the 17:00 NY roll."""
    ny = idx.tz_convert("America/New_York")
    return (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)


def panels(bars: pd.DataFrame, rule=None):
    """
    Long bar table (index = UTC time, columns o h l c v sym) to a dict of
    wide frames O H L C V. With rule='D' the bars are folded into trading
    days that end at the 17:00 New York roll.
    """
    out = {}
    if rule == "D":
        b = bars.copy()
        b["day"] = to_ny_day(b.index)
        g = b.groupby(["day", "sym"])
        d = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                  c=("c", "last"), v=("v", "sum"))
        for k in "ohlcv":
            out[k.upper()] = d[k].unstack("sym")
        # Drop holiday stubs: a day with almost no trading in most markets.
        n = out["C"].notna().sum(axis=1)
        keep = (n >= 0.5 * out["C"].shape[1]) & (out["C"].index.dayofweek < 5)
        out = {k: v[keep] for k, v in out.items()}
        # A market shut for a day it would normally trade (a local holiday)
        # keeps yesterday's price, so no indicator sees a hole.
        filled = out["C"].ffill(limit=5)
        hole = out["C"].isna() & filled.notna()
        for k in "OHL":
            out[k] = out[k].where(~hole, filled)
        out["C"] = filled
        out["V"] = out["V"].where(~hole, 0.0)
    else:
        for k in "ohlcv":
            out[k.upper()] = bars.pivot_table(index=bars.index, columns="sym",
                                              values=k, aggfunc="first")
    cols = [s for s in ALL if s in out["C"].columns]
    return {k: v[cols] for k, v in out.items()}


# ------------------------------------------------------------------- engine

def ewm_vol(r, span=60, min_periods=20):
    return r.ewm(span=span, min_periods=min_periods).std()


def run(sig, C, *, bars_per_year=252, scale=True, rolls=None,
        target_vol=TARGET_VOL, vol_span=None, cost_mult=1.0, fin=True,
        weights=None, agg="mean", buffer=0.25, name=None):
    """
    Turn a signal frame into a net return stream.

    sig      direction/weight per bar per symbol, known at that bar's close
    C        closes, same shape
    scale    vol-scale each instrument to target_vol (else sig is notional)
    rolls    number of 17:00 NY rolls crossed between bar t and t+1 (Series
             indexed like C); defaults to calendar nights for daily bars
    weights  optional per-symbol portfolio weights (default equal across
             symbols that have a position available)
    buffer   resize an open position only when the target differs from it by
             more than this fraction (sign changes and exits always go
             through) -- nobody re-trades a 3% size drift every day
    agg      "sum" adds the symbols' P&L instead of averaging it, for
             models whose weights are already whole-portfolio notionals
             (pairs, overlays)
    Returns a Result.
    """
    # A missing bar (holiday, feed gap, a market's closed hours) is not a
    # reason to close and re-open a position: carry price and signal over
    # short gaps so the gap costs nothing and earns nothing.
    gap = 5 if bars_per_year <= 260 else (24 if bars_per_year < 10000
                                          else 96)
    live = C.notna()
    C = C.ffill(limit=gap)
    sig = sig.reindex_like(C).astype(float).ffill(limit=gap)
    r = C.pct_change(fill_method=None)
    if scale:
        span = vol_span or (60 if bars_per_year <= 260 else 60 * 23)
        vol = ewm_vol(r, span) * math.sqrt(bars_per_year)
        lev = (target_vol / vol).clip(upper=MAX_LEV)
        w = sig * lev
    else:
        w = sig.copy()
    # Positions can only change on a bar where the market actually traded.
    w = w.where(live).ffill(limit=gap).where(C.notna())
    w = w.fillna(0.0)
    if buffer:
        w = pd.DataFrame(_buffered(w.to_numpy(float), buffer),
                         index=w.index, columns=w.columns)

    nxt = r.shift(-1).fillna(0.0)
    gross = w * nxt

    sb = pd.Series({s: SPREAD_BP.get(s, 5.0) for s in C.columns})
    side = (sb / 2 + SLIP_BP) / 1e4 * cost_mult
    trade = w.diff().abs()
    trade.iloc[0] = w.iloc[0].abs()
    cost = trade * side

    if fin:
        if rolls is None:
            gap = C.index.to_series().diff().shift(-1).dt.days.fillna(1)
            rolls = gap.clip(lower=1)
        fy = pd.Series({s: FIN_YR["fx" if GROUP_OF.get(s) == "fx" else
                                   "other"] for s in C.columns})
        financing = w.abs().mul(rolls, axis=0) * (fy / 365.0)
    else:
        financing = w * 0.0

    net = gross - cost - financing
    active = (w != 0) | (w.shift(1).fillna(0) != 0)

    def agg_(x):
        if agg == "sum":
            return x.sum(axis=1)
        if weights is None:
            return x.sum(axis=1) / C.notna().sum(axis=1).replace(0, np.nan)
        wt = pd.Series(weights).reindex(C.columns).fillna(0)
        return (x * wt).sum(axis=1)

    return Result(name=name, net=agg_(net).fillna(0.0),
                  gross=agg_(gross).fillna(0.0),
                  nofin=agg_(gross - cost).fillna(0.0),
                  per_sym=net, w=w, bars_per_year=bars_per_year,
                  turnover=float(trade.sum(axis=1).mean()),
                  exposure=float(active.mean().mean()))


@njit(cache=True)
def _buffered(w, band):
    out = np.zeros_like(w)
    n, m = w.shape
    for j in range(m):
        h = 0.0
        for i in range(n):
            t = w[i, j]
            if t == 0.0 or h == 0.0 or t * h < 0 or \
                    abs(t - h) > band * abs(h):
                h = t
            out[i, j] = h
    return out


class Result:
    def __init__(self, name, net, gross, nofin, per_sym, w, bars_per_year,
                 turnover, exposure):
        self.name, self.net, self.gross, self.nofin = name, net, gross, nofin
        self.per_sym, self.w = per_sym, w
        self.bpy, self.turnover, self.exposure = (bars_per_year, turnover,
                                                  exposure)

    def daily(self, which="net"):
        """Returns summed to trading days, for cross-model tests."""
        x = getattr(self, which)
        if self.bpy <= 260:
            return x
        return x.groupby(to_ny_day(x.index)).sum()


def sharpe(x, ppy=252):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * math.sqrt(ppy))


def nw_t(x, lags=5):
    """Newey-West t statistic of the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def max_dd(x):
    eq = np.cumsum(np.asarray(x, float))
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def score(res: Result, bench_daily=None):
    """One results row for a model."""
    d = res.daily()
    d = d[d.index >= d.index[0]]
    active = d[d != 0]
    first = active.index[0] if len(active) else d.index[0]
    d = d[d.index >= first]
    dev, hold = d[d.index < DEV_END], d[d.index >= DEV_END]
    gd = res.daily("gross")
    gd = gd[gd.index >= first]
    nf = res.daily("nofin")
    nf = nf[nf.index >= first]
    row = dict(
        model=res.name,
        sharpe=sharpe(d), sharpe_gross=sharpe(gd), sharpe_nofin=sharpe(nf),
        sharpe_dev=sharpe(dev), sharpe_hold=sharpe(hold),
        t=nw_t(d), t_dev=nw_t(dev), t_hold=nw_t(hold),
        ann_ret=float(d.mean() * 252), ann_vol=float(d.std() * math.sqrt(252)),
        max_dd=max_dd(d),
        turnover=res.turnover, exposure=res.exposure,
        sym_pos=float((res.per_sym.sum() > 0).mean()),
    )
    if bench_daily is not None:
        b = bench_daily.reindex(d.index).fillna(0.0)
        if b.std() > 0:
            beta = float(np.cov(d, b)[0, 1] / b.var())
            row["beta_long"] = beta
            row["alpha_t"] = nw_t(d - beta * b)
    return row


# --------------------------------------------------------- multiple testing

def reality_check(streams: dict, n_boot=2000, block=10, seed=0):
    """
    White's Reality Check on daily Sharpe across every model tested.

    Each stream is demeaned so none has an edge by construction; whole
    blocks of days are resampled together; the p value is how often the
    best of the demeaned models beats the best real one.
    """
    df = pd.DataFrame(streams).fillna(0.0)
    X = df.to_numpy()
    nd = len(X)
    sd = X.std(axis=0)
    sd[sd == 0] = np.nan
    obs = X.mean(axis=0) / sd
    best = int(np.nanargmax(obs))
    Xd = X - X.mean(axis=0)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / block))
    span = np.arange(block)
    mx = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, nd, size=nb)
        idx = (st[:, None] + span[None, :]).ravel()[:nd] % nd
        mx[b] = np.nanmax(Xd[idx].mean(axis=0) / sd)
    ann = math.sqrt(252)
    return (df.columns[best], float(obs[best] * ann),
            float((mx >= obs[best]).mean()), mx * ann)


def deflated_sharpe(sr_ann, n_obs, n_trials, var_trials, skew=0.0, kurt=3.0,
                    ppy=252):
    """
    Bailey & Lopez de Prado's deflated Sharpe ratio: the probability that a
    Sharpe this high is real after picking it as the best of n_trials.
    """
    from scipy.stats import norm
    sr = sr_ann / math.sqrt(ppy)
    v = var_trials / ppy
    g = 0.5772156649
    e_max = math.sqrt(max(v, 1e-12)) * (
        (1 - g) * norm.ppf(1 - 1 / n_trials)
        + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(norm.cdf((sr - e_max) * math.sqrt(n_obs - 1) / den))
