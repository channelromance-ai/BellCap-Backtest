"""
Shared indicators and building blocks for the quant battery.

Every function takes and returns wide frames (rows = bars, columns =
symbols) and uses only data up to each row, so a model built from them is
known at that bar's close. The engine (`bcbt.quant.run`) applies the one-bar
shift; nothing here should.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit


# ------------------------------------------------------------ basic series

def sma(x, n):
    return x.rolling(n, min_periods=n).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rstd(x, n):
    return x.rolling(n, min_periods=n).std()


def zscore(x, n):
    return (x - sma(x, n)) / rstd(x, n)


def ret(C, n=1):
    return C.pct_change(n, fill_method=None)


def logp(C):
    return np.log(C)


def true_range(D):
    pc = D["C"].shift(1)
    return np.maximum(D["H"] - D["L"],
                      np.maximum((D["H"] - pc).abs(), (D["L"] - pc).abs()))


def atr(D, n=14):
    return true_range(D).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(C, n=14):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False,
                                min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def stoch(D, n=14, k=3):
    lo = D["L"].rolling(n).min()
    hi = D["H"].rolling(n).max()
    return (100 * (D["C"] - lo) / (hi - lo).replace(0, np.nan)).rolling(k).mean()


def cci(D, n=20):
    tp = (D["H"] + D["L"] + D["C"]) / 3
    md = tp.rolling(n).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    return (tp - sma(tp, n)) / (0.015 * md)


def willr(D, n=14):
    hi = D["H"].rolling(n).max()
    lo = D["L"].rolling(n).min()
    return -100 * (hi - D["C"]) / (hi - lo).replace(0, np.nan)


def adx(D, n=14):
    h, l, c = D["H"], D["L"], D["C"]
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(D).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / tr
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, ndi


def roll_slope(x, n):
    """OLS slope of the last n points against time, per column."""
    t = np.arange(n) - (n - 1) / 2
    w = t / (t @ t)
    return x.rolling(n, min_periods=n).apply(lambda a: a @ w, raw=True)


def roll_linreg(x, n):
    """Fitted value at the last point, slope, and residual std."""
    t = np.arange(n, dtype=float)
    tm = t.mean()
    sxx = ((t - tm) ** 2).sum()
    m = sma(x, n)
    slope = roll_slope(x, n)
    fit_last = m + slope * (n - 1 - tm)
    # residual std via E[x^2] - fitted variance
    var_x = x.rolling(n).var(ddof=0)
    res_var = (var_x - slope ** 2 * sxx / n).clip(lower=0)
    return fit_last, slope, np.sqrt(res_var * n / max(n - 2, 1))


def r_squared(x, n):
    t = np.arange(n, dtype=float)
    sxx = ((t - t.mean()) ** 2).sum()
    slope = roll_slope(x, n)
    var_x = x.rolling(n).var(ddof=0)
    return (slope ** 2 * sxx / n / var_x.replace(0, np.nan)).clip(0, 1)


def rolling_pct_rank(x, n):
    """Percentile of the last value within the trailing window, 0..1."""
    return x.rolling(n, min_periods=n).rank(pct=True)


def rolling_corr(a, b, n):
    return a.rolling(n, min_periods=n).corr(b)


def rolling_beta(y, x, n):
    """Beta of y on x (Series or frame aligned on columns)."""
    cov = y.rolling(n, min_periods=n).cov(x)
    var = x.rolling(n, min_periods=n).var()
    return cov / var


def realized_vol(r, n):
    return rstd(r, n) * math.sqrt(252)


def kama(C, n=10, fast=2, slow=30):
    ch = (C - C.shift(n)).abs()
    vol = C.diff().abs().rolling(n).sum()
    er = (ch / vol.replace(0, np.nan)).fillna(0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    return adaptive_filter(C, sc)


def adaptive_filter(C, alpha):
    a = alpha.to_numpy(float)
    x = C.to_numpy(float)
    out = _adaptive(x, a)
    return pd.DataFrame(out, index=C.index, columns=C.columns)


@njit(cache=True)
def _adaptive(x, a):
    n, m = x.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        prev = np.nan
        for i in range(n):
            xi = x[i, j]
            if np.isnan(xi):
                out[i, j] = prev
                continue
            if np.isnan(prev):
                prev = xi
            else:
                ai = a[i, j]
                if not np.isnan(ai):
                    prev = prev + ai * (xi - prev)
            out[i, j] = prev
    return out


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return x.rolling(n, min_periods=n).apply(lambda a: a @ w, raw=True)


def hma(x, n):
    return wma(2 * wma(x, n // 2) - wma(x, n), int(math.sqrt(n)))


def zlema(x, n):
    lag = (n - 1) // 2
    return ema(x + (x - x.shift(lag)), n)


def efficiency_ratio(C, n=20):
    ch = (C - C.shift(n)).abs()
    vol = C.diff().abs().rolling(n).sum()
    return ch / vol.replace(0, np.nan)


# ------------------------------------------------------- position builders

@njit(cache=True)
def _machine(le, se, lx, sx):
    n, m = le.shape
    out = np.zeros((n, m))
    for j in range(m):
        p = 0.0
        for i in range(n):
            if p > 0 and lx[i, j]:
                p = 0.0
            elif p < 0 and sx[i, j]:
                p = 0.0
            if le[i, j] and not se[i, j]:
                p = 1.0
            elif se[i, j] and not le[i, j]:
                p = -1.0
            out[i, j] = p
    return out


def machine(long_entry, short_entry, long_exit, short_exit, like):
    """
    Stateful positions: enter on an entry flag, hold until the matching exit
    flag. Entries override exits on the same bar. All inputs boolean frames.
    """
    f = lambda x: (x.reindex_like(like).fillna(False).to_numpy(bool)  # noqa
                   if isinstance(x, pd.DataFrame) else
                   np.broadcast_to(np.asarray(x, bool), like.shape))
    out = _machine(f(long_entry), f(short_entry), f(long_exit),
                   f(short_exit))
    return pd.DataFrame(out, index=like.index, columns=like.columns)


def fade_band(z, entry=2.0, exit=0.0):
    """Mean-reversion positions from a z-score: in beyond +-entry, out at exit."""
    return machine(z < -entry, z > entry, z >= -exit, z <= exit, z)


def follow_band(z, entry=2.0, exit=0.0):
    """Trend positions from a z-score: in beyond +-entry, out back at exit."""
    return machine(z > entry, z < -entry, z <= exit, z >= -exit, z)


def hold_for(events, n):
    """Hold a signed event (+1/-1/0) for n bars; later events override."""
    e = events.fillna(0)
    out = e.replace(0, np.nan)
    out = out.ffill(limit=n - 1)
    return out.fillna(0)


def sign(x):
    return np.sign(x).fillna(0)


def clip1(x):
    return x.clip(-1, 1).fillna(0)


def xs_rank_signal(score, frac=1 / 3, cols=None):
    """Long the top `frac`, short the bottom `frac` of each row of `score`."""
    s = score if cols is None else score[cols]
    r = s.rank(axis=1, pct=True)
    n = s.notna().sum(axis=1)
    out = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    ok = n >= 3
    out[(r > 1 - frac) & ok.to_numpy()[:, None]] = 1.0
    out[(r <= frac) & ok.to_numpy()[:, None]] = -1.0
    return out.where(s.notna(), 0.0)


def xs_rank_groups(score, groups, frac=1 / 3):
    parts = [xs_rank_signal(score[[c for c in g if c in score.columns]], frac)
             for g in groups]
    return pd.concat(parts, axis=1).reindex(columns=score.columns).fillna(0)


def xs_demean(score, groups=None):
    if groups is None:
        return score.sub(score.mean(axis=1), axis=0)
    parts = []
    for g in groups:
        c = [x for x in g if x in score.columns]
        parts.append(score[c].sub(score[c].mean(axis=1), axis=0))
    return pd.concat(parts, axis=1).reindex(columns=score.columns)


def rebalance(sig, every):
    """Only allow the signal to change every `every` bars (e.g. 5 = weekly)."""
    keep = np.arange(len(sig)) % every == 0
    return sig.where(pd.Series(keep, index=sig.index), np.nan).ffill().fillna(0)


def month_end_rebalance(sig):
    idx = sig.index
    me = pd.Series(idx.to_period("M"), index=idx)
    last = me != me.shift(-1)
    return sig.where(last, np.nan).ffill().fillna(0)


def per_column(frame, fn, *args, **kw):
    """Apply a Series -> Series function to each column."""
    return pd.DataFrame({c: fn(frame[c], *args, **kw) for c in frame.columns},
                        index=frame.index)[frame.columns]


def yearly_refit(series, fit, apply, start_years=3, refit="YS"):
    """
    Walk-forward helper for one Series: at the start of each year from
    `start_years` in, call fit(train) on all data before it, then
    apply(model, test_slice_with_history) to get values for that year.
    """
    s = series.dropna()
    if len(s) < 300:
        return pd.Series(np.nan, index=series.index)
    years = sorted(set(s.index.year))
    out = []
    for y in years[start_years:]:
        tr = s[s.index.year < y]
        te_mask = s.index.year == y
        if te_mask.sum() == 0:
            continue
        model = fit(tr)
        if model is None:
            continue
        v = apply(model, s[s.index.year <= y])
        out.append(v[v.index.year == y])
    if not out:
        return pd.Series(np.nan, index=series.index)
    return pd.concat(out).reindex(series.index)
