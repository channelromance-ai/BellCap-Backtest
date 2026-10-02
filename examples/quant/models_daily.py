"""
Single-instrument daily models: mean reversion, momentum and trend,
breakouts, volatility, regime, statistical forecasting, technical and
distributional entries.

Textbook parameters throughout, chosen before looking at any result.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

from examples.quant import lib as L
from examples.quant.core import model


# =========================================================== mean reversion

@model("stat_mr")
def stat_mr(X):
    """Continuous fade of the 20-day z-score of log price."""
    return L.clip1(-L.zscore(np.log(X.C), 20) / 2)


@model("z_mr")
def z_mr(X):
    """Enter against |z20| > 2, exit when z crosses 0."""
    return L.fade_band(L.zscore(X.C, 20), 2.0, 0.0)


@model("boll_z")
def boll_z(X):
    """Bollinger(20,2): fade on the close back inside the band, exit mid."""
    z = L.zscore(X.C, 20)
    le = (z.shift(1) < -2) & (z >= -2)
    se = (z.shift(1) > 2) & (z <= 2)
    return L.machine(le, se, z >= 0, z <= 0, z)


@model("sd_band")
def sd_band(X):
    """50-day mean +-2.5 sd band; fade outside, exit inside +-0.5 sd."""
    return L.fade_band(L.zscore(X.C, 50), 2.5, 0.5)


def _ou_fit(x):
    """AR(1) on levels -> (mu, sigma_eq, half-life)."""
    y, z = x[1:], x[:-1]
    zm, ym = z.mean(), y.mean()
    b = ((z - zm) * (y - ym)).sum() / ((z - zm) ** 2).sum()
    a = ym - b * zm
    if not (0 < b < 1):
        return np.nan, np.nan, np.nan
    e = y - (a + b * z)
    mu = a / (1 - b)
    sig_eq = e.std() / math.sqrt(1 - b * b)
    return mu, sig_eq, -math.log(2) / math.log(b)


def _ou_frame(C, n=120):
    lp = np.log(C)
    Z = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    HL = Z.copy()
    for c in C.columns:
        v = lp[c].to_numpy()
        for i in range(n, len(v)):
            w = v[i - n:i + 1]
            if np.isnan(w).any():
                continue
            mu, se, hl = _ou_fit(w)
            if np.isfinite(se) and se > 0:
                Z.iat[i, Z.columns.get_loc(c)] = (v[i] - mu) / se
                HL.iat[i, HL.columns.get_loc(c)] = hl
    return Z, HL


@model("ou_mr")
def ou_mr(X):
    """Rolling 120-day OU fit; fade |(x-mu)/sigma_eq| > 1.5 when half-life
    under 30 days, exit at mu."""
    Z, HL = X.cached("ou120", lambda: _ou_frame(X.C, 120))
    ok = HL < 30
    return L.machine((Z < -1.5) & ok, (Z > 1.5) & ok, Z >= 0, Z <= 0, Z)


@model("halflife_mr")
def halflife_mr(X):
    """z-score over a lookback equal to the estimated half-life (2-60d)."""
    Z, HL = X.cached("ou120", lambda: _ou_frame(X.C, 120))
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    zs = {n: L.zscore(X.C, n) for n in (5, 10, 20, 40, 60)}
    hl = HL.clip(2, 60)
    pick = pd.DataFrame(np.nan, index=hl.index, columns=hl.columns)
    for n, z in zs.items():
        sel = (hl >= n * 0.6) & (hl < n * 1.5)
        pick = pick.where(~sel, z)
    out = L.fade_band(pick.fillna(0), 2.0, 0.0)
    return out.where((HL >= 2) & (HL <= 60), 0.0)


@model("ts_mr")
def ts_mr(X):
    """Time-series reversal: fade the 5-day return in units of its vol."""
    r5 = L.ret(X.C, 5)
    vol = L.rstd(X.R, 60) * math.sqrt(5)
    return L.clip1(-(r5 / vol) / 2)


@model("reg_to_mean")
def reg_to_mean(X):
    """Price > 2 residual sd from its 100-day regression line: fade to it."""
    fit, slope, sd = L.roll_linreg(np.log(X.C), 100)
    z = (np.log(X.C) - fit) / sd
    return L.fade_band(z, 2.0, 0.0)


@model("resid_rev")
def resid_rev(X):
    """Residual of the 20-day regression line, fade |z| > 1.5."""
    fit, slope, sd = L.roll_linreg(np.log(X.C), 20)
    z = (np.log(X.C) - fit) / sd
    return L.fade_band(z, 1.5, 0.0)


@model("fcst_err_rev")
def fcst_err_rev(X):
    """Fade large one-step errors of an EMA(10) forecast (z over 100d)."""
    err = X.C - L.ema(X.C, 10).shift(1)
    return L.fade_band(L.zscore(err, 100), 2.0, 0.0)


@model("ema_dist_z")
def ema_dist_z(X):
    """z (100d) of the distance from EMA20, fade +-2."""
    return L.fade_band(L.zscore(X.C / L.ema(X.C, 20) - 1, 100), 2.0, 0.0)


@model("price_ma_z")
def price_ma_z(X):
    """z (250d) of price / SMA20 - 1, fade +-2."""
    return L.fade_band(L.zscore(X.C / L.sma(X.C, 20) - 1, 250), 2.0, 0.0)


@model("ma_dist")
def ma_dist(X):
    """Fade price more than 3 ATR from SMA50; exit at the average."""
    d = (X.C - L.sma(X.C, 50)) / L.atr(X.D, 14)
    return L.fade_band(d, 3.0, 0.0)


@model("price_vwap_z")
def price_vwap_z(X):
    """z (100d) of price vs its 20-day tick-volume-weighted average."""
    tp = (X.D["H"] + X.D["L"] + X.D["C"]) / 3
    v = X.D["V"]
    vw = (tp * v).rolling(20).sum() / v.rolling(20).sum()
    return L.fade_band(L.zscore(X.C / vw - 1, 100), 2.0, 0.0)


@model("hist_pct_rev")
def hist_pct_rev(X):
    """20-day return below its 5th / above its 95th 250-day percentile:
    fade for 10 days."""
    p = L.rolling_pct_rank(L.ret(X.C, 20), 250)
    ev = (p < 0.05).astype(float) - (p > 0.95).astype(float)
    return L.hold_for(ev, 10)


@model("pct_entry")
def pct_entry(X):
    """One-day return beyond its 250-day 5th/95th percentile: fade 1 day."""
    p = L.rolling_pct_rank(X.R, 250)
    return (p < 0.05).astype(float) - (p > 0.95).astype(float)


@model("multi_candle")
def multi_candle(X):
    """Three-day return sign: fade it for one day."""
    return -L.sign(L.ret(X.C, 3))


@model("vol_mr_fade")
def vol_mr_fade(X):
    """When 10d vol > 1.5x its 1-year median, fade the 5-day move."""
    hv = L.rstd(X.R, 10)
    hi = hv > 1.5 * hv.rolling(250).median()
    return (-L.sign(L.ret(X.C, 5))).where(hi, 0.0)


@model("vol_shock")
def vol_shock(X):
    """Vol shock (5d vol > 2x 60d vol): fade the 5-day move for 5 days."""
    s = L.rstd(X.R, 5) / L.rstd(X.R, 60)
    ev = (-L.sign(L.ret(X.C, 5))).where(s > 2, 0.0)
    return L.hold_for(ev, 5)


# ======================================================= momentum and trend

@model("ts_mom")
def ts_mom(X):
    """Time-series momentum: sign of the 12-month return (Moskowitz et al.)."""
    return L.sign(L.ret(X.C, 252))


@model("trend_follow")
def trend_follow(X):
    """Multi-speed EWMA crossover trend (8/24, 16/48, 32/96), averaged."""
    s = 0
    for f, sl in ((8, 24), (16, 48), (32, 96)):
        x = (L.ema(X.C, f) - L.ema(X.C, sl)) / (L.rstd(X.C, 63))
        s = s + L.clip1(x / 1.0)
    return s / 3


@model("multi_ma")
def multi_ma(X):
    """Average of sign(price - SMA n) for n in 20, 50, 100, 200."""
    return sum(L.sign(X.C - L.sma(X.C, n)) for n in (20, 50, 100, 200)) / 4


@model("ma_slope")
def ma_slope(X):
    """Sign of the 10-day change of SMA50."""
    m = L.sma(X.C, 50)
    return L.sign(m - m.shift(10))


@model("slope_entry")
def slope_entry(X):
    """EMA20 5-day slope in ATR units: long > 0.1, short < -0.1."""
    e = L.ema(X.C, 20)
    s = (e - e.shift(5)) / 5 / L.atr(X.D, 14)
    return (s > 0.1).astype(float) - (s < -0.1).astype(float)


@model("linreg_slope")
def linreg_slope(X):
    """50-day OLS slope of log price with |t| > 2 sets the direction."""
    lp = np.log(X.C)
    fit, slope, sd = L.roll_linreg(lp, 50)
    t = slope / (sd / math.sqrt(((np.arange(50) - 24.5) ** 2).sum()))
    return (t > 2).astype(float) - (t < -2).astype(float)


@model("linreg_fcst")
def linreg_fcst(X):
    """Next value of the 20-day regression line vs price: sign."""
    lp = np.log(X.C)
    fit, slope, sd = L.roll_linreg(lp, 20)
    return L.sign(fit + slope - lp)


@model("poly_fcst")
def poly_fcst(X):
    """Quadratic fit to 20 days of log price, extrapolated one day: sign."""
    n = 20
    t = np.arange(n, dtype=float)
    V = np.vander(t, 3)
    P = np.linalg.pinv(V)                       # 3 x n
    nxt = np.array([n ** 2, n, 1.0]) @ P        # weights for next value
    lp = np.log(X.C)
    f = lp.rolling(n, min_periods=n).apply(lambda a: a @ nxt, raw=True)
    return L.sign(f - lp)


@model("trend_strength")
def trend_strength(X):
    """R^2 of the 60-day regression > 0.5: trade the slope's direction."""
    lp = np.log(X.C)
    r2 = L.r_squared(lp, 60)
    s = L.sign(L.roll_slope(lp, 60))
    return s.where(r2 > 0.5, 0.0)


@model("adx_filter")
def adx_filter(X):
    """EMA20/50 trend taken only while ADX(14) > 25."""
    a, pdi, ndi = L.adx(X.D)
    s = L.sign(L.ema(X.C, 20) - L.ema(X.C, 50))
    return s.where(a > 25, 0.0)


@model("ma_spread_z")
def ma_spread_z(X):
    """z (250d) of EMA10 - EMA50 spread: long > 1, short < -1, exit at 0."""
    return L.follow_band(L.zscore(L.ema(X.C, 10) - L.ema(X.C, 50), 250),
                         1.0, 0.0)


@model("macd")
def macd(X):
    """Sign of the MACD(12,26,9) histogram."""
    m = L.ema(X.C, 12) - L.ema(X.C, 26)
    return L.sign(m - L.ema(m, 9))


@model("risk_adj_mom")
def risk_adj_mom(X):
    """12-month return over 12-month vol (a Sharpe), clipped, as weight."""
    sr = L.ret(X.C, 252) / (L.rstd(X.R, 252) * math.sqrt(252))
    return L.clip1(sr)


@model("sharpe_mom_filter")
def sharpe_mom_filter(X):
    """TSMOM taken only when the trailing 6-month |Sharpe| > 0.5."""
    m = X.R.rolling(126).mean() / X.R.rolling(126).std() * math.sqrt(252)
    return L.sign(m).where(m.abs() > 0.5, 0.0)


@model("sortino_mom_filter")
def sortino_mom_filter(X):
    """As the Sharpe filter but with the Sortino ratio (downside dev)."""
    mu = X.R.rolling(126).mean()
    dd = np.sqrt((X.R.clip(upper=0) ** 2).rolling(126).mean())
    up = np.sqrt((X.R.clip(lower=0) ** 2).rolling(126).mean())
    long = (mu / dd * math.sqrt(252)) > 0.7
    short = (-mu / up * math.sqrt(252)) > 0.7
    return long.astype(float) - short.astype(float)


@model("mom_vol_filter")
def mom_vol_filter(X):
    """TSMOM only while 60-day vol is below its 1-year median."""
    v = L.rstd(X.R, 60)
    return L.sign(L.ret(X.C, 252)).where(v < v.rolling(250).median(), 0.0)


@model("trend_volregime")
def trend_volregime(X):
    """EMA50/200 trend unless 20d vol is above its 80th 2-year percentile."""
    p = L.rolling_pct_rank(L.rstd(X.R, 20), 500)
    return L.sign(L.ema(X.C, 50) - L.ema(X.C, 200)).where(p < 0.8, 0.0)


@model("vol_scaled_entry")
def vol_scaled_entry(X):
    """20-day return in sd units > 1: long; < -1: short; out inside 0."""
    z = L.ret(X.C, 20) / (L.rstd(X.R, 60) * math.sqrt(20))
    return L.follow_band(z, 1.0, 0.0)


@model("atr_norm_entry")
def atr_norm_entry(X):
    """(price - SMA50)/ATR > 2: long; < -2: short; exit at 0."""
    return L.follow_band((X.C - L.sma(X.C, 50)) / L.atr(X.D, 14), 2.0, 0.0)


@model("regime_trend")
def regime_trend(X):
    """Trend (EMA50/200) only when the 60-day efficiency ratio > 0.25."""
    er = L.efficiency_ratio(X.C, 60)
    return L.sign(L.ema(X.C, 50) - L.ema(X.C, 200)).where(er > 0.25, 0.0)


@model("serial_mom")
def serial_mom(X):
    """Follow yesterday's return when the 60-day lag-1 autocorr > 0.1."""
    ac = X.R.rolling(60).corr(X.R.shift(1))
    return L.sign(X.R).where(ac > 0.1, 0.0)


@model("serial_rev")
def serial_rev(X):
    """Fade yesterday's return when the 60-day lag-1 autocorr < -0.1."""
    ac = X.R.rolling(60).corr(X.R.shift(1))
    return (-L.sign(X.R)).where(ac < -0.1, 0.0)


@model("autocorr_entry")
def autocorr_entry(X):
    """sign(autocorr) x sign(last return), when |autocorr(60)| > 0.1."""
    ac = X.R.rolling(60).corr(X.R.shift(1))
    return (L.sign(ac) * L.sign(X.R)).where(ac.abs() > 0.1, 0.0)


@model("ret_autocorr_weekly")
def ret_autocorr_weekly(X):
    """Weekly-return autocorrelation (52 weeks) sets follow/fade of last week."""
    r5 = L.ret(X.C, 5)
    ac = r5.rolling(260).corr(r5.shift(5))
    return L.rebalance((L.sign(ac) * L.sign(r5)).where(ac.abs() > 0.1, 0.0),
                       5)


# ================================================================ breakouts

@model("quant_breakout")
def quant_breakout(X):
    """Close beyond the prior 50-day high/low; exit on the 25-day opposite."""
    hi50 = X.D["H"].rolling(50).max().shift(1)
    lo50 = X.D["L"].rolling(50).min().shift(1)
    hi25 = X.D["H"].rolling(25).max().shift(1)
    lo25 = X.D["L"].rolling(25).min().shift(1)
    return L.machine(X.C > hi50, X.C < lo50, X.C < lo25, X.C > hi25, X.C)


@model("donchian")
def donchian(X):
    """Turtle: 20-day channel breakout, exit on the 10-day opposite."""
    hi = X.D["H"].rolling(20).max().shift(1)
    lo = X.D["L"].rolling(20).min().shift(1)
    hx = X.D["H"].rolling(10).max().shift(1)
    lx = X.D["L"].rolling(10).min().shift(1)
    return L.machine(X.C > hi, X.C < lo, X.C < lx, X.C > hx, X.C)


@model("vol_breakout")
def vol_breakout(X):
    """A daily move beyond 2 sd (60d): follow it for 5 days."""
    z = X.R / L.rstd(X.R, 60).shift(1)
    return L.hold_for((z > 2).astype(float) - (z < -2).astype(float), 5)


@model("atr_breakout")
def atr_breakout(X):
    """Close more than 1.5 ATR beyond the prior close: follow 5 days."""
    a = L.atr(X.D, 14).shift(1)
    d = X.C - X.C.shift(1)
    return L.hold_for((d > 1.5 * a).astype(float) - (d < -1.5 * a)
                      .astype(float), 5)


@model("voladj_breakout")
def voladj_breakout(X):
    """Close beyond the 20-day high/low by at least 0.5 ATR; exit at SMA20."""
    a = L.atr(X.D, 14)
    hi = X.D["H"].rolling(20).max().shift(1)
    lo = X.D["L"].rolling(20).min().shift(1)
    m = L.sma(X.C, 20)
    return L.machine(X.C > hi + 0.5 * a, X.C < lo - 0.5 * a, X.C < m,
                     X.C > m, X.C)


@model("range_exp_breakout")
def range_exp_breakout(X):
    """Range > 2 ATR, close in the outer quarter: follow for 5 days."""
    a = L.atr(X.D, 14).shift(1)
    rng = X.D["H"] - X.D["L"]
    loc = (X.C - X.D["L"]) / rng.replace(0, np.nan)
    big = rng > 2 * a
    ev = (big & (loc > 0.75)).astype(float) - (big & (loc < 0.25)).astype(
        float)
    return L.hold_for(ev, 5)


@model("stat_range_breakout")
def stat_range_breakout(X):
    """Close above the 95th / below the 5th percentile of 100 closes;
    exit at the median."""
    p = L.rolling_pct_rank(X.C, 100)
    return L.machine(p > 0.95, p < 0.05, p < 0.5, p > 0.5, X.C)


@model("compress_expand")
def compress_expand(X):
    """10d vol in its lowest decile of the year, then the first 1-sd day
    sets the direction for 10 days."""
    hv = L.rstd(X.R, 10)
    comp = (L.rolling_pct_rank(hv, 250) < 0.1).shift(1).rolling(5).max() > 0
    z = X.R / L.rstd(X.R, 60).shift(1)
    ev = ((z > 1) & comp).astype(float) - ((z < -1) & comp).astype(float)
    return L.hold_for(ev, 10)


@model("bb_squeeze")
def bb_squeeze(X):
    """BB width at a 120-day low (within 10%): trade the band break, exit mid."""
    m, s = L.sma(X.C, 20), L.rstd(X.C, 20)
    w = 4 * s / m
    sq = (w <= 1.1 * w.rolling(120).min()).shift(1).rolling(10).max() > 0
    up, lo = m + 2 * s, m - 2 * s
    return L.machine(sq & (X.C > up), sq & (X.C < lo), X.C < m, X.C > m, X.C)


@model("keltner")
def keltner(X):
    """Keltner EMA20 +-2 ATR10 breakout, exit at EMA20."""
    e = L.ema(X.C, 20)
    a = L.atr(X.D, 10)
    return L.machine(X.C > e + 2 * a, X.C < e - 2 * a, X.C < e, X.C > e, X.C)


@model("atr_channel")
def atr_channel(X):
    """SMA50 +-3 ATR channel: in beyond it, out at SMA50."""
    m = L.sma(X.C, 50)
    a = L.atr(X.D, 14)
    return L.machine(X.C > m + 3 * a, X.C < m - 3 * a, X.C < m, X.C > m, X.C)


@model("hv_breakout")
def hv_breakout(X):
    """10d vol / 100d vol > 1.5: follow the 10-day return."""
    s = L.rstd(X.R, 10) / L.rstd(X.R, 100)
    return L.sign(L.ret(X.C, 10)).where(s > 1.5, 0.0)


@model("rv_breakout")
def rv_breakout(X):
    """Hourly realized vol of the day (sum of squares) z > 2 (100d):
    follow the day's direction for 5 days."""
    def f():
        hr = X.H["C"].pct_change(fill_method=None)
        from bcbt.quant import to_ny_day
        rv = (hr ** 2).groupby(to_ny_day(hr.index)).sum() ** 0.5
        return rv.reindex(X.C.index)
    rv = X.cached("rv_daily", f)
    z = L.zscore(rv, 100)
    ev = L.sign(X.R).where(z > 2, 0.0)
    return L.hold_for(ev, 5)


@model("evt_breakout")
def evt_breakout(X):
    """20-day range beyond the GPD 99% quantile of past 20-day ranges:
    follow the 20-day direction for 10 days."""
    rng = np.log(X.D["H"].rolling(20).max() / X.D["L"].rolling(20).min())
    q = X.cached("evt_q20", lambda: _evt_quantile(rng, 0.99, 750))
    ev = L.sign(L.ret(X.C, 20)).where(rng > q, 0.0)
    return L.hold_for(ev, 10)


def _evt_quantile(x, p, n, thresh_q=0.9, step=20):
    """Peaks-over-threshold GPD quantile from the trailing n values."""
    from scipy.stats import genpareto
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for c in x.columns:
        v = x[c].to_numpy()
        last = np.nan
        for i in range(n, len(v)):
            if (i - n) % step == 0:
                w = v[i - n:i]
                w = w[np.isfinite(w)]
                if len(w) > 200:
                    u = np.quantile(w, thresh_q)
                    ex = w[w > u] - u
                    try:
                        xi, _, beta = genpareto.fit(ex, floc=0)
                        pu = (w > u).mean()
                        last = u + genpareto.ppf(1 - (1 - p) / pu, xi, 0,
                                                 beta)
                    except Exception:
                        pass
            out.iat[i, out.columns.get_loc(c)] = last
    return out


@model("evt_tail_rev")
def evt_tail_rev(X):
    """Daily return beyond the GPD 99% tail of the last 3 years: fade 3 days."""
    q_up = X.cached("evt_r_up", lambda: _evt_quantile(X.R, 0.99, 750))
    q_dn = X.cached("evt_r_dn", lambda: _evt_quantile(-X.R, 0.99, 750))
    ev = (X.R < -q_dn).astype(float) - (X.R > q_up).astype(float)
    return L.hold_for(ev, 3)


@model("evt_tail_mom")
def evt_tail_mom(X):
    """As tail-event reversion, but follow the tail move for 3 days."""
    return -evt_tail_rev(X)


@model("evt_entry")
def evt_entry(X):
    """EVT entry: fade a 1-day move beyond the 97.5% GPD quantile, 1 day."""
    q_up = X.cached("evt_r_up975", lambda: _evt_quantile(X.R, 0.975, 750))
    q_dn = X.cached("evt_r_dn975", lambda: _evt_quantile(-X.R, 0.975, 750))
    return (X.R < -q_dn).astype(float) - (X.R > q_up).astype(float)


@model("range_compression")
def range_compression(X):
    """NR7 day: follow the next close outside its range for 5 days."""
    rng = X.D["H"] - X.D["L"]
    nr7 = rng <= rng.rolling(7).min()
    h7, l7 = X.D["H"].where(nr7).ffill(limit=3), X.D["L"].where(nr7).ffill(
        limit=3)
    ev = (X.C > h7.shift(1)).astype(float) - (X.C < l7.shift(1)).astype(float)
    return L.hold_for(ev.where(nr7.shift(1).rolling(3).max() > 0, 0.0), 5)


@model("range_expansion")
def range_expansion(X):
    """Widest range in 7 days: fade its direction next day (exhaustion)."""
    rng = X.D["H"] - X.D["L"]
    wr7 = rng >= rng.rolling(7).max()
    return (-L.sign(X.C - X.D["O"])).where(wr7, 0.0)


@model("hl_range")
def hl_range(X):
    """Range > 1.5 ATR with the close in the top/bottom 20%: follow 1 day."""
    a = L.atr(X.D, 14).shift(1)
    rng = X.D["H"] - X.D["L"]
    loc = (X.C - X.D["L"]) / rng.replace(0, np.nan)
    big = rng > 1.5 * a
    return (big & (loc > 0.8)).astype(float) - (big & (loc < 0.2)).astype(float)


@model("atr_compression")
def atr_compression(X):
    """ATR5/ATR50 < 0.6, then a close beyond the 10-day range: follow 10 days."""
    c = (L.atr(X.D, 5) / L.atr(X.D, 50) < 0.6).shift(1).rolling(5).max() > 0
    hi = X.D["H"].rolling(10).max().shift(1)
    lo = X.D["L"].rolling(10).min().shift(1)
    ev = ((X.C > hi) & c).astype(float) - ((X.C < lo) & c).astype(float)
    return L.hold_for(ev, 10)


@model("vcp")
def vcp(X):
    """Volatility contraction: 10d vol < 0.7 x 50d vol, close at a 20-day
    high (long only, Minervini-style); exit at the 10-day low."""
    c = L.rstd(X.R, 10) < 0.7 * L.rstd(X.R, 50)
    hi = X.D["H"].rolling(20).max().shift(1)
    lo = X.D["L"].rolling(10).min().shift(1)
    return L.machine(c & (X.C > hi), X.C * 0 > 1, X.C < lo, X.C * 0 > 1, X.C)


@model("vol_exp_cont")
def vol_exp_cont(X):
    """ATR5/ATR20 > 1.3 in the direction of the EMA20/50 trend: follow."""
    e = L.sign(L.ema(X.C, 20) - L.ema(X.C, 50))
    exp = L.atr(X.D, 5) / L.atr(X.D, 20) > 1.3
    agree = L.sign(L.ret(X.C, 5)) == e
    return e.where(exp & agree, 0.0)


@model("fractal_breakout")
def fractal_breakout(X):
    """Close beyond the last confirmed 5-bar fractal high/low; exit on the
    opposite fractal."""
    H, Lo = X.D["H"], X.D["L"]
    fh = (H.shift(2) == H.rolling(5).max()).astype(bool)
    fl = (Lo.shift(2) == Lo.rolling(5).min()).astype(bool)
    lvl_h = H.shift(2).where(fh).ffill()
    lvl_l = Lo.shift(2).where(fl).ffill()
    return L.machine(X.C > lvl_h, X.C < lvl_l, X.C < lvl_l, X.C > lvl_h, X.C)


@model("swing_level")
def swing_level(X):
    """Close beyond the last confirmed swing (5 bars each side); exit on
    the opposite swing."""
    k = 5
    H, Lo = X.D["H"], X.D["L"]
    sh = H.shift(k) == H.rolling(2 * k + 1).max()
    sl = Lo.shift(k) == Lo.rolling(2 * k + 1).min()
    lh = H.shift(k).where(sh).ffill()
    ll = Lo.shift(k).where(sl).ffill()
    return L.machine(X.C > lh, X.C < ll, X.C < ll, X.C > lh, X.C)


@model("regression_channel_bo")
def regression_channel_bo(X):
    """Close outside the 50-day regression line +-2 residual sd: follow;
    exit back at the line."""
    lp = np.log(X.C)
    fit, slope, sd = L.roll_linreg(lp, 50)
    z = (lp - fit.shift(1) - slope.shift(1)) / sd.shift(1)
    return L.follow_band(z, 2.0, 0.0)


@model("pred_interval_bo")
def pred_interval_bo(X):
    """Close outside the 95% prediction interval of an AR(1) on returns
    (250d): follow for 5 days."""
    r = X.R
    rl = r.shift(1)
    b = L.rolling_beta(r, rl, 250)
    a = r.rolling(250).mean() - b * rl.rolling(250).mean()
    resid_sd = (r - (a + b * rl)).rolling(250).std()
    f = (a + b * rl).shift(0)
    z = (r - f) / resid_sd.shift(1)
    return L.hold_for((z > 1.96).astype(float) - (z < -1.96).astype(float), 5)


# ============================================================== volatility

@model("vol_regime_switch")
def vol_regime_switch(X):
    """High-vol regime (60d vol in the top 3rd of 3 years): fade z20;
    otherwise 3-month momentum."""
    p = L.rolling_pct_rank(L.rstd(X.R, 60), 750)
    mr = L.clip1(-L.zscore(X.C, 20) / 2)
    tr = L.sign(L.ret(X.C, 63))
    return mr.where(p > 2 / 3, tr)


@model("atr_regime")
def atr_regime(X):
    """ATR14/ATR100 > 1.2: fade z20; < 0.8: follow 20-day breakouts."""
    q = L.atr(X.D, 14) / L.atr(X.D, 100)
    mr = L.clip1(-L.zscore(X.C, 20) / 2)
    tr = donchian(X)
    return mr.where(q > 1.2, tr.where(q < 0.8, 0.0))


@model("vol_regime_trend")
def vol_regime_trend(X):
    """Trend (EMA50/200) only in the low-vol half of the 2-year range."""
    p = L.rolling_pct_rank(L.rstd(X.R, 20), 500)
    return L.sign(L.ema(X.C, 50) - L.ema(X.C, 200)).where(p < 0.5, 0.0)


@model("vol_regime_mr")
def vol_regime_mr(X):
    """z20 fade only in the high-vol half of the 2-year range."""
    p = L.rolling_pct_rank(L.rstd(X.R, 20), 500)
    return L.fade_band(L.zscore(X.C, 20), 2.0, 0.0).where(p > 0.5, 0.0)


@model("volofvol")
def volofvol(X):
    """TSMOM, flat while vol-of-vol (sd of 20d vol over 60d) is in its top
    quintile of 2 years."""
    hv = L.rstd(X.R, 20)
    vv = L.rstd(hv, 60) / hv.rolling(60).mean()
    p = L.rolling_pct_rank(vv, 500)
    return L.sign(L.ret(X.C, 252)).where(p < 0.8, 0.0)


@model("vol_fcst_rev")
def vol_fcst_rev(X):
    """Index CFDs only: 20d vol > 1.5x the 3-year median (vol expected to
    fall) -> long for 20 days. Buy fear."""
    from bcbt.quant import GROUPS
    hv = L.rstd(X.R, 20)
    ev = (hv > 1.5 * hv.rolling(750).median()).astype(float)
    ev[[c for c in ev.columns if c not in GROUPS["index"]]] = 0.0
    return L.hold_for(ev, 20)


# ============================================== regime and change point

@model("cusum_event")
def cusum_event(X):
    """Event sampling: symmetric CUSUM filter (h = 2 x daily sd) marks
    events; follow the event direction for 5 days."""
    r = np.log(X.C).diff()
    h = (2 * L.rstd(r, 60)).to_numpy()
    v = r.to_numpy()
    out = np.zeros_like(v)
    for j in range(v.shape[1]):
        sp = sn = 0.0
        for i in range(v.shape[0]):
            x = v[i, j]
            if not np.isfinite(x) or not np.isfinite(h[i, j]):
                continue
            sp, sn = max(0.0, sp + x), min(0.0, sn + x)
            if sp > h[i, j]:
                out[i, j], sp = 1, 0.0
            elif sn < -h[i, j]:
                out[i, j], sn = -1, 0.0
    return L.hold_for(pd.DataFrame(out, index=X.C.index, columns=X.C.columns),
                      5)


@model("change_point")
def change_point(X):
    """Page CUSUM on standardised returns (k=0.5, h=5): after an upward
    (downward) shift in mean is detected, hold that direction 20 days."""
    z = (X.R / L.rstd(X.R, 250).shift(1)).to_numpy()
    out = np.zeros_like(z)
    for j in range(z.shape[1]):
        gp = gn = 0.0
        for i in range(z.shape[0]):
            x = z[i, j]
            if not np.isfinite(x):
                continue
            gp = max(0.0, gp + x - 0.5)
            gn = max(0.0, gn - x - 0.5)
            if gp > 5:
                out[i, j], gp, gn = 1, 0.0, 0.0
            elif gn > 5:
                out[i, j], gp, gn = -1, 0.0, 0.0
    return L.hold_for(pd.DataFrame(out, index=X.C.index, columns=X.C.columns),
                      20)


@model("structural_break")
def structural_break(X):
    """Mean of the last 60 days vs the 250 before differs with |t| > 3:
    trade the new direction."""
    r = X.R
    m1, s1 = r.rolling(60).mean(), r.rolling(60).std()
    m0 = r.shift(60).rolling(250).mean()
    s0 = r.shift(60).rolling(250).std()
    t = (m1 - m0) / np.sqrt(s1 ** 2 / 60 + s0 ** 2 / 250)
    return (t > 3).astype(float) - (t < -3).astype(float)


def _bocpd_runlength(x, hazard=1 / 250, rmax=300):
    """Adams-MacKay online change-point, Gaussian unknown mean (known var
    per standardised series). Returns the MAP run length per step."""
    n = len(x)
    out = np.full(n, np.nan)
    R = np.array([1.0])
    mu0, k0 = 0.0, 1.0
    mus, ks = np.array([mu0]), np.array([k0])
    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            continue
        var = 1.0 + 1.0 / ks
        pred = np.exp(-0.5 * (xt - mus) ** 2 / var) / np.sqrt(var)
        growth = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        R = np.r_[cp, growth]
        R /= R.sum()
        mus = np.r_[mu0, (ks * mus + xt) / (ks + 1)]
        ks = np.r_[k0, ks + 1]
        if len(R) > rmax:
            R, mus, ks = R[:rmax], mus[:rmax], ks[:rmax]
            R /= R.sum()
        out[t] = mus[np.argmax(R)]
    return out


@model("bayes_regime")
def bayes_regime(X):
    """Bayesian online change-point detection on standardised returns:
    trade the sign of the posterior mean of the current regime, when it is
    larger than 0.05 sd."""
    z = X.R / L.rstd(X.R, 250).shift(1)
    m = pd.DataFrame({c: _bocpd_runlength(z[c].to_numpy()) for c in z.columns},
                     index=z.index)
    return (m > 0.05).astype(float) - (m < -0.05).astype(float)


def _hmm_fit(x, k=2, iters=60, seed=0):
    """Gaussian HMM by Baum-Welch (scaled). Returns (pi, A, mu, sd)."""
    q = np.quantile(x, np.linspace(0.2, 0.8, k))
    sd = np.full(k, x.std())
    if k == 2:
        sd = np.array([x.std() * 0.7, x.std() * 1.5])
    A = np.full((k, k), 0.05 / (k - 1))
    np.fill_diagonal(A, 0.95)
    pi = np.full(k, 1 / k)
    return _baum_welch(x.astype(np.float64), pi, A, q.astype(np.float64),
                       sd.astype(np.float64), iters)


@njit(cache=True)
def _baum_welch(x, pi, A, mu, sd, iters):
    n, k = len(x), len(mu)
    for _ in range(iters):
        B = np.empty((n, k))
        for t in range(n):
            for s in range(k):
                B[t, s] = np.exp(-0.5 * ((x[t] - mu[s]) / sd[s]) ** 2) / sd[s] \
                    + 1e-300
        al = np.zeros((n, k))
        c = np.zeros(n)
        al[0] = pi * B[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]
            c[t] = al[t].sum()
            al[t] /= c[t]
        be = np.ones((n, k))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        for t in range(n):
            g[t] /= g[t].sum()
        xi = np.zeros((k, k))
        for t in range(n - 1):
            m = np.empty((k, k))
            tot = 0.0
            for i in range(k):
                for j in range(k):
                    m[i, j] = al[t, i] * A[i, j] * B[t + 1, j] * be[t + 1, j]
                    tot += m[i, j]
            xi += m / tot
        for i in range(k):
            A[i] = xi[i] / xi[i].sum()
        pi = g[0].copy()
        for s in range(k):
            w = g[:, s].sum()
            mu[s] = (g[:, s] * x).sum() / w
            sd[s] = np.sqrt((g[:, s] * (x - mu[s]) ** 2).sum() / w) + 1e-8
    return pi, A, mu, sd


def _hmm_filter(x, params):
    """Forward (filtered, causal) state probabilities."""
    pi, A, mu, sd = params
    k = len(mu)
    out = np.zeros((len(x), k))
    a = pi.copy()
    for t, xt in enumerate(x):
        b = np.exp(-0.5 * ((xt - mu) / sd) ** 2) / sd + 1e-300
        a = (a @ A) * b if t else a * b
        a /= a.sum()
        out[t] = a
    return out


def _hmm_signal(r, k):
    """Expected next-day return under the filtered state distribution,
    fitted on all data before each year (walk-forward)."""
    def fit(tr):
        x = tr.to_numpy()[-1500:]
        return _hmm_fit(x / x.std(), k), x.std()

    def apply(m, s):
        params, scl = m
        x = s.to_numpy() / scl
        p = _hmm_filter(x, params)
        pi, A, mu, sd = params
        exp_next = (p @ A) @ mu
        return pd.Series(exp_next, index=s.index)
    return L.yearly_refit(r, fit, apply)


@model("markov_switch")
def markov_switch(X):
    """2-state Markov-switching (Gaussian HMM) on daily returns, refit each
    year on prior data; trade the sign of the expected next-day return."""
    e = X.cached("hmm2", lambda: L.per_column(X.R, _hmm_signal, 2))
    return L.sign(e)


@model("hmm_entry")
def hmm_entry(X):
    """3-state HMM, same walk-forward; trade only when the expected
    next-day return is beyond 0.05 sd."""
    e = X.cached("hmm3", lambda: L.per_column(X.R, _hmm_signal, 3))
    return (e > 0.05).astype(float) - (e < -0.05).astype(float)


# ===================================================== statistical forecasts

@model("ar1_fcst")
def ar1_fcst(X):
    """Rolling 250-day AR(1) on returns: trade the forecast sign when it
    exceeds the one-way cost."""
    r = X.R
    rl = r.shift(1)
    b = L.rolling_beta(r, rl, 250)
    a = r.rolling(250).mean() - b * rl.rolling(250).mean()
    f = a + b * r
    return (f > 1e-4).astype(float) - (f < -1e-4).astype(float)


def _arima_sig(s):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    def fit(tr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return ARIMA(tr.to_numpy()[-750:] * 100, order=(1, 0, 1)
                             ).fit().params
            except Exception:
                return None

    def apply(p, s):
        # params: const, ar.L1, ma.L1, sigma2 -> one-step forecasts with the
        # fitted coefficients, innovations filtered forward
        c, phi, th = p[0], p[1], p[2]
        x = s.to_numpy() * 100
        f = np.zeros(len(x))
        e = 0.0
        prev = 0.0
        for t in range(len(x)):
            f[t] = c + phi * (prev - c) + th * e   # forecast of x[t]
            e = x[t] - f[t]
            prev = x[t]
        nxt = c + phi * (x - c) + th * np.r_[x - f]
        return pd.Series(nxt, index=s.index)
    return L.yearly_refit(s, fit, apply)


@model("arima_fcst")
def arima_fcst(X):
    """ARIMA(1,0,1) on daily returns, refit yearly on the prior 3 years;
    sign of the one-step forecast."""
    f = X.cached("arima", lambda: L.per_column(X.R, _arima_sig))
    return L.sign(f)


@model("exp_smooth")
def exp_smooth(X):
    """Holt linear-trend exponential smoothing (alpha .2, beta .05) of log
    price: sign of the smoothed trend."""
    lp = np.log(X.C).to_numpy()
    out = np.full_like(lp, np.nan)
    a, b = 0.2, 0.05
    for j in range(lp.shape[1]):
        lvl = tr = np.nan
        for i in range(lp.shape[0]):
            x = lp[i, j]
            if not np.isfinite(x):
                continue
            if not np.isfinite(lvl):
                lvl, tr = x, 0.0
                continue
            nl = a * x + (1 - a) * (lvl + tr)
            tr = b * (nl - lvl) + (1 - b) * tr
            lvl = nl
            out[i, j] = tr
    return L.sign(pd.DataFrame(out, index=X.C.index, columns=X.C.columns))


def _kalman_llt(y, q_lvl, q_slope, r):
    """Local linear trend Kalman filter; returns filtered level and slope.
    The noise variances may be scalars or arrays (one per step)."""
    n = len(y)
    q_lvl, q_slope, r = (np.broadcast_to(np.asarray(v, float), (n,))
                         for v in (q_lvl, q_slope, r))
    lv, sl = np.full(n, np.nan), np.full(n, np.nan)
    x = None
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    for t in range(n):
        if not (np.isfinite(y[t]) and np.isfinite(r[t])):
            continue
        if x is None:
            x = np.array([y[t], 0.0])
            P = np.eye(2) * 1e-2
            continue
        x = F @ x
        P = F @ P @ F.T + np.diag([q_lvl[t], q_slope[t]])
        S = P[0, 0] + r[t]
        K = P[:, 0] / S
        x = x + K * (y[t] - x[0])
        P = P - np.outer(K, P[0])
        lv[t], sl[t] = x[0], x[1]
    return lv, sl


@model("kalman_trend")
def kalman_trend(X):
    """Local-linear-trend Kalman filter on log price (noise scaled to each
    asset's trailing 1-year variance); trade the sign of the filtered
    slope."""
    lp = np.log(X.C)
    v = X.LR.rolling(250, min_periods=60).var()      # trailing, not full
    out = {}
    for c in lp.columns:
        vc = v[c].to_numpy()
        _, sl = _kalman_llt(lp[c].to_numpy(), 0.1 * vc, 1e-4 * vc, vc)
        out[c] = sl
    return L.sign(pd.DataFrame(out, index=lp.index))


@model("kalman_mr")
def kalman_mr(X):
    """Kalman local-level filter of log price; fade the deviation of price
    from the filtered level beyond 2 sd (100d)."""
    lp = np.log(X.C)
    v = X.LR.rolling(250, min_periods=60).var()
    out = {}
    for c in lp.columns:
        vc = v[c].to_numpy()
        lv, _ = _kalman_llt(lp[c].to_numpy(), 0.05 * vc, 0.0, vc)
        out[c] = lv
    dev = lp - pd.DataFrame(out, index=lp.index)
    return L.fade_band(L.zscore(dev, 100), 2.0, 0.0)


@model("state_space_trend")
def state_space_trend(X):
    """Unobserved-components local linear trend with noise variances fitted
    by maximum likelihood each year (statsmodels); sign of filtered slope."""
    def one(s):
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        import warnings
        lp = np.log(s)

        def fit(tr):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = UnobservedComponents(tr.to_numpy()[-750:] * 100,
                                             level="local linear trend")
                    return m.fit(disp=False).params
                except Exception:
                    return None

        def apply(p, s2):
            s2v = s2.to_numpy() * 100
            # params order: sigma2.irregular, sigma2.level, sigma2.trend
            _, sl = _kalman_llt(s2v, p[1], max(p[2], 1e-12), p[0])
            return pd.Series(sl, index=s2.index)
        return L.yearly_refit(lp, fit, apply)
    sl = X.cached("ucm", lambda: L.per_column(X.C, one))
    return L.sign(sl)


@model("roll_reg_entry")
def roll_reg_entry(X):
    """Rolling 500-day regression of next-day return on 1, 5 and 20-day
    returns; trade the forecast sign."""
    r1 = X.R
    r5 = L.ret(X.C, 5)
    r20 = L.ret(X.C, 20)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    n = 500
    for c in X.C.columns:
        Y = r1[c].shift(-1)
        F = pd.concat([r1[c], r5[c], r20[c]], axis=1).to_numpy()
        y = Y.to_numpy()
        for i in range(n + 20, len(y), 21):              # refit monthly
            tr = slice(i - n, i - 1)                     # y known to i-1
            A = F[tr]
            yy = y[tr]
            ok = np.isfinite(A).all(1) & np.isfinite(yy)
            if ok.sum() < 250:
                continue
            A1 = np.c_[np.ones(ok.sum()), A[ok]]
            beta = np.linalg.lstsq(A1, yy[ok], rcond=None)[0]
            seg = slice(i, min(i + 21, len(y)))
            pred = np.c_[np.ones(seg.stop - seg.start), F[seg]] @ beta
            out.iloc[seg, out.columns.get_loc(c)] = np.sign(
                np.nan_to_num(pred))
    return out


@model("quantile_reg")
def quantile_reg(X):
    """Quantile regression (yearly refit, prior 3 years) of the next 5-day
    return on z20 and the 20-day return: long when the 30th-percentile
    forecast is above 0, short when the 70th is below 0."""
    def one(c):
        from statsmodels.regression.quantile_regression import QuantReg
        import warnings
        C = X.C[c]
        z = L.zscore(C.to_frame(), 20)[c]
        m = L.ret(C.to_frame(), 20)[c]
        y = C.pct_change(5, fill_method=None).shift(-5)
        df = pd.DataFrame(dict(y=y, z=z, m=m)).dropna(subset=["z", "m"])
        out = pd.Series(0.0, index=C.index)
        years = sorted(set(df.index.year))
        for yv in years[3:]:
            tr = df[(df.index.year < yv) & (df.index.year >= yv - 3)].dropna()
            tr = tr.iloc[:-5]
            te = df[df.index.year == yv]
            if len(tr) < 300 or te.empty:
                continue
            A = np.c_[np.ones(len(tr)), tr[["z", "m"]].to_numpy()]
            At = np.c_[np.ones(len(te)), te[["z", "m"]].to_numpy()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lo = QuantReg(tr["y"].to_numpy(), A).fit(q=0.3).params
                hi = QuantReg(tr["y"].to_numpy(), A).fit(q=0.7).params
            s = (At @ lo > 0).astype(float) - (At @ hi < 0).astype(float)
            out[te.index] = s
        return out
    return pd.DataFrame({c: one(c) for c in X.C.columns})


@model("empirical_dist")
def empirical_dist(X):
    """Expanding empirical P(next day up | z20 decile); trade when > 53%
    or < 47% with 100+ observations in the bucket."""
    z = L.zscore(X.C, 20)
    nxt = (X.R.shift(-1) > 0).astype(float).where(X.R.shift(-1).notna())
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    edges = np.array([-np.inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, np.inf])
    for c in X.C.columns:
        b = np.digitize(z[c].to_numpy(), edges)
        y = nxt[c].to_numpy()
        cnt = np.zeros(len(edges) + 1)
        up = np.zeros(len(edges) + 1)
        for i in range(len(b)):
            if i >= 1 and np.isfinite(y[i - 1]) and np.isfinite(z[c].iat[i - 1]):
                cnt[b[i - 1]] += 1
                up[b[i - 1]] += y[i - 1]
            if not np.isfinite(z[c].iat[i]):
                continue
            k = b[i]
            if cnt[k] >= 100:
                p = up[k] / cnt[k]
                out.iat[i, out.columns.get_loc(c)] = (1.0 if p > 0.53 else
                                                     -1.0 if p < 0.47 else 0)
    return out


@model("bootstrap_dist")
def bootstrap_dist(X):
    """Stationary bootstrap of the last 60 daily returns: trade the sign of
    the mean when its 90% interval excludes zero (weekly decisions)."""
    rng = np.random.default_rng(0)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    R = X.R.to_numpy()
    for i in range(60, len(R), 5):
        w = R[i - 59:i + 1]
        for j in range(R.shape[1]):
            x = w[:, j]
            x = x[np.isfinite(x)]
            if len(x) < 50:
                continue
            idx = rng.integers(0, len(x), size=(500, len(x)))
            m = x[idx].mean(axis=1)
            lo, hi = np.quantile(m, [0.05, 0.95])
            out.iloc[i:i + 5, j] = 1.0 if lo > 0 else (-1.0 if hi < 0 else 0)
    return out


@model("monte_carlo_entry")
def monte_carlo_entry(X):
    """Block-bootstrap 5-day paths from the last 250 days; long when
    P(5-day return > 0) > 0.55, short when < 0.45 (weekly)."""
    rng = np.random.default_rng(1)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    R = X.R.to_numpy()
    for i in range(250, len(R), 5):
        w = R[i - 249:i + 1]
        for j in range(R.shape[1]):
            x = w[:, j]
            if np.isnan(x).mean() > 0.1:
                continue
            x = np.nan_to_num(x)
            st = rng.integers(0, len(x) - 5, size=1000)
            paths = np.stack([x[s:s + 5] for s in st]).sum(axis=1)
            p = (paths > 0).mean()
            out.iloc[i:i + 5, j] = 1.0 if p > 0.55 else (-1.0 if p < 0.45
                                                          else 0)
    return out


@model("mc_path_fcst")
def mc_path_fcst(X):
    """GBM paths with 60-day drift and vol: probability of touching +1 sd
    before -1 sd over 10 days; trade when > 0.55 / < 0.45."""
    mu = X.LR.rolling(60).mean()
    sd = X.LR.rolling(60).std()
    # Closed form for a driftful Brownian motion hitting +a before -a:
    a = sd * math.sqrt(10)
    nu = mu / sd ** 2
    p = (np.exp(2 * nu * a) - 1) / (np.exp(2 * nu * a) - np.exp(-2 * nu * a))
    return (p > 0.55).astype(float) - (p < 0.45).astype(float)


# ============================================================ technical

@model("rsi_quant")
def rsi_quant(X):
    """Connors RSI(2): long < 10, short > 90, exit through 50."""
    r = L.rsi(X.C, 2)
    return L.machine(r < 10, r > 90, r > 50, r < 50, r)


@model("stoch_quant")
def stoch_quant(X):
    """Stochastic(14,3) crosses up through 20: long until 80; mirror."""
    k = L.stoch(X.D, 14, 3)
    le = (k.shift(1) < 20) & (k >= 20)
    se = (k.shift(1) > 80) & (k <= 80)
    return L.machine(le, se, k > 80, k < 20, k)


@model("osc_pct")
def osc_pct(X):
    """RSI14 at its 250-day 5th percentile: long; 95th: short; out at 50th."""
    p = L.rolling_pct_rank(L.rsi(X.C, 14), 250)
    return L.machine(p < 0.05, p > 0.95, p > 0.5, p < 0.5, p)


@model("osc_composite")
def osc_composite(X):
    """Mean of z-scored RSI14, Stoch14, CCI20, Williams %R (250d z);
    fade beyond +-1.5, out at 0."""
    parts = [L.rsi(X.C, 14), L.stoch(X.D), L.cci(X.D), L.willr(X.D)]
    comp = sum(L.zscore(p, 250) for p in parts) / 4
    return L.fade_band(comp, 1.5, 0.0)


@model("confluence")
def confluence(X):
    """Five binary trend votes (SMA200, MACD>0, RSI>50, +DI>-DI with ADX>20,
    20-day breakout state): long on 4+, short on 1 or fewer."""
    a, pdi, ndi = L.adx(X.D)
    m = L.ema(X.C, 12) - L.ema(X.C, 26)
    votes = [X.C > L.sma(X.C, 200), m > 0, L.rsi(X.C, 14) > 50,
             (pdi > ndi) & (a > 20), donchian(X) > 0]
    up = sum(v.astype(float) for v in votes)
    return (up >= 4).astype(float) - (up <= 1).astype(float)


@model("factor_score")
def factor_score(X):
    """Time-series factor score: z(12m momentum) - z(5d reversal) - z(vol),
    each standardised over 3 years; weight = score/2 clipped."""
    mom = L.zscore(L.ret(X.C, 252), 750)
    rev = L.zscore(L.ret(X.C, 5), 750)
    vol = L.zscore(L.rstd(X.R, 60), 750)
    return L.clip1((mom - rev - vol) / 3)


@model("multi_factor_tech")
def multi_factor_tech(X):
    """Equal blend of trend (EWMA multi-speed), MACD, RSI(2) reversion and
    Bollinger fade signals."""
    return (trend_follow(X) + macd(X) + rsi_quant(X) + boll_z(X)) / 4


@model("bayes_agg")
def bayes_agg(X):
    """Naive-Bayes aggregation of 5 binary signals: each signal's expanding
    likelihood ratio P(s|up)/P(s|down) on next-day direction; trade when
    the posterior P(up) leaves 0.48-0.52."""
    sigs = [L.sign(L.ret(X.C, 252)), macd(X), L.sign(X.R),
            L.sign(-L.zscore(X.C, 20)), L.sign(X.C - L.sma(X.C, 200))]
    y = X.R.shift(-1)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for c in X.C.columns:
        S = np.stack([s[c].to_numpy() for s in sigs], axis=1)
        yy = y[c].to_numpy()
        # counts[k, state(0:-1,1:+1), outcome(0 down,1 up)]
        cnt = np.ones((S.shape[1], 2, 2))
        base = np.ones(2)
        for i in range(len(yy)):
            if i >= 1 and np.isfinite(yy[i - 1]) and yy[i - 1] != 0:
                o = int(yy[i - 1] > 0)
                base[o] += 1
                for k in range(S.shape[1]):
                    if S[i - 1, k] != 0:
                        cnt[k, int(S[i - 1, k] > 0), o] += 1
            if base.sum() < 250:
                continue
            lo = math.log(base[1] / base[0])
            for k in range(S.shape[1]):
                if S[i, k] != 0:
                    st = int(S[i, k] > 0)
                    lo += math.log((cnt[k, st, 1] / base[1]) /
                                   (cnt[k, st, 0] / base[0]))
            p = 1 / (1 + math.exp(-lo))
            out.iat[i, out.columns.get_loc(c)] = (1.0 if p > 0.52 else
                                                 -1.0 if p < 0.48 else 0.0)
    return out


# ===================================================== adaptive averages

@model("kama")
def kama_model(X):
    """Price vs Kaufman adaptive MA (10,2,30): sign."""
    return L.sign(X.C - L.kama(X.C))


@model("adaptive_ma")
def adaptive_ma(X):
    """VIDYA: EMA(20) whose speed scales with |CMO(9)|; sign(price - VIDYA)."""
    d = X.C.diff()
    up, dn = d.clip(lower=0).rolling(9).sum(), (-d).clip(lower=0).rolling(
        9).sum()
    cmo = ((up - dn) / (up + dn).replace(0, np.nan)).abs()
    return L.sign(X.C - L.adaptive_filter(X.C, (2 / 21) * cmo))


@model("hma")
def hma_model(X):
    """Hull MA(55) slope sign."""
    h = L.hma(X.C, 55)
    return L.sign(h - h.shift(1))


@model("zlema")
def zlema_model(X):
    """Price vs zero-lag EMA(50): sign."""
    return L.sign(X.C - L.zlema(X.C, 50))


@model("adaptive_channel")
def adaptive_channel(X):
    """Donchian whose lookback adapts to vol: 20 x (60d vol / 1y vol),
    clipped 10-60; exit at the half-length opposite channel."""
    q = (L.rstd(X.R, 60) / L.rstd(X.R, 250)).clip(0.5, 3)
    look = (20 * q).round().clip(10, 60)
    hi = {n: X.D["H"].rolling(n).max().shift(1) for n in range(10, 61, 5)}
    lo = {n: X.D["L"].rolling(n).min().shift(1) for n in range(10, 61, 5)}
    lk = (look / 5).round() * 5
    H = pd.DataFrame(np.nan, index=X.C.index, columns=X.C.columns)
    Lw = H.copy()
    Hx, Lx = H.copy(), H.copy()
    for n in range(10, 61, 5):
        m = lk == n
        H = H.where(~m, hi[n])
        Lw = Lw.where(~m, lo[n])
        hx = max(10, (n // 2) // 5 * 5)
        Hx = Hx.where(~m, hi[hx])
        Lx = Lx.where(~m, lo[hx])
    return L.machine(X.C > H, X.C < Lw, X.C < Lx, X.C > Hx, X.C)


# ==================================== fractal / memory / efficiency regime

def _hurst(x):
    """Variance-of-lagged-differences Hurst estimate."""
    lags = np.array([2, 4, 8, 16, 32])
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    if min(tau) <= 0:
        return np.nan
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def _roll_apply(frame, n, fn, step=5):
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    v = frame.to_numpy()
    for j in range(v.shape[1]):
        last = np.nan
        for i in range(n, len(v)):
            if (i - n) % step == 0:
                w = v[i - n + 1:i + 1, j]
                last = fn(w) if np.isfinite(w).all() else np.nan
            out.iat[i, j] = last
    return out


@model("hurst")
def hurst(X):
    """Rolling 250-day Hurst exponent: > 0.55 follow the 20-day return,
    < 0.45 fade z20."""
    H = X.cached("hurst", lambda: _roll_apply(np.log(X.C), 250, _hurst))
    tr = L.sign(L.ret(X.C, 20))
    mr = L.clip1(-L.zscore(X.C, 20) / 2)
    return tr.where(H > 0.55, mr.where(H < 0.45, 0.0))


def _sign_entropy(x, m=3):
    s = (np.diff(x) > 0).astype(int)
    if len(s) < m + 10:
        return np.nan
    codes = sum(s[i:len(s) - m + 1 + i] << i for i in range(m))
    p = np.bincount(codes, minlength=2 ** m) / len(codes)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / m)


@model("entropy_regime")
def entropy_regime(X):
    """Shannon entropy of 3-day up/down patterns over 60 days: in the lowest
    third of its 2-year range (ordered market) follow the 20-day trend;
    otherwise flat."""
    E = X.cached("entropy", lambda: _roll_apply(np.log(X.C), 60,
                                                 _sign_entropy))
    p = L.rolling_pct_rank(E, 500)
    return L.sign(L.ret(X.C, 20)).where(p < 1 / 3, 0.0)


@model("efficiency_ratio")
def efficiency_ratio(X):
    """Kaufman efficiency ratio(20) > 0.4: follow the 20-day direction."""
    er = L.efficiency_ratio(X.C, 20)
    return L.sign(L.ret(X.C, 20)).where(er > 0.4, 0.0)


def _vr(x, q=5):
    r = np.diff(x)
    if len(r) < 5 * q:
        return np.nan
    v1 = r.var()
    rq = x[q:] - x[:-q]
    return rq.var() / (q * v1) if v1 > 0 else np.nan


@model("variance_ratio")
def variance_ratio(X):
    """Lo-MacKinlay VR(5) over 250 days: > 1.1 follow the 5-day return,
    < 0.9 fade it."""
    V = X.cached("vr", lambda: _roll_apply(np.log(X.C), 250, _vr))
    r5 = L.sign(L.ret(X.C, 5))
    return r5.where(V > 1.1, (-r5).where(V < 0.9, 0.0))


@model("vr_mr")
def vr_mr(X):
    """Variance-ratio mean reversion: only the VR(5) < 0.9 side, fading the
    5-day return."""
    V = X.cached("vr", lambda: _roll_apply(np.log(X.C), 250, _vr))
    return (-L.sign(L.ret(X.C, 5))).where(V < 0.9, 0.0)


def _fdi(x):
    """Sevcik fractal dimension of a path."""
    n = len(x)
    y = (x - x.min()) / (x.max() - x.min() + 1e-12)
    t = np.linspace(0, 1, n)
    length = np.sqrt(np.diff(y) ** 2 + np.diff(t) ** 2).sum()
    return 1 + math.log(length) / math.log(2 * (n - 1))


@model("fractal_dim")
def fractal_dim(X):
    """Sevcik fractal dimension (30d): < 1.4 trending (follow 20d), > 1.6
    choppy (fade z20)."""
    F = X.cached("fdi", lambda: _roll_apply(np.log(X.C), 30, _fdi, step=1))
    tr = L.sign(L.ret(X.C, 20))
    mr = L.clip1(-L.zscore(X.C, 20) / 2)
    return tr.where(F < 1.4, mr.where(F > 1.6, 0.0))


def _gph_d(x):
    """Geweke Porter-Hudak long-memory d of returns."""
    r = np.diff(x)
    n = len(r)
    m = int(n ** 0.5)
    f = np.fft.fft(r - r.mean())
    I = (np.abs(f[1:m + 1]) ** 2) / (2 * math.pi * n)
    lam = 2 * math.pi * np.arange(1, m + 1) / n
    xreg = np.log(4 * np.sin(lam / 2) ** 2)
    return -np.polyfit(xreg, np.log(I + 1e-300), 1)[0]


@model("long_memory")
def long_memory(X):
    """GPH estimate of d on 500 days of returns: d > 0.1 follow the 20-day
    return, d < -0.1 fade it."""
    Dd = X.cached("gph", lambda: _roll_apply(np.log(X.C), 500, _gph_d,
                                             step=20))
    r20 = L.sign(L.ret(X.C, 20))
    return r20.where(Dd > 0.1, (-r20).where(Dd < -0.1, 0.0))


# ===================================================== pattern matching

def _analog(C, k=20, win=20, horizon=5, step=5, metric="euclid"):
    """Nearest past windows of normalised log price; average their next
    `horizon` returns (only fully known outcomes are used)."""
    lp = np.log(C)
    out = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for c in C.columns:
        x = lp[c].to_numpy()
        ok = np.isfinite(x)
        if ok.sum() < 1000:
            continue
        n = len(x)
        W = np.lib.stride_tricks.sliding_window_view(x, win)  # end = i+win-1
        Wn = W - W[:, -1:]
        sd = Wn.std(axis=1, keepdims=True)
        Wn = Wn / np.where(sd > 0, sd, np.nan)
        fut = np.full(n, np.nan)
        fut[:n - horizon] = x[horizon:] - x[:n - horizon]
        for i in range(750, n, step):
            q = Wn[i - win + 1]
            if not np.isfinite(q).all():
                continue
            last = i - horizon - win + 1           # outcome known by i
            cand = Wn[:last]
            cf = fut[win - 1:win - 1 + last]
            good = np.isfinite(cand).all(axis=1) & np.isfinite(cf)
            if good.sum() < 200:
                continue
            cand, cf = cand[good], cf[good]
            if metric == "corr":
                d = -(cand @ q) / win
            else:
                d = ((cand - q) ** 2).sum(axis=1)
            nn = np.argpartition(d, k)[:k]
            out.iloc[i:i + step, out.columns.get_loc(c)] = np.sign(
                cf[nn].mean())
    return out


@model("analog_match")
def analog_match(X):
    """Analog pattern matching: 20 nearest past 20-day normalised paths
    (Euclidean); trade the sign of their mean next-5-day return."""
    return X.cached("analog_e", lambda: _analog(X.C, 20, 20, 5))


@model("scenario_match")
def scenario_match(X):
    """Historical scenario matching on a state vector (5d, 20d, 60d return
    z-scores and vol percentile): 50 nearest past states, mean next 5-day
    return sign."""
    feats = [L.zscore(L.ret(X.C, 5), 250), L.zscore(L.ret(X.C, 20), 250),
             L.zscore(L.ret(X.C, 60), 250),
             L.rolling_pct_rank(L.rstd(X.R, 20), 250)]
    fut = np.log(X.C).shift(-5) - np.log(X.C)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for c in X.C.columns:
        F = np.stack([f[c].to_numpy() for f in feats], axis=1)
        y = fut[c].to_numpy()
        for i in range(1000, len(y), 5):
            q = F[i]
            if not np.isfinite(q).all():
                continue
            cand, cy = F[:i - 5], y[:i - 5]
            g = np.isfinite(cand).all(1) & np.isfinite(cy)
            if g.sum() < 300:
                continue
            d = ((cand[g] - q) ** 2).sum(1)
            nn = np.argpartition(d, 50)[:50]
            out.iloc[i:i + 5, out.columns.get_loc(c)] = np.sign(
                cy[g][nn].mean())
    return out


@model("nn_pattern")
def nn_pattern(X):
    """Nearest-neighbour pattern matching by correlation of 10-day return
    shapes (k = 30), next-5-day mean sign."""
    return X.cached("analog_c", lambda: _analog(X.C, 30, 10, 5,
                                                metric="corr"))


@model("dtw_pattern")
def dtw_pattern(X):
    """Dynamic-time-warping nearest neighbours (k = 20) of the last 15 days'
    normalised path among 1,500 sampled past windows; next-5-day sign."""
    from numba import njit

    @njit(cache=True)
    def dtw(a, b, w):
        n = len(a)
        D = np.full((n + 1, n + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(1, i - w), min(n, i + w) + 1):
                c = (a[i - 1] - b[j - 1]) ** 2
                D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[n, n]

    win, horizon, k = 15, 5, 20
    rng = np.random.default_rng(0)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    lp = np.log(X.C)
    for c in X.C.columns:
        x = lp[c].to_numpy()
        n = len(x)
        W = np.lib.stride_tricks.sliding_window_view(x, win)
        Wn = W - W[:, -1:]
        sd = Wn.std(axis=1, keepdims=True)
        Wn = Wn / np.where(sd > 0, sd, np.nan)
        fut = np.full(n, np.nan)
        fut[:n - horizon] = x[horizon:] - x[:n - horizon]
        for i in range(750, n, 10):
            q = Wn[i - win + 1]
            if not np.isfinite(q).all():
                continue
            last = i - horizon - win + 1
            idx = rng.choice(last, size=min(1500, last), replace=False)
            cf = fut[idx + win - 1]
            g = np.isfinite(Wn[idx]).all(1) & np.isfinite(cf)
            idx, cf = idx[g], cf[g]
            if len(idx) < 200:
                continue
            d = np.array([dtw(q, Wn[j], 3) for j in idx])
            nn = np.argpartition(d, k)[:k]
            out.iloc[i:i + 10, out.columns.get_loc(c)] = np.sign(cf[nn].mean())
    return out


# ============================================== candles and conditional

def _cond_prob_trade(state, R, min_n=100, hi=0.53, lo=0.47):
    """Expanding P(next day up | state); trade the side above hi/below lo."""
    out = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    y = R.shift(-1)
    for c in R.columns:
        s = state[c].to_numpy()
        yy = y[c].to_numpy()
        cnt, up = {}, {}
        for i in range(len(s)):
            if i >= 1 and np.isfinite(yy[i - 1]) and s[i - 1] == s[i - 1]:
                k = s[i - 1]
                cnt[k] = cnt.get(k, 0) + 1
                up[k] = up.get(k, 0) + (yy[i - 1] > 0)
            k = s[i]
            if k == k and cnt.get(k, 0) >= min_n:
                p = up[k] / cnt[k]
                out.iat[i, out.columns.get_loc(c)] = (1.0 if p > hi else
                                                     -1.0 if p < lo else 0.0)
    return out


@model("candle_sequence")
def candle_sequence(X):
    """Last 3 daily candles up/down (8 states): expanding conditional
    probability of an up day next."""
    u = (X.C > X.D["O"]).astype(int)
    st = (u + 2 * u.shift(1) + 4 * u.shift(2)).where(X.C.notna())
    return _cond_prob_trade(st, X.R)


@model("markov_candle")
def markov_candle(X):
    """3-state candle Markov chain (big up / small / big down by 0.5 sd):
    expanding transition probabilities, trade expected direction."""
    z = X.R / L.rstd(X.R, 60).shift(1)
    st = pd.DataFrame(np.select([z > 0.5, z < -0.5], [2, 0], 1),
                      index=z.index, columns=z.columns).where(z.notna())
    return _cond_prob_trade(st, X.R)


@model("pattern_cond_prob")
def pattern_cond_prob(X):
    """Candle patterns (inside, outside-up, outside-down, other) x trend
    sign: expanding conditional next-day probability."""
    H, Lo, O, C = X.D["H"], X.D["L"], X.D["O"], X.C
    inside = (H < H.shift(1)) & (Lo > Lo.shift(1))
    outside = (H > H.shift(1)) & (Lo < Lo.shift(1))
    pat = np.select([inside, outside & (C > O), outside & (C <= O)],
                    [1, 2, 3], 0)
    tr = (C > L.sma(C, 50)).astype(int)
    st = pd.DataFrame(pat, index=C.index, columns=C.columns) * 2 + tr
    return _cond_prob_trade(st.where(C.notna()), X.R)


@model("cond_prob")
def cond_prob(X):
    """Conditional probability on (trend sign, vol regime, last-day sign):
    expanding P(up next)."""
    tr = (X.C > L.sma(X.C, 100)).astype(int)
    vr = (L.rolling_pct_rank(L.rstd(X.R, 20), 250) > 0.5).astype(int)
    ld = (X.R > 0).astype(int)
    return _cond_prob_trade((tr * 4 + vr * 2 + ld).where(X.R.notna()), X.R)


@model("bayes_prob_update")
def bayes_prob_update(X):
    """Beta-binomial posterior of P(up next | trend state) with a Beta(50,50)
    prior and exponential forgetting (half-life 500 obs); trade when the
    posterior mean leaves 0.48-0.52."""
    tr = (X.C > L.sma(X.C, 100)).astype(int).where(X.C.notna())
    y = X.R.shift(-1)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    lam = 0.5 ** (1 / 500)
    for c in X.C.columns:
        a = np.full(2, 50.0)
        b = np.full(2, 50.0)
        s, yy = tr[c].to_numpy(), y[c].to_numpy()
        for i in range(len(s)):
            if i >= 1 and np.isfinite(yy[i - 1]) and s[i - 1] == s[i - 1]:
                k = int(s[i - 1])
                a[k] = 50 + lam * (a[k] - 50) + (yy[i - 1] > 0)
                b[k] = 50 + lam * (b[k] - 50) + (yy[i - 1] <= 0)
            if s[i] == s[i]:
                k = int(s[i])
                p = a[k] / (a[k] + b[k])
                out.iat[i, out.columns.get_loc(c)] = (1.0 if p > 0.52 else
                                                     -1.0 if p < 0.48 else 0.0)
    return out


@model("failed_breakout")
def failed_breakout(X):
    """Failed breakout: a close above the 20-day high that closes back
    inside within 2 days -> short for 5 days (and the mirror)."""
    hi = X.D["H"].rolling(20).max().shift(1)
    lo = X.D["L"].rolling(20).min().shift(1)
    bo_up = (X.C > hi)
    bo_dn = (X.C < lo)
    hi_lvl, lo_lvl = hi.where(bo_up).ffill(limit=2), lo.where(bo_dn).ffill(
        limit=2)
    fail_up = (bo_up.shift(1).fillna(False) | bo_up.shift(2).fillna(False)
               ) & (X.C < hi_lvl)
    fail_dn = (bo_dn.shift(1).fillna(False) | bo_dn.shift(2).fillna(False)
               ) & (X.C > lo_lvl)
    return L.hold_for(fail_dn.astype(float) - fail_up.astype(float), 5)


def _breakout_trades(X, n=20, hold=10):
    """Signed breakout events and their realised hold-period return."""
    hi = X.D["H"].rolling(n).max().shift(1)
    lo = X.D["L"].rolling(n).min().shift(1)
    ev = ((X.C > hi) & (X.C.shift(1) <= hi.shift(1))).astype(float) - \
         ((X.C < lo) & (X.C.shift(1) >= lo.shift(1))).astype(float)
    fwd = np.log(X.C).shift(-hold) - np.log(X.C)
    return ev, fwd


def _success_rate_filter(ev, fwd, hold, rule):
    out = pd.DataFrame(0.0, index=ev.index, columns=ev.columns)
    for c in ev.columns:
        e = ev[c].to_numpy()
        f = fwd[c].to_numpy()
        hist = []
        for i in range(len(e)):
            # outcomes become known `hold` days after the event
            j = i - hold
            if j >= 0 and e[j] != 0 and np.isfinite(f[j]):
                hist.append(float(e[j] * f[j] > 0))
            if e[i] != 0 and rule(hist):
                out.iat[i, out.columns.get_loc(c)] = e[i]
    return L.hold_for(out, hold)


@model("bayes_breakout")
def bayes_breakout(X):
    """20-day breakouts taken only when the Beta(5,5)-prior posterior mean
    of past breakout success (10-day follow-through) exceeds 0.55."""
    ev, fwd = _breakout_trades(X)
    return _success_rate_filter(ev, fwd, 10, lambda h: (5 + sum(h)) / (
        10 + len(h)) > 0.55)


@model("breakout_success")
def breakout_success(X):
    """20-day breakouts taken only when > 55% of the last 30 succeeded."""
    ev, fwd = _breakout_trades(X)
    return _success_rate_filter(ev, fwd, 10, lambda h: len(h) >= 30 and
                                np.mean(h[-30:]) > 0.55)


@model("cond_breakout")
def cond_breakout(X):
    """20-day breakouts only after compression (BB width in the lowest
    quintile of 250 days within the last 5 days); hold 10 days."""
    ev, _ = _breakout_trades(X)
    m, s = L.sma(X.C, 20), L.rstd(X.C, 20)
    comp = (L.rolling_pct_rank(s / m, 250) < 0.2).shift(1).rolling(5).max() > 0
    return L.hold_for(ev.where(comp, 0.0), 10)


@model("breakout_plain")
def breakout_plain(X):
    """Reference: every 20-day breakout, held 10 days."""
    ev, _ = _breakout_trades(X)
    return L.hold_for(ev, 10)
