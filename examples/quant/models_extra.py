"""
The remaining models: GARCH-family and range-based volatility estimators
(as sizing for the trend model and as a breakout trigger), regime-
conditional variants, support/resistance and level statistics, the
technical factor model, the range-expansion probability model and
intraday VWAP-deviation reversion.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from examples.quant import lib as L
from examples.quant.core import model
from examples.quant.models_risk import _base, BASE

warnings.filterwarnings("ignore")


# ============================================================ GARCH family

def _garch_vol(r, kind="GARCH"):
    """Conditional daily vol, parameters refit each year on the prior 3
    years (arch package), filtered forward with those fixed parameters."""
    from arch import arch_model

    def fit(tr):
        x = tr.to_numpy()[-750:] * 100
        try:
            if kind == "GARCH":
                m = arch_model(x, vol="GARCH", p=1, q=1, mean="Constant")
            else:
                m = arch_model(x, vol="EGARCH", p=1, o=1, q=1,
                               mean="Constant")
            return m.fit(disp="off").params.to_numpy(), x.var()
        except Exception:
            return None

    def apply(fp, s):
        p, v0 = fp
        x = s.to_numpy() * 100
        out = np.empty(len(x))
        if kind == "GARCH":
            mu, om, a, b = p
            h = v0
            for t in range(len(x)):
                out[t] = h
                e = x[t] - mu
                h = om + a * e * e + b * h
        else:
            mu, om, a, g, b = p
            lh = math.log(v0)
            ez = math.sqrt(2 / math.pi)
            for t in range(len(x)):
                out[t] = math.exp(lh)
                z = (x[t] - mu) / math.sqrt(out[t])
                lh = om + a * (abs(z) - ez) + g * z + b * lh
                lh = min(max(lh, -20), 20)
        # out[t] is the variance forecast for day t made at t-1; we want the
        # forecast for day t+1 known at t's close -> shift by one
        return pd.Series(np.sqrt(np.r_[out[1:], np.nan]) / 100, index=s.index)
    return L.yearly_refit(r, fit, apply)


def _garch_frame(X, kind):
    return X.cached(f"garch_{kind}", lambda: L.per_column(
        X.R.fillna(0).where(X.C.notna()), _garch_vol, kind))


def _sized(X, daily_vol):
    return _base(X) * (0.10 / (daily_vol * math.sqrt(252))).clip(upper=3) / \
        len(X.C.columns)


@model("garch_sizing", scale=False, agg="sum", base=BASE, norm=True)
def garch_sizing(X):
    """GARCH(1,1) volatility model: the trend model sized by the GARCH
    one-day-ahead vol forecast (yearly walk-forward fit)."""
    return _sized(X, _garch_frame(X, "GARCH"))


@model("egarch_sizing", scale=False, agg="sum", base=BASE, norm=True)
def egarch_sizing(X):
    """EGARCH(1,1,1) (asymmetric) volatility model as the sizing input."""
    return _sized(X, _garch_frame(X, "EGARCH"))


@model("garch_breakout")
def garch_breakout(X):
    """GARCH forecast breakout: a day moving more than 2x its GARCH forecast
    vol is followed for 5 days."""
    v = _garch_frame(X, "GARCH").shift(1)        # forecast made yesterday
    z = X.R / v
    return L.hold_for((z > 2).astype(float) - (z < -2).astype(float), 5)


@model("vol_clustering", scale=False, agg="sum", base=BASE, norm=True)
def vol_clustering(X):
    """Volatility clustering: size by a fast EWMA vol (span 10) that reacts
    within days to vol bursts, instead of the 60-day default."""
    v = X.R.ewm(span=10, min_periods=10).std()
    return _sized(X, v)


def _parkinson(D, n=20):
    hl = np.log(D["H"] / D["L"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * math.log(2)))


def _gk(D, n=20):
    hl = np.log(D["H"] / D["L"]) ** 2
    co = np.log(D["C"] / D["O"]) ** 2
    return np.sqrt((0.5 * hl - (2 * math.log(2) - 1) * co).clip(lower=0)
                   .rolling(n).mean())


def _yz(D, n=20):
    o = np.log(D["O"] / D["C"].shift(1))
    c = np.log(D["C"] / D["O"])
    u, d = np.log(D["H"] / D["O"]), np.log(D["L"] / D["O"])
    rs = (u * (u - c) + d * (d - c)).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((o.rolling(n).var() + k * c.rolling(n).var() +
                    (1 - k) * rs).clip(lower=0))


@model("range_vol_sizing", scale=False, agg="sum", base=BASE, norm=True)
def range_vol_sizing(X):
    """Range-based volatility: mean daily log range x 0.627 (20 days) as the
    sizing vol."""
    return _sized(X, np.log(X.D["H"] / X.D["L"]).rolling(20).mean() * 0.627)


@model("parkinson_sizing", scale=False, agg="sum", base=BASE, norm=True)
def parkinson_sizing(X):
    """Parkinson high-low volatility (20 days) as the sizing vol."""
    return _sized(X, _parkinson(X.D))


@model("gk_sizing", scale=False, agg="sum", base=BASE, norm=True)
def gk_sizing(X):
    """Garman-Klass OHLC volatility (20 days) as the sizing vol."""
    return _sized(X, _gk(X.D))


@model("yz_sizing", scale=False, agg="sum", base=BASE, norm=True)
def yz_sizing(X):
    """Yang-Zhang volatility (overnight + open-close + Rogers-Satchell, 20
    days) as the sizing vol."""
    return _sized(X, _yz(X.D))


# =================================================== regime-conditional

@model("regime_cond_mom")
def regime_cond_mom(X):
    """Regime-conditional momentum: 3-month time-series momentum only while
    the cross-asset average 60-day efficiency ratio is above its 2-year
    median (a trending market overall)."""
    er = L.efficiency_ratio(X.C, 60).mean(axis=1)
    ok = er > er.rolling(500, min_periods=250).median()
    m = ok.to_numpy().reshape(-1, 1)
    return L.sign(L.ret(X.C, 63)).where(lambda f: np.broadcast_to(m, f.shape),
                                        0.0)


@model("regime_cond_mr")
def regime_cond_mr(X):
    """Regime-conditional mean reversion: z20 fade only where the 250-day
    Hurst exponent is below 0.5."""
    from examples.quant.models_daily import _roll_apply, _hurst
    H = X.cached("hurst", lambda: _roll_apply(np.log(X.C), 250, _hurst))
    return L.fade_band(L.zscore(X.C, 20), 2.0, 0.0).where(H < 0.5, 0.0)


@model("regime_cond_bo")
def regime_cond_bo(X):
    """Regime-conditional breakout: 20-day Donchian only in the low half of
    the 2-year volatility range."""
    from examples.quant.models_daily import donchian
    p = L.rolling_pct_rank(L.rstd(X.R, 60), 500)
    return donchian(X).where(p < 0.5, 0.0)


@model("regime_cond_trend")
def regime_cond_trend(X):
    """Regime-conditional trend following: the multi-speed trend model, long
    only while the 200-day average rises and short only while it falls."""
    s = _base(X)
    sl = L.sign(L.sma(X.C, 200) - L.sma(X.C, 200).shift(20))
    return s.where(np.sign(s) == sl, 0.0)


# ============================================== support / resistance

def _touch_fade(X, above, below, hold=5):
    """Fade a rejection: high through a level above but close back under it
    -> short; low through a level below but close back over -> long."""
    H, Lo, C = X.D["H"], X.D["L"], X.C
    ev = ((Lo <= below) & (C > below)).astype(float) - \
         ((H >= above) & (C < above)).astype(float)
    return ev


@model("dyn_sr")
def dyn_sr(X):
    """Dynamic support/resistance: the last confirmed swing high/low (5 bars
    each side) as levels; fade rejections for 5 days."""
    k = 5
    H, Lo = X.D["H"], X.D["L"]
    lh = H.shift(k).where(H.shift(k) == H.rolling(2 * k + 1).max()).ffill()
    ll = Lo.shift(k).where(Lo.shift(k) == Lo.rolling(2 * k + 1).min()).ffill()
    return L.hold_for(_touch_fade(X, lh, ll), 5)


@model("ml_sr")
def ml_sr(X):
    """Machine-learned support/resistance: each month, k-means (k=6) on the
    prices of the last year's confirmed swing points gives the levels; fade
    rejections of the nearest level above/below for 5 days."""
    from sklearn.cluster import KMeans
    k = 5
    H, Lo, C = X.D["H"], X.D["L"], X.C
    sh = H.shift(k).where(H.shift(k) == H.rolling(2 * k + 1).max())
    sl = Lo.shift(k).where(Lo.shift(k) == Lo.rolling(2 * k + 1).min())
    above = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    below = above.copy()
    months = C.index.to_period("M")
    starts = np.flatnonzero(~pd.Series(months).duplicated().to_numpy())
    for c in C.columns:
        pts = pd.concat([sh[c], sl[c]]).dropna().sort_index()
        for si, st in enumerate(starts):
            if st < 252:
                continue
            en = starts[si + 1] if si + 1 < len(starts) else len(C)
            win = pts[(pts.index < C.index[st]) &
                      (pts.index >= C.index[st - 252])].to_numpy()
            if len(win) < 12:
                continue
            lv = np.sort(KMeans(6, n_init=3, random_state=0).fit(
                win.reshape(-1, 1)).cluster_centers_.ravel())
            px = C[c].iloc[st:en].to_numpy()
            prev = C[c].shift(1).iloc[st:en].to_numpy()
            ref = np.where(np.isfinite(prev), prev, px)
            idx_hi = np.searchsorted(lv, ref)
            ab = np.where(idx_hi < len(lv), lv[np.minimum(idx_hi,
                                                         len(lv) - 1)], np.nan)
            be = np.where(idx_hi > 0, lv[np.maximum(idx_hi - 1, 0)], np.nan)
            above.iloc[st:en, above.columns.get_loc(c)] = ab
            below.iloc[st:en, below.columns.get_loc(c)] = be
    return L.hold_for(_touch_fade(X, above, below), 5)


def _rejections(X):
    """Rejections of the prior 20-day high/low and their 5-day fade result."""
    hi = X.D["H"].rolling(20).max().shift(1)
    lo = X.D["L"].rolling(20).min().shift(1)
    ev = _touch_fade(X, hi, lo)
    fwd = np.log(X.C).shift(-5) - np.log(X.C)
    return ev, fwd, hi, lo


def _rate_filter(ev, fwd, rule, hold=5):
    out = pd.DataFrame(0.0, index=ev.index, columns=ev.columns)
    for c in ev.columns:
        e, f = ev[c].to_numpy(), fwd[c].to_numpy()
        hist = []
        for i in range(len(e)):
            j = i - hold
            if j >= 0 and e[j] != 0 and np.isfinite(f[j]):
                hist.append(float(e[j] * f[j] > 0))
            if e[i] != 0 and rule(hist):
                out.iat[i, out.columns.get_loc(c)] = e[i]
    return L.hold_for(out, hold)


@model("level_reaction_prob")
def level_reaction_prob(X):
    """Price-level reaction probability: fade rejections of the 20-day
    high/low only while > 55% of the asset's earlier rejections paid over
    5 days (at least 20 seen)."""
    ev, fwd, _, _ = _rejections(X)
    return _rate_filter(ev, fwd, lambda h: len(h) >= 20 and np.mean(h) > 0.55)


@model("bayes_sr")
def bayes_sr(X):
    """Bayesian support/resistance: the same rejections, taken while the
    Beta(5,5)-prior posterior mean of the fade's success exceeds 0.55."""
    ev, fwd, _, _ = _rejections(X)
    return _rate_filter(ev, fwd, lambda h: (5 + sum(h)) / (10 + len(h)) > 0.55)


@model("level_strength")
def level_strength(X):
    """Level-strength scoring: fade a rejection of the 20-day high/low only
    when that level has been tested (within 0.25 ATR) on 3+ days of the last
    60."""
    ev, _, hi, lo = _rejections(X)
    a = L.atr(X.D, 14)
    th = ((X.D["H"] - hi).abs() < 0.25 * a).rolling(60).sum()
    tl = ((X.D["L"] - lo).abs() < 0.25 * a).rolling(60).sum()
    ev = ev.where(((ev < 0) & (th >= 3)) | ((ev > 0) & (tl >= 3)), 0.0)
    return L.hold_for(ev, 5)


# ======================================================= misc daily models

@model("tech_factor")
def tech_factor(X):
    """Technical factor model: cross-sectional composite of 12-1 momentum,
    MACD histogram (vol-normalised) and 1-month trend ranks; long top third,
    short bottom third, weekly."""
    vol = L.rstd(X.R, 60)
    m = L.ema(X.C, 12) - L.ema(X.C, 26)
    f = [L.ret(X.C, 252) - L.ret(X.C, 21), (m - L.ema(m, 9)) / (X.C * vol),
         L.ret(X.C, 21) / vol]
    sc = sum(x.rank(axis=1, pct=True) for x in f) / 3
    return L.rebalance(L.xs_rank_signal(sc), 5)


@model("range_exp_prob")
def range_exp_prob(X):
    """Range-expansion probability: expanding P(next day's range > 1.2 ATR
    | NR7 / inside-day state); when a compressed state's probability is
    above the asset's unconditional rate, the next close beyond today's
    range is followed for 3 days."""
    H, Lo, C = X.D["H"], X.D["L"], X.C
    rng = H - Lo
    a = L.atr(X.D, 14)
    nr7 = rng <= rng.rolling(7).min()
    inside = (H < H.shift(1)) & (Lo > Lo.shift(1))
    st = (nr7.astype(int) * 2 + inside.astype(int)).where(C.notna())
    big = (rng.shift(-1) > 1.2 * a).astype(float).where(rng.shift(-1).notna())
    p = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    base = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    for c in C.columns:
        s, y = st[c].to_numpy(), big[c].to_numpy()
        cnt, hit = np.zeros(4), np.zeros(4)
        for i in range(len(s)):
            if i >= 1 and np.isfinite(y[i - 1]) and s[i - 1] == s[i - 1]:
                k = int(s[i - 1])
                cnt[k] += 1
                hit[k] += y[i - 1]
            if s[i] == s[i] and s[i] > 0 and cnt[int(s[i])] >= 50:
                p.iat[i, p.columns.get_loc(c)] = hit[int(s[i])] / cnt[int(s[i])]
                base.iat[i, p.columns.get_loc(c)] = hit.sum() / cnt.sum()
    arm = p > base                    # compression raises the odds
    ev = ((C > H.shift(1)) & arm.shift(1, fill_value=False)).astype(float) - \
         ((C < Lo.shift(1)) & arm.shift(1, fill_value=False)).astype(float)
    return L.hold_for(ev, 3)


# ================================================= intraday VWAP (15-min)

@model("vwap_dev", freq="M")
def vwap_dev(X):
    """VWAP deviation reversion: session VWAP (tick volume, from the 17:00
    roll) and its volume-weighted sd; fade |z| > 2 back to VWAP; flat from
    16:00 to the roll."""
    from examples.quant.models_intraday import nyt, window_mask, _col
    T = nyt(X, "M")
    d = T["day"].to_numpy()
    P = X.M
    tp = (P["H"] + P["L"] + P["C"]) / 3
    v = P["V"].fillna(0)
    pv = (tp * v).groupby(d).cumsum()
    pv2 = (tp * tp * v).groupby(d).cumsum()
    vv = v.groupby(d).cumsum().replace(0, np.nan)
    vw = pv / vv
    sd = np.sqrt((pv2 / vv - vw * vw).clip(lower=0))
    z = (P["C"] - vw) / sd.replace(0, np.nan)
    late = window_mask(T, 1600, 1700)
    nbars = v.groupby(d).cumcount()
    ok = ~late & (nbars.to_numpy() >= 8)        # 2 hours into the session
    pos = L.machine((z < -2) & _col(ok), (z > 2) & _col(ok),
                    (z >= 0) | _col(late), (z <= 0) | _col(late), z)
    return pos
