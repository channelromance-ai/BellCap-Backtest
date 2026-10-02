"""
Cross-asset daily models: pairs and statistical arbitrage, cross-sectional
ranking and relative strength, macro and intermarket relationships.

Pairs are chosen for an economic reason before any test (same market in
two wrappers, substitute crops, sovereign curves, commodity currencies),
never by scanning for the best-fitting pair.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from bcbt.quant import GROUPS, ALL
from examples.quant import lib as L
from examples.quant.core import model

warnings.filterwarnings("ignore", category=FutureWarning)

PAIRS = [("SPX500_USD", "NAS100_USD"), ("SPX500_USD", "US2000_USD"),
         ("FR40_EUR", "NL25_EUR"), ("UK100_GBP", "FR40_EUR"),
         ("EUR_USD", "GBP_USD"), ("EUR_JPY", "AUD_JPY"),
         ("CORN_USD", "WHEAT_USD"), ("CORN_USD", "SOYBN_USD"),
         ("USB10Y_USD", "DE10YB_EUR"), ("USB10Y_USD", "UK10YB_GBP"),
         ("DE10YB_EUR", "UK10YB_GBP"), ("USB02Y_USD", "USB10Y_USD")]

XPAIRS = [("AUD_USD", "XAU_USD"), ("USD_CAD", "WTICO_USD"),
          ("AUD_JPY", "SPX500_USD"), ("JP225_USD", "EUR_JPY"),
          ("XAU_USD", "USB10Y_USD"), ("AU200_AUD", "AUD_USD")]

PAIR_VOL = 0.10 / math.sqrt(252)


def beta_to(R, f, n):
    """Rolling beta of every column of R on one factor series f."""
    mr, mf = R.rolling(n).mean(), f.rolling(n).mean()
    cov = R.mul(f, axis=0).rolling(n).mean() - mr.mul(mf, axis=0)
    var = f.rolling(n).var(ddof=0)
    return cov.div(var, axis=0)


def _pair_weights(C, legs, pairs, n_norm=None):
    """
    legs[p] = (s, beta): spread position (+1 = long a / short beta*b) and
    hedge ratio per day. Each pair is scaled to 10% annual spread vol and the
    pairs are averaged -> portfolio notional per symbol.
    """
    R = C.pct_change(fill_method=None)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    n = 0
    for (a, b) in pairs:
        if (a, b) not in legs:
            continue
        s, beta = legs[(a, b)]
        s = s.fillna(0)
        beta = beta.ffill().fillna(1.0)
        spr = R[a] - beta.shift(1) * R[b]
        k = (PAIR_VOL / spr.ewm(span=60, min_periods=20).std()).clip(upper=3)
        # 0 x an undefined size is still no position (it must not wipe out
        # the other pairs' weights in the sum)
        W[a] += (s * k).fillna(0)
        W[b] -= (s * k * beta).fillna(0)
        n += 1
    return W / (n_norm or max(n, 1))


def _ols_spread(C, a, b, n):
    la, lb = np.log(C[a]), np.log(C[b])
    beta = la.rolling(n).cov(lb) / lb.rolling(n).var()
    spr = la - beta * lb
    z = (spr - spr.rolling(n).mean()) / spr.rolling(n).std()
    return z, beta


def _band(z, entry=2.0, exit=0.0):
    return L.fade_band(z.to_frame(), entry, exit).iloc[:, 0]


def _pairs_model(X, pairs, fn):
    legs = {}
    for a, b in pairs:
        if a in X.C and b in X.C:
            legs[(a, b)] = fn(a, b)
    return _pair_weights(X.C, legs, pairs)


# ================================================================== pairs

@model("spread_z", scale=False, agg="sum")
def spread_z(X):
    """Log-price spread on a rolling 120-day OLS hedge; enter |z| > 2,
    exit at 0."""
    def f(a, b):
        z, beta = _ols_spread(X.C, a, b, 120)
        return _band(z), beta
    return _pairs_model(X, PAIRS, f)


@model("dyn_hedge_pairs", scale=False, agg="sum")
def dyn_hedge_pairs(X):
    """Dynamic hedge ratio: 60-day rolling OLS, enter |z| > 2, exit 0."""
    def f(a, b):
        z, beta = _ols_spread(X.C, a, b, 60)
        return _band(z), beta
    return _pairs_model(X, PAIRS, f)


@model("coint_pairs", scale=False, agg="sum")
def coint_pairs(X):
    """Engle-Granger: trade the 250-day OLS spread (|z| > 2, exit 0) only in
    months when the residual ADF p-value on the prior 250 days is < 0.05."""
    from statsmodels.tsa.stattools import adfuller

    def f(a, b):
        la, lb = np.log(X.C[a]), np.log(X.C[b])
        z, beta = _ols_spread(X.C, a, b, 250)
        ok = pd.Series(False, index=X.C.index)
        months = X.C.index.to_period("M")
        first = ~pd.Series(months).duplicated().to_numpy()
        for i in np.flatnonzero(first):
            if i < 250:
                continue
            ya, yb = la.iloc[i - 250:i].to_numpy(), lb.iloc[i - 250:i].to_numpy()
            g = np.isfinite(ya) & np.isfinite(yb)
            if g.sum() < 200:
                continue
            bb = np.polyfit(yb[g], ya[g], 1)
            res = ya[g] - np.polyval(bb, yb[g])
            try:
                p = adfuller(res, maxlag=1, autolag=None)[1]
            except Exception:
                continue
            ok[months == months[i]] = p < 0.05
        return _band(z).where(ok, 0.0), beta
    return X.cached("coint_pairs", lambda: _pairs_model(X, PAIRS, f))


@model("relative_value", scale=False, agg="sum")
def relative_value(X):
    """Price-ratio relative value (hedge 1:1 in value): fade the 60-day z of
    log(a/b) beyond 2, exit at 0."""
    def f(a, b):
        r = np.log(X.C[a] / X.C[b])
        z = (r - r.rolling(60).mean()) / r.rolling(60).std()
        return _band(z), pd.Series(1.0, index=X.C.index)
    return _pairs_model(X, PAIRS, f)


@model("pairs_trading", scale=False, agg="sum")
def pairs_trading(X):
    """Gatev-Goetzmann-Rouwenhorst distance pairs: each half-year pick the 5
    closest normalised-price pairs per asset group over the past year;
    open at a 2 formation-sd divergence, close at the crossing."""
    C = X.C
    idx = C.index
    starts = [i for i in range(252, len(idx), 126)]
    legs_all = {}
    for g in GROUPS.values():
        cols = [c for c in g if c in C.columns]
        for st in starts:
            form = C[cols].iloc[st - 252:st]
            form = form.dropna(axis=1, thresh=240).ffill()
            if form.shape[1] < 2:
                continue
            nrm = form / form.iloc[0]
            cand = []
            cs = list(nrm.columns)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    d = ((nrm[cs[i]] - nrm[cs[j]]) ** 2).sum()
                    cand.append((d, cs[i], cs[j]))
            cand.sort()
            for d, a, b in cand[:5]:
                sd = (nrm[a] - nrm[b]).std()
                end = min(st + 126, len(idx))
                tr = C[[a, b]].iloc[st:end]
                base = C[[a, b]].iloc[st - 1]
                spr = (tr[a] / base[a] - tr[b] / base[b]) / sd
                pos = _band(spr, 2.0, 0.0)
                key = (a, b)
                s_prev, b_prev = legs_all.get(key, (pd.Series(0.0, index=idx),
                                                    pd.Series(1.0, index=idx)))
                s_prev = s_prev.copy()
                s_prev.loc[pos.index] = pos
                legs_all[key] = (s_prev, b_prev)
    # 5 pairs per group live at a time: a count known in advance (the number
    # of distinct pairs ever chosen is not)
    return _pair_weights(C, legs_all, list(legs_all), n_norm=5 * len(GROUPS))


@model("cross_asset_rv", scale=False, agg="sum")
def cross_asset_rv(X):
    """Cross-asset relative value on economically linked pairs (AUD/gold,
    CAD/oil, AUDJPY/S&P, Nikkei/EURJPY, gold/Treasuries, ASX/AUD): fade the
    120-day OLS spread z beyond 2."""
    def f(a, b):
        z, beta = _ols_spread(X.C, a, b, 120)
        return _band(z), beta
    return _pairs_model(X, XPAIRS, f)


@model("beta_neutral_pairs", scale=False, agg="sum")
def beta_neutral_pairs(X):
    """Return-beta hedged pairs: 60-day return regression beta; fade the
    20-day cumulative spread return when its 120-day z exceeds 2."""
    def f(a, b):
        ra, rb = X.R[a], X.R[b]
        beta = ra.rolling(60).cov(rb) / rb.rolling(60).var()
        spr = (ra - beta.shift(1) * rb).rolling(20).sum()
        z = (spr - spr.rolling(120).mean()) / spr.rolling(120).std()
        return _band(z), beta
    return _pairs_model(X, PAIRS, f)


@model("kalman_pairs", scale=False, agg="sum")
def kalman_pairs(X):
    """Kalman-filter hedge ratio and intercept (Chan, delta 1e-4); trade the
    standardised one-step forecast error beyond 1, exit at 0."""
    def f(a, b):
        y, x = np.log(X.C[a]).to_numpy(), np.log(X.C[b]).to_numpy()
        n = len(y)
        delta, ve = 1e-4, 1e-3
        Vw = delta / (1 - delta) * np.eye(2)
        th = np.zeros(2)
        P = np.zeros((2, 2))
        e_std = np.full(n, np.nan)
        beta = np.full(n, np.nan)
        init = False
        for t in range(n):
            if not (np.isfinite(y[t]) and np.isfinite(x[t])):
                continue
            F = np.array([x[t], 1.0])
            if not init:
                th = np.array([y[t] / x[t], 0.0])
                P = np.eye(2)
                init = True
            R = P + Vw
            yhat = F @ th
            Q = F @ R @ F + ve
            e = y[t] - yhat
            K = R @ F / Q
            th = th + K * e
            P = R - np.outer(K, F) @ R
            e_std[t] = e / math.sqrt(Q)
            beta[t] = th[0]
        z = pd.Series(e_std, index=X.C.index)
        z[: 250] = np.nan                                  # burn-in
        return _band(z, 1.0, 0.0), pd.Series(beta, index=X.C.index)
    return _pairs_model(X, PAIRS, f)


def _corr_cond(X, a, b):
    ra, rb = X.R[a], X.R[b]
    c20 = ra.rolling(20).corr(rb)
    c250 = ra.rolling(250).corr(rb)
    rel = np.log(X.C[a]).diff(20) - np.log(X.C[b]).diff(20)
    beta = pd.Series(1.0, index=X.C.index)
    return (c20 < c250 - 0.4) & (c250 > 0.5), rel, beta


@model("corr_reversion", scale=False, agg="sum")
def corr_reversion(X):
    """Correlation breakdown (20d corr 0.4 below its 250d level, normally
    > 0.5): bet the 20-day relative move reverts, 10 days."""
    def f(a, b):
        cond, rel, beta = _corr_cond(X, a, b)
        ev = (-np.sign(rel)).where(cond, 0.0).fillna(0)
        return L.hold_for(ev.to_frame(), 10).iloc[:, 0], beta
    return _pairs_model(X, PAIRS + XPAIRS, f)


@model("corr_breakout", scale=False, agg="sum")
def corr_breakout(X):
    """Same breakdown, but bet the divergence continues for 10 days."""
    return -corr_reversion(X)


@model("corr_spread", scale=False, agg="sum")
def corr_spread(X):
    """Spread z trading (120d OLS, |z| > 2) only while the 60-day return
    correlation of the pair is above 0.7."""
    def f(a, b):
        z, beta = _ols_spread(X.C, a, b, 120)
        hc = X.R[a].rolling(60).corr(X.R[b]) > 0.7
        return _band(z).where(hc, 0.0), beta
    return _pairs_model(X, PAIRS + XPAIRS, f)


# ===================================================== statistical arbitrage

def _pca_sscore(X, cols, k, look=252, resid_win=60, step=21):
    """Avellaneda-Lee: residuals of returns on the top-k PCA eigenportfolios,
    cumulated over 60 days and fitted with an OU -> s-score."""
    R = X.R[cols]
    S = pd.DataFrame(np.nan, index=R.index, columns=cols)
    Rv = R.to_numpy()
    E = None
    for i in range(look, len(R)):
        if (i - look) % step == 0:
            w = Rv[i - look:i]
            good = np.isfinite(w).all(axis=0)
            if good.sum() < k + 2:
                E = None
                continue
            wg = w[:, good]
            sd = wg.std(axis=0)
            Z = (wg - wg.mean(axis=0)) / sd
            vals, vecs = np.linalg.eigh(np.cov(Z.T))
            V = vecs[:, ::-1][:, :k] / sd[:, None]       # eigenportfolios
            E = (good, V)
        if E is None:
            continue
        good, V = E
        win = Rv[i - resid_win + 1:i + 1][:, good]
        if not np.isfinite(win).all():
            continue
        F = win @ V                                       # factor returns
        A = np.c_[np.ones(resid_win), F]
        coef = np.linalg.lstsq(A, win, rcond=None)[0]
        res = win - A @ coef
        Xc = np.cumsum(res, axis=0)
        y, z = Xc[1:], Xc[:-1]
        zm, ym = z.mean(0), y.mean(0)
        b = ((z - zm) * (y - ym)).sum(0) / ((z - zm) ** 2).sum(0)
        a = ym - b * zm
        e = y - (a + b * z)
        ok = (b > 0) & (b < 0.97)                          # half-life < ~23d
        mu = a / (1 - b)
        seq = e.std(0) / np.sqrt(np.clip(1 - b * b, 1e-9, None))
        s = (Xc[-1] - mu) / seq
        s[~ok] = np.nan
        S.iloc[i, np.flatnonzero(good)] = s
    return S


def _sscore_positions(S):
    return L.machine(S < -1.25, S > 1.25, S > -0.5, S < 0.5, S.fillna(0))


@model("stat_arb")
def stat_arb(X):
    """Avellaneda-Lee stat arb within each asset group: residuals after the
    first principal component, s-score entry 1.25, exit 0.5."""
    parts = []
    for g in GROUPS.values():
        cols = [c for c in g if c in X.C.columns]
        parts.append(_sscore_positions(X.cached(
            f"pca1_{g[0]}", lambda: _pca_sscore(X, cols, 1))))
    return pd.concat(parts, axis=1).reindex(columns=X.C.columns).fillna(0)


@model("pca_neutral")
def pca_neutral(X):
    """PCA factor-neutral: residuals of all 25 CFDs after 3 principal
    components, s-score entry 1.25, exit 0.5."""
    S = X.cached("pca3_all", lambda: _pca_sscore(X, list(X.C.columns), 3))
    return _sscore_positions(S)


def _group_factor(X):
    out = pd.DataFrame(index=X.C.index, columns=X.C.columns, dtype=float)
    for g in GROUPS.values():
        cols = [c for c in g if c in X.C.columns]
        f = X.R[cols].mean(axis=1)
        for c in cols:
            out[c] = f
    return out


@model("factor_resid_mr")
def factor_resid_mr(X):
    """Fade the 5-day residual return after a rolling 60-day beta to the
    asset's group (equal-weight) factor."""
    F = _group_factor(X)
    beta = X.R.rolling(60).cov(F) / F.rolling(60).var()
    res = (X.R - beta.shift(1) * F).rolling(5).sum()
    return L.clip1(-L.zscore(res, 120) / 2)


@model("resid_z")
def resid_z(X):
    """Residual z-score entry: 20-day cumulative residual vs the group factor
    (60d beta), z over 120 days, fade beyond 2, exit 0."""
    F = _group_factor(X)
    beta = X.R.rolling(60).cov(F) / F.rolling(60).var()
    res = (X.R - beta.shift(1) * F).rolling(20).sum()
    return L.fade_band(L.zscore(res, 120), 2.0, 0.0)


# ======================================================= cross-sectional

GRP = list(GROUPS.values())


@model("cs_mr")
def cs_mr(X):
    """Cross-sectional reversal: within each group, long the weakest third
    and short the strongest third of 5-day returns; weekly."""
    return L.rebalance(L.xs_rank_groups(-L.ret(X.C, 5), GRP), 5)


@model("cs_mom")
def cs_mom(X):
    """Cross-sectional momentum within groups, 3-month return, weekly."""
    return L.rebalance(L.xs_rank_groups(L.ret(X.C, 63), GRP), 5)


@model("momentum_factor")
def momentum_factor(X):
    """12-1 month momentum factor across all 25 CFDs (vol-adjusted returns),
    long top / short bottom third, monthly."""
    s = (L.ret(X.C, 252) - L.ret(X.C, 21)) / L.rstd(X.R, 252)
    return L.month_end_rebalance(L.xs_rank_signal(s))


@model("cs_rank")
def cs_rank(X):
    """Cross-sectional ranking on 1-month return, all assets, weekly."""
    return L.rebalance(L.xs_rank_signal(L.ret(X.C, 21)), 5)


@model("pct_rank")
def pct_rank(X):
    """Rank assets by where today's 6-month return sits in its own 3-year
    history; long top third, short bottom third, weekly."""
    p = L.rolling_pct_rank(L.ret(X.C, 126), 750)
    return L.rebalance(L.xs_rank_signal(p), 5)


@model("mom_rank")
def mom_rank(X):
    """Momentum ranking: 6-month return, all assets, monthly."""
    return L.month_end_rebalance(L.xs_rank_signal(L.ret(X.C, 126)))


@model("vol_rank")
def vol_rank(X):
    """Low-volatility ranking: long lowest-vol third, short highest-vol third
    (each risk-scaled), monthly."""
    return L.month_end_rebalance(L.xs_rank_signal(-L.rstd(X.R, 60)))


@model("mr_rank")
def mr_rank(X):
    """Mean-reversion ranking: 1-week return across all assets, weekly."""
    return L.rebalance(L.xs_rank_signal(-L.ret(X.C, 5)), 5)


def _composite(X):
    rk = lambda s: s.rank(axis=1, pct=True)   # noqa
    return (rk(L.ret(X.C, 126)) + rk(-L.ret(X.C, 5)) + rk(-L.rstd(X.R, 60))) / 3


@model("composite_rank")
def composite_rank(X):
    """Composite rank (6m momentum, 1w reversal, low vol), terciles, weekly."""
    return L.rebalance(L.xs_rank_signal(_composite(X)), 5)


@model("ls_rank")
def ls_rank(X):
    """Long/short ranking: composite score, top and bottom quintile only."""
    return L.rebalance(L.xs_rank_signal(_composite(X), frac=0.2), 5)


@model("mkt_neutral_rank")
def mkt_neutral_rank(X):
    """Market-neutral ranking: composite score ranked within each group."""
    return L.rebalance(L.xs_rank_groups(_composite(X), GRP), 5)


@model("beta_adj_rank")
def beta_adj_rank(X):
    """Beta-adjusted ranking: 6-month residual momentum vs the group factor
    (250d beta), within groups, monthly."""
    F = _group_factor(X)
    beta = X.R.rolling(250).cov(F) / F.rolling(250).var()
    res = (X.R - beta.shift(1) * F).rolling(126).sum()
    return L.month_end_rebalance(L.xs_rank_groups(res, GRP))


@model("voladj_rank")
def voladj_rank(X):
    """Volatility-adjusted ranking: 6-month return / 6-month vol, monthly."""
    s = L.ret(X.C, 126) / L.rstd(X.R, 126)
    return L.month_end_rebalance(L.xs_rank_signal(s))


@model("rs_quant")
def rs_quant(X):
    """Relative-strength line (asset / its group index) above its 50-day
    average: long; below: short; demeaned within the group."""
    out = []
    for g in GRP:
        cols = [c for c in g if c in X.C.columns]
        idx = np.exp(np.log(X.C[cols]).diff().mean(axis=1).cumsum())
        rs = X.C[cols].div(idx, axis=0)
        s = L.sign(rs - L.sma(rs, 50))
        out.append(s.sub(s.mean(axis=1), axis=0))
    return pd.concat(out, axis=1).reindex(columns=X.C.columns).fillna(0)


@model("index_rs")
def index_rs(X):
    """Index relative strength: equity indices ranked on 3-month return,
    long top 2, short bottom 2, weekly."""
    cols = GROUPS["index"]
    s = L.xs_rank_signal(L.ret(X.C[cols], 63), frac=0.25)
    return L.rebalance(s.reindex(columns=X.C.columns).fillna(0), 5)


@model("cross_asset_rs")
def cross_asset_rs(X):
    """Asset-class relative strength: average 6-month return per group; long
    every asset in the best group, short every asset in the worst; monthly."""
    m = L.ret(X.C, 126)
    gm = pd.DataFrame({g: m[[c for c in cs if c in m]].mean(axis=1)
                       for g, cs in GROUPS.items()})
    full = gm.notna().all(axis=1)
    best = gm.fillna(-np.inf).idxmax(axis=1).where(full)
    worst = gm.fillna(np.inf).idxmin(axis=1).where(full)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for g, cs in GROUPS.items():
        cs = [c for c in cs if c in out]
        out.loc[best == g, cs] = 1.0
        out.loc[worst == g, cs] = -1.0
    return L.month_end_rebalance(out.where(gm.notna().all(axis=1), 0.0))


@model("cluster_rs")
def cluster_rs(X):
    """Asset-cluster relative strength: each year cluster the CFDs (average
    linkage on 1 - correlation of the prior 2 years) into 6 clusters; long
    the cluster with the best 6-month return, short the worst; monthly."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    m = L.ret(X.C, 126)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for y in sorted(set(X.C.index.year))[2:]:
        tr = X.R[(X.R.index.year >= y - 2) & (X.R.index.year < y)]
        tr = tr.dropna(axis=1, thresh=int(len(tr) * 0.8))
        cr = tr.corr().fillna(0).to_numpy()
        dist = np.clip(1 - cr, 0, 2)
        np.fill_diagonal(dist, 0)
        lab = fcluster(linkage(squareform(dist, checks=False), "average"), 6,
                       "maxclust")
        cols = list(tr.columns)
        mask = X.C.index.year == y
        cm = pd.DataFrame({k: m.loc[mask, [c for c, l in zip(cols, lab)
                                           if l == k]].mean(axis=1)
                           for k in set(lab)})
        b = cm.fillna(-np.inf).idxmax(axis=1)
        w = cm.fillna(np.inf).idxmin(axis=1)
        for c, l in zip(cols, lab):
            out.loc[mask, c] = (b == l).astype(float) - (w == l).astype(float)
    return L.month_end_rebalance(out)


# ============================================================== currencies

CCY_PAIRS = {"EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
             "AUD_USD": ("AUD", "USD"), "USD_CAD": ("USD", "CAD"),
             "EUR_JPY": ("EUR", "JPY"), "AUD_JPY": ("AUD", "JPY")}


def _ccy_strength(X, n=20):
    """Least-squares currency strengths (sum to zero) from the six pairs'
    n-day log changes."""
    ccys = ["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]
    chg = np.log(X.C[list(CCY_PAIRS)]).diff(n)
    A = np.zeros((len(CCY_PAIRS) + 1, len(ccys)))
    for i, (p, (b, q)) in enumerate(CCY_PAIRS.items()):
        A[i, ccys.index(b)] = 1
        A[i, ccys.index(q)] = -1
    A[-1] = 1
    pinv = np.linalg.pinv(A)
    Y = np.c_[chg.to_numpy(), np.zeros(len(chg))]
    S = Y @ pinv.T
    return pd.DataFrame(S, index=chg.index, columns=ccys)


@model("ccy_strength")
def ccy_strength(X):
    """Currency strength meter (20-day, least squares over six pairs): trade
    every pair in the direction base strength - quote strength, scaled by the
    gap in strength-sd units; weekly."""
    S = _ccy_strength(X, 20)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    # typical strength size from the trailing two years only
    sd = np.sqrt((S ** 2).mean(axis=1).rolling(500, min_periods=120).mean())
    for p, (b, q) in CCY_PAIRS.items():
        out[p] = ((S[b] - S[q]) / sd).clip(-1, 1)
    return L.rebalance(out.fillna(0), 5)


# =================================================================== macro

def _usd(X):
    lc = np.log(X.C)
    return (-lc["EUR_USD"].diff() - lc["GBP_USD"].diff() -
            lc["AUD_USD"].diff() + lc["USD_CAD"].diff()) / 4


def _only(X, sigs: dict):
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for k, v in sigs.items():
        out[k] = v
    return out.fillna(0)


@model("dollar_factor")
def dollar_factor(X):
    """Dollar-index factor: synthetic DXY from four USD pairs; every asset
    takes sign(250d beta to the dollar) x sign(dollar 3-month trend)."""
    u = _usd(X)
    beta = beta_to(X.R, u, 250)
    tr = np.sign(u.rolling(63).sum())
    return L.sign(beta).mul(tr, axis=0).fillna(0)


@model("rates_diff")
def rates_diff(X):
    """Rates-differential proxy: EUR/USD follows the 20-day return of the
    Bund CFD minus the Treasury CFD (US yields rising relatively -> USD up);
    GBP/USD likewise with Gilts."""
    r = lambda s: np.log(X.C[s]).diff(20)   # noqa
    return _only(X, {
        "EUR_USD": np.sign(r("USB10Y_USD") - r("DE10YB_EUR")),
        "GBP_USD": np.sign(r("USB10Y_USD") - r("UK10YB_GBP"))})


@model("yield_curve")
def yield_curve(X):
    """Yield-curve signal: 2s10s slope change over 60 days from the bond CFDs
    (dy ~ -return/duration, durations 1.9 and 8.5); hold equity indices long
    while the curve is steepening, flat while it flattens."""
    d10 = -np.log(X.C["USB10Y_USD"]).diff(60) / 8.5
    d2 = -np.log(X.C["USB02Y_USD"]).diff(60) / 1.9
    steep = ((d10 - d2) > 0).astype(float)
    return _only(X, {c: steep for c in GROUPS["index"]})


@model("real_yield_proxy")
def real_yield_proxy(X):
    """Real-yield proxy (no TIPS data): gold follows the 20-day direction of
    10-year Treasury prices (falling yields -> long gold)."""
    return _only(X, {"XAU_USD": np.sign(np.log(X.C["USB10Y_USD"]).diff(20))})


@model("commodity_ccy")
def commodity_ccy(X):
    """Commodity currencies follow their commodity: AUD/USD by gold's 20-day
    trend, USD/CAD against oil's 20-day trend."""
    return _only(X, {
        "AUD_USD": np.sign(np.log(X.C["XAU_USD"]).diff(20)),
        "USD_CAD": -np.sign(np.log(X.C["WTICO_USD"]).diff(20))})


def _roro(X):
    z = lambda s: (np.log(X.C[s]).diff(20) /           # noqa
                   (X.R[s].rolling(250).std() * math.sqrt(20)))
    return (z("SPX500_USD") + z("AUD_JPY") + z("EUR_JPY") -
            z("USB10Y_USD")) / 4


@model("risk_on_off")
def risk_on_off(X):
    """Risk-on/off factor (20d z of S&P, AUDJPY, EURJPY, minus Treasuries):
    each asset takes sign(its 250d beta to the factor) x sign(factor)."""
    f = _roro(X)
    beta = beta_to(X.R, f.diff(), 250)
    return L.sign(beta).mul(np.sign(f), axis=0).fillna(0)


@model("intermarket_confirm")
def intermarket_confirm(X):
    """Intermarket confirmation: each asset's EMA50/200 trend is taken only
    when it agrees with the risk-on/off factor's trend through the asset's
    beta to that factor."""
    f = _roro(X)
    beta = beta_to(X.R, f.diff(), 250)
    tr = L.sign(L.ema(X.C, 50) - L.ema(X.C, 200))
    implied = L.sign(beta).mul(np.sign(f.rolling(50).mean()), axis=0)
    return tr.where(tr.to_numpy() == implied.to_numpy(), 0.0)


@model("macro_factor")
def macro_factor(X):
    """Four-factor macro model (dollar, risk-on/off, rates = Treasury CFD,
    commodities = average commodity CFD): 250-day betas times each factor's
    3-month trend, summed; trade the sign."""
    u = _usd(X)
    ro = _roro(X).diff()
    rt = np.log(X.C["USB10Y_USD"]).diff()
    cm = np.log(X.C[GROUPS["commodity"]]).diff().mean(axis=1)
    tot = 0
    for f in (u, ro, rt, cm):
        beta = beta_to(X.R, f, 250)
        tot = tot + beta.mul(np.sign(f.rolling(63).sum()), axis=0)
    return L.sign(tot)


@model("eq_bond_corr")
def eq_bond_corr(X):
    """Equity-bond correlation regime: 60-day corr(S&P, Treasuries) < 0 ->
    long both (they hedge); otherwise hold only whichever has a positive
    3-month return."""
    c = X.R["SPX500_USD"].rolling(60).corr(X.R["USB10Y_USD"])
    m = lambda s: np.log(X.C[s]).diff(63) > 0      # noqa
    neg = c < 0
    return _only(X, {
        "SPX500_USD": (neg | m("SPX500_USD")).astype(float),
        "USB10Y_USD": (neg | m("USB10Y_USD")).astype(float)})


@model("dollar_gold")
def dollar_gold(X):
    """Dollar-gold relationship: gold's EMA50/200 trend taken only when the
    synthetic dollar's 50-day trend points the other way."""
    u = _usd(X).rolling(50).sum()
    g = np.sign(L.ema(X.C[["XAU_USD"]], 50) - L.ema(X.C[["XAU_USD"]], 200)
                )["XAU_USD"]
    return _only(X, {"XAU_USD": g.where(np.sign(u) == -g, 0.0)})


@model("oil_ccy")
def oil_ccy(X):
    """Oil-currency relationship: USD/CAD's 20-day residual vs WTI (60d
    return beta), fade beyond 2 sd (120d), exit at 0."""
    ra, rb = X.R["USD_CAD"], X.R["WTICO_USD"]
    beta = ra.rolling(60).cov(rb) / rb.rolling(60).var()
    res = (ra - beta.shift(1) * rb).rolling(20).sum()
    z = (res - res.rolling(120).mean()) / res.rolling(120).std()
    return _only(X, {"USD_CAD": L.fade_band(z.to_frame(), 2, 0).iloc[:, 0]})


@model("vrp")
def vrp(X):
    """Volatility risk premium: long the S&P while VIX^2 exceeds 20-day
    realised variance (premium positive); flat when it inverts."""
    rv = X.R["SPX500_USD"].rolling(20).std() * math.sqrt(252) * 100
    return _only(X, {"SPX500_USD": (X.vix > rv).astype(float)})


@model("iv_rv")
def iv_rv(X):
    """Implied-vs-realised: z (250d) of VIX - 20d realised vol; long the S&P
    for 10 days when z > 1.5 (fear priced well above what is happening)."""
    rv = X.R["SPX500_USD"].rolling(20).std() * math.sqrt(252) * 100
    z = L.zscore((X.vix - rv).to_frame(), 250).iloc[:, 0]
    ev = (z > 1.5).astype(float).to_frame("SPX500_USD")
    return _only(X, {"SPX500_USD": L.hold_for(ev, 10)["SPX500_USD"]})


@model("sentiment_extreme_proxy")
def sentiment_extreme_proxy(X):
    """Sentiment extreme reversion, VIX as the fear gauge: VIX in the top 10%
    of its 1-year range -> long S&P and Nasdaq for 20 days."""
    p = L.rolling_pct_rank(X.vix.to_frame(), 250).iloc[:, 0]
    ev = (p > 0.9).astype(float)
    h = L.hold_for(ev.to_frame(), 20).iloc[:, 0]
    return _only(X, {"SPX500_USD": h, "NAS100_USD": h})


@model("sentiment_mom_div_proxy")
def sentiment_mom_div_proxy(X):
    """Sentiment-momentum divergence (VIX proxy): S&P at a 20-day high while
    VIX is also above its 20-day average -> short 5 days; S&P at a 20-day low
    with VIX below average -> long 5 days."""
    s = X.C["SPX500_USD"]
    hi = s >= s.rolling(20).max()
    lo = s <= s.rolling(20).min()
    vup = X.vix > X.vix.rolling(20).mean()
    ev = (lo & ~vup).astype(float) - (hi & vup).astype(float)
    return _only(X, {"SPX500_USD": L.hold_for(ev.to_frame(), 5).iloc[:, 0]})
