"""
Risk, sizing, trade-filter and portfolio-combination models.

None of these has an opinion about direction of its own. They change how
much of a base signal is held, which of its trades are taken, or how
several signals are blended, so each is scored against the thing it
modifies:

  * sizing schemes act on the multi-speed trend model's directions and are
    rescaled to the same 10% realised volatility before scoring (`norm`),
    so a scheme cannot look better just by taking more risk;
  * trade filters (expectancy, profit factor, Kelly, MAE/MFE ...) act on
    20-day Donchian breakout trades with a 2-ATR stop, a trade-level system
    where those ideas are defined;
  * portfolio models blend five base strategies.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bcbt.quant import SPREAD_BP, SLIP_BP, GROUPS, run
from examples.quant import lib as L
from examples.quant.core import model

BASE = "trend_follow"


def _base(X):
    from examples.quant.models_daily import trend_follow
    return X.cached("base_sig", lambda: trend_follow(X))


def _vol(X, span=60):
    return X.R.ewm(span=span, min_periods=20).std() * math.sqrt(252)


def _unit_strategy(X, sig):
    """Per-asset daily P&L of holding sig at 10%-vol size (gross)."""
    w = sig * (0.10 / _vol(X)).clip(upper=3)
    return (w * X.R.shift(-1)).shift(1)        # realised on day t


def _portfolio(X, w):
    """Portfolio daily return of a notional weight frame (gross, for the
    sizing rules' own state such as drawdown)."""
    return (w.shift(1) * X.R).sum(axis=1)


def _level(p):
    """Scale that puts the gross book at 10% annual vol, from the vol seen
    so far (expanding, from 20 days) -- never the full-sample figure."""
    sd = p.expanding(min_periods=20).std().shift(1) * math.sqrt(252)
    k = (0.10 / sd.where(sd > 0)).fillna(1.0)
    return k.clip(upper=100)


def _dd(series):
    eq = series.fillna(0).cumsum()
    return eq.cummax() - eq


# =================================================================== sizing

def _vt_weights(X):
    return _base(X) * (0.10 / _vol(X)).clip(upper=3) / len(X.C.columns)


@model("fixed_fractional", scale=False, agg="sum", base=BASE, norm=True)
def fixed_fractional(X):
    """Fixed-fractional with a fixed 2% stop: every position the same
    notional (risk 1% of equity per trade at a 2% stop), no vol scaling."""
    return _base(X) * 0.5 / len(X.C.columns)


@model("atr_sizing", scale=False, agg="sum", base=BASE, norm=True)
def atr_sizing(X):
    """ATR position sizing: risk 1% per position with a 2 x ATR(14) stop."""
    a = L.atr(X.D, 14) / X.C
    return _base(X) * (0.01 / (2 * a)).clip(upper=3) / len(X.C.columns)


@model("vol_target_sizing", scale=False, agg="sum", base=BASE, norm=True)
def vol_target_sizing(X):
    """Volatility-target sizing: each position at 10% annual vol (EWMA 60)."""
    return _vt_weights(X)


@model("vol_targeting_portfolio", scale=False, agg="sum", base=BASE,
       norm=True)
def vol_targeting_portfolio(X):
    """Portfolio volatility targeting: asset-level vol sizing, then the whole
    book scaled so its trailing 60-day realised vol is 10%."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    k = (0.10 / (p.rolling(60).std() * math.sqrt(252))).clip(upper=4)
    return w.mul(k, axis=0)


def _kelly_frac(X, sig, look=252):
    u = _unit_strategy(X, sig)
    mu, var = u.rolling(look).mean(), u.rolling(look).var()
    return mu / var


@model("kelly_sizing", scale=False, agg="sum", base=BASE, norm=True)
def kelly_sizing(X):
    """Kelly sizing: each asset's 10%-vol position scaled by its trailing
    1-year Kelly fraction mu/sigma^2 of the strategy's own returns (0-3)."""
    f = _kelly_frac(X, _base(X)).clip(0, 3 * 0.10 ** 2 * 252) / (
        0.10 ** 2 * 252)
    return _vt_weights(X) * f


@model("frac_kelly_sizing", scale=False, agg="sum", base=BASE, norm=True)
def frac_kelly_sizing(X):
    """Half-Kelly: as Kelly, halved and floored at a quarter size so a
    negative estimate does not switch the asset off entirely."""
    f = (_kelly_frac(X, _base(X)) / (0.10 ** 2 * 252) / 2).clip(0.25, 1.5)
    return _vt_weights(X) * f


def _erc(cov, iters=200):
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(iters):
        m = cov @ w
        rc = w * m
        w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
        w /= w.sum()
    return w


@model("rp_sizing", scale=False, agg="sum", base=BASE, norm=True)
def rp_sizing(X):
    """Risk-parity (equal risk contribution) sizing across assets, using the
    covariance of the strategy's per-asset returns over the past year;
    rebalanced monthly."""
    u = _unit_strategy(X, _base(X))
    me = L.month_end_rebalance(pd.DataFrame(1.0, index=X.C.index,
                                            columns=["x"])).index
    out = pd.DataFrame(np.nan, index=X.C.index, columns=X.C.columns)
    months = X.C.index.to_period("M")
    last = ~pd.Series(months).duplicated(keep="last").to_numpy()
    for i in np.flatnonzero(last):
        if i < 252:
            continue
        win = u.iloc[i - 251:i + 1].dropna(axis=1, thresh=200).fillna(0)
        if win.shape[1] < 3:
            continue
        w = _erc(win.cov().to_numpy())
        out.iloc[i, [X.C.columns.get_loc(c) for c in win.columns]] = \
            w * win.shape[1]
    k = out.ffill().fillna(0)
    return _vt_weights(X) * k


@model("corr_adj_sizing", scale=False, agg="sum", base=BASE, norm=True)
def corr_adj_sizing(X):
    """Correlation-adjusted sizing: each 10%-vol position divided by
    sqrt(1 + (N-1) x its average 1-year correlation with the other assets'
    strategy returns), monthly."""
    u = _unit_strategy(X, _base(X))
    n = len(X.C.columns)
    adj = pd.DataFrame(np.nan, index=X.C.index, columns=X.C.columns)
    months = X.C.index.to_period("M")
    last = ~pd.Series(months).duplicated(keep="last").to_numpy()
    for i in np.flatnonzero(last):
        if i < 252:
            continue
        c = u.iloc[i - 251:i + 1].corr()
        avg = (c.sum() - 1) / (c.notna().sum() - 1)
        adj.iloc[i] = (1 / np.sqrt(1 + (n - 1) * avg.clip(lower=0))
                       ).reindex(X.C.columns).to_numpy()
    return _vt_weights(X) * adj.ffill().fillna(1.0) * math.sqrt(n) / 2


@model("es_sizing", scale=False, agg="sum", base=BASE, norm=True)
def es_sizing(X):
    """Expected-shortfall sizing: notional inversely proportional to the
    asset's historical 97.5% one-day expected shortfall (1 year)."""
    def es(a):
        a = a[np.isfinite(a)]
        if len(a) < 100:
            return np.nan
        q = np.quantile(a, 0.025)
        return -a[a <= q].mean()
    e = X.R.rolling(252, min_periods=150).apply(es, raw=True)
    k = (0.10 / math.sqrt(252) * 2.3) / e          # ~ 10% vol for a normal
    return _base(X) * k.clip(upper=3) / len(X.C.columns)


@model("dd_adj_sizing", scale=False, agg="sum", base=BASE, norm=True)
def dd_adj_sizing(X):
    """Drawdown-adjusted sizing: book size x (1 - drawdown / 15%), floored at
    a quarter, using the full-size book's own drawdown (at 10% vol)."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    dd = _dd(p * _level(p))                 # drawdown of the 10%-vol book
    return w.mul((1 - dd / 0.15).clip(0.25, 1), axis=0)


@model("dynamic_risk_sizing", scale=False, agg="sum", base=BASE, norm=True)
def dynamic_risk_sizing(X):
    """Dynamic risk sizing: 1.5x after a positive trailing 60-day strategy
    Sharpe, 0.5x after a negative one."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    sr = p.rolling(60).mean() / p.rolling(60).std()
    return w.mul(np.where(sr > 0, 1.5, 0.5), axis=0)


@model("max_dd_aware", scale=False, agg="sum", base=BASE, norm=True)
def max_dd_aware(X):
    """Maximum-drawdown-aware: size shrinks linearly to zero as the book's
    drawdown within the calendar year approaches a 10% limit (each year a
    fresh account, as a funded account resets after a breach)."""
    w = _vt_weights(X)
    p = _portfolio(X, w) * _level(_portfolio(X, w))
    yr = p.index.year
    eq = p.fillna(0).groupby(yr).cumsum()
    dd = eq.groupby(yr).cummax() - eq
    return w.mul((1 - dd / 0.10).clip(0, 1), axis=0)


@model("capital_preservation", scale=False, agg="sum", base=BASE, norm=True)
def capital_preservation(X):
    """Capital preservation: half size beyond a 5% drawdown, flat beyond 8%,
    back to full only once the shadow strategy regains half the loss."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    k = _level(p)
    eq = (p * k).fillna(0).cumsum()
    peak = eq.cummax()
    dd = (peak - eq).to_numpy()
    scale = np.ones(len(dd))
    state = 1.0
    trough = 0.0
    for i in range(len(dd)):
        if dd[i] > 0.08:
            state, trough = 0.0, max(trough, dd[i])
        elif dd[i] > 0.05 and state == 1.0:
            state = 0.5
        elif state < 1.0 and dd[i] < trough / 2:
            state, trough = 1.0, 0.0
        if dd[i] == 0:
            state, trough = 1.0, 0.0
        scale[i] = state
    return w.mul(scale, axis=0)


@model("daily_loss_aware", scale=False, agg="sum", base=BASE, norm=True)
def daily_loss_aware(X):
    """Daily-loss-limit-aware: after a day losing more than 1.5% (book at
    10% vol), stand aside the next day."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    k = _level(p)
    bad = (p * k) < -0.015
    return w.mul((~bad).astype(float), axis=0)


@model("volscaled_prop", scale=False, agg="sum", base=BASE, norm=True)
def volscaled_prop(X):
    """Volatility-scaled prop-firm model: the book is scaled so its 99%
    one-day VaR (2.33 x trailing 20-day vol) is 1% -- half a typical 2%
    daily-loss allowance."""
    w = _vt_weights(X)
    p = _portfolio(X, w)
    k = 0.01 / (2.33 * p.rolling(20).std())
    return w.mul(k.clip(upper=50), axis=0)


@model("prop_firm_aware", scale=False, agg="sum", base=BASE, norm=True)
def prop_firm_aware(X):
    """Prop-firm constraint-aware: VaR-scaled book (as above), size cut
    linearly with drawdown toward a 10% limit, flat the day after a 2% loss."""
    w = volscaled_prop(X)
    p = _portfolio(X, w)
    dd = _dd(p)
    k = (1 - dd / 0.10).clip(0, 1) * (p > -0.02).astype(float)
    return w.mul(k, axis=0)


@model("kelly_opt_entry", scale=False, agg="sum", base=None, norm=True)
def kelly_opt_entry(X):
    """Kelly-optimised entry: hold each asset at its own full-Kelly weight
    mu/sigma^2 from its trailing 1-year return (direction and size)."""
    mu = X.R.rolling(252).mean()
    var = X.R.rolling(252).var()
    return (mu / var).clip(-20, 20) / len(X.C.columns)


@model("utility_max", scale=False, agg="sum", base=None, norm=True)
def utility_max(X):
    """Utility-maximising entry: mean-variance weight mu/(gamma sigma^2) with
    gamma 5 and the 12-month mean shrunk by half; only re-traded when the
    target moves by more than 25%."""
    mu = 0.5 * X.R.rolling(252).mean()
    var = X.R.rolling(60).var()
    return (mu / (5 * var)).clip(-4, 4) / len(X.C.columns)


@model("vol_targeting", scale=False, agg="sum", base=None, norm=True)
def vol_targeting(X):
    """Volatility targeting as a strategy: hold every CFD long at 10% vol
    (the risk premium, vol-timed), compared with the same basket held at
    fixed notional ("long_basket_fixed")."""
    return (0.10 / _vol(X)).clip(upper=3) / len(X.C.columns)


@model("long_basket_fixed", scale=False, agg="sum", base=None, norm=True)
def long_basket_fixed(X):
    """Reference: every CFD held long at the same fixed notional."""
    return pd.DataFrame(1.0 / len(X.C.columns), index=X.C.index,
                        columns=X.C.columns)


# ===================================================== signal-level filters

@model("drawdown_regime", base=BASE)
def drawdown_regime(X):
    """Drawdown-regime filter: an asset's trend position is switched off
    while that asset's own strategy is more than 2 vol-units under water,
    and back on when it recovers to half that."""
    u = _unit_strategy(X, _base(X)).fillna(0)
    eq = u.cumsum()
    dd = (eq.cummax() - eq) / 0.10
    on = pd.DataFrame(np.nan, index=X.C.index, columns=X.C.columns)
    on[dd > 0.2] = 0.0
    on[dd < 0.1] = 1.0
    return _base(X) * on.ffill().fillna(1.0)


@model("drawdown_aware_signal", base=BASE)
def drawdown_aware_signal(X):
    """Drawdown-aware signal: each asset's trend weight scaled by
    1 - its strategy drawdown / 3 vol-units (floor 0)."""
    u = _unit_strategy(X, _base(X)).fillna(0)
    eq = u.cumsum()
    dd = (eq.cummax() - eq) / 0.10
    return _base(X) * (1 - dd / 0.3).clip(0, 1)


@model("kelly_filter", base=BASE)
def kelly_filter(X):
    """Kelly-criterion filter: take an asset's trend signal only while its
    trailing 1-year Kelly fraction is positive."""
    return _base(X).where(_kelly_frac(X, _base(X)) > 0, 0.0)


@model("frac_kelly_filter", base=BASE)
def frac_kelly_filter(X):
    """Fractional-Kelly filter: only while half-Kelly would justify at least
    the 10%-vol size the engine takes."""
    f = _kelly_frac(X, _base(X)) / (0.10 ** 2 * 252)
    return _base(X).where(f / 2 >= 1, 0.0)


def _std_tail(X, fn):
    z = X.R / X.R.rolling(60).std().shift(1)
    return z.rolling(252, min_periods=150).apply(fn, raw=True)


@model("es_filter", base=BASE)
def es_filter(X):
    """Expected-shortfall filter: skip an asset whose 97.5% ES of
    vol-standardised returns (1 year) is above 1.25x the cross-sectional
    median -- fat tails relative to its own volatility."""
    def es(a):
        a = a[np.isfinite(a)]
        q = np.quantile(a, 0.025)
        return -a[a <= q].mean()
    e = X.cached("es_std", lambda: _std_tail(X, es))
    ok = e.le(1.25 * e.median(axis=1), axis=0) | e.isna()
    return _base(X).where(ok, 0.0)


@model("var_filter", base=BASE)
def var_filter(X):
    """VaR filter: the same with the 99% VaR of vol-standardised returns."""
    v = X.cached("var_std", lambda: _std_tail(
        X, lambda a: -np.quantile(a[np.isfinite(a)], 0.01)))
    ok = v.le(1.25 * v.median(axis=1), axis=0) | v.isna()
    return _base(X).where(ok, 0.0)


@model("inverse_vol_selection", base=BASE)
def inverse_vol_selection(X):
    """Inverse-volatility position selection: hold the trend signal only in
    the 12 CFDs with the lowest 60-day volatility."""
    rk = L.rstd(X.R, 60).rank(axis=1)
    return _base(X).where(rk <= 12, 0.0)


# ====================================================== trade-level system

def _trades(X, stop_atr=2.0, mae_q=None, mfe_q=None, filt=None):
    """
    20-day Donchian breakout trades, entry at the signal close, 2 x ATR stop
    (checked on daily lows/highs, filled at the stop), exit on the 10-day
    opposite channel. Optional: MAE stop / MFE target from earlier trades'
    quantiles (in ATR units), and a filter(history, context) -> bool that
    decides whether to take each new trade.
    Returns (positions frame, trade list per asset).
    """
    H, Lo, C = X.D["H"].to_numpy(), X.D["L"].to_numpy(), X.C.to_numpy()
    a = L.atr(X.D, 14).to_numpy()
    hi = X.D["H"].rolling(20).max().shift(1).to_numpy()
    lo = X.D["L"].rolling(20).min().shift(1).to_numpy()
    hx = X.D["H"].rolling(10).max().shift(1).to_numpy()
    lx = X.D["L"].rolling(10).min().shift(1).to_numpy()
    adx = L.adx(X.D)[0].to_numpy()
    volp = L.rolling_pct_rank(L.rstd(X.R, 20), 250).to_numpy()
    n, m = C.shape
    pos = np.zeros((n, m))
    exitpx = np.full((n, m), np.nan)
    trades = {}
    for j in range(m):
        hist = []
        side, entry, stop, risk, mae, mfe, ctx = 0, 0, 0, 0, 0, 0, None
        for i in range(1, n):
            c = C[i, j]
            if not np.isfinite(c) or not np.isfinite(a[i, j]):
                continue
            if side != 0:
                adv = (entry - Lo[i, j]) if side > 0 else (H[i, j] - entry)
                fav = (H[i, j] - entry) if side > 0 else (entry - Lo[i, j])
                mae, mfe = max(mae, adv / risk * stop_atr), max(
                    mfe, fav / risk * stop_atr)
                lim_stop = stop
                if mae_q is not None:
                    q = _q([t["mae"] for t in hist if t["r"] > 0], mae_q)
                    if q is not None:
                        alt = entry - side * q * risk / stop_atr
                        lim_stop = max(stop, alt) if side > 0 else min(stop,
                                                                       alt)
                hit = (Lo[i, j] <= lim_stop) if side > 0 else (H[i, j] >=
                                                               lim_stop)
                tgt = None
                if mfe_q is not None:
                    q = _q([t["mfe"] for t in hist], mfe_q)
                    if q is not None:
                        tgt = entry + side * q * risk / stop_atr
                tgt_hit = tgt is not None and ((H[i, j] >= tgt) if side > 0
                                               else (Lo[i, j] <= tgt))
                ch_exit = (c < lx[i, j]) if side > 0 else (c > hx[i, j])
                if hit or tgt_hit or ch_exit:
                    px = lim_stop if hit else (tgt if tgt_hit else c)
                    r = side * (px - entry) / risk
                    hist.append(dict(r=r, mae=mae, mfe=mfe, ctx=ctx, i=i))
                    exitpx[i, j] = px
                    side = 0
                    pos[i, j] = 0
                    continue
                pos[i, j] = side
                continue
            s = 1 if c > hi[i, j] else (-1 if c < lo[i, j] else 0)
            if s == 0:
                continue
            ctx = dict(adx=adx[i, j] > 25, volhi=volp[i, j] > 0.5)
            if filt is not None and not filt(hist, ctx):
                continue
            side, entry = s, c
            risk = stop_atr * a[i, j]
            stop = entry - side * risk
            mae = mfe = 0.0
            pos[i, j] = side
        trades[X.C.columns[j]] = hist
    return (pd.DataFrame(pos, index=X.C.index, columns=X.C.columns),
            pd.DataFrame(exitpx, index=X.C.index, columns=X.C.columns),
            trades)


def _q(xs, q):
    return float(np.quantile(xs, q)) if len(xs) >= 20 else None


def _trade_sig(X, **kw):
    """Positions from the trade loop, as a signal frame. Stop and target
    fills inside the bar are approximated by the engine's close-to-close
    accounting with the exit on that bar (the exit price is recorded for the
    R statistics but the engine charges the close)."""
    pos, _, _ = _trades(X, **kw)
    return pos


@model("trade_base")
def trade_base(X):
    """Reference trade system for the filters: 20-day Donchian breakout,
    2-ATR stop, 10-day opposite-channel exit."""
    return X.cached("trade_base", lambda: _trade_sig(X))


def _f_expectancy(n, thr=0.0):
    def f(hist, ctx):
        if len(hist) < n:
            return True
        return np.mean([t["r"] for t in hist[-n:]]) > thr
    return f


@model("expectancy_filter", base="trade_base")
def expectancy_filter(X):
    """Expectancy filter: take a new breakout only while the asset's last 30
    trades have positive mean R."""
    return _trade_sig(X, filt=_f_expectancy(30))


@model("pf_filter", base="trade_base")
def pf_filter(X):
    """Profit-factor filter: only while the last 30 trades' profit factor is
    above 1.2."""
    def f(hist, ctx):
        if len(hist) < 30:
            return True
        r = np.array([t["r"] for t in hist[-30:]])
        loss = -r[r < 0].sum()
        return loss == 0 or r[r > 0].sum() / loss > 1.2
    return _trade_sig(X, filt=f)


@model("r_multiple_dist", base="trade_base")
def r_multiple_dist(X):
    """R-multiple distribution model: take trades only while the last 50
    R-multiples have a positive mean AND a 25th percentile above -1 (the
    stop is doing its job, no gap-through losses)."""
    def f(hist, ctx):
        if len(hist) < 50:
            return True
        r = np.array([t["r"] for t in hist[-50:]])
        return r.mean() > 0 and np.quantile(r, 0.25) > -1.0
    return _trade_sig(X, filt=f)


@model("prob_payoff", base="trade_base")
def prob_payoff(X):
    """Probability x payoff: p(win) x average win - p(loss) x average loss
    over all earlier trades of the asset must be positive."""
    def f(hist, ctx):
        if len(hist) < 20:
            return True
        r = np.array([t["r"] for t in hist])
        w, l = r[r > 0], r[r <= 0]
        if not len(w) or not len(l):
            return True
        p = len(w) / len(r)
        return p * w.mean() + (1 - p) * l.mean() > 0
    return _trade_sig(X, filt=f)


@model("cond_expectancy", base="trade_base")
def cond_expectancy(X):
    """Conditional expectancy: expectancy of earlier trades taken in the same
    volatility regime (above/below median) must be positive."""
    def f(hist, ctx):
        same = [t["r"] for t in hist if t["ctx"]["volhi"] == ctx["volhi"]]
        return len(same) < 20 or np.mean(same) > 0
    return _trade_sig(X, filt=f)


@model("cond_r_multiple", base="trade_base")
def cond_r_multiple(X):
    """Conditional R-multiple: mean R of earlier trades in the same ADX
    regime (> 25 or not) must be positive."""
    def f(hist, ctx):
        same = [t["r"] for t in hist if t["ctx"]["adx"] == ctx["adx"]]
        return len(same) < 20 or np.mean(same) > 0
    return _trade_sig(X, filt=f)


@model("mae_model", base="trade_base")
def mae_model(X):
    """Maximum-adverse-excursion model: the stop is tightened to the 80th
    percentile MAE of the asset's earlier winning trades."""
    return _trade_sig(X, mae_q=0.8)


@model("mfe_model", base="trade_base")
def mfe_model(X):
    """Maximum-favourable-excursion model: take profit at the median MFE of
    the asset's earlier trades."""
    return _trade_sig(X, mfe_q=0.5)


@model("risk_of_ruin", base="trade_base")
def risk_of_ruin(X):
    """Risk-of-ruin filter: from the last 50 trades' win rate and payoff,
    the probability of a 20R drawdown (1% risk per trade, 20% ruin) must be
    under 5%; otherwise skip new trades."""
    def f(hist, ctx):
        if len(hist) < 50:
            return True
        r = np.array([t["r"] for t in hist[-50:]])
        w, l = r[r > 0], -r[r <= 0]
        if not len(w) or not len(l):
            return True
        p, b = len(w) / len(r), w.mean() / max(l.mean(), 1e-9)
        edge = p * b - (1 - p)
        if edge <= 0:
            return False
        # Balsara-style approximation: ((1 - e)/(1 + e))^units
        e = edge / b if b > 0 else 0
        ror = ((1 - e) / (1 + e)) ** 20 if e < 1 else 0.0
        return ror < 0.05
    return _trade_sig(X, filt=f)


@model("ev_optimization")
def ev_optimization(X):
    """Expected-value optimisation: per asset, long / short / flat by which
    has the higher expanding-window expected next-day return (conditional on
    trend state x z20 bucket) after the round-trip cost."""
    tr = (X.C > L.sma(X.C, 100)).astype(int)
    zb = pd.DataFrame(np.digitize(L.zscore(X.C, 20).fillna(0), [-1, 0, 1]),
                      index=X.C.index, columns=X.C.columns)
    st = (tr * 4 + zb).where(X.C.notna())
    y = X.R.shift(-1)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for c in X.C.columns:
        cost = (SPREAD_BP.get(c, 5) + 2 * SLIP_BP) / 1e4
        s, yy = st[c].to_numpy(), y[c].to_numpy()
        sm, cnt = np.zeros(8), np.zeros(8)
        for i in range(len(s)):
            if i >= 1 and np.isfinite(yy[i - 1]) and s[i - 1] == s[i - 1]:
                k = int(s[i - 1])
                sm[k] += yy[i - 1]
                cnt[k] += 1
            if s[i] == s[i]:
                k = int(s[i])
                if cnt[k] >= 100:
                    mu = sm[k] / cnt[k]
                    out.iat[i, out.columns.get_loc(c)] = (
                        1.0 if mu > cost else -1.0 if mu < -cost else 0.0)
    return out


# ================================================= portfolio combinations

def _strategies(X):
    from examples.quant import models_daily as D, models_xasset as XA
    def f():
        return {"trend": D.trend_follow(X), "mr": D.z_mr(X),
                "xsmom": XA.cs_mom(X), "breakout": D.breakout_plain(X),
                "macd": D.macd(X)}
    return X.cached("strategies5", f)


def _strategy_returns(X):
    def f():
        return pd.DataFrame({k: run(s, X.C).net for k, s in
                             _strategies(X).items()})
    return X.cached("strategies5_ret", f)


def _blend(X, W):
    """Combine the five strategies' signals with per-day weights W (date x
    strategy, summing to the number of strategies' share)."""
    S = _strategies(X)
    out = 0
    for k, s in S.items():
        out = out + s.mul(W[k], axis=0)
    return out


def _monthly(W):
    return L.month_end_rebalance(W.fillna(0)).shift(1).fillna(0)


@model("portfolio_signal")
def portfolio_signal(X):
    """Portfolio-level signal model: equal blend of trend, z-score mean
    reversion, cross-sectional momentum, breakout and MACD."""
    R = _strategy_returns(X)
    return _blend(X, pd.DataFrame(0.2, index=R.index, columns=R.columns))


@model("risk_parity_alloc")
def risk_parity_alloc(X):
    """Risk-parity signal allocation: the five strategies weighted for equal
    risk contribution (1-year covariance), monthly."""
    R = _strategy_returns(X)
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for i in range(252, len(R), 21):
        W.iloc[i] = _erc(R.iloc[i - 252:i].cov().to_numpy() + 1e-12 *
                         np.eye(R.shape[1]))
    return _blend(X, W.ffill().fillna(0.2))


def _trailing(R, fn, look=252):
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for i in range(look, len(R), 21):
        W.iloc[i] = fn(R.iloc[i - look:i])
    W = W.ffill().fillna(1 / R.shape[1])
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


@model("risk_adj_selection")
def risk_adj_selection(X):
    """Risk-adjusted signal selection: each month hold only the strategy with
    the best trailing 1-year Sharpe."""
    R = _strategy_returns(X)
    return _blend(X, _trailing(R, lambda w: (w.mean() / w.std() ==
                                             (w.mean() / w.std()).max()
                                             ).astype(float)))


@model("corr_adj_selection")
def corr_adj_selection(X):
    """Correlation-adjusted selection: strategy weights proportional to
    1 / (1 + average correlation with the others), trailing year."""
    R = _strategy_returns(X)

    def f(w):
        c = w.corr()
        avg = (c.sum() - 1) / (len(c) - 1)
        return 1 / (1 + avg.clip(lower=0))
    return _blend(X, _trailing(R, f))


@model("voladj_selection")
def voladj_selection(X):
    """Volatility-adjusted selection: strategy weights proportional to
    1 / trailing-year volatility of each strategy's returns."""
    R = _strategy_returns(X)
    return _blend(X, _trailing(R, lambda w: 1 / w.std()))


@model("dynamic_weighting")
def dynamic_weighting(X):
    """Dynamic signal weighting: weights proportional to the positive part of
    each strategy's trailing 1-year Sharpe."""
    R = _strategy_returns(X)
    return _blend(X, _trailing(R, lambda w: (w.mean() / w.std()).clip(
        lower=0)))


@model("regime_weighting")
def regime_weighting(X):
    """Regime-dependent weighting: when the average asset's 20-day vol
    percentile is above 0.5, 60% mean reversion / 10% each other; otherwise
    60% trend / 10% each other."""
    p = L.rolling_pct_rank(L.rstd(X.R, 20), 250).mean(axis=1).shift(1)
    R = _strategy_returns(X)
    W = pd.DataFrame(0.1, index=R.index, columns=R.columns)
    W.loc[p > 0.5, "mr"] = 0.6
    W.loc[p <= 0.5, "trend"] = 0.6
    return _blend(X, W)


@model("bma")
def bma(X):
    """Bayesian model averaging: weights proportional to exp(t/2) of each
    strategy's trailing 2-year mean return (a BIC-style posterior)."""
    R = _strategy_returns(X)

    def f(w):
        t = w.mean() / (w.std() / math.sqrt(len(w)))
        return np.exp((t / 2).clip(-20, 20))
    return _blend(X, _trailing(R, f, look=504))


@model("ensemble_tech")
def ensemble_tech(X):
    """Ensemble technical model: mean of 10 technical signals (MACD, RSI2,
    stochastic, Bollinger, Donchian, Keltner, multi-MA, ADX trend, KAMA,
    Hull)."""
    from examples.quant import models_daily as D
    sigs = [D.macd(X), D.rsi_quant(X), D.stoch_quant(X), D.boll_z(X),
            D.donchian(X), D.keltner(X), D.multi_ma(X), D.adx_filter(X),
            D.kama_model(X), D.hma_model(X)]
    return sum(sigs) / len(sigs)


@model("ensemble_macro")
def ensemble_macro(X):
    """Ensemble macro model: mean of dollar factor, risk-on/off, four-factor
    macro, commodity currencies, rates differential and yield curve."""
    from examples.quant import models_xasset as XA
    sigs = [XA.dollar_factor(X), XA.risk_on_off(X), XA.macro_factor(X),
            XA.commodity_ccy(X), XA.rates_diff(X), XA.yield_curve(X)]
    return sum(sigs) / len(sigs)


@model("ensemble_xasset")
def ensemble_xasset(X):
    """Ensemble cross-asset model: mean of cross-sectional momentum, the
    momentum factor, PCA stat arb, asset-class relative strength and the
    currency-strength model."""
    from examples.quant import models_xasset as XA
    sigs = [XA.cs_mom(X), XA.momentum_factor(X), XA.stat_arb(X),
            XA.cross_asset_rs(X), XA.ccy_strength(X)]
    return sum(sigs) / len(sigs)


@model("multi_asset_ensemble")
def multi_asset_ensemble(X):
    """Multi-asset signal ensemble: trend, cross-sectional momentum, the
    macro factor model and risk-on/off, equally blended."""
    from examples.quant import models_daily as D, models_xasset as XA
    return (D.trend_follow(X) + XA.cs_mom(X) + XA.macro_factor(X) +
            XA.risk_on_off(X)) / 4
