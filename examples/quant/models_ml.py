"""
Machine-learning and walk-forward models.

Every learner is trained walk-forward: refit at the start of each year on
all earlier data (or a rolling window where the model says so), predict
that year, never look at it again. The panel pools all 25 CFDs so each fit
sees ~40,000+ rows. Labels that span several days are cut off before the
training end so no training label overlaps the test year (purging).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from bcbt.quant import SPREAD_BP, SLIP_BP
from examples.quant import lib as L
from examples.quant.core import model

warnings.filterwarnings("ignore")
FIRST_TEST_YEAR = 2008


# ================================================================ features

def features(X):
    """Long panel (date, sym) of standardised features, known at the close."""
    def f():
        C, R = X.C, X.R
        lp = np.log(C)
        F = {}
        vol20 = L.rstd(R, 20)
        for n in (1, 2, 3, 5, 10, 20, 60, 120, 252):
            F[f"r{n}"] = (lp - lp.shift(n)) / (vol20 * math.sqrt(n))
        F["vol5"] = L.rstd(R, 5) / L.rstd(R, 250)
        F["vol20"] = vol20 / L.rstd(R, 250)
        F["vol60"] = L.rstd(R, 60) / L.rstd(R, 250)
        F["z20"] = L.zscore(C, 20)
        F["z50"] = L.zscore(C, 50)
        F["rsi14"] = (L.rsi(C, 14) - 50) / 10
        F["rsi2"] = (L.rsi(C, 2) - 50) / 25
        m = L.ema(C, 12) - L.ema(C, 26)
        F["macdh"] = (m - L.ema(m, 9)) / (vol20 * C)
        F["atr"] = L.atr(X.D, 14) / C / vol20
        F["dhi"] = (C / X.D["H"].rolling(20).max() - 1) / vol20
        F["dlo"] = (C / X.D["L"].rolling(20).min() - 1) / vol20
        F["er"] = L.efficiency_ratio(C, 20)
        F["skew"] = R.rolling(60).skew()
        F["volz"] = L.zscore(X.D["V"], 60)
        F["clv"] = ((C - X.D["L"]) - (X.D["H"] - C)) / (X.D["H"] - X.D["L"]
                                                        ).replace(0, np.nan)
        F["ffd"] = L.zscore(ffd(lp, 0.4), 250)
        spx = R["SPX500_USD"]
        usd = (-R["EUR_USD"] - R["GBP_USD"] - R["AUD_USD"] + R["USD_CAD"]) / 4
        for k, s in (("spx1", spx), ("spx5", spx.rolling(5).sum()),
                     ("usd5", usd.rolling(5).sum())):
            F[k] = pd.DataFrame({c: s / s.rolling(250).std() for c in C})
        dow = pd.DataFrame({c: C.index.dayofweek for c in C}, index=C.index)
        F["dow"] = (dow - 2) / 2
        P = pd.concat({k: v.stack(future_stack=True) for k, v in F.items()},
                      axis=1)
        P.index.names = ["date", "sym"]
        y1 = R.shift(-1).stack(future_stack=True)
        y5 = (lp.shift(-5) - lp).stack(future_stack=True)
        P["y1"], P["y5"] = y1, y5
        P = P.replace([np.inf, -np.inf], np.nan)
        feats = [c for c in P.columns if not c.startswith("y")]
        P[feats] = P[feats].clip(-6, 6)
        return P.dropna(subset=feats)
    return X.cached("ml_features", f)


FEATS = None


def feat_cols(P):
    return [c for c in P.columns if not c.startswith("y")]


def ffd(lp, d, thresh=1e-4, max_w=500):
    """Fixed-width-window fractional differentiation (Lopez de Prado)."""
    w = [1.0]
    k = 1
    while k < max_w:
        nw = -w[-1] * (d - k + 1) / k
        if abs(nw) < thresh:
            break
        w.append(nw)
        k += 1
    w = np.array(w[::-1])
    n = len(w)
    return lp.rolling(n, min_periods=n).apply(lambda a: a @ w, raw=True)


def walk_forward(X, make, target="y1", kind="clf", horizon=1, train_years=None,
                 rows=None, out="proba", key=None):
    """
    Fit `make()` each year on earlier rows, predict that year. Returns a wide
    frame (date x sym) of P(up) for classifiers or the forecast for
    regressors.
    """
    def run():
        P = features(X) if rows is None else rows
        P = P.dropna(subset=[target])
        fc = feat_cols(features(X))
        dates = P.index.get_level_values("date")
        res = []
        for y in range(FIRST_TEST_YEAR, 2021):
            start = pd.Timestamp(f"{y}-01-01")
            end = pd.Timestamp(f"{y + 1}-01-01")
            # purge: training labels must end before the test year begins
            cut = start - pd.tseries.offsets.BDay(horizon)
            tr = (dates < cut)
            if train_years:
                tr &= dates >= start - pd.DateOffset(years=train_years)
            te = (dates >= start) & (dates < end)
            if tr.sum() < 5000 or te.sum() == 0:
                continue
            Xtr, ytr = P.loc[tr, fc].to_numpy(), P.loc[tr, target].to_numpy()
            Xte = P.loc[te, fc].to_numpy()
            m = make()
            if kind == "clf":
                m.fit(Xtr, (ytr > 0).astype(int))
                pr = m.predict_proba(Xte)[:, 1] if out == "proba" else \
                    m.decision_function(Xte)
            else:
                m.fit(Xtr, ytr)
                pr = m.predict(Xte)
            res.append(pd.Series(pr, index=P.index[te]))
        s = pd.concat(res)
        return s.unstack("sym").reindex(index=X.C.index, columns=X.C.columns)
    return X.cached(key, run) if key else run()


def p_to_sig(p, band=0.0):
    return (p > 0.5 + band).astype(float) - (p < 0.5 - band).astype(float)


def _scaled(model_):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), model_)


# ===================================================== classifiers (1-day)

def _logit():
    from sklearn.linear_model import LogisticRegression
    return _scaled(LogisticRegression(C=0.05, max_iter=500))


def _lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return _scaled(LinearDiscriminantAnalysis())


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=200, n_jobs=4,
                                  random_state=0)


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=3, min_samples_leaf=200,
                                          random_state=0)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=50, n_jobs=4, random_state=0,
                         verbosity=0)


class _Sub:
    """Fit on a random subsample (for learners that scale badly)."""

    def __init__(self, m, n=15000, seed=0):
        self.m, self.n, self.seed = m, n, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        i = rng.choice(len(X), size=min(self.n, len(X)), replace=False)
        self.m.fit(X[i], y[i])
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)

    def decision_function(self, X):
        return self.m.decision_function(X)


def _svm():
    from sklearn.svm import SVC
    return _Sub(_scaled(SVC(C=0.5, kernel="rbf", gamma="scale")), 12000)


def _knn():
    from sklearn.neighbors import KNeighborsClassifier
    return _Sub(_scaled(KNeighborsClassifier(n_neighbors=250)), 60000)


def _mlp():
    from sklearn.neural_network import MLPClassifier
    return _scaled(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                 early_stopping=True, max_iter=200,
                                 random_state=0))


def _p(X, name, make, **kw):
    return walk_forward(X, make, key=f"p_{name}", **kw)


@model("ml_logit")
def ml_logit(X):
    """Logistic regression (L2, C=0.05) on 30 features; next-day direction."""
    return p_to_sig(_p(X, "logit", _logit))


@model("ml_lda")
def ml_lda(X):
    """Linear discriminant analysis on the same features."""
    return p_to_sig(_p(X, "lda", _lda))


@model("ml_rf")
def ml_rf(X):
    """Random forest (200 trees, depth 6, leaf 200)."""
    return p_to_sig(_p(X, "rf", _rf))


@model("ml_gbm")
def ml_gbm(X):
    """Histogram gradient boosting (200 rounds, depth 3)."""
    return p_to_sig(_p(X, "gbm", _gbm))


@model("ml_xgb")
def ml_xgb(X):
    """XGBoost (300 rounds, depth 3, subsampled)."""
    return p_to_sig(_p(X, "xgb", _xgb))


@model("ml_svm")
def ml_svm(X):
    """RBF support-vector machine on a 12,000-row subsample per fit; sign of
    the decision function."""
    d = walk_forward(X, _svm, out="decision", key="d_svm")
    return L.sign(d)


@model("ml_knn")
def ml_knn(X):
    """k-nearest neighbours (k=250) on standardised features."""
    return p_to_sig(_p(X, "knn", _knn))


@model("ml_mlp")
def ml_mlp(X):
    """Feed-forward neural network (32-16, early stopping)."""
    return p_to_sig(_p(X, "mlp", _mlp))


@model("ml_ensemble")
def ml_ensemble(X):
    """Average probability of logistic, random forest and boosting."""
    p = (_p(X, "logit", _logit) + _p(X, "rf", _rf) + _p(X, "gbm", _gbm)) / 3
    return p_to_sig(p)


@model("ml_voting")
def ml_voting(X):
    """Majority vote of logistic, RF, boosting, kNN and SVM."""
    votes = [p_to_sig(_p(X, "logit", _logit)), p_to_sig(_p(X, "rf", _rf)),
             p_to_sig(_p(X, "gbm", _gbm)), p_to_sig(_p(X, "knn", _knn)),
             L.sign(walk_forward(X, _svm, out="decision", key="d_svm"))]
    return L.sign(sum(votes))


@model("ml_stacked")
def ml_stacked(X):
    """Stacked ensemble: a logistic meta-model, refit yearly, on the base
    learners' out-of-sample probabilities from earlier years."""
    base = {k: _p(X, k, f) for k, f in (("logit", _logit), ("rf", _rf),
                                        ("gbm", _gbm), ("knn", _knn))}
    from sklearn.linear_model import LogisticRegression
    y = (X.R.shift(-1) > 0).astype(float).where(X.R.shift(-1).notna())
    stack = pd.concat({k: v.stack() for k, v in base.items()}, axis=1)
    yy = y.stack().reindex(stack.index)
    out = pd.Series(np.nan, index=stack.index)
    dates = stack.index.get_level_values(0)
    for yr in range(FIRST_TEST_YEAR + 2, 2021):
        st = pd.Timestamp(f"{yr}-01-01")
        tr = (dates < st - pd.tseries.offsets.BDay(1)) & yy.notna().to_numpy()
        te = (dates >= st) & (dates < pd.Timestamp(f"{yr + 1}-01-01"))
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        m = LogisticRegression(C=1.0).fit(stack[tr].to_numpy(), yy[tr])
        out[te] = m.predict_proba(stack[te].to_numpy())[:, 1]
    return p_to_sig(out.unstack().reindex(index=X.C.index, columns=X.C.columns))


@model("ml_prob_threshold")
def ml_prob_threshold(X):
    """Probability-threshold entry: trade only when the logistic + boosting
    average probability is beyond 0.55 / 0.45."""
    p = (_p(X, "logit", _logit) + _p(X, "gbm", _gbm)) / 2
    return p_to_sig(p, 0.05)


@model("ml_ev_threshold")
def ml_ev_threshold(X):
    """Expected-value threshold: boosted regression of next-day return; trade
    only when |forecast| exceeds the round-trip cost."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = walk_forward(X, lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
        random_state=0), target="y1", kind="reg", key="r_gbm1")
    cost = pd.Series({c: (SPREAD_BP.get(c, 5) + 2 * SLIP_BP) / 1e4
                      for c in X.C.columns})
    return np.sign(f).where(f.abs().gt(cost, axis=1), 0.0).fillna(0)


@model("ml_feature_return")
def ml_feature_return(X):
    """Feature-engineered return model: boosted regression of the next-day
    return; trade its sign."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = walk_forward(X, lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
        random_state=0), target="y1", kind="reg", key="r_gbm1")
    return L.sign(f)


# =========================================== 5-day and event-based learners

def _hold5(sig):
    return L.rebalance(sig, 5)


@model("ml_prob_return")
def ml_prob_return(X):
    """ML probability of return: boosting classifier of the 5-day direction,
    rebalanced weekly."""
    p = walk_forward(X, _gbm, target="y5", horizon=5, key="p_gbm5")
    return _hold5(p_to_sig(p, 0.02))


@model("ml_expected_return")
def ml_expected_return(X):
    """ML expected return: boosted regression of the 5-day return, weekly."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = walk_forward(X, lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
        random_state=0), target="y5", kind="reg", horizon=5, key="r_gbm5")
    return _hold5(L.sign(f))


def _event_rows(X, ev, hold):
    """Rows of the feature panel on event days, with the label = the
    event-direction return over `hold` days > 0, plus the side as a feature."""
    P = features(X)
    lp = np.log(X.C)
    fwd = (lp.shift(-hold) - lp)
    side = ev.stack()
    side = side[side != 0]
    rows = P.reindex(side.index).dropna(subset=feat_cols(P))
    rows = rows.copy()
    rows["side"] = side.reindex(rows.index)
    rows["yev"] = (fwd.stack().reindex(rows.index) * rows["side"])
    return rows


def _event_learner(X, ev, hold, key):
    rows = _event_rows(X, ev, hold)

    def run():
        fc = feat_cols(features(X)) + ["side"]
        dates = rows.index.get_level_values("date")
        out = pd.Series(np.nan, index=rows.index)
        for y in range(FIRST_TEST_YEAR, 2021):
            st = pd.Timestamp(f"{y}-01-01")
            tr = (dates < st - pd.tseries.offsets.BDay(hold)) & \
                rows["yev"].notna().to_numpy()
            te = (dates >= st) & (dates < pd.Timestamp(f"{y + 1}-01-01"))
            if tr.sum() < 500 or te.sum() == 0:
                continue
            m = _gbm()
            m.set_params(min_samples_leaf=50)
            m.fit(rows.loc[tr, fc].to_numpy(), (rows.loc[tr, "yev"] > 0)
                  .astype(int))
            out[te] = m.predict_proba(rows.loc[te, fc].to_numpy())[:, 1]
        return out
    p = X.cached(key, run)
    take = (p > 0.5).astype(float) * rows["side"]
    w = take.unstack("sym").reindex(index=X.C.index, columns=X.C.columns)
    return L.hold_for(w.fillna(0), hold)


@model("ml_prob_breakout")
def ml_prob_breakout(X):
    """ML probability of breakout success: boosting on 20-day breakout days
    predicts 10-day follow-through; take breakouts with p > 0.5."""
    from examples.quant.models_daily import _breakout_trades
    ev, _ = _breakout_trades(X)
    return _event_learner(X, ev, 10, "ev_bo")


@model("ml_prob_reversal")
def ml_prob_reversal(X):
    """ML probability of reversal: on |z20| > 2 days, boosting predicts
    whether a fade pays over 5 days; take fades with p > 0.5."""
    z = L.zscore(X.C, 20)
    ev = (z < -2).astype(float) - (z > 2).astype(float)
    return _event_learner(X, ev, 5, "ev_rev")


@model("meta_label")
def meta_label(X):
    """Meta-labeling (Lopez de Prado): primary = 20-day Donchian side on
    entry days; a boosted secondary model decides whether to act (10-day
    outcome)."""
    from examples.quant.models_daily import donchian
    d = donchian(X)
    ev = d.where(d != d.shift(1), 0.0)
    return _event_learner(X, ev, 10, "ev_meta")


@model("ml_risk_reward")
def ml_risk_reward(X):
    """ML risk/reward: two boosted regressions predict the 5-day maximum
    favourable up move and down move (in vol units); long when up/down >
    1.3, short when < 1/1.3; weekly."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    P = features(X)
    lp = np.log(X.C)
    vol = L.rstd(X.R, 20)
    up = (np.log(X.D["H"]).rolling(5).max().shift(-5) - lp) / vol
    dn = (lp - np.log(X.D["L"]).rolling(5).min().shift(-5)) / vol
    rows = P.copy()
    rows["yu"] = up.stack().reindex(rows.index)
    rows["yd"] = dn.stack().reindex(rows.index)
    mk = lambda: HistGradientBoostingRegressor(               # noqa
        max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
        random_state=0)
    fu = walk_forward(X, mk, target="yu", kind="reg", horizon=5, rows=rows,
                      key="r_up5")
    fd = walk_forward(X, mk, target="yd", kind="reg", horizon=5, rows=rows,
                      key="r_dn5")
    ratio = fu / fd
    return _hold5((ratio > 1.3).astype(float) - (ratio < 1 / 1.3).astype(float))


@model("triple_barrier")
def triple_barrier(X):
    """Triple-barrier model: labels from +-1.5 x 20-day sd barriers with a
    10-day vertical barrier (first touch, from daily highs/lows); a boosting
    classifier predicts the upper-vs-lower outcome; trade p > 0.55 / < 0.45
    weekly."""
    def labels():
        H, Lo, C = X.D["H"].to_numpy(), X.D["L"].to_numpy(), X.C.to_numpy()
        sd = L.rstd(X.R, 20).to_numpy()
        n, m = C.shape
        out = np.full((n, m), np.nan)
        for j in range(m):
            for i in range(n - 10):
                if not np.isfinite(sd[i, j]) or not np.isfinite(C[i, j]):
                    continue
                up = C[i, j] * (1 + 1.5 * sd[i, j])
                dn = C[i, j] * (1 - 1.5 * sd[i, j])
                lab = 0.0
                for k in range(i + 1, i + 11):
                    hu, hd = H[k, j] >= up, Lo[k, j] <= dn
                    if hu and hd:
                        lab = -1.0          # both in one day: assume the stop
                        break
                    if hu:
                        lab = 1.0
                        break
                    if hd:
                        lab = -1.0
                        break
                if lab == 0.0:
                    lab = np.sign(C[i + 10, j] - C[i, j])
                out[i, j] = lab
        return pd.DataFrame(out, index=X.C.index, columns=X.C.columns)
    lab = X.cached("tb_labels", labels)
    rows = features(X).copy()
    rows["ytb"] = lab.stack().reindex(rows.index)
    p = walk_forward(X, _gbm, target="ytb", horizon=10, rows=rows,
                     key="p_tb")
    return _hold5(p_to_sig(p, 0.05))


# ================================================= sequence models (torch)

def _seq_panel(X, win=20):
    """Per-symbol sequences of 6 standardised daily inputs."""
    P = features(X)
    cols = ["r1", "r5", "r20", "vol20", "z20", "rsi14"]
    W = {c: P[c].unstack("sym").reindex(index=X.C.index, columns=X.C.columns)
         for c in cols}
    A = np.stack([W[c].to_numpy() for c in cols], axis=-1)   # T x S x F
    y = (X.R.shift(-1) > 0).astype(float).where(X.R.shift(-1).notna()
                                                ).to_numpy()
    return A, y


def _torch_walk(X, build, key, win=20, epochs=4):
    def run():
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(4)
        A, y = _seq_panel(X, win)
        T, S, F = A.shape
        idx = X.C.index
        out = np.full((T, S), np.nan)
        # sample ends: (t, s) with a full finite window
        ok = np.zeros((T, S), bool)
        fin = np.isfinite(A).all(axis=2)
        cs = np.cumsum(np.vstack([np.zeros((1, S)), fin]), axis=0)
        ok[win - 1:] = (cs[win:] - cs[:-win]) == win
        for yr in range(FIRST_TEST_YEAR, 2021):
            st = np.searchsorted(idx, pd.Timestamp(f"{yr}-01-01"))
            en = np.searchsorted(idx, pd.Timestamp(f"{yr + 1}-01-01"))
            tr = [(t, s) for t in range(win - 1, st - 1) for s in range(S)
                  if ok[t, s] and np.isfinite(y[t, s])]
            te = [(t, s) for t in range(st, en) for s in range(S) if ok[t, s]]
            if len(tr) < 5000 or not te:
                continue
            rng = np.random.default_rng(yr)
            tr = [tr[i] for i in rng.choice(len(tr), min(40000, len(tr)),
                                            replace=False)]
            Xtr = torch.tensor(np.stack([A[t - win + 1:t + 1, s]
                                         for t, s in tr]), dtype=torch.float32)
            ytr = torch.tensor([y[t, s] for t, s in tr], dtype=torch.float32)
            net = build(F)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                                   weight_decay=1e-4)
            lossf = torch.nn.BCEWithLogitsLoss()
            for _ in range(epochs):
                perm = torch.randperm(len(Xtr))
                for b in range(0, len(Xtr), 512):
                    i = perm[b:b + 512]
                    opt.zero_grad()
                    loss = lossf(net(Xtr[i]).squeeze(-1), ytr[i])
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                Xte = torch.tensor(np.stack([A[t - win + 1:t + 1, s]
                                             for t, s in te]),
                                   dtype=torch.float32)
                pr = torch.sigmoid(net(Xte).squeeze(-1)).numpy()
            for (t, s), p in zip(te, pr):
                out[t, s] = p
        return pd.DataFrame(out, index=idx, columns=X.C.columns)
    return X.cached(key, run)


def _lstm(F):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(F, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(h[:, -1])
    return Net()


def _transformer(F):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(F, 16)
            self.pos = nn.Parameter(torch.zeros(1, 20, 16))
            layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.1,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 1)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            h = self.enc(self.inp(x) + self.pos)
            return self.head(h[:, -1])
    return Net()


@model("ml_lstm")
def ml_lstm(X):
    """LSTM (16 units) over 20-day sequences of 6 inputs; next-day
    direction, yearly walk-forward refit on 40,000 sampled sequences."""
    return p_to_sig(_torch_walk(X, _lstm, "p_lstm"))


@model("ml_transformer")
def ml_transformer(X):
    """One-layer Transformer encoder (2 heads, d=16) over the same 20-day
    sequences."""
    return p_to_sig(_torch_walk(X, _transformer, "p_tfm"))


# ===================================================== reinforcement learning

@model("rl_qlearn")
def rl_qlearn(X):
    """Tabular Q-learning: state = (trend sign, z20 bucket, vol regime,
    current position), actions short/flat/long, reward = next-day return
    minus cost on changes; trained each year on earlier data (10 passes,
    epsilon-greedy), then acted greedily on the test year."""
    tr_s = (X.C > L.sma(X.C, 50)).astype(int)
    zb = pd.DataFrame(np.digitize(L.zscore(X.C, 20).fillna(0), [-1.5, -0.5,
                                                                 0.5, 1.5]),
                      index=X.C.index, columns=X.C.columns)
    vr = (L.rolling_pct_rank(L.rstd(X.R, 20), 250) > 0.5).astype(int)
    state = (tr_s * 10 + zb * 2 + vr).to_numpy()            # < 20
    r = X.R.shift(-1).fillna(0).to_numpy()
    cost = np.array([(SPREAD_BP.get(c, 5) / 2 + SLIP_BP) / 1e4
                     for c in X.C.columns])
    vol = (L.rstd(X.R, 60).bfill()).to_numpy()
    years = X.C.index.year.to_numpy()
    out = np.zeros_like(r)
    rng = np.random.default_rng(0)
    acts = np.array([-1.0, 0.0, 1.0])
    for y in range(FIRST_TEST_YEAR, 2021):
        Q = np.zeros((20, 3, 3))
        tr = np.flatnonzero(years < y)[:-1]     # last reward is in year y
        te = np.flatnonzero(years == y)
        for _ in range(10):
            for j in range(r.shape[1]):
                pos = 1
                for i in tr:
                    s = state[i, j]
                    a = rng.integers(3) if rng.random() < 0.1 else \
                        int(np.argmax(Q[s, pos]))
                    rew = (acts[a] * r[i, j] - abs(acts[a] - acts[pos]) *
                           cost[j]) / vol[i, j]
                    s2 = state[min(i + 1, len(r) - 1), j]
                    Q[s, pos, a] += 0.01 * (rew + 0.9 * Q[s2, a].max() -
                                            Q[s, pos, a])
                    pos = a
        for j in range(r.shape[1]):
            pos = 1
            for i in te:
                a = int(np.argmax(Q[state[i, j], pos]))
                out[i, j] = acts[a]
                pos = a
    return pd.DataFrame(out, index=X.C.index, columns=X.C.columns)


# ================================================= online / window learners

@model("online_learning")
def online_learning(X):
    """Online learning: SGD logistic regression updated every day with the
    day's realised labels (predict, then learn), from 2007 on."""
    from sklearn.linear_model import SGDClassifier
    P = features(X)
    fc = feat_cols(P)
    P = P.dropna(subset=["y1"]).sort_index()
    m = SGDClassifier(loss="log_loss", alpha=1e-3, learning_rate="constant",
                      eta0=1e-3, random_state=0)
    out = {}
    first = True
    for d, g in P.groupby(level="date"):
        Xd = g[fc].to_numpy()
        if not first and d >= pd.Timestamp("2008-01-01"):
            out[d] = pd.Series(m.predict_proba(Xd)[:, 1],
                               index=g.index.get_level_values("sym"))
        m.partial_fit(Xd, (g["y1"] > 0).astype(int), classes=[0, 1])
        first = False
    p = pd.DataFrame(out).T.reindex(index=X.C.index, columns=X.C.columns)
    return p_to_sig(p)


@model("expanding_window")
def expanding_window(X):
    """Expanding-window ridge regression of next-day return on the features,
    refit yearly; trade the sign."""
    from sklearn.linear_model import Ridge
    f = walk_forward(X, lambda: _scaled(Ridge(alpha=100.0)), kind="reg",
                     key="r_ridge_exp")
    return L.sign(f)


@model("rolling_window")
def rolling_window(X):
    """Rolling-window (3 years) ridge regression, refit yearly."""
    from sklearn.linear_model import Ridge
    f = walk_forward(X, lambda: _scaled(Ridge(alpha=100.0)), kind="reg",
                     train_years=3, key="r_ridge_roll")
    return L.sign(f)


@model("frac_diff")
def frac_diff(X):
    """Fractionally differentiated price (d = 0.4, fixed window): stationary
    yet memory-preserving; fade its 250-day z beyond 2."""
    z = L.zscore(ffd(np.log(X.C), 0.4), 250)
    return L.fade_band(z, 2.0, 0.0)


# ============================================= walk-forward parameter choice

def _wf_select(X, cands: dict, lookback_years=3, every="YS"):
    """Each year, per symbol, pick the candidate signal with the best net
    Sharpe over the previous `lookback_years`; trade it for the year."""
    from bcbt.quant import run
    nets = {k: run(s, X.C).per_sym for k, s in cands.items()}
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    yrs = X.C.index.year
    for y in range(FIRST_TEST_YEAR, 2021):
        tr = (yrs < y) & (yrs >= y - lookback_years)
        te = yrs == y
        for c in X.C.columns:
            best, bs = None, -np.inf
            for k, n in nets.items():
                x = n.loc[tr, c]
                s = x.mean() / x.std() if x.std() > 0 else -np.inf
                if s > bs:
                    best, bs = k, s
            if best is not None:
                out.loc[te, c] = cands[best].loc[te, c]
    return out


@model("wf_optimized")
def wf_optimized(X):
    """Walk-forward optimised moving-average crossover: each year pick the
    best of 9 fast/slow EMA pairs per CFD on the prior 3 years."""
    def f():
        c = {f"{a}/{b}": L.sign(L.ema(X.C, a) - L.ema(X.C, b))
             for a in (5, 10, 20) for b in (50, 100, 200)}
        return _wf_select(X, c)
    return X.cached("wf_ma", f)


@model("wf_momentum")
def wf_momentum(X):
    """Walk-forward momentum: lookback 1, 3, 6 or 12 months chosen per CFD
    each year on the prior 3 years."""
    return X.cached("wf_mom", lambda: _wf_select(
        X, {n: L.sign(L.ret(X.C, n)) for n in (21, 63, 126, 252)}))


@model("wf_mr")
def wf_mr(X):
    """Walk-forward mean reversion: z lookback 10/20/50 x entry 1.5/2/2.5
    chosen per CFD each year."""
    return X.cached("wf_mr", lambda: _wf_select(
        X, {f"{n}/{e}": L.fade_band(L.zscore(X.C, n), e, 0.0)
            for n in (10, 20, 50) for e in (1.5, 2.0, 2.5)}))


@model("wf_breakout")
def wf_breakout(X):
    """Walk-forward breakout: Donchian 10/20/55/100 (exit at half) chosen per
    CFD each year."""
    def don(n):
        hi = X.D["H"].rolling(n).max().shift(1)
        lo = X.D["L"].rolling(n).min().shift(1)
        hx = X.D["H"].rolling(n // 2).max().shift(1)
        lx = X.D["L"].rolling(n // 2).min().shift(1)
        return L.machine(X.C > hi, X.C < lo, X.C < lx, X.C > hx, X.C)
    return X.cached("wf_bo", lambda: _wf_select(
        X, {n: don(n) for n in (10, 20, 55, 100)}))


@model("rolling_param")
def rolling_param(X):
    """Rolling-parameter model: every month choose the momentum lookback
    (1/3/6/12 months) with the best trailing 1-year net Sharpe per CFD."""
    from bcbt.quant import run
    cands = {n: L.sign(L.ret(X.C, n)) for n in (21, 63, 126, 252)}
    nets = {k: run(s, X.C).per_sym for k, s in cands.items()}
    sh = {k: n.rolling(252).mean() / n.rolling(252).std() for k, n in
          nets.items()}
    best = pd.concat(sh, axis=1)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for c in X.C.columns:
        b = best.xs(c, axis=1, level=1).fillna(-np.inf).idxmax(axis=1)
        b = L.month_end_rebalance(b.to_frame().astype(float).fillna(0)
                                  ).iloc[:, 0]
        for k, s in cands.items():
            out[c] = out[c].where(b != k, s[c])
    return out


@model("adaptive_param")
def adaptive_param(X):
    """Adaptive parameter: the trend lookback scales inversely with the
    volatility regime (short when vol is high): 20-120 days."""
    q = (L.rstd(X.R, 20) / L.rstd(X.R, 250)).clip(0.5, 2)
    look = (60 / q).round(-1).clip(20, 120)
    out = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
    for n in range(20, 121, 10):
        out = out.where(look != n, L.sign(L.ret(X.C, n)))
    return out
