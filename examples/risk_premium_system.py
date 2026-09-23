"""
A share-timing system driven by other markets, not by share prices.

What the design period (1990-2010) showed, measured on month-end dates so
that no two observations overlap:

  The volatility premium -- what insurance costs, minus how turbulent the
  last month actually was -- is the strongest single signal. When it is in
  its top fifth the next month returns +2.08%; in its bottom fifth, +0.14%.
  Correlation with next month's return +0.215 across 252 independent
  months. The economics are ordinary: when people are paying a lot to be
  protected, the reward for carrying risk instead is large.

  Credit spreads widening predicts weaker returns, with the sign expected.
  Over three months the correlation is -0.136. Bond investors are lending
  rather than owning, so they react to trouble sooner.

  Those two disagree about timing on purpose. The volatility premium is
  contrarian: it is highest when everyone is frightened but nothing has
  broken. The credit signal is confirming: it turns negative while things
  are actually deteriorating. A rule that uses both should be in the market
  when fear is priced but not yet justified, and out when the bond market
  says the fear is real.

Rules of the test:

  * Thresholds are fixed and readable, or use an expanding median, never a
    quantile of the whole history. Using the full sample to decide what
    counted as "expensive insurance" in 1994 is peeking.
  * Cash earns the actual fed funds rate. A rule that sits out has to be
    credited for the interest it would have earned, and in 1990 that was 8%.
  * Costs are charged on turnover at 5 basis points.
  * Design is 1990-2010. The holdout, 2011 onward, is opened once, at the
    end, for one pre-committed candidate.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bcbt import macro, universe as U

warnings.filterwarnings("ignore")

DESIGN_START, DESIGN_END = "1990-01-01", "2010-12-31"
HOLD_START = "2011-01-01"
COST_BPS = 5.0
EQ, CASH = "EQ", "CASH"

CANDIDATE = "vol premium positive AND credit not widening"


def prepare() -> pd.DataFrame:
    """Prices for the engine: the share index and an interest-earning cash."""
    df = macro.load()
    px = pd.DataFrame(index=df.index)
    px[EQ] = df["total"]
    rate = df["fed_funds"].reindex(df.index).ffill().fillna(0.0) / 100.0
    px[CASH] = (1.0 + rate / 252.0).cumprod()
    for c in ("vol_premium", "credit_chg_1m", "credit_chg_3m",
              "credit_spread", "vix", "realised_vol"):
        px[c] = df[c]
    # An expanding median: what counted as a high premium judged only
    # against the premiums already seen.
    px["vp_median"] = px["vol_premium"].expanding(250).median()
    return px.dropna(subset=[EQ, CASH])


def _sig(hist, col):
    v = hist[col].iloc[-1]
    return None if pd.isna(v) else float(v)


def make(rule):
    """Wrap a True/False decision into portfolio weights."""
    def fn(hist):
        on = rule(hist)
        if on is None:
            return {EQ: 1.0}          # no signal yet: default to invested
        return {EQ: 1.0} if on else {CASH: 1.0}
    return fn


def r_vp_positive(h):
    v = _sig(h, "vol_premium")
    return None if v is None else v > 0


def r_vp_above_median(h):
    v, m = _sig(h, "vol_premium"), _sig(h, "vp_median")
    return None if v is None or m is None else v > m


def r_credit_calm_1m(h):
    v = _sig(h, "credit_chg_1m")
    return None if v is None else v <= 0


def r_credit_calm_3m(h):
    v = _sig(h, "credit_chg_3m")
    return None if v is None else v <= 0


def r_both(h):
    a, b = r_vp_positive(h), r_credit_calm_3m(h)
    if a is None or b is None:
        return None
    return a and b


def r_both_median(h):
    a, b = r_vp_above_median(h), r_credit_calm_3m(h)
    if a is None or b is None:
        return None
    return a and b


def r_either(h):
    a, b = r_vp_positive(h), r_credit_calm_3m(h)
    if a is None or b is None:
        return None
    return a or b


def r_vix_below_realised(h):
    """The opposite side of the premium: insurance cheaper than reality."""
    v = _sig(h, "vol_premium")
    return None if v is None else v > -2.0


SYSTEMS = {
    "vol premium positive": make(r_vp_positive),
    "vol premium above its own median": make(r_vp_above_median),
    "credit not widening (1m)": make(r_credit_calm_1m),
    "credit not widening (3m)": make(r_credit_calm_3m),
    "vol premium positive AND credit not widening": make(r_both),
    "vol premium above median AND credit calm": make(r_both_median),
    "either of the two": make(r_either),
    "insurance not much cheaper than reality": make(r_vix_below_realised),
}

BENCH = {
    "buy and hold": lambda h: {EQ: 1.0},
    "cash only": lambda h: {CASH: 1.0},
}


def invested_share(px, fn, start, end):
    sub = px
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]
    vals = []
    for day in U.rebalance_dates(sub.index, "M"):
        hist = px.loc[:day]
        if len(hist) < 300:
            continue
        vals.append(float(fn(hist).get(EQ, 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def excess_stats(rec, cash_ret, label):
    """
    Score a portfolio on what it earned ABOVE cash.

    Measuring return per unit of wobble on raw returns rewards any rule
    that sits in cash, because cash has a return and almost no wobble. Run
    that way, "hold cash forever" scores 14 and wins, which is how this was
    caught. The comparison that means something is the reward for taking
    risk, so cash is subtracted first and a cash-only portfolio correctly
    scores zero.
    """
    r = rec["net"].fillna(0.0)
    r = r[r.index >= r.ne(0).idxmax()]
    if len(r) < 250:
        return dict(label=label, years=0)
    c = cash_ret.reindex(r.index).fillna(0.0)
    ex = r - c
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    years = len(r) / 252.0
    ann = eq.iloc[-1] ** (1.0 / years) - 1.0
    vol = ex.std(ddof=1) * np.sqrt(252)
    dd = (eq / peak - 1.0).min()
    return dict(
        label=label, years=round(years, 1),
        annual_pct=100 * ann,
        vol_pct=100 * r.std(ddof=1) * np.sqrt(252),
        risk_adj=(ex.mean() * 252) / vol if vol > 1e-9 else 0.0,
        worst_fall_pct=100 * dd,
        calmar=ann / abs(dd) if dd < 0 else np.nan,
        turnover_yr=rec["turnover"].sum() / years,
    )


def table(px, start, end, title):
    rows, daily = [], {}
    cash_ret = px[CASH].pct_change()
    for name, fn in {**SYSTEMS, **BENCH}.items():
        rec = U.run_portfolio(px, fn, freq="M", cost_bps=COST_BPS,
                              start=start, end=end)
        s = excess_stats(rec, cash_ret, name)
        if not s.get("years"):
            continue
        s["in_mkt_pct"] = 100 * invested_share(px, fn, start, end)
        rows.append(s)
        r = rec["net"]
        r = r[r.index >= r.ne(0).idxmax()]
        # The search is scored on excess return too, for the same reason.
        daily[name] = r - cash_ret.reindex(r.index).fillna(0.0)
    d = pd.DataFrame(rows).set_index("label")
    cols = ["annual_pct", "vol_pct", "risk_adj", "worst_fall_pct", "calmar",
            "turnover_yr", "in_mkt_pct"]
    print(f"\n{title}")
    print(d[cols].sort_values("risk_adj", ascending=False).round(3).to_string())
    return d, daily


def score_search(daily, n_boot=5000, block=21, seed=5):
    M = pd.DataFrame(daily).dropna()
    X = (M - M.mean()).to_numpy()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    span = np.arange(block)
    mx = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.integers(0, n, size=nb)[:, None] + span[None, :]
               ).ravel()[:n] % n
        S = X[idx]
        sd = S.std(axis=0, ddof=1)
        mx[b] = np.nanmax(np.where(sd > 0, S.mean(axis=0) / sd, -np.inf))
    obs = M.mean() / M.std(ddof=1)
    return obs.idxmax(), float(obs.max()), float((mx >= obs.max()).mean()), mx


def main() -> int:
    pd.set_option("display.width", 220)
    px = prepare()
    print(f"data {px.index[0].date()} -> {px.index[-1].date()}")

    dd, ddaily = table(px, DESIGN_START, DESIGN_END,
                       f"--- design {DESIGN_START} -> {DESIGN_END} ---")
    # "cash only" is a reference point, not a competing rule; scoring it
    # as one asks whether doing nothing beats doing nothing.
    best, val, p, null = score_search(
        {k: v for k, v in ddaily.items() if k != "cash only"})
    print(f"\n  scoring the search over {len(ddaily)} rules")
    print(f"    best: {best}  ({val:.4f} per unit of daily wobble)")
    print(f"    luck alone: median {np.median(null):.4f}, "
          f"top 5% above {np.percentile(null, 95):.4f}")
    print(f"    p = {p:.4f}  ", end="")
    print("-> better than luck" if p < 0.05 else "-> NOT better than luck")
    print(f"\n  PRE-COMMITTED CANDIDATE: {CANDIDATE}")

    hd, _ = table(px, HOLD_START, None,
                  f"--- HOLDOUT {HOLD_START} onward, one look ---")

    print("\n" + "=" * 100)
    print("THE CANDIDATE, DESIGN vs HOLDOUT")
    print("=" * 100)
    print(f"  {'':<34}{'a year':>9}{'wobble':>9}{'risk-adj':>10}"
          f"{'worst':>9}{'in mkt':>9}")
    for lab, frame in (("design", dd), ("HOLDOUT", hd)):
        for who in (CANDIDATE, "buy and hold"):
            if who not in frame.index:
                continue
            r = frame.loc[who]
            print(f"  {lab + ': ' + who:<34}{r['annual_pct']:>8.2f}%"
                  f"{r['vol_pct']:>8.1f}%{r['risk_adj']:>10.2f}"
                  f"{r['worst_fall_pct']:>8.1f}%{r['in_mkt_pct']:>8.0f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
