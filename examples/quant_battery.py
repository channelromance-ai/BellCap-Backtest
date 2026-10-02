"""
The quant battery: every model on the user's list through one engine, one
cost model and one set of tests.

    python -m examples.quant_battery            # run (cached per model)
    python -m examples.quant_battery --force    # recompute everything
    python -m examples.quant_battery --report   # rebuild the report only

Data: OANDA CFD quotes for 25 instruments (8 equity indices, 6 FX pairs,
gold, oil, gas, 4 grains/softs, 4 government bonds), 1-minute bars folded to
15-minute, hourly and daily, Jan 2005 - May 2020; VIX daily for the
implied-volatility models.

Every model is judged on:
  net Sharpe          after spread + slippage on every change of position
                      and overnight financing on everything held over the
                      17:00 New York roll
  development/hold-out  2005-2012 vs 2013-2020: an edge must show in both
  Newey-West t        of the daily net return
  alpha t vs long     regression on a long-only, vol-scaled holding of the
                      same instruments: is it more than being long?
  multiple testing    White's Reality Check on the best model, and the
                      deflated Sharpe ratio for 359 trials

Overlays, filters and sizing schemes are also compared with the model they
modify. Results go to out/quant/ (csv) and docs/quant_battery.md.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bcbt import quant as Q                                   # noqa: E402
from examples.quant.core import MODELS, Ctx, evaluate         # noqa: E402
from examples.quant import (models_daily, models_xasset,      # noqa: E402,F401
                            models_intraday, models_ml, models_risk,
                            models_extra)
from examples.quant.registry import entries, REFERENCES       # noqa: E402

OUT = "out/quant"
CACHE = os.path.join(OUT, "streams")
REPORT = "docs/quant_battery.md"


# ------------------------------------------------------------------ running

def _norm(x, target=0.10):
    sd = x.std() * math.sqrt(252)
    return x * (target / sd) if sd > 0 else x


def run_one(key, X, force=False):
    path = os.path.join(CACHE, f"{key}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    res = evaluate(key, X)
    m = MODELS[key]
    net, gross, nofin = res.daily("net"), res.daily("gross"), res.daily(
        "nofin")
    if m["norm"]:
        k = 0.10 / (net.std() * math.sqrt(252))
        net, gross, nofin = net * k, gross * k, nofin * k
    syms = list(res.w.columns[(res.w.abs().mean() > 1e-3).to_numpy()])
    out = dict(key=key, net=net, gross=gross, nofin=nofin, syms=syms,
               turnover=res.turnover, exposure=res.exposure,
               sym_pos=float((res.per_sym.sum() > 0).mean()),
               secs=time.time() - t0)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def bench_stream(X, syms, cache={}):
    """Long-only, vol-scaled holding of the same instruments (net)."""
    k = tuple(sorted(syms))
    if k not in cache:
        sig = pd.DataFrame(0.0, index=X.C.index, columns=X.C.columns)
        sig[list(k)] = 1.0
        cache[k] = Q.run(sig, X.C).net
    return cache[k]


# ------------------------------------------------------------------ scoring

def _stats(d):
    d = d.dropna()
    act = d[d != 0]
    if len(act) < 50:
        return None
    d = d[d.index >= act.index[0]]
    dev, hold = d[d.index < Q.DEV_END], d[d.index >= Q.DEV_END]
    return d, dev, hold


def score(o, X):
    s = _stats(o["net"])
    if s is None:
        return dict(model=o["key"], n_days=0)
    d, dev, hold = s
    g = o["gross"].reindex(d.index)
    nf = o["nofin"].reindex(d.index)
    row = dict(model=o["key"], n_days=len(d),
               start=d.index[0].date(),
               sharpe=Q.sharpe(d), sharpe_gross=Q.sharpe(g),
               sharpe_nofin=Q.sharpe(nf),
               sharpe_dev=Q.sharpe(dev), sharpe_hold=Q.sharpe(hold),
               t=Q.nw_t(d), t_dev=Q.nw_t(dev), t_hold=Q.nw_t(hold),
               t_gross=Q.nw_t(g),
               ann_ret=float(d.mean() * 252),
               ann_vol=float(d.std() * math.sqrt(252)),
               max_dd=Q.max_dd(d), worst_day=float(d.min()),
               turnover=o["turnover"], exposure=o["exposure"],
               sym_pos=o["sym_pos"], n_syms=len(o["syms"]))
    if o["syms"]:
        b = bench_stream(X, o["syms"]).reindex(d.index).fillna(0)
        if b.std() > 0:
            beta = float(np.cov(d, b)[0, 1] / b.var())
            row["beta_long"] = beta
            row["alpha_t"] = Q.nw_t(d - beta * b)
            row["corr_long"] = float(np.corrcoef(d, b)[0, 1])
    return row


def prop_pass(d, days=60, target=0.08, max_loss=0.10, day_loss=0.05):
    """Share of 60-day challenges (one starting every 21 days) that reach
    +8% before a 10% loss from the start or a 5% losing day, with the
    stream at 10% annual vol."""
    x = _norm(d.dropna())
    v = x.to_numpy()
    res = []
    for s in range(0, len(v) - days, 21):
        eq, out = 0.0, "timeout"
        for r in v[s:s + days]:
            eq += r
            if r <= -day_loss or eq <= -max_loss:
                out = "bust"
                break
            if eq >= target:
                out = "pass"
                break
        res.append(out)
    res = np.array(res)
    return float((res == "pass").mean()), float((res == "bust").mean())


def verdict(r, n_trials):
    if not r or r.get("n_days", 0) == 0:
        return "no trades"
    sh, sg = r["sharpe"], r["sharpe_gross"]
    both = r["sharpe_dev"] > 0 and r["sharpe_hold"] > 0
    alpha_ok = r.get("alpha_t", r["t"]) >= 2
    if sh > 0 and r["t"] >= 2 and both and alpha_ok and r.get("dsr", 0) > 0.95:
        return "EDGE (survives multiple testing)"
    if sh > 0 and r["t"] >= 2 and both and alpha_ok:
        return "significant alone, not after 359 trials"
    if sh > 0 and r["t"] >= 2 and both:
        return "positive, but it is the long bias (alpha t < 2)"
    if sh > 0 and both:
        return "positive both halves, not significant"
    if sh > 0:
        return "positive overall, fails one half"
    if sg > 0.3 and r["t_gross"] >= 2:
        return "gross edge, eaten by costs"
    if sg > 0:
        return "no edge (gross barely positive)"
    return "no edge"


# ------------------------------------------------------------------ report

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def write_report(rows, entries_, rc, extras):
    R = {r["model"]: r for r in rows}
    L = []
    w = L.append
    w("# Quant battery: 385 retail CFD models, one test\n")
    w(extras["intro"])
    w("\n## Verdict count\n")
    w("| Verdict | Models |\n|---|---|")
    for k, v in extras["verdicts"].items():
        w(f"| {k} | {v} |")
    w("")
    w(extras["mt"])
    for title, body in extras["sections"]:
        w(f"\n## {title}\n")
        w(body)
    w("\n## Every model on the list\n")
    w("Sharpe ratios are annualised from daily net returns. *Gross* is "
      "before any cost, *no-fin* after spread and slippage but before "
      "overnight financing. *Dev* is 2005-2012, *hold* 2013-May 2020. "
      "*α t* is the Newey-West t of the return left after regressing on a "
      "long-only holding of the same instruments.\n")
    w("| # | Model | Test | Net | Gross | No-fin | Dev | Hold | t | α t | "
      "Verdict |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (name, key, note) in enumerate(entries_, 1):
        if key is None:
            w(f"| {i} | {name} | — | | | | | | | | not testable: {note} |")
            continue
        r = R.get(key, {})
        v = r.get("verdict", "error")
        extra = f" ({note})" if note else ""
        w(f"| {i} | {name} | `{key}` | {fmt(r.get('sharpe'))} | "
          f"{fmt(r.get('sharpe_gross'))} | {fmt(r.get('sharpe_nofin'))} | "
          f"{fmt(r.get('sharpe_dev'))} | {fmt(r.get('sharpe_hold'))} | "
          f"{fmt(r.get('t'), 1)} | {fmt(r.get('alpha_t'), 1)} | {v}{extra} |")
    w("\n## How each model is defined\n")
    w("One line per test, from its docstring. Code: `examples/quant/`.\n")
    for k in sorted(MODELS):
        doc = " ".join(MODELS[k]["doc"].split())
        w(f"- `{k}` ({MODELS[k]['freq']}): {doc}")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------- report text

FAMILY = {"models_daily": "Daily single-market (mean reversion, trend, "
                          "breakout, volatility, regime, forecasting, "
                          "technical, patterns)",
          "models_xasset": "Cross-asset: pairs, stat arb, ranking, relative "
                           "strength, macro",
          "models_intraday": "Intraday: sessions, seasonality, gaps, news, "
                             "lead-lag, CFD execution, tick-volume flow, "
                             "jumps, multi-timeframe",
          "models_ml": "Machine learning and walk-forward",
          "models_risk": "Risk, sizing, trade filters, portfolio blends",
          "models_extra": "Volatility estimators, regime-conditional, "
                          "support/resistance and others"}


def family(key):
    mod = MODELS[key]["fn"].__module__.rsplit(".", 1)[-1]
    return FAMILY.get(mod, mod)


def table(df, cols, heads, nd=2):
    out = ["| " + " | ".join(heads) + " |",
           "|" + "---|" * len(heads)]
    for k, r in df.iterrows():
        cells = [f"`{k}`"] + [fmt(r.get(c), nd) if not isinstance(
            r.get(c), str) else r.get(c) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_extras(df, meta, entries_):
    listed = list(dict.fromkeys(k for _, k, _ in entries_ if k))
    T = df.loc[[k for k in listed if k in df.index]].copy()
    T = T[T["n_days"] > 0]
    N = len(T)
    n_t2 = int((T["t"] >= 2).sum())
    n_t2_both = int(((T["t"] >= 2) & (T["sharpe_dev"] > 0) &
                     (T["sharpe_hold"] > 0)).sum())
    n_pos = int((T["sharpe"] > 0).sum())
    n_gpos = int((T["sharpe_gross"] > 0).sum())
    best = T["sharpe"].idxmax()
    intro = f"""
**Data.** OANDA's own CFD quotes (1-minute bars with tick volume) for 25
instruments: S&P 500, Nasdaq 100, Russell 2000, FTSE 100, CAC 40, Nikkei
225, ASX 200, AEX; EUR/USD, GBP/USD, AUD/USD, USD/CAD, EUR/JPY, AUD/JPY;
gold, WTI, natural gas, corn, wheat, soybeans, sugar; US 2-year and
10-year Treasury, Bund and Gilt CFDs. January 2005 to May 2020, folded to
15-minute, hourly and daily (17:00 New York roll) bars, plus daily VIX.
Fetched from the public FutureSharks/financial-data repository by
`bcbt/oanda_data.py` (Dukascopy and Yahoo are blocked in this environment).

**One engine for every model** (`bcbt/quant.py`). A model only states the
position it wants at each bar's close; the engine holds it to the next
close, sizes every instrument to the same 10% annual volatility (capped at
3x leverage, resized only on a 25% drift), and charges:

* the spread plus 1bp slippage on every change of position, per instrument
  (1bp S&P, 1.2bp EUR/USD, 3bp gold, 5bp WTI, 10-20bp grains, 2-4bp bonds:
  typical retail quotes, not the tightest advertised);
* overnight financing on everything held over the 17:00 New York roll, three
  nights at weekends: a 2.5% a year markup on indices, commodities and
  bonds and 1% on FX, charged long and short. The benchmark rate itself is
  left out (near zero for USD, EUR and JPY most of this sample).

**Every model is judged the same way.** Net Sharpe; the same before
financing and before any cost; 2005-2012 vs 2013-2020 (an edge must show
in both); the Newey-West t of daily net returns; and an alpha t after
regressing on a long-only holding of the same instruments, so that a
model which is mostly long the S&P in a bull market is not credited with
skill. Then the multiple-testing correction for having tried {N} distinct
models. Parameters are textbook values fixed before any result was seen;
everything that learns (ML, walk-forward, Bayesian, seasonal) is fitted
only on data before the year it trades. A look-ahead audit
(`examples/quant_audit.py`) re-runs every model on data cut at mid-2014 and
checks every signal before the cut is identical.

**Result.** {n_pos} of {N} models have a positive net Sharpe; {n_gpos} are
positive before costs. {n_t2} have a net t-statistic of 2 or more, and
{n_t2_both} of those are positive in both halves. The best net Sharpe is
`{best}` at {T.loc[best, "sharpe"]:.2f}.
"""
    vc = T["verdict"].value_counts()
    nr = len([1 for _, k, _ in entries_ if k is None])
    verdicts = {**vc.to_dict(), "not testable on this data": nr}
    exp_t2 = 0.025 * N
    mt = f"""
## Multiple testing

With {N} models and no edge anywhere, about {exp_t2:.0f} would still show a
one-sided t of 2 by luck (2.5% each). {n_t2} did. White's Reality Check on
the best daily net stream (`{meta["best"]}`, Sharpe {meta["best_sr"]:.2f})
against the best of {N} demeaned, block-bootstrapped streams: p =
{meta["p_rc"]:.3f}. The deflated Sharpe ratio, which asks how likely the
top Sharpe is to be real once it is known to be the best of {N} tries with
this spread of results, is in the `dsr` column of `out/quant/results.csv`;
a model needs dsr > 0.95 to be called an edge here.
"""
    secs = []
    # 1. cost
    G = T.sort_values("sharpe_gross", ascending=False).head(20).copy()
    G["spread_drag"] = G["sharpe_gross"] - G["sharpe_nofin"]
    G["fin_drag"] = G["sharpe_nofin"] - G["sharpe"]
    secs.append(("Before costs: the 20 best gross Sharpe ratios", 
                 "What the signal earns before the broker is paid, and what "
                 "spread and overnight financing take back.\n\n" +
                 table(G, ["sharpe_gross", "spread_drag", "fin_drag",
                           "sharpe", "turnover", "verdict"],
                       ["Model", "Gross", "− spread", "− financing", "Net",
                        "Turnover/day", "Verdict"])))
    # 2. best net
    B = T.sort_values("sharpe", ascending=False).head(20)
    secs.append(("After costs: the 20 best net Sharpe ratios",
                 "β and α t are against a long-only, vol-scaled holding of the "
                 "same instruments; dsr is the deflated Sharpe ratio.\n\n" +
                 table(B, ["sharpe", "sharpe_dev", "sharpe_hold", "t",
                           "beta_long", "alpha_t", "dsr", "verdict"],
                       ["Model", "Net", "Dev", "Hold", "t", "β long", "α t",
                        "dsr", "Verdict"])))
    # 3. families
    T["family"] = [family(k) for k in T.index]
    F = T.groupby("family").agg(models=("sharpe", "size"),
                                median_net=("sharpe", "median"),
                                median_gross=("sharpe_gross", "median"),
                                best_net=("sharpe", "max"),
                                positive_net=("sharpe",
                                              lambda x: int((x > 0).sum())))
    lines = ["| Family | Models | Median net | Median gross | Best net | "
             "Net > 0 |", "|---|---|---|---|---|---|"]
    for k, r in F.iterrows():
        lines.append(f"| {k} | {int(r.models)} | {r.median_net:.2f} | "
                     f"{r.median_gross:.2f} | {r.best_net:.2f} | "
                     f"{int(r.positive_net)} |")
    secs.append(("By family", "\n".join(lines)))
    # 4. sizing
    S = df[[MODELS[k]["norm"] for k in df.index]].copy()
    S = S[S["n_days"] > 0].sort_values("sharpe", ascending=False)
    if len(S):
        secs.append((
            "Sizing and risk schemes on the same trend signals",
            "Each scheme sizes the multi-speed trend model's positions (or, "
            "for the Kelly/utility entries and the long baskets, its own) and "
            "is rescaled to 10% realised volatility, so the comparison is at "
            "equal risk. *Prop pass / bust* is the share of 60-day windows, "
            "one starting every month, that reach +8% before a 10% loss or a "
            "5% losing day.\n\n" +
            table(S, ["sharpe", "sharpe_dev", "sharpe_hold", "max_dd",
                      "worst_day", "prop_pass", "prop_bust"],
                  ["Scheme", "Net", "Dev", "Hold", "Max DD", "Worst day",
                   "Prop pass", "Prop bust"], nd=3)))
    # 5. overlays
    O = df[[bool(MODELS[k]["base"]) and not MODELS[k]["norm"]
            for k in df.index]].copy()
    O = O[O["n_days"] > 0]
    if len(O):
        O["base"] = [MODELS[k]["base"] for k in O.index]
        secs.append((
            "Filters and overlays against the model they modify",
            "Δ is the change in net Sharpe versus the unfiltered base, "
            "overall and in each half. A filter that helps should help in "
            "both.\n\n" +
            table(O.sort_values("d_sharpe_vs_base", ascending=False),
                  ["base", "sharpe", "d_sharpe_vs_base", "d_dev_vs_base",
                   "d_hold_vs_base"],
                  ["Filter", "Base", "Net", "Δ", "Δ dev", "Δ hold"])))
    # 6. untestable
    U = [f"- **{n}**: {note}" for n, k, note in entries_ if k is None]
    secs.append(("Not testable on this data", "\n".join(U)))
    secs.append(("Caveats", """
* OANDA's quotes are one broker's mid prices. Costs are charged as typical
  retail spreads per instrument, not the spread actually quoted at each
  minute, so the cost of trading through news and the rollover hour is
  understated.
* Commodity CFDs roll between futures contracts; any roll gaps in OANDA's
  series are in the returns.
* Stops and targets in the trade-level filters (MAE/MFE, expectancy,
  R-multiple) are tracked on daily highs and lows, but the engine books the
  exit at that day's close.
* Proxies stand in where the real input is not in the data (no consensus
  forecasts, positioning, order book, TIPS or sector CFDs); those rows say
  so, and a proxy failing is weaker evidence than the real input failing.
* Intraday Sharpe ratios are very negative for many hourly and 15-minute
  models because they trade 25 instruments every bar against a spread;
  the gross column shows what the signal itself was worth.
"""))
    return dict(intro=intro, verdicts=verdicts, mt=mt, sections=secs)


def report():
    df = pd.read_csv(os.path.join(OUT, "results.csv"), index_col=0)
    with open(os.path.join(OUT, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    E = entries()
    rows = df.reset_index().to_dict("records")
    write_report(rows, E, None, build_extras(df, meta, E))
    print("wrote", REPORT)


# --------------------------------------------------------------------- main
# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()

    X = Ctx()
    E = entries()
    keys = list(dict.fromkeys([k for _, k, _ in E if k] + REFERENCES))
    todo = args.only or keys
    streams, errors = {}, {}
    for k in todo:
        t = time.time()
        try:
            streams[k] = run_one(k, X, force=args.force or bool(args.only))
            print(f"{k:28s} {time.time() - t:6.1f}s", flush=True)
        except Exception as e:                       # report, keep going
            errors[k] = repr(e)
            traceback.print_exc()
            print(f"{k:28s} ERROR {e!r}", flush=True)
    # score everything that has a cached stream, not just this run's keys
    for k in keys:
        if k not in streams and os.path.exists(os.path.join(CACHE,
                                                            f"{k}.pkl")):
            streams[k] = run_one(k, X)

    rows = [score(o, X) for o in streams.values()]
    tested = [k for k in dict.fromkeys(k for _, k, _ in E if k)
              if k in streams]
    N = len(tested)

    # Multiple testing over the distinct tests on the list
    daily = {k: streams[k]["net"] for k in tested}
    best, best_sr, p_rc, null = Q.reality_check(daily, n_boot=2000)
    srs = np.array([r["sharpe"] for r in rows if r.get("n_days")
                    and r["model"] in tested])
    srs = srs[np.isfinite(srs)]
    var_tr = float(np.var(srs))
    for r in rows:
        if not r.get("n_days"):
            continue
        d = streams[r["model"]]["net"]
        d = d[d.index >= pd.Timestamp(r["start"])]
        from scipy.stats import skew, kurtosis
        r["dsr"] = Q.deflated_sharpe(r["sharpe"], len(d), N, var_tr,
                                     float(skew(d)),
                                     float(kurtosis(d, fisher=False)))
        r["verdict"] = verdict(r, N)
        if MODELS[r["model"]]["norm"] or r["model"] in (
                "vol_target_sizing",):
            r["prop_pass"], r["prop_bust"] = prop_pass(streams[r["model"]]
                                                       ["net"])
        base = MODELS[r["model"]]["base"]
        if base:
            b = next((x for x in rows if x["model"] == base), None)
            if b and b.get("n_days"):
                r["d_sharpe_vs_base"] = r["sharpe"] - b["sharpe"]
                r["d_dev_vs_base"] = r["sharpe_dev"] - b["sharpe_dev"]
                r["d_hold_vs_base"] = r["sharpe_hold"] - b["sharpe_hold"]
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(os.path.join(OUT, "results.csv"))
    with open(os.path.join(OUT, "meta.pkl"), "wb") as f:
        pickle.dump(dict(best=best, best_sr=best_sr, p_rc=p_rc, null=null,
                         N=N, var_tr=var_tr, errors=errors), f)
    print(df[["sharpe", "sharpe_gross", "sharpe_dev", "sharpe_hold", "t",
              "alpha_t", "dsr", "verdict"]].sort_values("sharpe").to_string())
    print("reality check: best", best, f"{best_sr:.2f}", "p =", p_rc)
    print("errors:", errors)
    report()


if __name__ == "__main__":
    main()
