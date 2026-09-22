"""
Does the pullback edge survive real exchange volume?

The rule's whole result came from one filter. Every EMA-50 cross passing the
macro test and the entry window returned +0.018% over 283 trades; the 45 that
also cleared RVOL > 1.3x and the ADR veto returned +0.159%. So the strategy
is, in effect, a volume filter with a crossover attached.

That was measured on a CFD feed whose "volume" is broker-side activity, not
exchange prints. A 55-day check against E-mini futures volume put the RVOL
correlation at 0.72 and found only 68% of proxy-flagged bars were also
flagged on real volume, which leaves roughly a third of the entries in doubt.

SPY and QQQ settle it. They track the same indices, trade with real
consolidated volume, and cover the same three years. Three questions:

  1. How well does the proxy agree with real volume over the FULL sample,
     rather than the 55 days yfinance serves?
  2. Run end to end on real prints, does the edge survive -- with the same
     controls, the same parameter search and the same Reality Check?
  3. Do the two agree trade by trade, or is the CFD result selecting a
     different set of bars entirely?

Run:  python examples/validate_real_volume.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

from bcbt import metrics, twelvedata

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "standalone"))
sys.path.insert(0, _ROOT)          # so `examples` resolves when run directly

import intraday_pullback as ip  # noqa: E402

from examples.pullback_real_data import cost_bps_for, rth_5min  # noqa: E402

PAIRS = [("SPX500", "SPY"), ("NAS100", "QQQ")]

# Round-trip friction for the ETFs: a penny spread on a ~$550 SPY is under
# 0.2bp, but retail pays commission and rarely gets the touch, so 1bp round
# trip (cost_bps=0.5 one way) is the honest conservative figure.
ETF_COST_BPS = 0.5

RNG = np.random.default_rng(31337)
N_DRAWS = 400


def random_control(prepared, cfg, n, mask, draws=N_DRAWS):
    """Same trade count and risk model, random timing inside `mask`."""
    sl = prepared["sl_dist"].to_numpy(float)
    pool = np.flatnonzero(mask & np.isfinite(sl) & (sl > 0))
    if pool.size < n or n == 0:
        return np.array([])
    frame = prepared.copy()
    out = np.empty(draws)
    for i in range(draws):
        flag = np.zeros(len(frame), bool)
        flag[RNG.choice(pool, size=n, replace=False)] = True
        frame["entry_signal"] = flag
        t = ip.run_backtest(frame, cfg)
        out[i] = t["return_pct"].mean() if len(t) else np.nan
    return out[np.isfinite(out)]


def describe(tag, trades):
    if trades.empty:
        print(f"  {tag}: no trades")
        return None
    r = trades["return_pct"].to_numpy()
    lo, hi = metrics.bootstrap_ci(r / 100.0, seed=1)
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    print(f"  {tag}: n={len(r):>4}  mean {r.mean():+.4f}%  "
          f"total {r.sum():+.2f}%  win {100 * (r > 0).mean():.1f}%  "
          f"t={t_stat:+.2f}")
    print(f"        95% CI [{100 * lo:+.4f}%, {100 * hi:+.4f}%]   "
          f"expectancy {trades['r_multiple'].mean():+.3f}R")
    return r


def main() -> int:
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------- Q1
    print("=" * 76)
    print("1. PROXY vs REAL VOLUME, full sample")
    print("=" * 76)
    for cfd, etf in PAIRS:
        cfd_bars = rth_5min(cfd)
        etf_bars = twelvedata.load(etf)
        j = pd.DataFrame({
            "proxy": cfd_bars["volume"],
            "real": etf_bars["volume"],
        }).dropna()
        j = j[(j["proxy"] > 0) & (j["real"] > 0)]
        j["slot"] = j.index.hour * 60 + j.index.minute

        for col in ("proxy", "real"):
            base = j.groupby("slot")[col].transform(
                lambda s: s.shift(1).rolling(20, min_periods=20).mean())
            j[col + "_rvol"] = j[col] / base
        k = j.dropna()

        lvl = np.corrcoef(np.log(k["proxy"]), np.log(k["real"]))[0, 1]
        rv = np.corrcoef(k["proxy_rvol"], k["real_rvol"])[0, 1]
        a, b = k["proxy_rvol"] > 1.3, k["real_rvol"] > 1.3
        print(f"\n  {cfd} vs {etf}: {len(k):,} overlapping 5-min bars")
        print(f"    log-volume correlation : {lvl:.3f}")
        print(f"    RVOL correlation       : {rv:.3f}")
        print(f"    flagged by proxy       : {100 * a.mean():.1f}%")
        print(f"    flagged by real        : {100 * b.mean():.1f}%")
        print(f"    agree (both or neither): {100 * (a == b).mean():.1f}%")
        print(f"    proxy-flagged that are also real-flagged: "
              f"{100 * b[a].mean():.1f}%")

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 76)
    print("2. THE RULE ON REAL PRINTS")
    print("=" * 76)
    results = {}
    for _cfd, etf in PAIRS:
        bars = twelvedata.load(etf)
        cfg = ip.Config(cost_bps=ETF_COST_BPS)
        prep = ip.prepare_data(bars, cfg)
        trades = ip.run_backtest(prep, cfg)
        results[etf] = (bars, prep, trades, cfg)

        print(f"\n  --- {etf} ({len(bars):,} bars, "
              f"{bars.index.normalize().nunique()} sessions) ---")
        r = describe(etf, trades)
        if r is None:
            continue
        print("        exits: " + ", ".join(
            f"{k2}={v}" for k2, v in trades["reason"].value_counts().items()))

        # Same funnel as the CFD run, to see whether the filter bites the
        # same way on real prints.
        macro = prep["macro_bull"] & prep["cross_up"]
        inwin = macro & prep["in_window"]
        print(f"        funnel: cross {int(prep['cross_up'].sum())} -> "
              f"macro {int(macro.sum())} -> window {int(inwin.sum())} -> "
              f"RVOL {int((inwin & prep['rvol_ok']).sum())} -> "
              f"ADR {int(prep['raw_signal'].sum())}")

        n_sig = int(prep["entry_signal"].sum())
        ctrl_a = random_control(prep, cfg, n_sig,
                                prep["in_window"].to_numpy(bool))
        ctrl_b = random_control(
            prep, cfg, n_sig,
            (prep["in_window"] & prep["macro_bull"]).to_numpy(bool))
        loose = ip.Config(cost_bps=ETF_COST_BPS, rvol_mult=0.0,
                          adr_exhaustion=99.0)
        t_c = ip.run_backtest(ip.prepare_data(bars, loose), loose)

        mean = trades["return_pct"].mean()
        for label, arr in (("A random in-window     ", ctrl_a),
                           ("B random while macro OK", ctrl_b)):
            if arr.size:
                p = (1 + (arr >= mean).sum()) / (1 + arr.size)
                print(f"        {label}: {arr.mean():+.4f}% "
                      f"(sd {arr.std():.4f})  p={p:.3f}")
        if len(t_c):
            print(f"        C no vetoes            : "
                  f"{t_c['return_pct'].mean():+.4f}%  (n={len(t_c)})")

        # By year and out of sample.
        t = trades.copy()
        t["entry_time"] = pd.DatetimeIndex(t["entry_time"])
        by = t.groupby(t["entry_time"].dt.year)["return_pct"].agg(
            ["count", "mean"])
        print("        by year: " + "  ".join(
            f"{y}: n={int(v['count'])} {v['mean']:+.3f}%"
            for y, v in by.iterrows()))
        ins = t[t["entry_time"] < "2025-06-01"]["return_pct"]
        oos = t[t["entry_time"] >= "2025-06-01"]["return_pct"]
        if len(ins) > 4 and len(oos) > 4:
            print(f"        in-sample  n={len(ins):>3} {ins.mean():+.4f}%   "
                  f"out-sample n={len(oos):>3} {oos.mean():+.4f}%")

    # ------------------------------------------------- parameter search
    print("\n" + "=" * 76)
    print("PARAMETER SEARCH + REALITY CHECK, real volume")
    print("=" * 76)
    grid = list(itertools.product(
        (1.0, 1.3, 1.8), (0.70, 0.85, 1.00),
        ((2.0, 4.0), (2.5, 5.0), (3.0, 6.0)), (20, 50)))
    for _, etf in PAIRS:
        bars = results[etf][0]
        rows, per_day = [], {}
        for rv, adr, (sl, tp), fast in grid:
            cfg = ip.Config(cost_bps=ETF_COST_BPS, rvol_mult=rv,
                            adr_exhaustion=adr, atr_sl_mult=sl,
                            atr_tp_mult=tp, ema_fast=fast)
            t = ip.run_backtest(ip.prepare_data(bars, cfg), cfg)
            if len(t) < 30:
                continue
            tag = f"rv{rv}/adr{adr}/{sl}-{tp}/ema{fast}"
            rows.append(dict(cell=tag, n=len(t),
                             mean_pct=t["return_pct"].mean(),
                             win=100.0 * (t["return_pct"] > 0).mean()))
            d = t.copy()
            d["day"] = pd.DatetimeIndex(d["entry_time"]).normalize()
            per_day[tag] = d.groupby("day")["return_pct"].agg(["sum", "count"])

        if len(per_day) < 2:
            print(f"\n  {etf}: too few populated cells")
            continue
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        print(f"\n  --- {etf}: {len(df)} cells, "
              f"{int((df['mean_pct'] > 0).sum())} positive ---")
        print(df.head(5).round(4).to_string(index=False))
        name, val, p, null = metrics.reality_check(per_day, n_boot=4000)
        print(f"    Reality Check: best {name} {val:+.4f}%")
        print(f"      null median {np.median(null):+.4f}%, "
              f"95th {np.percentile(null, 95):+.4f}%   p={p:.3f}  ", end="")
        print("SURVIVES" if p < 0.05 else "-> search noise")

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 76)
    print("3. DO THE TWO DATASETS PICK THE SAME TRADES?")
    print("=" * 76)
    for cfd, etf in PAIRS:
        cfd_bars = rth_5min(cfd)
        ccfg = ip.Config(cost_bps=cost_bps_for(cfd, cfd_bars))
        c_tr = ip.run_backtest(ip.prepare_data(cfd_bars, ccfg), ccfg)
        e_tr = results[etf][2]
        if c_tr.empty or e_tr.empty:
            continue
        cset = set(pd.DatetimeIndex(c_tr["entry_time"]))
        eset = set(pd.DatetimeIndex(e_tr["entry_time"]))
        both = cset & eset
        print(f"  {cfd} {len(cset)} trades, {etf} {len(eset)} trades, "
              f"same entry bar: {len(both)} "
              f"({100 * len(both) / max(len(cset), 1):.0f}% of the CFD set)")
        cd = set(pd.DatetimeIndex(c_tr["entry_time"]).normalize())
        ed = set(pd.DatetimeIndex(e_tr["entry_time"]).normalize())
        print(f"    same session: {len(cd & ed)} of {len(cd)} CFD sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
