"""
Troubleshoot gold_combos.py before believing it.

That run produced 57 passers against 0 by luck, 53 of them still positive
on 2024-26 -- far stronger than anything else this repo has found, which
is exactly what the old bugs looked like. Checks:

  1  Look-ahead: corrupt every price after a decision and re-run the signal
     code (bcbt.audit.future_poison). Any changed decision = peeking.
  2  Trend: split each top combination into longs and shorts, and compare
     with buying at exactly the same timestamps. Gold doubled in 2024-26.
  3  Execution: the best combinations fade sharp moves, where a closing
     quote can be momentary. Re-score with the entry delayed 1 and 2 bars.
  4  Singles: do the models already show this alone, without pairing?
  5  Year by year.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_battery as B                      # noqa: E402
import examples.gold_combos as C                       # noqa: E402
import examples.gold_zones as G                        # noqa: E402
from bcbt.audit import future_poison                   # noqa: E402

TOP = [
    ("Overextension fade (2.5 ATR from EMA20)", "Bollinger + RSI reversal", "1h"),
    ("Overextension fade (2.5 ATR from EMA20)", "VWAP 2-sigma fade", "1h"),
    ("VWAP 2-sigma fade", "Bollinger + RSI reversal", "1h"),
    ("EMA 9/21 cross", "MACD signal-line cross", "4h"),
    ("EMA 9/21 cross", "MACD zero-line cross", "to_roll"),
    ("EMA50 reclaim/rejection", "VWAP cross", "4h"),
    ("EMA50 reclaim/rejection", "RSI 50 cross", "to_roll"),
]


def pair_signals(M, a, b):
    rl = {k: pd.Series(M[k][1]).rolling(C.WINDOW, min_periods=1).max()
          .to_numpy().astype(bool) for k in (a, b)}
    rs = {k: pd.Series(M[k][2]).rolling(C.WINDOW, min_periods=1).max()
          .to_numpy().astype(bool) for k in (a, b)}
    lg = (M[a][1] & rl[b]) | (M[b][1] & rl[a])
    sh = (M[a][2] & rs[b]) | (M[b][2] & rs[a])
    return lg & ~(lg & sh), sh & ~(lg & sh)


def fwd_delayed(m15, delay, hz_bars):
    """Long-side net bp entering at the open `delay` bars later than usual."""
    o = m15["open"].to_numpy(np.float64)
    c = m15["close"].to_numpy(np.float64)
    sp = (m15["spread"] * G.POINT).to_numpy(np.float64)
    day = m15["server"].dt.normalize().to_numpy()
    n = len(c)
    last = pd.Series(np.arange(n)).groupby(day).transform("max").to_numpy()
    ei = np.clip(np.arange(n) + 1 + delay, 0, n - 1)
    xi = last if hz_bars is None else np.minimum(ei + hz_bars - 1, last)
    ok = (np.arange(n) + 1 + delay < n) & (last[ei] == last) & (xi >= ei)
    gross = 1e4 * (c[xi] / o[ei] - 1)
    cost = 1e4 * (sp[ei] + G.COMMISSION) / o[ei]
    return np.where(ok, gross - cost, np.nan), np.where(ok, -gross - cost,
                                                         np.nan), \
        np.where(ok, gross, np.nan)


def main() -> int:
    pd.set_option("display.width", 230)
    m5, m15, h4 = G.load()
    m15 = m15[m15["server"] >= m5["server"].iloc[0]].reset_index(drop=True)
    d = B.indicators(m15, h4)
    M = B.models(d)
    t = m15["server"].to_numpy()
    yr = m15["server"].dt.year.to_numpy()
    hz = C.HORIZONS

    print("=" * 100)
    print("1. LOOK-AHEAD: corrupt the future at each decision, re-run")
    print("=" * 100)
    sub = m15[(m15["server"] >= "2024-01-01") & (m15["server"] < "2024-07-01")]
    data = sub.set_index(sub["server"].dt.tz_localize("UTC"))[
        ["server", "open", "high", "low", "close", "tick_volume",
         "spread"]].copy()
    for col in ("open", "high", "low", "close", "tick_volume", "spread"):
        data[col] = data[col].astype("float64")
    for a, b, _ in TOP[:2] + TOP[3:5]:
        def fn(df, a=a, b=b):
            x = df.reset_index(drop=True).copy()
            dd = B.indicators(x, h4)
            MM = B.models(dd)
            lg, sh = pair_signals(MM, a, b)
            idx = np.flatnonzero(lg | sh)
            ts = pd.to_datetime(x["server"].to_numpy()[idx]).tz_localize("UTC")
            return pd.DataFrame(dict(entry_ts=ts,
                                     d=np.where(lg[idx], 1, -1)))
        r = future_poison(fn, data, probes=8)
        print(f"  {a[:34]:<34} + {b[:28]:<28} {'LEAK' if r['leak'] else 'clean'}"
              f"  probes {r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}"
              f"  decisions {r.get('n_clean')}")

    print("\n" + "=" * 100)
    print("2-3. TREND AND EXECUTION, per top combination (bp per signal, "
          "after cost; day-averaged)")
    print("=" * 100)
    rows = []
    for a, b, h in TOP:
        lg, sh = pair_signals(M, a, b)
        for delay in (0, 1, 2):
            L, S, gross = fwd_delayed(m15, delay, hz[h])
            il, is_ = np.flatnonzero(lg), np.flatnonzero(sh)
            vl, vs = L[il], S[is_]
            both = np.r_[vl, vs]
            # "Always long at the same timestamps": gold's own drift.
            drift = np.r_[gross[il], gross[is_]]
            rows.append(dict(
                combo=f"{a[:30]} + {b[:24]}", hz=h, delay=delay,
                longs=len(il), shorts=len(is_),
                long_bp=np.nanmean(vl), short_bp=np.nanmean(vs),
                all_bp=np.nanmean(both),
                buy_same_times_bp=np.nanmean(drift)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("4. SINGLE MODELS, no pairing (same scoring, top 12 by dev t)")
    print("=" * 100)
    fwd = C.forward_returns(m15)
    dayidx = pd.factorize(m15["server"].dt.normalize())[0]
    nd = dayidx.max() + 1
    is_dev = t < C.DEV_END
    rows = []
    for k, (_fam, lg, sh) in M.items():
        for h, f in fwd.items():
            s = C.score(np.flatnonzero(lg), np.flatnonzero(sh), f, dayidx, nd,
                        is_dev)
            rows.append(dict(model=k, hz=h, **s))
    single = pd.DataFrame(rows)
    print(single.sort_values("dev_t", ascending=False).head(12)
          [["model", "hz", "dev_n", "dev_bp", "dev_t", "val_bp", "val_t"]]
          .round(2).to_string(index=False))
    print(f"\n  single models passing t >= 3 on development: "
          f"{((single['dev_t'] >= 3) & (single['dev_bp'] > 0)).sum()} of "
          f"{len(single)}")

    print("\n" + "=" * 100)
    print("5. YEAR BY YEAR (bp per signal after cost, day-averaged)")
    print("=" * 100)
    rows = {}
    for a, b, h in TOP:
        lg, sh = pair_signals(M, a, b)
        L, S, _ = fwd_delayed(m15, 0, hz[h])
        v = np.where(lg, L, np.where(sh, S, np.nan))
        s = pd.Series(v).groupby([yr, dayidx]).mean().groupby(level=0).mean()
        rows[f"{a[:22]}+{b[:16]} {h}"] = s
    print(pd.DataFrame(rows).T.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
