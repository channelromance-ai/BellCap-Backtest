"""
Try to make the pullback rule work, without fooling ourselves doing it.

Searching until something looks good always succeeds, and the thing it finds
is usually luck. The protection is procedural, and it has to be set up before
any numbers are seen:

  * The last fifteen months are locked away and not looked at. Every choice
    is made on the earlier period alone.
  * The whole search is scored against what searching alone produces, so a
    good-looking best cell has to beat the best cell a search this wide finds
    in data with no edge in it.
  * Exactly one candidate goes forward to the locked-away data, once. If it
    fails there, it failed -- going back for a second candidate is how the
    holdout stops being a holdout.

What is being changed, and why -- all three come from the original's own
trade log rather than from guessing:

  1. Shorts. The rule was long-only across two indices that rose the whole
     sample, so it could not be told apart from owning the market.
  2. Nearer targets. The original always aimed at twice its risk and reached
     it 5 times in 29; the best tenth of trades only ran about 1.7-2.0 times
     risk before the bell. Aiming shorter should convert more of them.
  3. Trailing stops, as an alternative to sitting until 15:45 -- which is how
     the largest share of trades ended.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from bcbt import metrics, twelvedata
from bcbt.strategies import pullback_v2 as v2

SPLIT = "2025-06-01"          # everything from here on is not looked at
ETFS = ["SPY", "QQQ"]
COST_BPS = 0.5
RNG = np.random.default_rng(7)

# The search space. Deliberately modest: every extra axis makes the best cell
# look better by luck alone, and the Reality Check charges for all of them.
DIRECTIONS = ["long", "both"]
RVOLS = [1.0, 1.3]
STOPS = [1.5, 2.5]
EXITS = [
    ("target 1.0R", dict(rr=1.0, trail=0.0)),
    ("target 1.5R", dict(rr=1.5, trail=0.0)),
    ("target 2.0R", dict(rr=2.0, trail=0.0)),
    ("trail 1.0R", dict(rr=0.0, trail=1.0)),
    ("trail 1.5R", dict(rr=0.0, trail=1.5)),
    ("target 2R + trail 1R", dict(rr=2.0, trail=1.0)),
]


def pooled(prepared, design_only, **kw):
    """Run one setting on both ETFs and stack the trades."""
    out = []
    for etf, df in prepared.items():
        sub = df[df.index < SPLIT] if design_only else df[df.index >= SPLIT]
        if len(sub) < 500:
            continue
        t = v2.run(etf, sub, cost_bps=COST_BPS, **kw)
        if len(t):
            out.append(t)
    return pd.concat(out) if out else pd.DataFrame(columns=v2.COLUMNS)


def score(trades):
    if trades is None or len(trades) < 20:
        return None
    r = trades["return_pct"].to_numpy()
    return dict(n=len(r), mean=r.mean(), total=r.sum(),
                win=100.0 * (r > 0).mean(),
                t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def main() -> int:
    pd.set_option("display.width", 220)
    prepared = {e: v2.prepare(twelvedata.load(e)) for e in ETFS}

    design_sessions = sum(
        df[df.index < SPLIT].index.normalize().nunique()
        for df in prepared.values())
    hold_sessions = sum(
        df[df.index >= SPLIT].index.normalize().nunique()
        for df in prepared.values())
    print(f"design: {design_sessions} ETF-sessions before {SPLIT}")
    print(f"locked: {hold_sessions} ETF-sessions from {SPLIT} -- not touched "
          f"until one candidate is chosen\n")

    # ---- baseline: the rule as it was, on the design period only
    base = pooled(prepared, True, direction="long", rvol_mult=1.3,
                  atr_sl=2.5, rr=2.0, trail=0.0)
    b = score(base)
    print("=" * 92)
    print("BASELINE (original rule), design period, SPY+QQQ pooled")
    print("=" * 92)
    print(f"  n={b['n']}  mean {b['mean']:+.4f}%  total {b['total']:+.2f}%  "
          f"win {b['win']:.1f}%  t={b['t']:+.2f}")
    print("  exits: " + ", ".join(
        f"{k}={v}" for k, v in base["reason"].value_counts().items()))

    # ---- the search
    print("\n" + "=" * 92)
    print("SEARCH, design period only")
    print("=" * 92)
    rows, per_day = [], {}
    for direction, rv, sl, (ename, ekw) in itertools.product(
            DIRECTIONS, RVOLS, STOPS, EXITS):
        t = pooled(prepared, True, direction=direction, rvol_mult=rv,
                   atr_sl=sl, **ekw)
        s = score(t)
        if s is None:
            continue
        tag = f"{direction}/rv{rv}/sl{sl}/{ename}"
        rows.append(dict(cell=tag, **s))
        d = t.copy()
        d["day"] = pd.DatetimeIndex(d["entry_time"]).normalize()
        per_day[tag] = d.groupby("day")["return_pct"].agg(["sum", "count"])

    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(f"  {len(df)} settings ran, {int((df['mean'] > 0).sum())} made money")
    print(df.head(8).round(4).to_string(index=False))
    print("  ...")
    print(df.tail(3).round(4).to_string(index=False))

    name, val, p, null = metrics.reality_check(per_day, n_boot=5000)
    print(f"\n  best on design: {name}  {val:+.4f}%")
    print(f"  what searching {len(per_day)} settings finds by luck alone: "
          f"median {np.median(null):+.4f}%, top 5% above "
          f"{np.percentile(null, 95):+.4f}%")
    print(f"  p = {p:.3f}  ", end="")
    survived = p < 0.05
    print("-> better than luck" if survived
          else "-> NOT better than luck")

    # ---- one candidate, one look
    print("\n" + "=" * 92)
    print("THE LOCKED-AWAY PERIOD -- one candidate, one look")
    print("=" * 92)
    if not survived:
        print("  The search did not beat luck on the design period, so there")
        print("  is no honest candidate to carry forward. Reporting the best")
        print("  cell on the holdout anyway, clearly labelled as a curiosity")
        print("  and not as evidence.\n")

    parts = name.split("/")
    direction = parts[0]
    rv = float(parts[1][2:])
    sl = float(parts[2][2:])
    ekw = dict(EXITS)[parts[3]]

    ins = pooled(prepared, True, direction=direction, rvol_mult=rv,
                 atr_sl=sl, **ekw)
    oos = pooled(prepared, False, direction=direction, rvol_mult=rv,
                 atr_sl=sl, **ekw)
    si, so = score(ins), score(oos)
    print(f"  candidate: {name}")
    if si:
        print(f"    design : n={si['n']:>4}  mean {si['mean']:+.4f}%  "
              f"total {si['total']:+6.2f}%  win {si['win']:4.1f}%  "
              f"t={si['t']:+5.2f}")
    if so:
        lo, hi = metrics.bootstrap_ci(
            oos["return_pct"].to_numpy() / 100.0, seed=3)
        print(f"    LOCKED : n={so['n']:>4}  mean {so['mean']:+.4f}%  "
              f"total {so['total']:+6.2f}%  win {so['win']:4.1f}%  "
              f"t={so['t']:+5.2f}")
        print(f"             95% range for the true average: "
              f"[{100 * lo:+.4f}%, {100 * hi:+.4f}%]")
        print("    exits: " + ", ".join(
            f"{k}={v}" for k, v in oos["reason"].value_counts().items()))
        by = oos.copy()
        by["side"] = by["dir"]
        g = by.groupby("side")["return_pct"].agg(["count", "mean"])
        print("    by side: " + "  ".join(
            f"{k}: n={int(v['count'])} {v['mean']:+.4f}%"
            for k, v in g.iterrows()))
    else:
        print("    LOCKED : too few trades to judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
