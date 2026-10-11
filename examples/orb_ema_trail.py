"""
NAS100 ORB + 12 EMA with a 1R trail: the full study, in the order that
keeps it honest.

    python -m bcbt.fetch --symbols NAS100 --start 2013-01-01 --ask \\
        --out data/nas_m1_long.parquet
    python examples/orb_ema_trail.py --data data/nas_m1_long.parquet

Rules (bcbt/strategies/orb_trail.py has the full statement): range 09:30-
10:00 ET, side from the 10:00 close against the 12 EMA on five-minute bars,
stop beyond the far side of the range, breakeven at +1R then a 1R trail,
flat at the close. One trade a day.

The rules were fixed before any of this ran, so the specification is scored
first and on its own, with no tuning. Everything after that pushes on it:

  1  the data, and how many sessions it really holds
  2  the specification as written: mean R with a bootstrap interval, by
     year, by side, by exit, and how much rests on bars that cannot say
     which came first
  3  is the EMA doing anything? The same days traded long-only, short-only,
     against the EMA, and on a coin. A trailing stop makes the mirror of a
     trade a different trade (its stop is the other side of the range), so
     both sides are actually simulated every day rather than sign-flipped
  4  is the trail doing anything? The same entries under other exits
  5  is the result a peak or a plateau? A grid around the specification,
     scored with White's Reality Check, and a pick-on-the-first-half,
     judge-on-the-second test
  6  costs, and the cost at which it stops paying
  7  the optional filters from the rules, judged out of sample
  8  what it does to an account at 0.5% and 1% risk, under a leverage cap
  9  a look-ahead probe on the real data

Writes out/orb_ema_trail/<sym>/report.md, trades.csv and charts.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bcbt import audit, metrics, store                  # noqa: E402
from bcbt.fills import warm                             # noqa: E402
from bcbt.strategies import orb_trail                   # noqa: E402

SPEC = dict(or_min=30, delay=0, ema="all12", buf=0.0, act_r=1.0,
            trail_r=1.0, rr=0.0)

# Round trip in index points. Dukascopy's own NAS100 spread is ~1 point in
# New York hours; retail CFD brokers are wider, and a stop fills worse than
# its level in a fast market. 1.5 is the repo's conservative default.
COSTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


class Report:
    def __init__(self):
        self.lines = []

    def h(self, text, level=2):
        self.add(f"\n{'#' * level} {text}\n")

    def add(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, df, floatfmt=".3f"):
        """A markdown table without pulling in `tabulate`."""
        def cell(v):
            if isinstance(v, (float, np.floating)):
                return "" if not math.isfinite(v) and math.isnan(v) \
                    else format(v, floatfmt)
            return str(v)
        head = [str(df.index.name or "")] + [str(c) for c in df.columns]
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
        for k, r in df.iterrows():
            lines.append("| " + " | ".join([str(k)] + [cell(v) for v in r])
                         + " |")
        self.add("\n".join(lines))
        self.add()

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def stats(t, label=""):
    """One row of headline numbers, with the bootstrap interval."""
    if t is None or len(t) == 0:
        return dict(label=label, n=0)
    s = metrics.summary(t)
    lo, hi = metrics.bootstrap_ci(t["r"])
    return dict(label=label, n=s["n"], win_pct=s["win"], avg_r=s["avg_r"],
                ci_lo=lo, ci_hi=hi, t=s["t"], total_r=s["total_r"],
                pf=s["pf"], max_dd_r=s["max_dd_r"],
                avg_bp=float(t["bp"].mean()))


def gross(t, cost):
    """The same trades with the cost put back, for questions about signal."""
    return t.assign(r=t["r_gross"], bp=t["bp"] + 1e4 * cost / t["entry"])


def rows(items):
    return pd.DataFrame(items).set_index("label")


# ------------------------------------------------------------------ sizing


def equity_path(t, risk_pct, lev_cap=None):
    """
    Compound the trades at a fixed share of equity risked.

    Position = equity x risk / stop distance, so notional over equity is
    risk x entry / stop. A tight stop asks for a lot of leverage; with a cap
    the position is cut back and that day earns less than its R.
    """
    r = t["r"].to_numpy(float)
    scale = np.ones_like(r)
    if lev_cap is not None:
        lev = risk_pct * t["entry"].to_numpy(float) / t["risk"].to_numpy(float)
        scale = np.minimum(1.0, lev_cap / lev)
    ret = risk_pct * r * scale
    eq = np.cumprod(1.0 + ret)
    return pd.Series(eq, index=pd.to_datetime(t["day"])), float((scale < 1).mean())


def path_stats(eq):
    peak = eq.cummax()
    dd = 1.0 - eq / peak
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0
    # Longest stretch below a previous high, in calendar days.
    under = dd > 0
    longest, start = 0, None
    for ts, u in under.items():
        if u and start is None:
            start = ts
        elif not u and start is not None:
            longest = max(longest, (ts - start).days)
            start = None
    if start is not None:
        longest = max(longest, (eq.index[-1] - start).days)
    ye = eq.groupby(eq.index.year).last()
    yearly = ye / ye.shift(1).fillna(1.0)
    return dict(final_x=float(eq.iloc[-1]), cagr_pct=100 * cagr,
                max_dd_pct=100 * float(dd.max()), longest_dd_days=longest,
                worst_year_pct=100 * float((yearly - 1).min()))


def mc_drawdowns(t, risk_pct, lev_cap, n=5000, seed=0):
    """Max drawdown over reshuffled trade orders: how bad it can plausibly get."""
    r = t["r"].to_numpy(float)
    lev = risk_pct * t["entry"].to_numpy(float) / t["risk"].to_numpy(float)
    ret = risk_pct * r * np.minimum(1.0, lev_cap / lev)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        eq = np.cumprod(1.0 + rng.choice(ret, size=len(ret), replace=True))
        out[i] = 1.0 - (eq / np.maximum.accumulate(eq)).min()
    return 100 * np.percentile(out, [50, 95, 99])


def losing_streak(r):
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


# ------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nas_m1_long.parquet")
    p.add_argument("--sym", default="NAS100")
    p.add_argument("--cost", type=float, default=1.5)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--lev-cap", type=float, default=20.0)
    p.add_argument("--start", default=None, help="first session to trade")
    p.add_argument("--out", default="out/orb_ema_trail")
    a = p.parse_args(argv)

    pd.set_option("display.width", 220)
    warm()
    out = os.path.join(a.out, a.sym)
    os.makedirs(out, exist_ok=True)
    R = Report()
    t0 = time.time()

    m1 = store.load_m1(a.sym, src=a.data)
    days, arr = orb_trail.prepare(m1)
    # Let the EMA settle on a month of history before the first trade.
    first_ok = sorted(days)[min(20, len(days) - 1)]
    if a.start:
        first_ok = max(first_ok, pd.Timestamp(a.start, tz=store.ET))
    days = {k: v for k, v in days.items() if k >= first_ok}
    ranges = orb_trail.opening_ranges(days, 30)
    cost = a.cost

    def run(**kw):
        cfg = dict(SPEC)
        cfg.update(kw)
        c = cfg.pop("cost", cost)
        rg = ranges if cfg["or_min"] == 30 else None
        return orb_trail.run(a.sym, days, arr, c, ranges=rg, **cfg)

    R.add(f"# {a.sym} ORB + 12 EMA, 1R trail: backtest report")
    R.add()
    R.add(f"Data: `{a.data}` (Dukascopy {a.sym} CFD, bid, 1-minute). "
          f"Fills on 1-minute bars; signals on 5-minute bars. Cost "
          f"{cost} index points per round trip unless stated. Generated "
          f"{pd.Timestamp.now():%Y-%m-%d %H:%M}.")

    # ------------------------------------------------------------- 1 data
    R.h("1. Data")
    sess = sorted(days)
    complete = sum(v is not None for v in ranges.values())
    yrs = pd.Series([d.year for d in sess]).value_counts().sort_index()
    rth_min = pd.Series({d: days[d]["i1"] - days[d]["i0"] + 1 for d in sess})
    exp_min = pd.Series({d: days[d]["close_min"] - days[d]["open_min"] for d in sess})
    cover = (rth_min / exp_min).groupby([d.year for d in sess]).mean()
    rng = pd.Series({d: v[0] - v[1] for d, v in ranges.items() if v is not None})
    R.add(f"- Sessions: {len(sess)} real NYSE sessions, "
          f"{sess[0].date()} to {sess[-1].date()}")
    R.add(f"- Sessions with all six range bars present: {complete} "
          f"({len(sess) - complete} dropped for feed gaps)")
    R.add(f"- Opening ranges of zero width (a frozen feed): "
          f"{int((rng <= 0).sum())}")
    R.add(f"- Opening range width, points: median {rng.median():.1f}, "
          f"10th pct {rng.quantile(.1):.1f}, 90th pct {rng.quantile(.9):.1f}")
    R.add()
    R.table(pd.DataFrame({"sessions": yrs,
                          "minute_coverage_pct": 100 * cover,
                          "median_or_pts": rng.groupby(rng.index.year).median()}),
            floatfmt=".1f")

    # ------------------------------------------------------------- 2 spec
    R.h("2. The specification, as written")
    spec = run()
    spec.to_csv(os.path.join(out, "trades.csv"), index=False)
    opt = run(stop_first=False)
    R.table(rows([stats(spec, "spec (stop-first)"),
                  stats(opt, "spec (stop-last)")]), floatfmt=".3f")
    R.add("*stop-first* reads every ambiguous minute bar against the trade; "
          "*stop-last* reads it in the trade's favour. The truth lies "
          f"between. {int(spec['ambiguous'].sum())} of {len(spec)} trades "
          "touch such a bar.")
    R.add()

    yr = spec.assign(year=pd.to_datetime(spec["day"]).dt.year)
    R.add("**By year**")
    R.table(rows([stats(g, y) for y, g in yr.groupby("year")]))
    R.add("**By side**")
    R.table(rows([stats(g, s) for s, g in spec.groupby("dir")]))
    R.add("**By exit**")
    ex = spec.groupby("reason").agg(n=("r", "size"), avg_r=("r", "mean"),
                                    total_r=("r", "sum"))
    ex["share_pct"] = 100 * ex["n"] / len(spec)
    R.table(ex)

    reach = {f"reached +{k}R": 100 * (spec["mfe_r"] >= k).mean()
             for k in (0.5, 1, 2, 3, 5)}
    R.add("**How far trades ran** (% of trades whose best excursion got there): "
          + ", ".join(f"{k} {v:.1f}%" for k, v in reach.items()))
    R.add()
    risk = spec["risk"]
    R.add(f"**Stop distance (1R)**: median {risk.median():.1f} pts, "
          f"10th pct {risk.quantile(.1):.1f}, 90th pct {risk.quantile(.9):.1f}. "
          f"Cost per trade: median {(cost / risk).median():.3f}R, mean "
          f"{(cost / risk).mean():.3f}R. "
          f"{int((risk < 10).sum())} trades had a stop under 10 points.")
    R.add()
    srt = spec.sort_values("day")
    h1, h2 = srt.iloc[:len(srt) // 2], srt.iloc[len(srt) // 2:]
    R.add("**Halves**")
    R.table(rows([stats(h1, f"{h1['day'].iloc[0]} to {h1['day'].iloc[-1]}"),
                  stats(h2, f"{h2['day'].iloc[0]} to {h2['day'].iloc[-1]}")]))

    # ------------------------------------------------------------- 3 direction
    R.h("3. Does the EMA call the direction?")
    R.add("Judged on **gross** R, before costs. The two sides of one day do "
          "not have the same stop: when the 10:00 close is above the EMA it "
          "is usually near the top of the range, so the long stop is far and "
          "the short stop is near. A fixed cost is then a small slice of the "
          "long's R and a large slice of the short's, which makes any side "
          "the EMA picks look better *net* even on a random walk -- this "
          "test was checked against one and failed exactly that way before "
          "it was moved to gross. The cost side of that geometry is real "
          "and is in every other section; it just is not a forecast.")
    R.add()
    lg, sh = run(force=1), run(force=-1)
    anti = run(force=-2)
    both = sorted(set(lg["day"]) & set(sh["day"]) & set(spec["day"]))
    L = lg.set_index("day").loc[both, "r_gross"].to_numpy()
    S = sh.set_index("day").loc[both, "r_gross"].to_numpy()
    pick = (spec.set_index("day").loc[both, "dir"] == "long").to_numpy()
    rule_mean = float(np.where(pick, L, S).mean())
    rng_ = np.random.default_rng(1)
    draws = np.array([np.where(rng_.random(len(both)) < .5, L, S).mean()
                      for _ in range(20000)])
    p_dir = float((1 + (draws >= rule_mean).sum()) / (1 + len(draws)))
    # The coin's own share of longs matters in a rising market, so also
    # compare against coins that go long as often as the EMA does.
    q = pick.mean()
    draws_q = np.array([np.where(rng_.random(len(both)) < q, L, S).mean()
                        for _ in range(20000)])
    p_dir_q = float((1 + (draws_q >= rule_mean).sum()) / (1 + len(draws_q)))
    side_rows = []
    for t, name in ((spec, "EMA side (the rule)"), (anti, "opposite of the EMA"),
                    (lg, "always long"), (sh, "always short")):
        g = stats(gross(t, cost), name)
        side_rows.append(dict(label=name, n=g["n"], gross_avg_r=g["avg_r"],
                              gross_ci_lo=g["ci_lo"], gross_ci_hi=g["ci_hi"],
                              gross_avg_bp=g["avg_bp"],
                              net_avg_r=float(t["r"].mean()),
                              median_risk_pts=float(t["risk"].median())))
    R.table(rows(side_rows))
    R.add(f"Random side on the same {len(both)} days, gross: mean "
          f"{draws.mean():+.4f}R (sd {draws.std():.4f}); the rule "
          f"{rule_mean:+.4f}R sits at the {100 * (draws < rule_mean).mean():.1f}th "
          f"percentile, p = {p_dir:.4f}.")
    R.add(f"Against a coin that goes long {100 * q:.0f}% of the time, as the "
          f"EMA does: p = {p_dir_q:.4f}.")
    # R divides by the stop, so on the tight-stop side every point of
    # friction -- and every ambiguous bar read against the trade -- is
    # magnified. Index basis points are the same trades in a unit that
    # treats both sides alike.
    Lb = lg.set_index("day").loc[both, "bp"].to_numpy() + \
        1e4 * cost / lg.set_index("day").loc[both, "entry"].to_numpy()
    Sb = sh.set_index("day").loc[both, "bp"].to_numpy() + \
        1e4 * cost / sh.set_index("day").loc[both, "entry"].to_numpy()
    rule_bp = float(np.where(pick, Lb, Sb).mean())
    draws_b = np.array([np.where(rng_.random(len(both)) < .5, Lb, Sb).mean()
                        for _ in range(20000)])
    p_dir_bp = float((1 + (draws_b >= rule_bp).sum()) / (1 + len(draws_b)))
    R.add(f"The same test in gross index basis points per trade, which does "
          f"not magnify friction on tight stops: rule {rule_bp:+.2f}bp, "
          f"coin {draws_b.mean():+.2f}bp, p = {p_dir_bp:.4f}.")
    R.add()

    # ------------------------------------------------------------- 4 exits
    R.h("4. Does the trail add anything? Same entries, other exits")
    mg = [
        ("spec: BE at 1R, trail 1R", {}),
        ("initial stop only, out at the close", dict(act_r=0.0)),
        ("BE at 1R, no trail", dict(trail_r=0.0)),
        ("BE at 1R, trail 0.5R", dict(trail_r=0.5)),
        ("BE at 1R, trail 1.5R", dict(trail_r=1.5)),
        ("BE at 1R, trail 2R", dict(trail_r=2.0)),
        ("trail 1R from +0.5R", dict(act_r=0.5, trail_r=1.0)),
        ("trail 1R from +2R", dict(act_r=2.0, trail_r=1.0)),
        ("fixed target 1R", dict(act_r=0.0, rr=1.0)),
        ("fixed target 2R", dict(act_r=0.0, rr=2.0)),
        ("fixed target 3R", dict(act_r=0.0, rr=3.0)),
    ]
    mres, mday = [], {}
    for name, kw in mg:
        t = run(**kw)
        mres.append(stats(t, name))
        mday[name] = metrics.per_day_table(t)
    R.table(rows(mres))
    nm, val, p_m, _ = metrics.reality_check(mday, n_boot=5000)
    R.add(f"Best exit: {nm} at {val:+.4f}R; Reality Check over these "
          f"{len(mday)} exits p = {p_m:.3f}.")

    # ------------------------------------------------------------- 5 grid
    R.h("5. Peak or plateau? A grid around the specification")
    grid = dict(
        or_min=[15, 30, 60],
        delay=[0, 5],
        ema=["all9", "all12", "all20", "rth12"],
        mgmt=[(1.0, 1.0), (1.0, 0.0), (1.0, 2.0), (0.0, 0.0)],
    )
    cells, per_day, per_day_h1, per_day_h2 = [], {}, {}, {}
    split = h2["day"].iloc[0]
    for om, dl, em, (ar, tr) in itertools.product(*grid.values()):
        t = run(or_min=om, delay=dl, ema=em, act_r=ar, trail_r=tr)
        if len(t) < 50:
            continue
        tag = f"OR{om}/+{dl}m/{em}/act{ar:g}-trail{tr:g}"
        s = stats(t, tag)
        a1, a2 = t[t["day"] < split], t[t["day"] >= split]
        s["h1_avg_r"], s["h2_avg_r"] = a1["r"].mean(), a2["r"].mean()
        cells.append(s)
        per_day[tag] = metrics.per_day_table(t)
        per_day_h1[tag] = metrics.per_day_table(a1)
        per_day_h2[tag] = metrics.per_day_table(a2)
    G = rows(cells).sort_values("avg_r", ascending=False)
    spec_tag = "OR30/+0m/all12/act1-trail1"
    rank = list(G.index).index(spec_tag) + 1
    R.add(f"{len(G)} cells. Positive mean R in {int((G['avg_r'] > 0).sum())}; "
          f"lower bootstrap bound above zero in {int((G['ci_lo'] > 0).sum())}. "
          f"The specification ranks {rank} of {len(G)}.")
    R.add()
    R.table(G.head(12)[["n", "win_pct", "avg_r", "ci_lo", "ci_hi", "t",
                        "h1_avg_r", "h2_avg_r"]])
    nm, val, p_g, null = metrics.reality_check(per_day, n_boot=5000)
    R.add(f"Reality Check over all {len(per_day)} cells: best {nm} at "
          f"{val:+.4f}R; the best of {len(per_day)} cells on edgeless data "
          f"has median {np.median(null):+.4f}R; p = {p_g:.3f}.")
    best_h1 = max(per_day_h1, key=lambda k: per_day_h1[k]["sum"].sum()
                  / max(per_day_h1[k]["count"].sum(), 1))
    oos = per_day_h2[best_h1]
    R.add(f"Picked on the first half: {best_h1} "
          f"({G.loc[best_h1, 'h1_avg_r']:+.4f}R). In the second half it made "
          f"{oos['sum'].sum() / oos['count'].sum():+.4f}R per trade, against "
          f"{G.loc[spec_tag, 'h2_avg_r']:+.4f}R for the specification.")
    G.to_csv(os.path.join(out, "grid.csv"))

    # ------------------------------------------------------------- 6 costs
    R.h("6. Costs")
    cr = [stats(run(cost=c), f"{c:g} pts") for c in COSTS]
    R.table(rows(cr))
    inv = 1.0 / spec["risk"]
    be_cost = float((spec["r_gross"]).sum() / inv.sum())
    R.add(f"Break-even round-trip cost: {be_cost:.2f} index points per trade.")

    # ------------------------------------------------------------- 7 filters
    R.h("7. The optional filters, judged out of sample")
    R.add("Range width is today's 09:30-10:00 range over the median of the "
          "previous 20. Each filter is judged on whether it helps in both "
          "halves, not on its full-sample number.")
    R.add()
    ratios = orb_trail.range_ratio(ranges, 20)
    spec_r = spec.assign(or_ratio=spec["day"].map(
        {k.date(): v for k, v in ratios.items()}))
    spec_r["q"] = pd.qcut(spec_r["or_ratio"], 5,
                          labels=["narrowest", "2", "3", "4", "widest"])
    R.add("**Specification by range width (quintiles of today / 20-day median)**")
    R.table(rows([stats(g, str(qq)) for qq, g in
                  spec_r.groupby("q", observed=True)]))
    spec_r["rq"] = pd.qcut(spec_r["risk"], 5, labels=["tightest", "2", "3",
                                                       "4", "widest"])
    R.add("**Specification by stop distance (quintiles)**")
    R.table(rows([stats(g, str(qq)) for qq, g in
                  spec_r.groupby("rq", observed=True)]))
    fl = [("none", {}),
          ("range 0.5-2x median", dict(or_band=(0.5, 2.0))),
          ("range 0.67-1.5x median", dict(or_band=(0.67, 1.5))),
          ("range under median", dict(or_band=(0.0, 1.0))),
          ("range over median", dict(or_band=(1.0, 99.0))),
          ("stop at least 20 pts", dict(min_risk=20.0))]
    fr = []
    for name, kw in fl:
        t = run(ratios=ratios if "or_band" in kw else None, **kw)
        s = stats(t, name)
        s["h1_avg_r"] = t[t["day"] < split]["r"].mean()
        s["h2_avg_r"] = t[t["day"] >= split]["r"].mean()
        fr.append(s)
    R.table(rows(fr)[["n", "avg_r", "ci_lo", "ci_hi", "h1_avg_r", "h2_avg_r"]])
    R.add("The news filter is not tested: it needs an event calendar this "
          "repo does not have.")

    # ------------------------------------------------------------- 8 sizing
    R.h("8. On an account")
    R.add(f"Compounded at a fixed share of equity risked per trade, cost "
          f"{cost} pts. Leverage cap {a.lev_cap:g}:1 -- when the stop is so "
          f"tight that the risk budget needs more notional than that, the "
          f"position is cut back.")
    R.add()
    acc = []
    for rp in (0.005, 0.01):
        for cap in (None, a.lev_cap):
            eq, bound = equity_path(spec, rp, cap)
            s = path_stats(eq)
            s["label"] = f"{100 * rp:g}% risk, " + ("no cap" if cap is None
                                                    else f"cap {cap:g}:1")
            s["capped_trades_pct"] = 100 * bound
            if cap is not None:
                q50, q95, q99 = mc_drawdowns(spec, rp, cap)
                s["mc_dd_median_pct"], s["mc_dd_p95_pct"] = q50, q95
            acc.append(s)
    R.table(rows(acc), floatfmt=".2f")
    R.add(f"Longest run of losing trades: {losing_streak(spec['r'])}.")
    lev = a.risk * spec["entry"] / spec["risk"]
    R.add(f"Notional needed at {100 * a.risk:g}% risk: median "
          f"{lev.median():.1f}x equity, 90th pct {lev.quantile(.9):.1f}x, "
          f"max {lev.max():.0f}x.")
    R.add(f"On Nasdaq futures a 1R of {risk.median():.0f} points is "
          f"${20 * risk.median():,.0f} a contract on NQ and "
          f"${2 * risk.median():,.0f} on MNQ, so 1% risk with one MNQ needs "
          f"about ${200 * risk.median():,.0f} of equity.")

    # ------------------------------------------------------------- 9 audit
    R.h("9. Look-ahead probe")
    tail = m1[m1.index >= m1.index[-1] - pd.Timedelta(days=200)]

    def decide(frame):
        d, ar_ = orb_trail.prepare(frame)
        t = orb_trail.run(a.sym, d, ar_, cost)
        return t[["entry_ts", "entry", "stop", "risk", "sig_ema"]].assign(
            side=np.where(t["dir"] == "long", 1.0, -1.0))

    rep = audit.future_poison(decide, tail, probes=8)
    R.add(f"Corrupted everything after {rep.get('probes_tried', 1)} of the "
          f"last 200 days' entries and re-ran: "
          + ("**a decision changed -- something reads ahead**: "
             + rep["detail"] if rep["leak"]
             else "no decision made at or before the cut changed."))

    # ------------------------------------------------------------- verdict
    R.h("Verdict")
    s0 = stats(spec)
    R.add(f"- Specification: {s0['n']} trades, mean {s0['avg_r']:+.4f}R, "
          f"95% CI [{s0['ci_lo']:+.4f}, {s0['ci_hi']:+.4f}]"
          + (" -- **the interval contains zero**" if s0["ci_lo"] < 0 < s0["ci_hi"]
             else ""))
    R.add(f"- Halves: {h1['r'].mean():+.4f}R then {h2['r'].mean():+.4f}R")
    R.add(f"- EMA direction vs a coin on the same days (gross R): "
          f"p = {p_dir:.4f}; in basis points p = {p_dir_bp:.4f}")
    R.add(f"- Grid of {len(per_day)} variants, Reality Check p = {p_g:.3f}")
    R.add(f"- Break-even cost {be_cost:.2f} pts against {cost} assumed")
    R.add(f"- Ambiguous-bar spread: {stats(opt)['avg_r'] - s0['avg_r']:+.4f}R")

    _charts(spec, lg, sh, anti, out)
    R.add()
    R.add(f"_Ran in {time.time() - t0:.0f}s._")
    R.save(os.path.join(out, "report.md"))
    print(f"\nwrote {out}/report.md")


def _charts(spec, lg, sh, anti, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for t, name, lw in ((spec, "rule (EMA side)", 2.0),
                        (anti, "opposite of EMA", 1.0),
                        (lg, "always long", 1.0), (sh, "always short", 1.0)):
        e = metrics.equity(t)
        ax.plot(e.index, e.values, lw=lw, label=name)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("cumulative R (net of costs)")
    ax.legend()
    ax.set_title("NAS100 ORB + 12 EMA, 1R trail")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "equity.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(spec["r"].clip(-2, 8), bins=80)
    ax.set_xlabel("R per trade (clipped to -2..8)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r_hist.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
