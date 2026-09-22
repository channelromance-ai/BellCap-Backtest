"""
A worked search, end to end, showing the shape every study should take.

Run it with:  python examples/search.py

It sweeps the ORB rule over a grid, prints the table everyone wants to see,
and then prints the two numbers that decide whether the table means
anything: the bootstrap interval on the best cell, and the Reality Check
p-value for the search as a whole.

The point of keeping this in the repo is the order of operations. Pick the
best cell first and you will always find one. Score the search first and
you find out whether picking was worth doing.
"""
from __future__ import annotations

import itertools
import pandas as pd

from bcbt import metrics, store
from bcbt.strategies import orb_ema

# Measured from Dukascopy ask data, not assumed. The Nasdaq is wider
# overnight (~1.46) than during New York hours (~0.96); this uses the
# conservative end.
COST = {"SPX500": 0.6, "NAS100": 1.5}

OR_MINS = [5, 15, 30]
STOPS = [("or", {}), ("bar", dict(buffer_frac=0.10)), ("atr", dict(atr_k=1.0))]
EXITS = [("1R", dict(rr=1.0)), ("1.5R", dict(rr=1.5)), ("2R", dict(rr=2.0)),
         ("EMA", dict(rr=None, ema_exit=True))]


def main():
    pd.set_option("display.width", 200)

    for sym in ("SPX500", "NAS100"):
        m1 = store.load_m1(sym)
        days, arr = orb_ema.prepare(m1)
        print(f"\n{'=' * 78}\n{sym}: {len(days)} sessions\n{'=' * 78}")

        rows, per_day = [], {}
        for om, (sm, skw), (en, ekw) in itertools.product(
                OR_MINS, STOPS, EXITS):
            t = orb_ema.run(sym, days, arr, COST[sym], or_min=om,
                            stop_mode=sm, **skw, **ekw)
            if t.empty:
                continue
            tag = f"OR{om}/{sm}/{en}"
            s = metrics.summary(t)
            s["cell"] = tag
            rows.append(s)
            per_day[tag] = metrics.per_day_table(t)

        df = pd.DataFrame(rows).set_index("cell").sort_values(
            "avg_r", ascending=False)
        print(df.round(3).to_string())

        best_cell = df.index[0]
        best = orb_ema.run(
            sym, days, arr, COST[sym],
            **_kw(best_cell))
        lo, hi = metrics.bootstrap_ci(best["r"])
        print(f"\nbest cell {best_cell}: mean R {best['r'].mean():+.4f}, "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        if lo < 0 < hi:
            print("  the interval contains zero, so the cell is not "
                  "distinguishable from no edge")

        name, val, p, null = metrics.reality_check(per_day)
        print(f"\nReality Check over {len(per_day)} cells")
        print(f"  best            : {name}  mean R {val:+.4f}")
        print(f"  null best median: {pd.Series(null).median():+.4f}")
        print(f"  p               : {p:.3f}", end="  ")
        print("survives the search" if p < 0.05
              else "-> indistinguishable from search noise")

        flip = metrics.coin_flip(best, COST[sym])
        print(f"\ncoin flip on the same entries: rule {flip['real']:+.4f} vs "
              f"random {flip['flip_mean']:+.4f}  p={flip['p']:.3f}")


def _kw(cell):
    om, sm, en = cell.split("/")
    kw = dict(or_min=int(om[2:]), stop_mode=sm)
    kw.update(dict(STOPS)[sm])
    kw.update(dict(EXITS)[en])
    return kw


if __name__ == "__main__":
    main()
