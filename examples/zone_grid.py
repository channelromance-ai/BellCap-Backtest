"""
The zone-retest rule across every sensible way of managing the trade.

This also answers the question directly: does aiming to keep money rather
than make it change the outcome? A small target with a high hit rate is the
preservation end; a distant target taken rarely is the other. Both are run,
and so is moving the stop to breakeven early, which is the purest expression
of "do not lose".

Design is the first 60% of each instrument's history; the rest is not opened
until one candidate has been chosen.

Held back deliberately: the headline before this was a t-statistic of 15,
produced entirely by letting trades happen inside the breakout hour before
that hour had closed. One hour of hindsight. The numbers here are what is
left once the clock is honest, and they are the real ones.
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.zone_retest import (FX, INDICES, evaluate, load,  # noqa: E402
                                  score, signals, split_date)

warnings.filterwarnings("ignore")

RR = [0.5, 1.0, 1.5, 2.0, 3.0]
BE = [0.0, 0.5, 1.0]
TFS = [5, 15]


def main() -> int:
    pd.set_option("display.width", 210)
    syms = INDICES + FX
    data = {s: load(s) for s in syms}
    cuts = {s: split_date(data[s]) for s in syms}

    print("=" * 92)
    print("DESIGN PERIOD, pooled over 2 indices and 7 currency majors")
    print("=" * 92)

    rows = []
    for tf in TFS:
        sg = {s: signals(s, data[s], entry_tf=tf) for s in syms}
        for rr, be in itertools.product(RR, BE):
            parts = []
            for s in syms:
                g = sg[s]
                g = g[g.entry_ts < cuts[s]]   # both already tz-aware
                t = evaluate(s, data[s], g, rr=rr, be_at=be)
                if len(t):
                    parts.append(t)
            if not parts:
                continue
            allt = pd.concat(parts)
            sc = score(allt, f"tf{tf}/target{rr}R/be{be}")
            if sc:
                rows.append(sc)

    d = pd.DataFrame(rows).set_index("label").sort_values("avg_r",
                                                          ascending=False)
    print(d.round(4).to_string())
    print(f"\n  settings with positive expectancy: "
          f"{int((d['avg_r'] > 0).sum())} of {len(d)}")

    print("\n" + "=" * 92)
    print("DOES AIMING TO KEEP MONEY BEAT AIMING TO MAKE IT?")
    print("=" * 92)
    print("  (5-minute entries, stop left alone, target varied)")
    print(f"  {'target':>8}{'trades':>9}{'win%':>8}{'avg R':>9}{'t':>7}"
          f"{'profit factor':>15}")
    sg5 = {s: signals(s, data[s], entry_tf=5) for s in syms}
    for rr in RR:
        parts = []
        for s in syms:
            g = sg5[s]
            g = g[g.entry_ts < cuts[s]]
            t = evaluate(s, data[s], g, rr=rr, be_at=0.0)
            if len(t):
                parts.append(t)
        sc = score(pd.concat(parts), "")
        print(f"  {rr:>7.1f}R{sc['trades']:>9}{sc['win_pct']:>8.1f}"
              f"{sc['avg_r']:>9.4f}{sc['t']:>7.2f}{sc['pf']:>15.3f}")

    print("\n  A higher hit rate is bought, not earned: a nearer target wins")
    print("  more often and wins less each time. What matters is whether the")
    print("  average trade improves, and that is the avg R column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
