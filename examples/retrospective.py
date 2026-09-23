"""
What went wrong across everything tested here, and what to do about it.

Ten strategy families have been run through this repo. None survived. That
is a legitimate outcome, but the WAY they failed repeats, and the repetition
is more useful than any single result.

Three patterns, in order of how much time they cost.

1. The edge is real and it is the size of the spread.

   Three separate ideas, arrived at independently, landed in the same place:
   a genuine effect that costs about as much to harvest as it pays. This is
   not bad luck, it is what an efficiently priced market looks like from the
   inside -- competition pushes mispricing down to the cost of removing it.
   Each one took hours to establish. `cost_feasibility` answers it in a
   second, and is the first thing that should be run on any new idea.

2. Every design-period winner failed out of sample. Every single one.

   Not "most". The best cell on the design data never once held up. In the
   worst case the cross-market signals did not merely fade, they INVERTED:
   signals that predicted falls in 1990-2010 predicted rises in 2011-2026.

3. The damaging bugs were all plumbing, and they all produced beautiful
   numbers rather than crashes.

   Hourly bar labels, timezone-aware indexes turning into objects, a
   breakeven latch not carried between windows, an ATR that made every quiet
   overnight hour look like a consolidation, a risk measure that rewarded
   sitting in cash. Not one raised an error. Every one of them made the
   result look better.

The through-line: a backtest fails loudly when the code is broken and
quietly when the METHOD is broken, so the checks have to be aimed at the
method.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import os
import sys

from bcbt import store
from bcbt.audit import cost_feasibility, report_feasibility

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


# Measured gross edge against measured cost, from the commits in this repo.
# The ratio is the whole story: above 1 the idea pays for itself.
RECORD = [
    ("index pairs, gap 1.5", 0.033, 0.058),
    ("index pairs, gap 2.5", 0.112, 0.059),
    ("index pairs, gap 3.0", 0.312, 0.059),
    ("index pairs, intraday", 0.0031, 0.0140),
    ("zone retest, wick entry", 2.700, 4.300),
    ("zone retest, close entry", 1.400, 4.000),
    ("zone retest, wide zones only", 2.200, 1.800),
    ("overnight drift, every night", 5.630, 2.000),
]

# Every candidate that was named before the holdout was opened.
HOLDOUTS = [
    ("ORB + EMA confirmation", "beat search p=0.58", "never cleared"),
    ("intraday pullback (CFD)", "+0.159%/trade", "artifact: proxy volume"),
    ("intraday pullback (real vol)", "+0.056%/trade", "-0.011%/trade"),
    ("overnight, weak-close filter", "+0.047%/night", "+0.023%/night"),
    ("defensive 200-day switch", "risk-adj 0.97", "0.72 vs hold 0.78"),
    ("trend, multi-asset", "better on 13/13", "6/13 once cash counted"),
    ("cross-market risk premium", "corr +0.215", "corr +0.002, signs flip"),
    ("CFD pair reversion", "+0.253%/trade", "+0.036%/trade"),
    ("zone retest", "best t 2.3 in-sample", "best t 0.18"),
]


def main() -> int:
    pd.set_option("display.width", 210)

    print("=" * 94)
    print("1. THE EDGE IS REAL AND IT IS THE SIZE OF THE SPREAD")
    print("=" * 94)
    d = pd.DataFrame(RECORD, columns=["idea", "gross", "cost"])
    d["ratio"] = (d["gross"] / d["cost"]).round(2)
    d["pays?"] = np.where(d["ratio"] > 1, "yes", "no")
    print(d.to_string(index=False))
    print(f"\n  ideas whose edge exceeds their cost: "
          f"{int((d['ratio'] > 1).sum())} of {len(d)}")
    print(f"  median ratio: {d['ratio'].median():.2f}")
    print("\n  Of the three that clear, one fires seven times a year, one is")
    print("  the overnight trade that needs 252 round trips, and one is the")
    print("  wide-zone filter found after the fact. None is a business.")

    print("\n" + "=" * 94)
    print("2. EVERY PRE-COMMITTED CANDIDATE FAILED OUT OF SAMPLE")
    print("=" * 94)
    h = pd.DataFrame(HOLDOUTS, columns=["candidate", "design", "holdout"])
    print(h.to_string(index=False))
    print(f"\n  survived: 0 of {len(h)}")

    print("\n" + "=" * 94)
    print("3. THE CHECK THAT SHOULD RUN FIRST, ON REAL STOPS")
    print("=" * 94)
    rows = []
    specs = [("SPX500", 0.6), ("NAS100", 1.2)]
    try:
        import examples.zone_retest as ZR
        for sym, spread in specs:
            m1 = store.load_m1(sym)
            sg = ZR.signals(sym, m1, entry_tf=5)
            if len(sg):
                rows.append(cost_feasibility(sg["risk"], spread, rr=2.0,
                                             label=f"{sym} zone stops"))
    except Exception as exc:                            # noqa: BLE001
        print(f"  (live check skipped: {exc})")

    # Illustrative stop sizes on the S&P, where the spread is 0.6 points.
    for pts in (5, 10, 20, 40, 80):
        rows.append(cost_feasibility(np.full(500, float(pts)), 0.6, rr=2.0,
                                     label=f"a {pts}-point stop"))
    print(report_feasibility(rows).to_string(index=False))
    print("\n  breakeven_win_free_pct is what a 2:1 payoff needs with no")
    print("  costs: one win in three. breakeven_win_pct is what it needs")
    print("  after the spread. The gap is what the idea must beat before it")
    print("  has earned anything at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
