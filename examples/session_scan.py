"""
Open in Asia, close in New York: is there a direction worth holding?

No signal, no BellCap -- just the clock. For every FX pair and gold on The5ers'
server, and for each window below, the direction is chosen on 2000-2012 and
judged on 2013-2026, which it never saw. Longs and shorts are equally
eligible; whichever way the window moved in the design years is the bet.

  entry (New York)  19:00, 23:00          = 18:00, 22:00 Winnipeg
  exit  (New York)  08:00, 10:00, 12:00, 16:00  (next day)
                                          = 07:00, 09:00, 11:00, 15:00 Wpg

Every window sits inside one The5ers trading day (19:00 to 16:00 New York
never crosses the 17:00 rollover), so no swap is charged. Costs are The5ers'
measured spread plus $4 a lot commission.

About 30 instruments x 8 windows is ~240 tests, so a handful will look
excellent by luck. The bar is set for that, before looking:

  holdout (2013-26) net positive with t above 3.5 -- roughly what 240
  tests need to keep the chance of a single false alarm near 5%;
  still positive in 2021-26; positive in at least 9 of the 14 holdout years.

Only years with genuinely hourly data count (5,000+ bars a year).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

ENTRIES = (19, 23)
EXITS = (8, 10, 12, 16)
DESIGN_END = 2013
T_BAR = 3.5
DEFAULT_COST_BP = 1.3          # pairs without a measured cost


COMMISSION_BP = 0.4           # $4 a lot round trip, about 0.4bp of notional


def load():
    a = pd.read_parquet("data/the5ers_h1_all.parquet")
    a["ny"] = a["server"] - pd.Timedelta(hours=7)
    a = a[a["ny"] >= "2000-01-01"]
    a["yr"] = a["ny"].dt.year
    # Genuinely hourly years only. Judged per trading day, not per year: a
    # yearly bar count silently dropped 2026 because the year is not over.
    days = a.groupby(["sym", "yr"])["ny"].apply(
        lambda s: s.dt.normalize().nunique())
    bars = a.groupby(["sym", "yr"]).size()
    good = (bars / days >= 18).rename("ok").reset_index()
    a = a.merge(good[good["ok"]][["sym", "yr"]], on=["sym", "yr"])
    # Spread by New York hour, measured on The5ers Jun-Sep 2026. A long pays
    # the spread at the hour it buys, a short at the hour it buys back.
    sp = pd.read_csv("data/the5ers_spread_by_hour.csv")
    spread = {(r.sym, r.ny_hour): r.spread_bp for r in sp.itertuples()}
    return a, spread


def window_returns(g, entry_h, exit_h):
    """Return in bp from the open at entry_h on day D (Sun-Thu evening) to
    the open at exit_h on day D+1, long."""
    g = g.set_index("ny")["open"]
    idx = g.index
    ent = g[(idx.hour == entry_h) & (idx.weekday.isin([6, 0, 1, 2, 3]))]
    ex_time = ent.index.normalize() + pd.Timedelta(days=1) + \
        pd.Timedelta(hours=exit_h)
    ex = g.reindex(ex_time)
    r = 1e4 * (ex.to_numpy() / ent.to_numpy() - 1)
    out = pd.Series(r, index=ent.index.normalize() + pd.Timedelta(days=1))
    return out.dropna()


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 \
        else np.nan


def main() -> int:
    pd.set_option("display.width", 230)
    a, spread = load()
    rows = []
    for sym, g in a.groupby("sym"):
        for e in ENTRIES:
            for x in EXITS:
                r = window_returns(g, e, x)
                if len(r) < 500:
                    continue
                yrs = r.index.year
                des, hold = r[yrs < DESIGN_END], r[yrs >= DESIGN_END]
                if len(des) < 250 or len(hold) < 250:
                    continue
                d = 1 if des.mean() > 0 else -1          # chosen on design only
                paid_at = e if d > 0 else x
                sp = spread.get((sym, paid_at))
                if sp is None or not np.isfinite(sp):
                    continue                             # not tradeable then
                cost = sp + COMMISSION_BP
                net = d * hold - cost
                rec = net[net.index.year >= 2021]
                by_yr = net.groupby(net.index.year).mean()
                rows.append(dict(
                    sym=sym, entry_wpg=f"{(e - 1) % 24:02d}:00",
                    exit_wpg=f"{x - 1:02d}:00",
                    bet="long" if d > 0 else "short",
                    design_t=abs(tstat(des)), cost_bp=cost,
                    hold_days=len(hold), hold_net_bp=net.mean(),
                    hold_t=tstat(net), hold_right_pct=100 * (d * hold > 0)
                    .mean(), recent_net_bp=rec.mean(),
                    yrs_pos=f"{int((by_yr > 0).sum())}/{len(by_yr)}",
                    yrs_pos_n=int((by_yr > 0).sum()),
                    full_net_bp=(d * r - cost).mean()))
    d = pd.DataFrame(rows)
    print(f"tests run: {len(d)} ({d['sym'].nunique()} instruments x up to "
          f"{len(ENTRIES) * len(EXITS)} windows)")

    print("\n" + "=" * 110)
    print("HOW MANY WINDOWS HELD UP OUT OF SAMPLE (direction picked on "
          "2000-12, judged on 2013-26, after cost)")
    print("=" * 110)
    print(f"  holdout net positive: {(d['hold_net_bp'] > 0).sum()} of "
          f"{len(d)}   (a coin would give about half)")
    print(f"  holdout t above 2:    {(d['hold_t'] > 2).sum()}   "
          f"(luck alone: about {0.023 * len(d):.0f})")
    print(f"  holdout t above {T_BAR}:  {(d['hold_t'] > T_BAR).sum()}")
    print(f"  holdout t below -2 (flipped): {(d['hold_t'] < -2).sum()}")

    ok = d[(d["hold_t"] > T_BAR) & (d["recent_net_bp"] > 0)
           & (d["yrs_pos_n"] >= 9)]
    print("\n  PASSING EVERY PRE-SET TEST:")
    print(ok.drop(columns="yrs_pos_n").round(2).to_string(index=False)
          if len(ok) else "  none")

    print("\n  the 15 strongest by holdout t, for context (most will be luck):")
    print(d.sort_values("hold_t", ascending=False).head(15)
          .drop(columns="yrs_pos_n").round(2).to_string(index=False))

    print("\n  does picking a direction on 2000-12 predict 2013-26 at all?")
    print(f"  correlation between design strength and holdout t: "
          f"{d['design_t'].corr(d['hold_t']):+.2f}")
    strong = d[d["design_t"] > 3]
    print(f"  windows that were strong in design (t>3): {len(strong)}; of "
          f"those, holdout positive after cost: "
          f"{(strong['hold_net_bp'] > 0).sum()}")
    d.to_csv("data/session_scan_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
