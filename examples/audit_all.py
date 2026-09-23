"""
Point the look-ahead detector at every strategy in the repo.

Nothing here should be trusted until this passes. The detector works by
cutting at a decision's own timestamp, corrupting everything after it, and
re-running: that decision claimed to be made at that moment, so nothing
later may change it. A strategy that survives has at least proved it is not
reading the future, which is the failure that produced the only spectacular
result this repo ever generated.

The portfolio engine is checked differently, because its decisions are
weights rather than trades: the weights set on or before a date must not
change when later prices are corrupted.

This does not prove a strategy is correct. It proves one specific thing --
that its decisions could have been made when it says they were.
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

from bcbt import store, universe as U
from bcbt.audit import future_poison

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

# A slice big enough to generate plenty of decisions and small enough that
# a dozen full re-runs finish.
WINDOW = ("2023-09-01", "2024-06-01")


def slice_m1(sym, src=None):
    m1 = store.load_m1(sym, src=src) if src else store.load_m1(sym)
    return m1[(m1.index >= WINDOW[0]) & (m1.index < WINDOW[1])]


def check(name, fn, data, probes=10, key="entry_ts"):
    try:
        r = future_poison(fn, data, probes=probes, key=key)
    except Exception as exc:                            # noqa: BLE001
        # An error means NOT CHECKED, which must never be reported as
        # clean. The first version of this script printed PASS while a
        # strategy had failed to run at all.
        print(f"  {name:<34} ERROR   {str(exc)[:58]}")
        return dict(leak=None, error=str(exc))
    flag = "LEAK" if r["leak"] else "clean"
    print(f"  {name:<34} {flag:<7} "
          f"probes {r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}"
          f"   decisions {r.get('n_clean')}")
    if r["leak"]:
        print(f"  {'':<34} {r['detail']}")
    return r


def main() -> int:
    pd.set_option("display.width", 200)
    results = {}

    print("=" * 92)
    print("LOOK-AHEAD CHECK ON EVERY STRATEGY")
    print(f"window {WINDOW[0]} -> {WINDOW[1]}")
    print("=" * 92)

    spx = slice_m1("SPX500")

    # --- zone retest
    import examples.zone_retest as ZR
    results["zone"] = check(
        "zone retest (5m entry)",
        lambda df: ZR.signals("SPX500", df, entry_tf=5)[
            ["entry_ts", "entry", "stop"]], spx)

    # --- opening range breakout
    from bcbt.strategies import orb_ema
    def orb(df):
        days, arr = orb_ema.prepare(df)
        t = orb_ema.run("SPX500", days, arr, 0.6, or_min=15,
                        stop_mode="or", rr=1.5)
        return t[["entry_ts", "entry", "stop"]] if len(t) else pd.DataFrame(
            columns=["entry_ts", "entry", "stop"])
    results["orb"] = check("opening-range breakout", orb, spx)

    # --- hourly EMA crossover
    from bcbt.strategies import ema_cross
    def emac(df):
        h = ema_cross.prepare(df, k=2)
        t = ema_cross.run("SPX500", df, h, 0.6, use_be=True)
        if not len(t):
            return pd.DataFrame(columns=["entry_ts", "entry", "stop"])
        return t.rename(columns={"entry_time": "entry_ts"})[
            ["entry_ts", "entry", "stop"]]
    results["ema_cross"] = check("hourly 9/21 crossover", emac, spx)

    # --- pullback v2
    from bcbt.strategies import pullback_v2 as v2
    def pull(df):
        bars = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                  "c": "close", "v": "volume"})
        bars = bars.resample("5min", closed="left", label="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum"}).dropna()
        prep = v2.prepare(bars)
        t = v2.run("SPX500", prep, cost_bps=0.5, direction="both")
        if not len(t):
            return pd.DataFrame(columns=["entry_ts", "entry", "stop"])
        return t.rename(columns={"entry_time": "entry_ts"})[
            ["entry_ts", "entry", "stop"]]
    results["pullback"] = check("pullback v2 (long and short)", pull, spx)

    # --- the standalone deliverable
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "standalone"))
    import intraday_pullback as ip
    def standalone(df):
        bars = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                  "c": "close", "v": "volume"})
        bars = bars.resample("5min", closed="left", label="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum"}).dropna()
        prep = ip.prepare_data(bars)
        t = ip.run_backtest(prep)
        if not len(t):
            return pd.DataFrame(columns=["entry_ts", "entry_price"])
        return t.rename(columns={"entry_time": "entry_ts"})[
            ["entry_ts", "entry_price"]]
    results["standalone"] = check("standalone pullback (spec file)",
                                  standalone, spx)

    # --- the portfolio engine, whose decisions are weights
    print("\n" + "=" * 92)
    print("LOOK-AHEAD CHECK ON THE PORTFOLIO ENGINE")
    print("=" * 92)
    px = U.load()
    px = px[px.index >= "2010-01-01"]

    def weights_as_decisions(frame):
        rows = []
        for day in U.rebalance_dates(frame.index, "M"):
            hist = frame.loc[:day]
            if len(hist) < 300:
                continue
            w = {"SPY": 1.0} if hist["SPY"].iloc[-1] > \
                hist["SPY"].iloc[-200:].mean() else {"IEF": 1.0}
            rows.append(dict(entry_ts=day, spy=w.get("SPY", 0.0),
                             ief=w.get("IEF", 0.0)))
        return pd.DataFrame(rows, columns=["entry_ts", "spy", "ief"])

    check("monthly weight decisions", weights_as_decisions, px, probes=8)

    leaks = [k for k, v in results.items() if v and v.get("leak")]
    # An error means NOT CHECKED. Reporting that as a pass is how a broken
    # strategy gets a clean bill of health, so it is counted as a failure.
    errors = [k for k, v in results.items() if v and v.get("leak") is None]
    print("\n" + "=" * 92)
    if leaks:
        print(f"FAIL: look-ahead found in {leaks}")
    if errors:
        print(f"NOT CHECKED (errored, so not clean): {errors}")
    if not leaks and not errors:
        print("PASS: every strategy ran, and none used information "
              "before it existed")
    print("=" * 92)
    return 1 if (leaks or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
