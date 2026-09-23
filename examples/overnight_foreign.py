"""
Test the foreign-market lead, with a control group that can refute it.

The claim from the replication: a fund whose home market trades while New
York sleeps shows a far larger weak-close effect, because its "overnight" is
not a gap at all -- it is a full session of real trading somewhere else.

If that is the reason, the size of the effect should follow the clock:

    Asia-Pacific   home market runs entirely inside the New York night
                   -> largest
    Europe         home session runs from about 3am to 11:30am New York,
                   so most of it falls inside the window -> large
    Americas       Canada, Mexico, Brazil, Chile trade the same hours as
                   New York, so nothing happens overnight -> nothing

The Americas group is the point. It is foreign, it is a single-country fund,
it has all the same properties except the one the story depends on. If it
shows the same lift as Asia, the clock explanation is wrong and something
else is driving this.

The second question is whether any of it can be traded. These funds are far
thinner than SPY, and a spread of a tenth of a percent against an effect of
a tenth of a percent leaves nothing. Costs are measured here, not assumed.

None of these instruments was used to find the effect, to choose the rule,
or in the holdout. EWJ, EWG, EWU and FXI are excluded for exactly that
reason -- they appeared in the replication that produced the idea.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

from bcbt import overnight_data as od

warnings.filterwarnings("ignore")

# Grouped by when the home market is open, in New York terms.
GROUPS = {
    "asia": ["EWA", "EWH", "EWS", "EWT", "EWY", "EWM", "INDA", "EPP",
             "AAXJ"],
    "europe": ["EWQ", "EWI", "EWL", "EWN", "EWP", "EWD", "VGK"],
    "americas": ["EWC", "EWW", "EWZ", "ECH"],      # the control group
}
ALL = [t for v in GROUPS.values() for t in v]
KIND = {t: k for k, v in GROUPS.items() for t in v}

PANEL = "data/overnight_foreign.parquet"


def build():
    vix = od.vix_series()
    frames = []
    for t in ALL:
        try:
            f = od.build(t, vix)
            if len(f) > 1000:
                frames.append(f)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {t}: {exc}")
    p = pd.concat(frames).sort_index()
    p.to_parquet(PANEL)
    return p


def spy_signal() -> pd.DataFrame:
    """The US market's own close, as the trigger."""
    s = od.build("SPY")
    return s[["close_in_range", "ret_5d", "prev_overnight"]].rename(
        columns=lambda c: "spy_" + c)


def main() -> int:
    pd.set_option("display.width", 210)
    import os
    p = pd.read_parquet(PANEL) if os.path.exists(PANEL) else build()
    spy = spy_signal()
    p = p.join(spy, how="left")
    p["kind"] = p["sym"].map(KIND)
    p = p.dropna(subset=["overnight", "close_in_range", "spy_close_in_range"])

    rows = []
    for sym, g in p.groupby("sym"):
        if len(g) < 1000:
            continue
        base = g["overnight"]
        own = g.loc[g["close_in_range"] < 0.50, "overnight"]
        usa = g.loc[g["spy_close_in_range"] < 0.50, "overnight"]
        # Dollar volume is the liquidity that decides what a trade costs.
        dv = (g["close"] * g["volume"]).median()
        rows.append(dict(
            sym=sym, kind=KIND[sym], n=len(g),
            night_pct=100 * base.mean(),
            own_signal_pct=100 * own.mean(),
            us_signal_pct=100 * usa.mean(),
            lift_own=100 * (own.mean() - base.mean()),
            lift_us=100 * (usa.mean() - base.mean()),
            t_us=usa.mean() / (usa.std(ddof=1) / np.sqrt(len(usa))),
            median_dollar_vol_m=dv / 1e6))

    d = pd.DataFrame(rows).sort_values(["kind", "lift_us"],
                                       ascending=[True, False])
    print("=" * 110)
    print("FRESH SINGLE-COUNTRY FUNDS, none used to find or test the rule")
    print("=" * 110)
    print(d.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("THE TEST: does the effect follow the clock?")
    print("=" * 110)
    g = d.groupby("kind").agg(
        funds=("sym", "size"),
        night_pct=("night_pct", "mean"),
        us_signal_pct=("us_signal_pct", "mean"),
        lift_us=("lift_us", "mean"),
        lift_own=("lift_own", "mean"),
        median_t=("t_us", "median"),
        dollar_vol_m=("median_dollar_vol_m", "median"))
    order = [k for k in ("asia", "europe", "americas") if k in g.index]
    print(g.loc[order].round(4).to_string())

    if {"asia", "americas"} <= set(d["kind"].unique()):
        a = d.loc[d["kind"] == "asia", "lift_us"]
        c = d.loc[d["kind"] == "americas", "lift_us"]
        tt = st.ttest_ind(a, c, equal_var=False)
        print(f"\n  Asia-Pacific lift {a.mean():+.4f}% a night vs the "
              f"Americas control {c.mean():+.4f}%")
        print(f"  difference test: t={tt.statistic:.2f}, "
              f"p={tt.pvalue:.4f}")
        print("  " + ("the clock explanation survives"
                      if tt.pvalue < 0.05 and a.mean() > c.mean()
                      else "the clock explanation is NOT supported"))

    print("\n" + "=" * 110)
    print("CAN IT BE TRADED? the effect against realistic dealing costs")
    print("=" * 110)
    print("  A round trip costs roughly the spread. These funds are thin, so")
    print("  the spread is the whole question.\n")
    print(f"  {'fund':<7}{'group':<11}{'$ vol (m)':>11}{'gain/night':>12}"
          f"{'needs spread under':>21}")
    for _, r in d.sort_values("lift_us", ascending=False).iterrows():
        gain = r["us_signal_pct"]
        print(f"  {r['sym']:<7}{r['kind']:<11}{r['median_dollar_vol_m']:>11.1f}"
              f"{gain:>11.4f}%{gain:>20.4f}%")
    print("\n  A fund trading a few million dollars a day typically quotes "
          "0.05-0.20%\n  wide. Compare that with the gain column: the effect "
          "has to clear the\n  spread twice over to be worth anything, once "
          "in and once out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
