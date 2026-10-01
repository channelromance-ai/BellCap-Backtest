"""
A long-only index system built from index_ev.py's findings, under the
user's prop-firm limits. Fixed before running:

  Base      long every trading day, opened after the 17:00 New York roll
            and closed before the next one: no financing, nothing held
            over the weekend; one round trip of spread a day.
  Boost     DOUBLE size on (a) the day after a day more than 2 sigma down
            (sigma = std of the previous 20 daily returns) and (b) turn-of-
            month days (last trading day + first three of the next month).
  Variants  base only; boost days only (flat otherwise); base + boost.
  Markets   NAS100 (QQQ 1999-), SP500 (SPY 1993-), and a 50/50 mix.
  Size      exposure E = index notional per $1 of account: 0.25, 0.5, 1, 2.
  Limits    3% daily loss; 6% max loss measured from the START balance
            (The5ers) and, separately, trailing from the high-water mark.
            Replayed from every month start, one year each: breached, hit
            +10% first, or neither.
  Check     the same base system on The5ers' own hourly bars (2020-26):
            buy the first bar after the roll, sell the last bar before it.

Daily ETF closes include dividends (a CFD long is usually credited them;
if The5ers does not, subtract ~1.3% / 0.6% a year at E = 1). They are
16:00 New York closes rather than the 17:00 roll -- a close approximation.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

COST_BP = {"QQQ": 0.65, "SPY": 0.90}        # The5ers round trip, measured
EXPOSURES = (0.25, 0.5, 1.0, 2.0)


def daily(tk):
    u = pd.read_parquet("data/universe_daily.parquet")[tk].dropna()
    u = u[u.index.weekday < 5]
    r = u.pct_change()
    vol = r.rolling(20).std().shift(1)
    dip = (r < -2 * vol).shift(1, fill_value=False)       # yesterday
    ym = pd.Series(u.index.year * 100 + u.index.month, index=u.index)
    pos = ym.groupby(ym).cumcount()
    rev = ym.groupby(ym).cumcount(ascending=False)
    tom = (rev == 0) | (pos <= 2)
    return pd.DataFrame(dict(r=r, dip=dip, tom=tom,
                             cost=COST_BP[tk] / 1e4)).dropna()


def weights(df, variant):
    boost = (df["dip"] | df["tom"]).astype(float)
    if variant == "base only":
        return pd.Series(1.0, index=df.index)
    if variant == "boost days only":
        return 2.0 * boost
    return 1.0 + boost                                     # base + boost


def account_returns(df, variant, E):
    w = weights(df, variant) * E
    # cost only on days actually traded
    return w * df["r"] - (w > 0) * w * df["cost"]


def stats(ret):
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(252)
    return dict(cagr_pct=100 * cagr, vol_pct=100 * vol,
                sharpe=(ret.mean() * 252) / vol if vol else np.nan,
                worst_dd_pct=100 * dd, worst_day_pct=100 * ret.min(),
                days_in_mkt_pct=100 * (ret != 0).mean())


def prop_replay(ret, target=0.10, max_loss=0.06, daily_loss=0.03):
    """From every 21st day, one year forward: which comes first?"""
    r = ret.to_numpy()
    out = {"static": [], "trailing": []}
    for s in range(0, len(r) - 252, 21):
        for mode in ("static", "trailing"):
            eq, peak, res = 1.0, 1.0, "neither"
            for x in r[s:s + 252]:
                if x <= -daily_loss:
                    res = "daily limit"
                    break
                eq *= 1 + x
                peak = max(peak, eq)
                floor = (1 - max_loss) if mode == "static" else \
                    peak * (1 - max_loss)
                if eq <= floor:
                    res = "max loss"
                    break
                if eq >= 1 + target:
                    res = "target"
                    break
            out[mode].append(res)
    res = {}
    for mode, lst in out.items():
        s = pd.Series(lst)
        res[f"{mode}_target_pct"] = 100 * (s == "target").mean()
        res[f"{mode}_breach_pct"] = 100 * s.isin(["max loss",
                                                  "daily limit"]).mean()
    return res


def the5ers_check():
    a = pd.read_parquet("data/the5ers_idx_h1.parquet")
    rows = []
    for sym, tk in (("NAS100", "QQQ"), ("SP500", "SPY")):
        g = a[a["sym"] == sym].copy()
        g["sday"] = g["server"].dt.normalize()
        per = g.groupby("sday").size()
        g = g[g["sday"].isin(per[per >= 18].index)]
        sess = g.groupby("sday").agg(o=("open", "first"), c=("close", "last"))
        sess = sess[pd.DatetimeIndex(sess.index).weekday < 5]
        r_cfd = sess["c"] / sess["o"] - 1 - COST_BP[tk] / 1e4
        d = daily(tk)
        d = d[d.index >= sess.index.min()]
        r_etf = d["r"] - d["cost"]
        for name, r in (("The5ers CFD sessions", r_cfd),
                        (f"{tk} daily closes", r_etf)):
            st = stats(r)
            rows.append(dict(market=sym, source=name, days=len(r),
                             cagr_pct=st["cagr_pct"],
                             worst_dd_pct=st["worst_dd_pct"],
                             avg_day_bp=1e4 * r.mean()))
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 230)
    data = {"NAS100": daily("QQQ"), "SP500": daily("SPY")}
    both = data["NAS100"].join(data["SP500"], lsuffix="_n", rsuffix="_s",
                               how="inner")
    rows, era_rows = [], []
    for variant in ("base only", "base + boost", "boost days only"):
        for mkt in ("NAS100", "SP500", "50/50"):
            for E in EXPOSURES:
                if mkt == "50/50":
                    dn = both.rename(columns=lambda c: c[:-2] if c.endswith(
                        "_n") else c + "_x")[["r", "dip", "tom", "cost"]]
                    ds = both.rename(columns=lambda c: c[:-2] if c.endswith(
                        "_s") else c + "_x")[["r", "dip", "tom", "cost"]]
                    ret = 0.5 * account_returns(dn, variant, E) + \
                        0.5 * account_returns(ds, variant, E)
                else:
                    ret = account_returns(data[mkt], variant, E)
                row = dict(variant=variant, market=mkt, exposure=E,
                           **stats(ret), **prop_replay(ret))
                rows.append(row)
                if E == 1.0:
                    for era, sel in (("before 2010", ret.index.year < 2010),
                                     ("2010-26", ret.index.year >= 2010)):
                        st = stats(ret[sel])
                        era_rows.append(dict(variant=variant, market=mkt,
                                             era=era,
                                             cagr_pct=st["cagr_pct"],
                                             worst_dd_pct=st["worst_dd_pct"],
                                             sharpe=st["sharpe"]))
    res = pd.DataFrame(rows)
    print("=" * 140)
    print("LONG-ONLY INDEX SYSTEM, whole history (QQQ 1999-, SPY 1993-, "
          "50/50 1999-)")
    print("  replay = from every month start, one year: % reaching +10% "
          "first / % breaching 6% (static or trailing) or the 3% daily limit")
    print("=" * 140)
    print(res.round(2).to_string(index=False))
    print("\n" + "=" * 100)
    print("BY ERA at exposure 1x")
    print("=" * 100)
    print(pd.DataFrame(era_rows).round(2).to_string(index=False))
    print("\n" + "=" * 100)
    print("REALITY CHECK: base system on The5ers' own prices vs ETF closes, "
          "2020-26, exposure 1x")
    print("=" * 100)
    print(the5ers_check().round(2).to_string(index=False))
    res.to_csv("data/index_long_system.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
