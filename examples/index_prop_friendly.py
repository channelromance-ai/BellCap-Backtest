"""
Making the long-only index system prop-friendly. Fixed before running.

Strategies (from index_long_system.py, long by day, no financing):
  S1  boost days only, SP500   (day after a -2 sigma day + turn of month,
                                double size; flat otherwise)
  S2  base + boost, SP500
  S3  base + boost, NAS100

Layers, each aimed at a reason prop accounts break:
  none    fixed exposure 0.25x (index notional per $1 of account)
  VOL     size so the expected daily swing is TV of the account, using the
          last 20 days' volatility (known before the day); TV = 0.25% or
          0.35%; capped at 3x. Crashes tend to arrive when volatility is
          already high, so size is already smaller.
  TREND   only long when the index closed above its 200-day average the
          day before.
  CUSHION size multiplied by the share of the 6% loss allowance still left
          (at -3% half size, at -4.5% quarter size), measured from the
          start balance (static) or from the high-water mark (trailing).

Judged by The5ers one-step rules: +10% target, 6% max loss, 3% daily loss,
no time limit. Replayed from every month start since 1993 (SP500) / 1999
(NAS100): outcome within 1 year and within 3 years, and months to pass.
Daily closes include dividends; the daily-loss check uses close-to-close
moves, so an intraday dip that recovered by the close is not counted.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.index_long_system as L                 # noqa: E402

TARGET, MAX_LOSS, DAILY = 0.10, 0.06, 0.03


def base_exposure(df, strategy):
    if strategy.startswith("S1"):
        return L.weights(df, "boost days only")
    return L.weights(df, "base + boost")


def exposure(df, strategy, layer, tv):
    w = base_exposure(df, strategy)
    sig = df["r"].rolling(20).std().shift(1)
    price = (1 + df["r"]).cumprod()
    trend_ok = (price > price.rolling(200).mean()).shift(1, fill_value=False)
    if "VOL" in layer:
        e = w * np.minimum(3.0, tv / sig)
    else:
        e = w * 0.25
    if "TREND" in layer:
        e = e * trend_ok
    return e.fillna(0.0)


def replay(df, e, cushion, mode, horizon):
    r, cost = df["r"].to_numpy(), df["cost"].to_numpy()
    ex = e.to_numpy()
    out = []
    for s in range(250, len(r) - 21, 21):
        eq, peak, res, days = 1.0, 1.0, "open", None
        for k in range(s, min(s + horizon, len(r))):
            floor = (1 - MAX_LOSS) if mode == "static" else \
                peak * (1 - MAX_LOSS)
            scale = 1.0
            if cushion:
                scale = max(0.0, min(1.0, (eq - floor) / (eq * MAX_LOSS)))
            x = ex[k] * scale
            day = x * r[k] - (x > 0) * x * cost[k]
            if day <= -DAILY:
                res, days = "fail", k - s
                break
            eq *= 1 + day
            peak = max(peak, eq)
            if eq <= floor:
                res, days = "fail", k - s
                break
            if eq >= 1 + TARGET:
                res, days = "pass", k - s
                break
        out.append((res, days))
    o = pd.DataFrame(out, columns=["res", "days"])
    passed = o[o["res"] == "pass"]
    return dict(pass_pct=100 * (o["res"] == "pass").mean(),
                fail_pct=100 * (o["res"] == "fail").mean(),
                open_pct=100 * (o["res"] == "open").mean(),
                months_to_pass=passed["days"].median() / 21
                if len(passed) else np.nan)


def main() -> int:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    data = {"SP500": L.daily("SPY"), "NAS100": L.daily("QQQ")}
    strategies = {"S1 boost days only, SP500": "SP500",
                  "S2 base+boost, SP500": "SP500",
                  "S3 base+boost, NAS100": "NAS100"}
    layers = [("none (0.25x)", None), ("VOL", 0.0025), ("VOL", 0.0035),
              ("TREND (0.25x)", None), ("VOL+TREND", 0.0025),
              ("VOL+TREND", 0.0035)]
    rows = []
    for sname, mkt in strategies.items():
        df = data[mkt]
        for layer, tv in layers:
            e = exposure(df, sname, layer, tv or 0.0)
            ret = e * df["r"] - (e > 0) * e * df["cost"]
            st = L.stats(ret)
            for cushion in (False, True):
                for mode in ("static", "trailing"):
                    one = replay(df, e, cushion, mode, 252)
                    three = replay(df, e, cushion, mode, 756)
                    rows.append(dict(
                        strategy=sname, layer=layer + (f" {tv:.2%}" if tv
                                                       else ""),
                        cushion="yes" if cushion else "no", limit=mode,
                        cagr_pct=st["cagr_pct"], worst_dd_pct=st["worst_dd_pct"],
                        pass_1y=one["pass_pct"], fail_1y=one["fail_pct"],
                        pass_3y=three["pass_pct"], fail_3y=three["fail_pct"],
                        open_3y=three["open_pct"],
                        months_to_pass=three["months_to_pass"]))
    res = pd.DataFrame(rows)
    print("=" * 150)
    print("PROP-FRIENDLY VERSIONS  (cagr / worst_dd: whole history without the "
          "cushion; pass/fail: % of monthly starts; months_to_pass: median)")
    print("=" * 150)
    print(res.round(1).to_string(index=False))
    print("\nBEST pass-to-fail ratios within 3 years (static limit, as The5ers):")
    s = res[res["limit"] == "static"].copy()
    s["pass_per_fail"] = s["pass_3y"] / s["fail_3y"].replace(0, np.nan)
    print(s.sort_values("pass_per_fail", ascending=False).head(10)
          .round(2).to_string(index=False))
    res.to_csv("data/index_prop_friendly.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
