"""
The European-morning EUR/USD short, built as a system and tested hard.

The claim: EUR/USD tends to drift down between 03:00 and 08:00 New York,
while Europe is at work and America is not. It came out of the home-hours
test (home_hours_long.py), where it was one of three windows fixed before
26 years of The5ers' hourly data were downloaded, and it was the only one
to hold: 53% of mornings, +1.5 pips a morning after cost, positive in 22 of
27 years -- but only +0.6 pips since 2021.

Everything below was written before it was run.

PART 1 -- is the drift real, or something else wearing its clothes?
  a  Placebo: the same 5-hour short starting at every hour of the day. If
     EUR/USD simply fell over the period, every window shows a drift and
     03:00 is not special.
  b  Boundaries moved by an hour either way. A real effect should not
     depend on the exact hour.
  c  Year by year, and whether it only appears in years the euro fell.
  d  A second, independent price source (Dukascopy minutes, 2024-26) on
     the same mornings.

PART 2 -- the trade.
  Short at the 03:00 New York open. Stop 20 pips above entry. Exit at the
  08:00 open. Variant: a 40-pip target (2:1) as well. Stops of 15, 30 and
  50 pips are shown for context and NOT chosen from. Hourly bars cannot say
  whether a stop or target came first inside one hour; the stop is assumed
  first, and the result is re-run on minute bars where they exist.

PART 3 -- The5ers one-step, from where the account actually is.
  $5,000 account at +4.0% and +4.5%. Target $5,500, floor $4,700, daily loss
  $150 from the start-of-day balance. Risk 1.5% of the starting balance per
  trade, lots rounded DOWN to 0.01, $4 commission per lot and the measured
  spread paid on every trade. One trade per morning, stop at the target.
  Replayed on the real sequence of mornings from every start date since
  2000, against the same trade with a random direction each morning.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PIP = 1e-4
COMMISSION_PER_LOT = 4.0          # $ round trip, from The5ers' deal history
SPREAD = 0.0000053                # measured time-weighted EUR/USD spread
ACCOUNT = 5000.0
TARGET, FLOOR, DAILY = 5500.0, 4700.0, 150.0
DAYS_LEFT = 70


def load_h1():
    a = pd.read_parquet("data/the5ers_h1.parquet")
    e = a[a["sym"] == "EURUSD"].copy()
    e["ny"] = e["server"] - pd.Timedelta(hours=7)
    e = e[(e["ny"] >= "2000-01-01") & (e["ny"].dt.weekday < 5)]
    e = e.set_index("ny").sort_index()[["open", "high", "low", "close"]]
    return e


# ------------------------------------------------------------------ part 1

def window_moves(h1, start_h, length=5):
    """Short return in pips for each weekday window [start_h, start_h+len)."""
    hrs = h1.index.hour
    out = []
    for day, g in h1.groupby(h1.index.normalize()):
        start = day + pd.Timedelta(hours=start_h)
        idx = pd.date_range(start, periods=length, freq="h")
        if idx[-1].normalize() != day:
            continue                          # would run into the next day
        # Never straddle the 17:00 rollover, where spreads explode.
        if any(t.hour == 17 for t in idx):
            continue
        if not idx.isin(g.index).all():
            continue
        o = g.at[idx[0], "open"]
        c = g.at[idx[-1], "close"]
        out.append((day, (o - c) / PIP))
    del hrs
    return pd.DataFrame(out, columns=["day", "pips"])


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    return dict(n=n, down_pct=100 * (x > 0).mean(), avg_pips=x.mean(),
                t=x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def part1(h1):
    print("=" * 96)
    print("PART 1a. PLACEBO: a 5-hour short starting at every hour (pips)")
    print("=" * 96)
    rows = []
    for sh in range(24):
        w = window_moves(h1, sh)
        if len(w) < 500:
            continue
        rec = w[w["day"].dt.year >= 2021]
        rows.append(dict(start_ny=f"{sh:02d}:00", **{
            f"all_{k}": v for k, v in stats(w["pips"]).items()},
            recent_avg=rec["pips"].mean(),
            recent_t=stats(rec["pips"])["t"]))
    p = pd.DataFrame(rows).set_index("start_ny")
    print(p.round(2).to_string())
    rank = p["all_t"].rank(ascending=False)
    print(f"\n  03:00 ranks {int(rank['03:00'])} of {len(p)} windows on "
          f"strength over 2000-26, and "
          f"{int(p['recent_t'].rank(ascending=False)['03:00'])} of {len(p)} "
          f"since 2021")
    print(f"  average of all windows: {p['all_avg_pips'].mean():+.2f} pips "
          f"(the general drift that any window would pick up)")

    print("\n" + "=" * 96)
    print("PART 1b. BOUNDARIES MOVED")
    print("=" * 96)
    rows = []
    for sh, ln in [(2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 4), (4, 5)]:
        w = window_moves(h1, sh, ln)
        rec = w[w["day"].dt.year >= 2021]
        rows.append(dict(window=f"{sh:02d}:00-{sh + ln:02d}:00",
                         **stats(w["pips"]), recent_avg=rec["pips"].mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 96)
    print("PART 1c. YEAR BY YEAR, against what the euro did that year")
    print("=" * 96)
    w = window_moves(h1, 3)
    w["yr"] = w["day"].dt.year
    yr = w.groupby("yr")["pips"].agg(["count", "mean"])
    yclose = h1["close"].groupby(h1.index.year).last()
    yopen = h1["open"].groupby(h1.index.year).first()
    yr["euro_year_pct"] = 100 * (yclose / yopen - 1)
    print(yr.round(2).T.to_string())
    print(f"\n  years the morning short was positive: "
          f"{int((yr['mean'] > 0).sum())} of {len(yr)}")
    print(f"  in years the euro ROSE: "
          f"{int(((yr['mean'] > 0) & (yr['euro_year_pct'] > 0)).sum())} of "
          f"{int((yr['euro_year_pct'] > 0).sum())} positive")
    print(f"  in years the euro FELL: "
          f"{int(((yr['mean'] > 0) & (yr['euro_year_pct'] < 0)).sum())} of "
          f"{int((yr['euro_year_pct'] < 0).sum())} positive")
    print(f"  correlation of the morning result with the euro's year: "
          f"{yr['mean'].corr(yr['euro_year_pct']):+.2f}")
    # Resample whole years, so a good run of years is not counted as many
    # independent mornings.
    rng = np.random.default_rng(3)
    years = yr.index.to_numpy()
    by_year = {y: w.loc[w["yr"] == y, "pips"].to_numpy() for y in years}
    boots = [np.concatenate([by_year[y] for y in rng.choice(years,
             len(years))]).mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  resampling whole years: average {w['pips'].mean():+.2f} pips, "
          f"95% range {lo:+.2f} to {hi:+.2f}")

    print("\n" + "=" * 96)
    print("PART 1d. A SECOND PRICE SOURCE (Dukascopy minutes, same mornings)")
    print("=" * 96)
    try:
        import examples.zone_retest as ZR
        m1 = ZR.load("EURUSD").tz_localize(None)
        rows = []
        for day, g in m1.groupby(m1.index.normalize()):
            if day.weekday() >= 5:
                continue
            s = g.loc[day + pd.Timedelta(hours=3):
                      day + pd.Timedelta(hours=7, minutes=59)]
            if len(s) < 250:
                continue
            rows.append((day, (s["o"].iloc[0] - s["c"].iloc[-1]) / PIP))
        d = pd.DataFrame(rows, columns=["day", "duka"]).set_index("day")
        t5 = window_moves(h1, 3).set_index("day")["pips"].rename("the5ers")
        both = d.join(t5, how="inner")
        print(f"  mornings in both: {len(both)}   correlation "
              f"{both['duka'].corr(both['the5ers']):.3f}")
        print(f"  average: Dukascopy {both['duka'].mean():+.2f} pips, "
              f"The5ers {both['the5ers'].mean():+.2f} pips")
        print(f"  largest disagreement on one morning: "
              f"{(both['duka'] - both['the5ers']).abs().max():.1f} pips")
    except Exception as exc:                            # noqa: BLE001
        print(f"  skipped: {exc}")


# ------------------------------------------------------------------ part 2

def trades(h1, stop_pips=20.0, target_r=None, direction=None, rng=None):
    """
    One trade per morning. Returns price move in pips (positive = profit,
    before commission and spread) and how it ended. `direction` of None is
    the system (short); 'random' flips a coin each morning.
    """
    rows = []
    for day, g in h1.groupby(h1.index.normalize()):
        idx = pd.date_range(day + pd.Timedelta(hours=3), periods=5, freq="h")
        if not idx.isin(g.index).all():
            continue
        bars = g.loc[idx]
        d = -1
        if direction == "random":
            d = 1 if rng.random() < 0.5 else -1
        entry = bars["open"].iloc[0]
        stop = entry - d * stop_pips * PIP
        tgt = entry + d * target_r * stop_pips * PIP if target_r else None
        px, why = bars["close"].iloc[-1], "time"
        for _, b in bars.iterrows():
            # Stop checked first: where one hour holds both, assume the worse.
            if (d < 0 and b["high"] >= stop) or (d > 0 and b["low"] <= stop):
                px = (max(stop, b["open"]) if d < 0 else
                      min(stop, b["open"]))
                why = "stop"
                break
            if tgt is not None and ((d < 0 and b["low"] <= tgt) or
                                    (d > 0 and b["high"] >= tgt)):
                px, why = tgt, "target"
                break
        rows.append(dict(day=day, d=d, entry=entry,
                         pips=d * (px - entry) / PIP, why=why))
    return pd.DataFrame(rows)


def minute_trades(m1, stop_pips=20.0, target_r=None):
    """The same trade on 1-minute bars, where order inside an hour is known."""
    rows = []
    for day, g in m1.groupby(m1.index.normalize()):
        if day.weekday() >= 5:
            continue
        s = g.loc[day + pd.Timedelta(hours=3):
                  day + pd.Timedelta(hours=7, minutes=59)]
        if len(s) < 250:
            continue
        entry = s["o"].iloc[0]
        stop = entry + stop_pips * PIP
        tgt = entry - target_r * stop_pips * PIP if target_r else None
        px, why = s["c"].iloc[-1], "time"
        for o, hi, lo in zip(s["o"], s["h"], s["l"], strict=True):
            if hi >= stop:
                px, why = max(stop, o), "stop"
                break
            if tgt is not None and lo <= tgt:
                px, why = tgt, "target"
                break
        rows.append(dict(day=day, pips=(entry - px) / PIP, why=why))
    return pd.DataFrame(rows)


def cost_pips():
    """Commission and spread per trade, expressed in pips."""
    return COMMISSION_PER_LOT / 10.0 + SPREAD / PIP


def part2(h1):
    print("\n" + "=" * 96)
    print("PART 2. THE TRADE: short 03:00, out 08:00 (R = multiples of risk,"
          " after costs)")
    print("=" * 96)
    c = cost_pips()
    print(f"  cost per trade: {c:.2f} pips (commission {COMMISSION_PER_LOT/10:.2f}"
          f" + spread {SPREAD / PIP:.2f})")
    rows = []
    keep = {}
    for stop in (15, 20, 30, 50):
        for tr in (None, 2.0):
            t = trades(h1, stop, tr)
            t["r"] = (t["pips"] - c) / stop
            t["era"] = pd.cut(t["day"].dt.year, [1999, 2007, 2014, 2020, 2026],
                              labels=["2000-07", "2008-14", "2015-20",
                                      "2021-26"])
            label = f"{stop} pip stop" + (", 2:1 target" if tr else "")
            if stop == 20:
                keep[label] = t
            for era, g in list(t.groupby("era")) + [("ALL", t)]:
                r = g["r"].to_numpy()
                rows.append(dict(system=label, era=era, n=len(r),
                                 win_pct=100 * (r > 0).mean(),
                                 avg_r=r.mean(),
                                 t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
                                 stopped_pct=100 * (g["why"] == "stop").mean()))
    d = pd.DataFrame(rows)
    print(d[d["system"].str.startswith("20")].round(3).to_string(index=False))
    print("\n  for context only -- other stop sizes, whole period and 2021-26:")
    ctx = d[~d["system"].str.startswith("20") & d["era"].isin(["ALL",
                                                               "2021-26"])]
    print(ctx.round(3).to_string(index=False))

    # Losing streaks: what the account has to sit through.
    t = keep["20 pip stop"]
    loss = (t["r"] <= 0).to_numpy()
    runs, cur = [], 0
    for x in loss:
        cur = cur + 1 if x else 0
        runs.append(cur)
    t2 = t[t["day"].dt.year >= 2021]
    loss2 = (t2["r"] <= 0).to_numpy()
    runs2, cur = [], 0
    for x in loss2:
        cur = cur + 1 if x else 0
        runs2.append(cur)
    print(f"\n  longest run of losing mornings, 20-pip stop: {max(runs)} "
          f"(2000-26), {max(runs2)} (2021-26)")

    print("\n  hourly vs minute resolution, same mornings (2024-26):")
    try:
        import examples.zone_retest as ZR
        m1 = ZR.load("EURUSD").tz_localize(None)
        for tr in (None, 2.0):
            mt = minute_trades(m1, 20.0, tr).set_index("day")
            ht = trades(h1, 20.0, tr).set_index("day")
            both = mt.join(ht, lsuffix="_min", rsuffix="_hour", how="inner")
            print(f"    {'2:1 target' if tr else 'time exit '}: mornings "
                  f"{len(both)}, avg pips minute {both['pips_min'].mean():+.2f}"
                  f" vs hourly {both['pips_hour'].mean():+.2f}, "
                  f"outcome differs on {100 * (both['why_min'] != both['why_hour']).mean():.1f}%")
    except Exception as exc:                            # noqa: BLE001
        print(f"    skipped: {exc}")
    return keep


# ------------------------------------------------------------------ part 3

def lots_for(risk_dollars, stop_pips):
    return np.floor(risk_dollars / (stop_pips * 10.0) / 0.01) * 0.01


def replay(pips_seq, start_bal, risk_pct, stop_pips):
    """Run the evaluation over one real sequence of mornings."""
    bal = start_bal
    lots = lots_for(ACCOUNT * risk_pct / 100, stop_pips)
    for k, p in enumerate(pips_seq):
        day_start = bal
        bal += lots * 10.0 * p - lots * COMMISSION_PER_LOT \
            - lots * 10.0 * SPREAD / PIP
        if bal >= TARGET:
            return "pass", k + 1
        if bal <= FLOOR or day_start - bal >= DAILY:
            return "fail", k + 1
    return "open", len(pips_seq)


def part3(h1):
    print("\n" + "=" * 96)
    print("PART 3. THE5ERS ONE-STEP ON $5,000, replayed on real mornings")
    print("=" * 96)
    stop = 20.0
    sys_t = trades(h1, stop).reset_index(drop=True)
    sys_t2 = trades(h1, stop, 2.0).reset_index(drop=True)
    rng = np.random.default_rng(11)
    rand = [trades(h1, stop, None, "random", rng).reset_index(drop=True)
            for _ in range(10)]
    print(f"  lots at 1% / 1.5% / 2% risk with a {stop:.0f}-pip stop: "
          f"{lots_for(50, stop):.2f} / {lots_for(75, stop):.2f} / "
          f"{lots_for(100, stop):.2f}")

    rows = []
    for start_pct in (4.0, 4.5):
        start_bal = ACCOUNT * (1 + start_pct / 100)
        for risk in (1.0, 1.5, 2.0):
            for label, seqs in (("system, time exit", [sys_t]),
                                ("system, 2:1 target", [sys_t2]),
                                ("random direction", rand)):
                for era_name, y0 in (("starts 2000-26", 2000),
                                     ("starts 2021-26", 2021)):
                    res = []
                    for s in seqs:
                        starts = s.index[(s["day"].dt.year >= y0)
                                         & (s.index < len(s) - DAYS_LEFT)]
                        for i0 in starts[::5]:
                            seq = s["pips"].iloc[i0:i0 + DAYS_LEFT].to_numpy()
                            res.append(replay(seq, start_bal, risk, stop)[0])
                    res = pd.Series(res)
                    rows.append(dict(start=f"+{start_pct}%", risk_pct=risk,
                                     system=label, starts=era_name,
                                     pass_pct=100 * (res == "pass").mean(),
                                     fail_pct=100 * (res == "fail").mean(),
                                     open_pct=100 * (res == "open").mean(),
                                     runs=len(res)))
    d = pd.DataFrame(rows)
    print(d.round(1).to_string(index=False))


def part4(h1):
    print("\n" + "=" * 96)
    print("PART 4. LOOK-AHEAD CHECK")
    print("=" * 96)
    from bcbt.audit import future_poison
    h = h1.loc["2023-01-01":"2024-12-31"].copy()
    h.index = h.index.tz_localize("America/New_York", ambiguous="NaT",
                                  nonexistent="NaT")
    h = h[h.index.notna()]

    def decide(df):
        t = trades(df.tz_localize(None), 20.0)
        t = t.assign(entry_ts=(t["day"] + pd.Timedelta(hours=3))
                     .dt.tz_localize("America/New_York"))
        return t[["entry_ts", "d", "entry"]]
    r = future_poison(decide, h, probes=10)
    print(f"  {'LEAK' if r['leak'] else 'clean'}   probes "
          f"{r.get('probes_fired', 0)}/{r.get('probes_tried', 0)}   "
          f"decisions {r.get('n_clean')}")
    return not r["leak"]


def main() -> int:
    pd.set_option("display.width", 220)
    h1 = load_h1()
    part1(h1)
    part2(h1)
    part3(h1)
    ok = part4(h1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
