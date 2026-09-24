"""
The NAS100 long, set up as the counterpart to the European-morning EUR/USD
short, and closed inside the same The5ers trading day.

The idea: EUR/USD drifts down while Europe works and New York sleeps, which
is a dollar that firms overnight. The inverse on the Nasdaq is a long. But
the repo already knows where index returns come from (overnight.py and the
commits around it): the NIGHT carries the index and the cash session adds
roughly nothing -- QQQ +72.8% overnight against +10.6% in the daytime over
763 nights. So "long NAS100, out before the session ends" is two different
trades depending on which session is meant, and both are tested:

  night   in 23:00 New York (22:00 Winnipeg, the EUR entry time), out at the
          09:00 open (08:00 Winnipeg), before the cash open.
          Variant: out at 10:00, which takes the 09:30 open with it.
          Context: in at 19:00 (after the 18:00 reopen), out at 09:00.
  day     in at the 10:00 open, out at the 16:00 open (the cash close).
          Hourly bars cannot enter at 09:30; with minute data this would.

The5ers' day resets at 17:00 New York, so every one of these opens and
closes inside one trading day and never holds through the rollover.

Fixed before running:

  Units   basis points of the entry price, and stops as a percentage of it,
          because the Nasdaq went from about 1,500 to above 20,000 and a
          fixed point stop means nothing across that.
  Stops   0.5%, 0.75% and 1.0% below entry. Stop checked first inside
          every hour; a gap through it fills at the open.
  Cost    SPREAD_PTS points a round trip (above Dukascopy's measured 1.46
          overnight, which is a floor), and double that for sensitivity.

The comparison that matters: the index went up, so ANY long made money and
a random-direction baseline flatters it. The fair baseline keeps the same
sequence of nights and replaces this window's average with the average of
every window of the same length -- what holding the index for that many
hours would have paid regardless of the clock. Only the difference is
timing.

Then the two together, since the EUR short and this long would be open on
the same nights: how their results correlate, and The5ers replays with both
on. Two losers at 1.5% each is $150 -- exactly the daily limit.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.eur_morning as EM                      # noqa: E402

PANEL = "data/the5ers_h1.parquet"
NAMES = ("NAS100", "US100", "USTEC", "NDX", "NAS100.cash")
SPREAD_PTS = 2.0
STOPS = (0.5, 0.75, 1.0)                  # % of entry
WINDOWS = {                               # name: (entry NY hour, exit NY hour)
    "night 22:00-08:00 Wpg": (23, 9),
    "night 22:00-09:00 Wpg": (23, 10),
    "night 18:00-08:00 Wpg": (19, 9),
    "day 09:00-15:00 Wpg": (10, 16),
}
MAIN = "night 22:00-08:00 Wpg"
ERAS = ([1999, 2007, 2014, 2020, 2026],
        ["2000-07", "2008-14", "2015-20", "2021-26"])


def load_h1():
    """Hourly NAS100 in New York time: The5ers' panel if it has it, else
    the Dukascopy minute store resampled (a much shorter sample)."""
    if os.path.exists(PANEL):
        a = pd.read_parquet(PANEL)
        for name in NAMES:
            e = a[a["sym"] == name]
            if len(e):
                print(f"  data: The5ers hourly {name}, server time - 7h")
                e = e.assign(ny=e["server"] - pd.Timedelta(hours=7))
                e = e[e["ny"] >= "2000-01-01"]
                return e.set_index("ny").sort_index()[
                    ["open", "high", "low", "close"]]
    from bcbt import store
    m1 = store.load_m1("NAS100").tz_localize(None)
    print("  data: Dukascopy NAS100 minutes resampled to hours -- short "
          "sample, read the eras with care")
    h = m1.resample("h").agg({"o": "first", "h": "max", "l": "min",
                              "c": "last"}).dropna()
    return h.rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close"})


_GRID = {}


def grid(h):
    """h on a gapless hourly clock, missing hours NaN, cached per frame."""
    key = id(h)
    if key not in _GRID:
        full = pd.date_range(h.index[0].floor("h"), h.index[-1].floor("h"),
                             freq="h")
        g = h[~h.index.duplicated()].reindex(full)
        _GRID[key] = (full[0], {c: g[c].to_numpy() for c in g.columns})
    return _GRID[key]


def window_trades(h, entry_h, exit_h, stop_pct=None, direction=None,
                  rng=None):
    """
    One trade per weekday D, in at entry_h (on D-1 if entry_h > exit_h),
    out at the open of exit_h on D. bp of entry, positive = profit,
    pre-cost. Stop checked first inside every hour; a gap fills at the open.
    """
    t0, a = grid(h)
    days = pd.bdate_range(h.index[0].normalize() + pd.Timedelta(days=1),
                          h.index[-1].normalize())
    overnight = entry_h > exit_h
    start = days + pd.Timedelta(hours=entry_h) - (
        pd.Timedelta(days=1) if overnight else pd.Timedelta(0))
    # The week opens Sunday 18:00 and ends Friday 17:00.
    ok = start.weekday != 5
    if overnight:
        ok &= start.weekday != 4
    n = (exit_h - entry_h) % 24
    pos = ((start - t0) // pd.Timedelta(hours=1)).to_numpy()
    ok &= (pos >= 0) & (pos + n <= len(a["open"]))
    days, pos = days[ok], pos[ok]
    ix = pos[:, None] + np.arange(n)
    o, hi, lo, c = (a[k][ix] for k in ("open", "high", "low", "close"))
    # Every hour present, and not a flat holiday print from a CFD feed.
    ok = ~np.isnan(o).any(1) & ~np.isnan(c).any(1) & \
        (np.nanmax(hi, 1) > np.nanmin(lo, 1))
    days, o, hi, lo, c = days[ok], o[ok], hi[ok], lo[ok], c[ok]
    d = np.ones(len(days))
    if direction == "random":
        d = np.where(rng.random(len(days)) < 0.5, 1.0, -1.0)
    entry = o[:, 0]
    px = c[:, -1].copy()
    why = np.full(len(days), "time", dtype=object)
    if stop_pct is not None:
        stop = entry * (1 - d * stop_pct / 100)
        hit = np.where(d[:, None] > 0, lo <= stop[:, None],
                       hi >= stop[:, None])
        any_hit = hit.any(1)
        first = hit.argmax(1)
        o_first = o[np.arange(len(days)), first]
        fill = np.where(d > 0, np.minimum(stop, o_first),
                        np.maximum(stop, o_first))
        px = np.where(any_hit, fill, px)
        why[any_hit] = "stop"
    return pd.DataFrame(dict(day=days, entry=entry,
                             bp=d * 1e4 * (px - entry) / entry, why=why))


def cost_bp(t, mult=1.0):
    return 1e4 * SPREAD_PTS * mult / t["entry"]


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


# ------------------------------------------------------------------ part 1

def part1(h):
    print("=" * 100)
    print("PART 1a. HOUR BY HOUR: average long return, bp, New York hour")
    print("=" * 100)
    x = pd.DataFrame({"bp": 1e4 * (h["close"] / h["open"] - 1),
                      "hr": h.index.hour, "yr": h.index.year},
                     index=h.index)
    x = x[x.index.weekday < 5]
    prof = pd.DataFrame({
        "all": x.groupby("hr")["bp"].mean(),
        "t_all": x.groupby("hr")["bp"].apply(tstat),
        "2021-26": x[x["yr"] >= 2021].groupby("hr")["bp"].mean()})
    prof.index = [f"{hh:02d} NY" for hh in prof.index]
    print(prof.round(2).T.to_string())

    print("\n" + "=" * 100)
    print("PART 1b. PLACEBO: a 10-hour long starting at every hour, never "
          "across 17:00 (bp, no stop, no cost)")
    print("=" * 100)
    rows = []
    for sh in range(24):
        eh = (sh + 10) % 24
        span = [(sh + k) % 24 for k in range(10)]
        if 17 in span:
            continue
        w = window_trades(h, sh, eh)
        if len(w) < 250:
            continue
        rec = w[w["day"].dt.year >= 2021]
        rows.append(dict(start_ny=f"{sh:02d}:00", n=len(w),
                         up_pct=100 * (w["bp"] > 0).mean(),
                         avg_bp=w["bp"].mean(), t=tstat(w["bp"]),
                         recent_bp=rec["bp"].mean() if len(rec) else np.nan))
    p = pd.DataFrame(rows).set_index("start_ny")
    print(p.round(2).to_string())
    if "23:00" in p.index:
        print(f"\n  23:00 ranks {int(p['t'].rank(ascending=False)['23:00'])}"
              f" of {len(p)} on strength; the average 10-hour long made "
              f"{p['avg_bp'].mean():+.2f} bp -- the drift any window gets")

    print("\n" + "=" * 100)
    print("PART 1c. YEAR BY YEAR, main night window vs the cash day (bp)")
    print("=" * 100)
    rows = {}
    for name in (MAIN, "day 09:00-15:00 Wpg"):
        w = window_trades(h, *WINDOWS[name])
        rows[name] = w.groupby(w["day"].dt.year)["bp"].mean()
    yr = pd.DataFrame(rows)
    yr["index_year_pct"] = 100 * (h["close"].groupby(h.index.year).last()
                                  / h["open"].groupby(h.index.year).first()
                                  - 1)
    print(yr.round(1).T.to_string())
    for name in (MAIN, "day 09:00-15:00 Wpg"):
        down = yr["index_year_pct"] < 0
        print(f"  {name}: positive in {int((yr[name] > 0).sum())} of "
              f"{len(yr)} years, and in {int((yr[name] > 0)[down].sum())} "
              f"of {int(down.sum())} years the index fell")


# ------------------------------------------------------------------ part 2

def part2(h):
    print("\n" + "=" * 100)
    print(f"PART 2. THE TRADE (R after {SPREAD_PTS:.1f} points of spread; "
          f"'x2 cost' doubles it)")
    print("=" * 100)
    rows, keep = [], {}
    for name, (eh, xh) in WINDOWS.items():
        for stop in STOPS:
            t = window_trades(h, eh, xh, stop)
            sbp = 100.0 * stop
            t["r"] = (t["bp"] - cost_bp(t)) / sbp
            t["r2"] = (t["bp"] - cost_bp(t, 2)) / sbp
            t["era"] = pd.cut(t["day"].dt.year, *ERAS[:1], labels=ERAS[1])
            keep[(name, stop)] = t
            for era, g in list(t.groupby("era")) + [("ALL", t)]:
                if len(g) < 30:
                    continue
                rows.append(dict(window=name, stop_pct=stop, era=era,
                                 n=len(g), win_pct=100 * (g["r"] > 0).mean(),
                                 avg_r=g["r"].mean(), t=tstat(g["r"]),
                                 avg_r_x2cost=g["r2"].mean(),
                                 stopped_pct=100 * (g["why"] == "stop")
                                 .mean()))
    d = pd.DataFrame(rows)
    print(d[d["era"] == "ALL"].round(3).to_string(index=False))
    print("\n  by era, 0.75% stop:")
    e = d[(d["stop_pct"] == 0.75) & (d["era"] != "ALL")]
    print(e.pivot(index="window", columns="era", values="avg_r")
          .round(3).to_string())

    print("\n  timing vs drift (no stop, bp a night, pre-cost): the window's"
          " own average against every same-length long")
    for name, (eh, xh) in WINDOWS.items():
        ln = (xh - eh) % 24
        own = window_trades(h, eh, xh)["bp"]
        others = []
        for sh in range(24):
            span = [(sh + k) % 24 for k in range(ln)]
            if 17 not in span:
                others.append(window_trades(h, sh, (sh + ln) % 24)["bp"]
                              .mean())
        print(f"    {name:24s} own {own.mean():+6.2f}   all {ln}-hour "
              f"windows {np.nanmean(others):+6.2f}   timing "
              f"{own.mean() - np.nanmean(others):+6.2f}")
    return keep


# ------------------------------------------------------------------ part 3

def replay_r(r_seq, start_bal, risk_pct):
    """The5ers one-step on a sequence of per-day R (one or more trades)."""
    bal, risk = start_bal, EM.ACCOUNT * risk_pct / 100
    for k, r in enumerate(r_seq):
        day_start = bal
        bal += r * risk
        if bal >= EM.TARGET:
            return "pass", k + 1
        if bal <= EM.FLOOR or day_start - bal >= EM.DAILY:
            return "fail", k + 1
    return "open", len(r_seq)


def pass_table(seqsets, risks):
    rows = []
    for start_pct in (4.0, 4.5):
        start_bal = EM.ACCOUNT * (1 + start_pct / 100)
        for label, seqs in seqsets.items():
            for risk in risks:
                for era_name, y0 in (("2000-26", 2000), ("2021-26", 2021)):
                    res = []
                    for s in seqs:
                        s = s.reset_index(drop=True)
                        starts = s.index[(s["day"].dt.year >= y0)
                                         & (s.index < len(s) - EM.DAYS_LEFT)]
                        for i0 in starts[::5]:
                            seq = s["r"].iloc[i0:i0 + EM.DAYS_LEFT]
                            res.append(replay_r(seq.to_numpy(), start_bal,
                                                risk)[0])
                    if not res:
                        continue
                    res = pd.Series(res)
                    rows.append(dict(start=f"+{start_pct}%", system=label,
                                     risk_pct=f"{risk:g}", starts=era_name,
                                     pass_pct=100 * (res == "pass").mean(),
                                     fail_pct=100 * (res == "fail").mean(),
                                     open_pct=100 * (res == "open").mean()))
    return pd.DataFrame(rows)


def part3(h, keep, stop=0.75):
    print("\n" + "=" * 100)
    print(f"PART 3. THE5ERS ONE-STEP, $5,000, {MAIN}, {stop}% stop")
    print("=" * 100)
    eh, xh = WINDOWS[MAIN]
    sys_t = keep[(MAIN, stop)]
    sbp = 100.0 * stop
    # Same nights, the window's own average swapped for the plain drift.
    drift = np.nanmean([window_trades(h, sh, (sh + 10) % 24)["bp"].mean()
                        for sh in range(24)
                        if 17 not in [(sh + k) % 24 for k in range(10)]])
    flat = sys_t.assign(r=sys_t["r"] - sys_t["r"].mean()
                        + (drift - cost_bp(sys_t).mean()) / sbp)
    rng = np.random.default_rng(7)
    rand = []
    for _ in range(6):
        t = window_trades(h, eh, xh, stop, "random", rng)
        rand.append(t.assign(r=(t["bp"] - cost_bp(t)) / sbp))
    tab = pass_table({"system": [sys_t], "timing removed": [flat],
                      "random direction": rand}, (0.75, 1.0, 1.5))
    print(tab.round(1).to_string(index=False))


def part4(keep, stop=0.75):
    print("\n" + "=" * 100)
    print("PART 4. WITH THE EUR/USD SHORT ON THE SAME NIGHTS")
    print("=" * 100)
    try:
        import examples.eur_morning_late as EL
        eur = EL.overnight_trades(EL.load_all_hours(), 23, 8, 40.0)
    except Exception as exc:                            # noqa: BLE001
        print(f"  skipped: {exc}")
        return
    eur = eur.assign(r=(eur["pips"] - EM.cost_pips()) / 40.0)
    nas = keep[(MAIN, stop)]
    both = nas.set_index("day")[["r"]].join(
        eur.set_index("day")[["r"]], lsuffix="_nas", rsuffix="_eur",
        how="inner")
    print(f"  nights with both: {len(both)}   correlation of results "
          f"{both['r_nas'].corr(both['r_eur']):+.2f}")
    print(f"  both stopped/losing the same night: "
          f"{100 * ((both['r_nas'] < 0) & (both['r_eur'] < 0)).mean():.1f}%"
          f"   combined R a night {both.sum(axis=1).mean():+.3f}")
    comb = both.sum(axis=1).rename("r").reset_index()
    solo = eur[["day", "r"]]
    tab = pass_table({"EUR short alone": [solo],
                      "EUR short + NAS long": [comb]}, (0.75, 1.0, 1.5))
    print("  risk_pct is per trade; two trades a night risk twice that")
    print(tab.round(1).to_string(index=False))


def main() -> int:
    pd.set_option("display.width", 230)
    h = load_h1()
    part1(h)
    keep = part2(h)
    part3(h, keep)
    part4(keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
