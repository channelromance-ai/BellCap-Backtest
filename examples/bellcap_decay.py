"""
Does BellCap's macro read pay off overnight, and when?

BellCap publishes its ideas at about 21:31 UTC (17:31 New York, 16:31
Winnipeg), after the New York close. Two questions:

  b  Does the read's edge fade through the night? Track every idea from the
     first hourly open after publication (18:00 New York) to the next New
     York close, hour by hour, in the idea's direction.
  a  Is "open in Asia, close in New York" a trade? Enter at 19:00 or 23:00
     New York (18:00 / 22:00 Winnipeg), exit at 08:00, 12:00 or 16:00 New
     York, after The5ers' measured cost. All inside one The5ers trading day,
     so no swap.

Only the ideas BellCap actually logged are used (35 nights from 15 August
2026). Rebuilding older ideas is not honest: the carry component reads
policy rates from today's config, so a rebuilt 2025 idea would know 2026's
rates. Ideas on one night share currencies and move together, so a NIGHT is
the unit of evidence: ideas are averaged within the night first.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = r"C:\Users\Me\Documents\GitHub\BellCap-Terminal\macro.db"
NY_OFFSET = pd.Timedelta(hours=7)          # server = New York + 7
EXCLUDE = {"BTCUSD"}                       # trades weekends; different costs


def publication_times(c):
    """
    When each asof's ideas were really written, in New York time.

    `asof` is a data date, not a publish time: on 18 of 38 logged runs the
    evening run labelled its ideas with the NEXT day's date, and one manual
    run wrote at 19:02 New York. Ideas are INSERT OR REPLACE, so the last
    run for an asof is the one whose ideas are in the table. Nights with no
    run record fall back to the scheduled 17:31 New York on the asof date.
    """
    runs = pd.read_sql("select ts, asof as run_asof from runs "
                       "where asof is not null", c)
    runs["ny"] = (pd.to_datetime(runs["ts"], format="ISO8601", utc=True)
                  .dt.tz_convert("America/New_York").dt.tz_localize(None))
    return runs.groupby("run_asof")["ny"].max()


def first_clean_hour(pub):
    """The first hourly open after publication, skipping 18:00 New York --
    the hour after the rollover, when sell-side prices recover from the
    spread blow-out and rise on about 80% of days for every instrument."""
    start = pub.ceil("h")
    if start == pub:
        start += pd.Timedelta(hours=1)
    if start.hour in (17, 18):
        start = start.normalize() + pd.Timedelta(hours=19)
    return start


def load():
    c = sqlite3.connect(DB)
    ideas = pd.read_sql("select asof, pair, direction, score, conviction, "
                        "passed, shortlisted from ideas", c)
    ideas = ideas[~ideas["pair"].isin(EXCLUDE)]
    pub = publication_times(c)
    default = pd.to_datetime(ideas["asof"]) + pd.Timedelta(hours=17,
                                                           minutes=31)
    ideas["published"] = ideas["asof"].map(pub).fillna(default)
    ideas["start"] = ideas["published"].map(first_clean_hour)
    h = pd.read_parquet("data/bellcap_h1.parquet")
    h["ny"] = h["server"] - NY_OFFSET
    costs = pd.read_csv("data/the5ers_costs_all.csv", index_col=0)["cost_bp"]
    return ideas, h, costs


def paths(ideas, h, n_bars=22):
    """Signed cumulative move (bp) at each hour from the first clean hour
    after publication (normally 19:00 New York) to the next NY close."""
    series = {p: g.set_index("ny").sort_index() for p, g in h.groupby("pair")}
    rows = []
    for r in ideas.itertuples(index=False):
        g = series.get(r.pair)
        if g is None:
            continue
        after = g[g.index >= r.start].head(n_bars)
        if len(after) < n_bars:
            continue
        d = 1 if r.direction == "long" else -1
        ref = after["open"].iloc[0]
        cum = d * 1e4 * (after["close"].to_numpy() / ref - 1)
        row = dict(asof=r.asof, pair=r.pair, passed=r.passed,
                   shortlisted=r.shortlisted, d=d)
        for k, (t, v) in enumerate(zip(after.index, cum, strict=True)):
            row[f"h{k}"] = v
            row[f"t{k}"] = t
        rows.append(row)
    return pd.DataFrame(rows)


def by_night(p, cols):
    return p.groupby("asof")[cols].mean()


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 \
        else np.nan


def trade(ideas, h, costs, entry_ny, exit_ny):
    """Enter at the open of entry_ny (evening after publication), exit at
    the open of exit_ny the next morning/afternoon, New York hours."""
    series = {p: g.set_index("ny").sort_index() for p, g in h.groupby("pair")}
    rows = []
    for r in ideas.itertuples(index=False):
        g = series.get(r.pair)
        if g is None:
            continue
        seg = g[g.index >= r.start]
        if seg.empty:
            continue
        # Next occurrence of the entry hour after publication, then the next
        # occurrence of the exit hour after that.
        ent = seg[seg.index.hour == entry_ny]
        if ent.empty:
            continue
        t0 = ent.index[0]
        ex = seg[(seg.index > t0) & (seg.index.hour == exit_ny)]
        if ex.empty or ex.index[0] - t0 > pd.Timedelta(hours=30):
            continue
        d = 1 if r.direction == "long" else -1
        gross = d * 1e4 * (ex["open"].iloc[0] / ent["open"].iloc[0] - 1)
        cost = costs.get(r.pair, 1.2)
        rows.append(dict(asof=r.asof, pair=r.pair, passed=r.passed,
                         shortlisted=r.shortlisted, gross=gross,
                         net=gross - cost))
    return pd.DataFrame(rows)


def main() -> int:
    pd.set_option("display.width", 220)
    ideas, h, costs = load()
    print(f"ideas: {len(ideas)} over {ideas['asof'].nunique()} nights "
          f"({ideas['asof'].min()} to {ideas['asof'].max()}); passed "
          f"{int(ideas['passed'].sum())}, shortlisted "
          f"{int(ideas['shortlisted'].sum())}")

    late = (ideas.groupby("asof")["start"].first().dt.hour != 19).sum()
    print(f"nights whose first clean hour is not 19:00 New York (late manual "
          f"runs): {late}")
    p = paths(ideas, h)
    hcols = [f"h{k}" for k in range(22)]
    # Clock time at the END of each bar, for the usual 19:00 start.
    labels = [f"{(20 + k) % 24:02d}" for k in range(22)]
    print("\n" + "=" * 110)
    print("b. CUMULATIVE MOVE IN THE IDEA'S DIRECTION, bp, from the first clean"
          " hour after publication (19:00 NY / 18:00 Wpg). Nights are the"
          " unit.")
    print("=" * 110)
    out = {}
    for name, sel in (("all ideas", p),
                      ("passed gates", p[p["passed"] == 1]),
                      ("shortlist", p[p["shortlisted"] == 1])):
        n = by_night(sel, hcols)
        out[name] = n
        res = pd.DataFrame({
            "avg_bp": n.mean().to_numpy(),
            "t": [tstat(n[c]) for c in hcols],
            "nights_right_pct": [100 * (n[c] > 0).mean() for c in hcols],
        }, index=[f"{lab}:00 NY / {(int(lab) - 1) % 24:02d}:00 Wpg"
                  for lab in labels]).T
        print(f"\n  {name}: {len(sel)} ideas, {len(n)} nights")
        print(res.iloc[:, ::2].round(2).to_string())

    print("\n  by session, marginal move added in each (nights):")
    for name in ("all ideas", "passed gates", "shortlist"):
        n = out[name]
        seg = {"Asia 19-02": n["h6"], "London 02-08": n["h12"] - n["h6"],
               "New York 08-17": n["h21"] - n["h12"]}
        print(f"    {name:<13} " + "   ".join(
            f"{k} {v.mean():+6.2f}bp (t {tstat(v):+.2f}, right "
            f"{100 * (v > 0).mean():.0f}%)" for k, v in seg.items()))

    print("\n" + "=" * 110)
    print("a. OPEN IN ASIA, CLOSE IN NEW YORK: after The5ers cost, bp per idea"
          " (nights are the unit)")
    print("=" * 110)
    rows = []
    for e in (19, 23):
        for x in (8, 12, 16):
            t = trade(ideas, h, costs, e, x)
            for name, sel in (("all", t), ("passed", t[t["passed"] == 1]),
                              ("shortlist", t[t["shortlisted"] == 1])):
                nn = sel.groupby("asof")[["gross", "net"]].mean()
                rows.append(dict(
                    entry=f"{e:02d}:00 NY ({(e - 1) % 24:02d}:00 Wpg)",
                    exit=f"{x:02d}:00 NY ({x - 1:02d}:00 Wpg)", ideas=name,
                    trades=len(sel), nights=len(nn),
                    gross_bp=nn["gross"].mean(), net_bp=nn["net"].mean(),
                    t_net=tstat(nn["net"]),
                    nights_up_pct=100 * (nn["net"] > 0).mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n  for scale, BellCap's own multi-day outcomes (outcomes table):")
    c = sqlite3.connect(DB)
    o = pd.read_sql("select i.asof, i.pair, i.direction, i.shortlisted, "
                    "o.fwd_1d, o.fwd_3d, o.fwd_5d, o.fwd_10d from ideas i "
                    "join outcomes o on o.asof = i.asof and "
                    "o.symbol = i.pair", c)
    if len(o):
        s = np.where(o["direction"] == "long", 1, -1)
        for col in ("fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d"):
            v = o[col] * s
            ok = v.notna()
            nn = (v[ok].groupby(o.loc[ok, "asof"]).mean())
            print(f"    {col}: ideas {ok.sum()}, nights {len(nn)}, avg "
                  f"{v[ok].mean():+.4f}, right {100 * (v[ok] > 0).mean():.1f}%"
                  f", night t {tstat(nn):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
