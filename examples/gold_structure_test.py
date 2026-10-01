"""
The user's structure theory for gold range breakouts, tested.

From an annotated chart (22 Sep 2026): inside a range, the early touches of
the edges define it; LATER dips that stop short of the bottom (higher lows)
show buyers stepping in earlier, building toward an upside break (mirror
for the downside: lower highs). Entry on the breakout, stop under the last
higher low, target the swing high from before the drop into the range,
only with the 4-hour trend, preferably in Asia, else New York. An earlier
failed poke through the top (the user's "?") was unexplained.

Fixed before running, on every breakout from gold_ranges.py's live ranges,
2005-2026 (develop 2005-2015, one look at 2016-2026):

  structure   the last two confirmed swing lows (2 bars each side, confirmed
              before the breakout) inside the range are RISING and the later
              one is above the bottom 20% of the range (mirror for downside)
  poke        earlier in the range a bar's wick went more than 30% of the
              range height beyond the breakout edge and closed back inside
  session     breakout bar: Asia (19:00-02:00 NY), London (02:00-08:00),
              New York (08:00-17:00)
  H4          the H4 50-EMA agrees with the breakout

  Q1  does structure separate breakouts that hold from false ones?
  Q2  how should the "?" be read: do breakouts after a failed poke hold
      more often or less?
  Q3  the user's full setup: breakout + structure + H4, Asia or New York,
      next-bar entry, stop under the last higher low (lower high) less 10%
      of the range height, target the extreme of the 48 bars before the
      range began (skipped if that is not beyond the entry); and the same
      with a fixed 2R target. Out by the 17:00 New York roll.
  luck        the full setup's trades with directions randomised

False breakout = a close back inside the range within 2 hours (known only
afterwards; used for explanation, never as an entry rule).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_ranges as R                       # noqa: E402
import examples.gold_ranges_test as T                  # noqa: E402
import examples.gold_zones as G                        # noqa: E402

CACHE = "data/gold_live_ranges.npz"


def live(m):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if len(z["start"]) == len(m):
            return z["start"], z["top"], z["bot"]
    st, tp, bt = R.live_ranges_fast(m)
    np.savez(CACHE, start=st, top=tp, bot=bt)
    return st, tp, bt


def swings(h, l, n=2):
    """Confirmed swing points: index j is a swing low if l[j] is the minimum
    of j-n..j+n; it is only KNOWN at bar j+n."""
    ll = pd.Series(l).rolling(2 * n + 1, center=True).min().to_numpy()
    hh = pd.Series(h).rolling(2 * n + 1, center=True).max().to_numpy()
    return np.flatnonzero(l == ll), np.flatnonzero(h == hh)


def features(m, bo, sw_lo, sw_hi, n=2):
    o, h, l, c = (m[x].to_numpy(np.float64) for x in ("open", "high", "low",
                                                      "close"))
    hour = m["server"].dt.hour.to_numpy()
    rows = []
    for r in bo.itertuples(index=False):
        k, d, s, e, top, bot = r.k, r.d, r.start, r.end, r.top, r.bot
        H = top - bot
        # swings inside the range, confirmed by bar k-1 at the latest
        if d > 0:
            sw = sw_lo[(sw_lo >= s) & (sw_lo + n <= k - 1)]
            vals = l[sw]
            structure = len(sw) >= 2 and vals[-1] > vals[-2] and \
                vals[-1] > bot + 0.2 * H
            struct_level = vals[-1] if len(sw) else np.nan
            poke = bool(np.any((h[s:e + 1] > top + 0.3 * H)
                               & (c[s:e + 1] <= top)))
            before = h[max(0, s - 48):s]
            target = before.max() if len(before) else np.nan
        else:
            sw = sw_hi[(sw_hi >= s) & (sw_hi + n <= k - 1)]
            vals = h[sw]
            structure = len(sw) >= 2 and vals[-1] < vals[-2] and \
                vals[-1] < top - 0.2 * H
            struct_level = vals[-1] if len(sw) else np.nan
            poke = bool(np.any((l[s:e + 1] < bot - 0.3 * H)
                               & (c[s:e + 1] >= bot)))
            before = l[max(0, s - 48):s]
            target = before.min() if len(before) else np.nan
        false_bo = bool(any(bot <= c[j] <= top
                            for j in range(k + 1, min(k + 9, len(c)))))
        hr = hour[k]
        session = "asia" if 2 <= hr <= 8 else ("london" if 9 <= hr <= 14
                                               else ("new_york" if 15 <= hr
                                                     <= 23 else "rollover"))
        rows.append(dict(k=k, d=d, top=top, bot=bot, H=H,
                         structure=structure, struct_level=struct_level,
                         poke=poke, target=target, false_bo=false_bo,
                         session=session))
    return pd.DataFrame(rows)


def run_trade(m, sp, i, d, stop, target):
    """Enter next open; stop / target / roll; R after cost, or NaN."""
    o, h, l, c = (m[x].to_numpy(np.float64) for x in ("open", "high", "low",
                                                      "close"))
    day = m["server"].dt.normalize().to_numpy()
    n = len(c)
    j = i + 1
    if j >= n or not np.isfinite(stop) or not np.isfinite(target):
        return np.nan
    entry = o[j]
    risk = (entry - stop) * d
    cost = sp[j] + G.COMMISSION
    if risk <= 2 * cost or (target - entry) * d <= 0:
        return np.nan
    k, px = j, np.nan
    while k < n and day[k] == day[j]:
        if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
            px = min(stop, o[k]) if d > 0 else max(stop, o[k])
            break
        if (d > 0 and h[k] >= target) or (d < 0 and l[k] <= target):
            px = max(target, o[k]) if d > 0 else min(target, o[k])
            break
        k += 1
    if np.isnan(px):
        px = c[min(k, n) - 1]
    return (d * (px - entry) - cost) / risk


def pf(x):
    x = np.asarray(x)[np.isfinite(x)]
    w, lo = x[x > 0].sum(), -x[x <= 0].sum()
    return w / lo if lo > 0 else np.inf


def main() -> int:
    pd.set_option("display.width", 230)
    t0 = time.time()
    m, h4 = T.load()
    st, tp, bt = live(m)
    bo = R.breakouts(m, st, tp, bt)
    sw_lo, sw_hi = swings(m["high"].to_numpy(), m["low"].to_numpy())
    f = features(m, bo, sw_lo, sw_hi)
    t = m["server"].to_numpy()
    f["h4"] = G.h4_bias(h4, t[f["k"]] + np.timedelta64(15, "m")) == f["d"]
    f["dev"] = t[f["k"]] < T.DEV_END
    _, sp = T.forward(m)
    # Far-edge stop / 2R: the baseline style from gold_ranges_test.
    f["r_edge"] = [run_trade(m, sp, r.k, r.d, r.bot if r.d > 0 else r.top,
                             (m["open"].iat[r.k + 1] if r.k + 1 < len(m)
                              else np.nan) + r.d * 2 * abs(
                                 (m["open"].iat[r.k + 1] if r.k + 1 < len(m)
                                  else np.nan) - (r.bot if r.d > 0 else r.top)))
                   for r in f.itertuples()]
    print(f"{len(f):,} breakouts  [{time.time() - t0:.0f}s]")

    def table(col, label):
        out = []
        for val, g in f.groupby(col):
            out.append(dict(**{label: val}, breakouts=len(g),
                            false_rate=f"{g['false_bo'].mean():.0%}",
                            pf_dev=pf(g.loc[g['dev'], 'r_edge']),
                            pf_val=pf(g.loc[~g['dev'], 'r_edge'])))
        print(pd.DataFrame(out).round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("Q1. DOES YOUR STRUCTURE PREDICT WHICH BREAKOUTS HOLD?  (profit factor "
          "= far-edge stop, 2R)")
    print("=" * 100)
    table("structure", "higher lows / lower highs")
    print("\n  with the 4-hour trend agreeing only:")
    g = f[f["h4"]]
    out = []
    for val, gg in g.groupby("structure"):
        out.append(dict(structure=val, breakouts=len(gg),
                        false_rate=f"{gg['false_bo'].mean():.0%}",
                        pf_dev=pf(gg.loc[gg['dev'], 'r_edge']),
                        pf_val=pf(gg.loc[~gg['dev'], 'r_edge'])))
    print(pd.DataFrame(out).round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("Q2. HOW TO READ THE '?': an earlier failed poke through the same edge")
    print("=" * 100)
    table("poke", "earlier failed poke")

    print("\n" + "=" * 100)
    print("SESSION and 4-HOUR TREND on their own")
    print("=" * 100)
    table("session", "session")
    table("h4", "4h trend agrees")

    print("\n" + "=" * 100)
    print("Q3. YOUR FULL SETUP: breakout + structure + 4h trend, Asia or New "
          "York")
    print("=" * 100)
    pick = f[f["structure"] & f["h4"] & f["session"].isin(["asia",
                                                           "new_york"])]
    rows = []
    rng = np.random.default_rng(5)
    for tgt_name in ("prior swing", "fixed 2R"):
        rs, rr_rand = [], []
        for r in pick.itertuples():
            stop = r.struct_level - r.d * 0.1 * r.H
            entry_next = m["open"].iat[r.k + 1] if r.k + 1 < len(m) else np.nan
            tgt = r.target if tgt_name == "prior swing" else \
                entry_next + r.d * 2 * abs(entry_next - stop)
            rs.append(run_trade(m, sp, r.k, r.d, stop, tgt))
            # luck: same moment, random direction, same distances mirrored
            dd = 1 if rng.random() < 0.5 else -1
            if np.isfinite(entry_next):
                st_r = entry_next - dd * abs(entry_next - stop)
                tg_r = entry_next + dd * abs(tgt - entry_next)
                rr_rand.append(run_trade(m, sp, r.k, dd, st_r, tg_r))
            else:
                rr_rand.append(np.nan)
        rs, rr_rand = np.array(rs), np.array(rr_rand)
        dev = pick["dev"].to_numpy()
        for sess in ("asia + new_york", "asia", "new_york"):
            ss = np.ones(len(pick), bool) if sess == "asia + new_york" else \
                (pick["session"] == sess).to_numpy()
            for part, pm in (("2005-15", dev), ("2016-26", ~dev)):
                x = rs[ss & pm]
                xr = rr_rand[ss & pm]
                rows.append(dict(target=tgt_name, session=sess, period=part,
                                 trades=int(np.isfinite(x).sum()),
                                 # wins among trades actually taken
                                 # (skipped trades are NaN, not losses)
                                 win_pct=100 * (x[np.isfinite(x)] > 0)
                                 .mean() if np.isfinite(x).any()
                                 else np.nan,
                                 avg_R=np.nanmean(x), pf=pf(x),
                                 pf_random_dir=pf(xr)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    f.to_csv("data/gold_structure_features.csv", index=False)
    print(f"\ndone [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
