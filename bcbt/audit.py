"""
Two checks that would have caught most of what went wrong in this repo.

Looking back over everything tested here, the damaging mistakes fall into
two groups, and neither was a mistake of reasoning. They were mistakes of
plumbing that produced beautiful numbers.

  Using information before it existed. The worst case: hourly bars are
  labelled with the time they START, so reading a breakout from the bar's
  label let trades happen inside the breakout hour, before that hour had
  closed and confirmed anything. That single hour of hindsight turned a dead
  strategy into one with a t-statistic of 15. A tz-aware index silently
  turning into objects, and a breakeven latch not carried between windows,
  are the same disease in milder form.

  Not checking whether the idea could pay for itself. Three separate
  strategies -- index pairs, the zone retest, overnight drift -- turned out
  to have a real edge that was roughly the size of the spread. Each took
  hours to find out. The arithmetic takes a second and could have been done
  first.

`future_poison` is the general answer to the first. Corrupt everything after
some moment, re-run, and compare the decisions made BEFORE that moment. They
must be identical: a decision made on Tuesday cannot depend on Thursday. If
they differ, something is reading ahead, and the check says where.

`cost_feasibility` is the answer to the second. Given how big the stop is and
how wide the spread is, it says what the strategy must achieve merely to
break even, and how big an edge it needs to be worth trading at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _poison_once(signal_fn, data, cut, noise, jump, seed, key):
    """One cut: corrupt everything after it and compare what came before."""
    """
    Detect information leaking backwards in time.

    `signal_fn(frame) -> DataFrame` produces decisions, each stamped with
    the moment it was made in column `key`. The same function is run twice:
    once on the real data, once on data whose values after `cut` have been
    replaced with noise. Every decision dated at or before `cut` must be
    identical in both runs.

    The corruption has to bite from the very first bar after `cut`, or a
    rule that peeks only one bar ahead slips through: a random walk that
    starts at 1.0 barely changes the next value. So the future is displaced
    by `jump` immediately and then wanders by `noise` a bar.

    One multiplier is shared across all price columns, so a bar stays a
    coherent bar -- high above close above low. Corrupting each column
    separately makes nonsense bars, which strategies then reject for
    reasons that have nothing to do with looking ahead.

    Returns a report. `leak` is True when the two disagree, which means the
    strategy could not have made those decisions at the time it claims.
    """
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(cut)
    if data.index.tz is not None and cut.tz is None:
        cut = cut.tz_localize(data.index.tz)

    clean = signal_fn(data)

    spoiled = data.copy()
    after = spoiled.index > cut
    n_after = int(after.sum())
    if n_after == 0:
        raise ValueError("cut is at or past the end of the data")
    mult = (1.0 + jump) * np.exp(
        np.cumsum(rng.normal(0.0, noise, size=n_after)))
    for col in spoiled.columns:
        # pandas' own check, not numpy's: numpy cannot interpret a
        # timezone-aware datetime dtype and raises instead of saying no.
        # That confusion is itself one of the recurring faults this module
        # exists to catch.
        if not pd.api.types.is_numeric_dtype(spoiled[col]):
            continue
        spoiled.loc[after, col] = (
            spoiled.loc[after, col].to_numpy(float) * mult)
    poisoned = signal_fn(spoiled)

    a = clean[clean[key] <= cut] if len(clean) else clean
    b = poisoned[poisoned[key] <= cut] if len(poisoned) else poisoned

    same_count = len(a) == len(b)
    detail = ""
    if same_count and len(a):
        cols = [c for c in a.columns if c in b.columns
                and pd.api.types.is_numeric_dtype(a[c])]
        aa = a.reset_index(drop=True)
        bb = b.reset_index(drop=True)
        mism = {}
        for c in cols:
            d = ~np.isclose(aa[c].to_numpy(float), bb[c].to_numpy(float),
                            rtol=1e-9, atol=1e-12, equal_nan=True)
            if d.any():
                mism[c] = int(d.sum())
        leak = bool(mism)
        if mism:
            detail = ", ".join(f"{k}: {v} differ" for k, v in mism.items())
    else:
        leak = True
        detail = f"{len(a)} decisions clean vs {len(b)} poisoned"

    return dict(leak=leak, n_clean=len(a), n_poisoned=len(b),
                cut=cut, detail=detail)


def future_poison(signal_fn, data: pd.DataFrame, cut=None, probes=15,
                  noise=0.05, jump=0.25, seed=0, key="entry_ts"):
    """
    Probe for look-ahead by cutting at the decisions themselves.

    The obvious design -- cut at a dozen arbitrary points -- does not work,
    and was caught here failing against a leak known to exist. A rule that
    peeks one hour ahead only misbehaves when a cut lands inside that hour
    AND a decision sits between the cut and the hour's end. On a year of
    minute data that is a few minutes of sensitivity out of 280,000, so the
    leak passes and the check reports clean. A reassuring check that cannot
    see the bug it was built for is worse than none.

    So the cut is placed AT a decision's own timestamp. Everything strictly
    after it is corrupted, and that decision must survive unchanged: it
    claimed to be made at that moment, so nothing later can matter to it.
    A sample of decisions is probed, because each one costs a full re-run.

    Returns a report; `leak` is True if any probe changed a decision that
    had already been made.
    """
    if cut is not None:
        return _poison_once(signal_fn, data, cut, noise, jump, seed, key)

    clean = signal_fn(data)
    if clean is None or len(clean) == 0:
        return dict(leak=False, probes_fired=0, probes_tried=0,
                    n_clean=0, n_poisoned=0, cut=None,
                    detail="no decisions to probe")

    ts = pd.DatetimeIndex(clean[key]).sort_values()
    # Skip the very first and last: too little history or future to judge.
    lo, hi = int(len(ts) * 0.15), int(len(ts) * 0.9)
    pool = ts[lo:hi] if hi > lo else ts
    if len(pool) == 0:
        pool = ts
    step = max(1, len(pool) // max(1, probes))
    points = list(pool[::step])[:probes]

    fired, first = 0, None
    for i, c in enumerate(points):
        try:
            r = _poison_once(signal_fn, data, c, noise, jump, seed + i, key)
        except ValueError:
            continue
        if r["leak"]:
            fired += 1
            first = first or r
    if first:
        first["probes_fired"] = fired
        first["probes_tried"] = len(points)
        return first
    return dict(leak=False, probes_fired=0, probes_tried=len(points),
                n_clean=len(clean), n_poisoned=len(clean), cut=None,
                detail="")


def cost_feasibility(risk, spread, rr=2.0, label=""):
    """
    What this idea must do just to break even, given the spread.

    `risk` is the stop distance per trade in the instrument's own points,
    `spread` the round-trip cost in the same units. Everything is expressed
    in R -- multiples of the risk taken -- because that is the only way a
    Nasdaq stop and a euro stop are comparable.

    The number that matters is `breakeven_win_pct`: with a target of `rr`
    and the spread taken out, how often the trade must win. Three separate
    strategies in this repo turned out to have a genuine edge that was
    roughly the size of their spread, each discovered after hours of work.
    This takes a second and can be done first.
    """
    risk = np.asarray(risk, float)
    risk = risk[np.isfinite(risk) & (risk > 0)]
    if risk.size == 0:
        return dict(label=label, n=0)
    cost_r = spread / risk
    c = float(np.mean(cost_r))
    # Win w of the time for rr, lose 1 otherwise, pay c either way:
    #   w*rr - (1-w) - c = 0
    be = (1.0 + c) / (1.0 + rr)
    return dict(
        label=label, n=int(risk.size),
        median_risk=float(np.median(risk)),
        cost_r=c,
        cost_r_median=float(np.median(cost_r)),
        breakeven_win_pct=100.0 * be,
        breakeven_win_free_pct=100.0 / (1.0 + rr),
        extra_wins_needed_pct=100.0 * (be - 1.0 / (1.0 + rr)),
        edge_needed_r=c,
    )


def report_feasibility(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    cols = ["label", "n", "median_risk", "cost_r", "breakeven_win_pct",
            "breakeven_win_free_pct", "extra_wins_needed_pct"]
    return d[cols].round(4)
