"""
Minute-resolution fill engine.

Every strategy in this package reads its signals off a slow chart -- 5-minute
bars, hourly bars -- and resolves what happened to the position on the
1-minute bars underneath. That separation is the point. A backtest that asks
a 5-minute candle whether the stop or the target came first has to guess, and
the guess is worth about a third of the result.

The rules, all of which cost the strategy performance:

  * The stop is checked before the target. Where one minute bar contains
    both, the stop is taken and the trade is flagged ambiguous. A minute bar
    spanning both levels is rare -- typically a handful of trades in a
    thousand -- and counting them is how you know it stayed rare.
  * A bar that opens through a level fills at the open, not at the level.
    That is worse than the level for a stop and better for a target, which
    is what actually happens.
  * Arming breakeven does not protect the bar it armed on: if the same
    minute that reached the trigger also traded back to the entry, the
    trade exits at breakeven.

Everything here is nopython numba over plain float64 arrays. It is roughly
sixty times faster than the equivalent numpy, which matters because a
variant grid calls it tens of thousands of times.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Exit reasons, as small ints so they can cross the numba boundary.
OPEN = -1
STOP = 0
TARGET = 1
BREAKEVEN = 2
TRAIL = 3
REASONS = {OPEN: "open", STOP: "stop", TARGET: "target", BREAKEVEN: "be",
           TRAIL: "trail"}


@njit(cache=True)
def resolve(o, h, l, i0, i1, d, entry, stop, risk, target, be_mult,
            armed_in=False, trail_mult=0.0, best_in=0.0):
    """
    Walk 1-minute bars i0..i1 for one open position.

    Callers that resolve a trade in pieces -- hour by hour, say -- must
    thread `armed_in` and the returned stop back in on the next call.
    Breakeven is a latch: once the trade has been 1R onside it stays
    protected, and a caller that forgets makes the position re-earn that
    protection every window, which quietly changes the strategy.

    Parameters
    ----------
    o, h, l : float64[:]
        Open, high and low of the 1-minute series.
    i0, i1 : int
        Inclusive bounds of the window to scan.
    d : int
        +1 for a long, -1 for a short.
    entry, stop, risk : float
        Fill price, initial stop price, and abs(entry - stop).
    target : float
        Take-profit price, or 0.0 for no fixed target.
    be_mult : float
        Move the stop to entry once the trade is this many R onside.
        0.0 disables the breakeven move.
    trail_mult : float
        Follow price with a stop this many R behind the best level reached
        since entry. 0.0 disables trailing. The trail only ever tightens.
    best_in : float
        Best price seen so far, for a trade being resolved in pieces. Pass
        the value returned by the previous call; 0.0 means "start from the
        entry".

    Returns
    -------
    (exit_index, exit_price, reason, ambiguous, cur_stop, armed, best)
        exit_index is -1 and reason is OPEN if the window ended with the
        position still open; cur_stop, armed and best carry forward.
    """
    cur_stop = entry if armed_in else stop
    armed = armed_in
    best = best_in if best_in != 0.0 else entry

    for j in range(i0, i1 + 1):
        if d > 0:
            hit_stop = l[j] <= cur_stop
            hit_tgt = target > 0.0 and h[j] >= target
        else:
            hit_stop = h[j] >= cur_stop
            hit_tgt = target > 0.0 and l[j] <= target

        # One bar holding both levels cannot say which came first, so the
        # answer always goes against the position.
        if hit_stop and hit_tgt:
            return j, cur_stop, (BREAKEVEN if armed else STOP), True, cur_stop, armed, best

        if hit_stop:
            if d > 0:
                px = o[j] if o[j] < cur_stop else cur_stop
            else:
                px = o[j] if o[j] > cur_stop else cur_stop
            return j, px, (BREAKEVEN if armed else STOP), False, cur_stop, armed, best

        if hit_tgt:
            if d > 0:
                px = o[j] if o[j] > target else target
            else:
                px = o[j] if o[j] < target else target
            return j, px, TARGET, False, cur_stop, armed, best

        if be_mult > 0.0 and not armed:
            if d > 0:
                reached = h[j] >= entry + be_mult * risk
            else:
                reached = l[j] <= entry - be_mult * risk
            if reached:
                armed = True
                cur_stop = entry
                # The bar that armed breakeven may also have come back to it.
                if d > 0 and l[j] <= entry:
                    return j, entry, BREAKEVEN, False, cur_stop, armed, best
                if d < 0 and h[j] >= entry:
                    return j, entry, BREAKEVEN, False, cur_stop, armed, best

        if trail_mult > 0.0:
            # Tighten behind the best level reached. Applied after this bar
            # has been judged, so the trail never exits on the same bar that
            # created the high it is measured from.
            if d > 0:
                if h[j] > best:
                    best = h[j]
                lifted = best - trail_mult * risk
                if lifted > cur_stop:
                    cur_stop = lifted
            else:
                if l[j] < best:
                    best = l[j]
                lifted = best + trail_mult * risk
                if lifted < cur_stop:
                    cur_stop = lifted

    return -1, 0.0, OPEN, False, cur_stop, armed, best


@njit(cache=True)
def resolve_trail(o, h, l, i0, i1, d, entry, stop, risk, act_r, trail_r,
                  target=0.0, stop_first=True):
    """
    Walk 1-minute bars i0..i1 under a stop that is managed in two stages.

    Until the trade has been `act_r` R onside the initial stop stays where
    it is. From the bar that reaches that point the stop moves to entry,
    then follows the best price since entry at `trail_r` R behind it, and
    only ever tightens. This differs from `resolve(trail_mult=...)`, which
    trails from the first bar and so tightens the stop long before the
    trade has earned anything.

    `act_r` 0.0 leaves the initial stop alone all day; `trail_r` 0.0 with
    `act_r` > 0 is a plain breakeven move. `target` 0.0 means no target.

    The intrabar order problem. A minute bar that pushes the stop up (by
    arming, or by a new extreme) and also trades back through the new stop
    cannot say which came first. With `stop_first` the exit is taken at the
    new stop; without it the bar is read low-then-high and the position
    survives. Either way the bar is flagged ambiguous, so running both
    bounds the effect. The same flag covers a bar holding both stop and
    target, and is returned set if any bar of the trade was such a bar.

    Returns
    -------
    (exit_index, exit_price, reason, ambiguous, armed, mfe, mae)
        reason is STOP before the trade armed and TRAIL after it; OPEN with
        exit_index -1 if the window ended first. mfe and mae are the best
        and worst excursions in price points, both >= 0.
    """
    cur = stop
    armed = False
    best = entry
    mfe = 0.0
    mae = 0.0
    amb = False

    for j in range(i0, i1 + 1):
        if d > 0:
            fav = h[j] - entry
            adv = entry - l[j]
            hit_stop = l[j] <= cur
            hit_tgt = target > 0.0 and h[j] >= target
        else:
            fav = entry - l[j]
            adv = h[j] - entry
            hit_stop = h[j] >= cur
            hit_tgt = target > 0.0 and l[j] <= target
        if fav > mfe:
            mfe = fav
        if adv > mae:
            mae = adv
        why = TRAIL if armed else STOP

        if hit_stop and hit_tgt:
            if stop_first:
                return j, cur, why, True, armed, mfe, mae
            return j, target, TARGET, True, armed, mfe, mae

        if hit_stop:
            if d > 0:
                px = o[j] if o[j] < cur else cur
            else:
                px = o[j] if o[j] > cur else cur
            return j, px, why, amb, armed, mfe, mae

        if hit_tgt:
            if d > 0:
                px = o[j] if o[j] > target else target
            else:
                px = o[j] if o[j] < target else target
            return j, px, TARGET, amb, armed, mfe, mae

        if act_r <= 0.0:
            continue

        if d > 0:
            if h[j] > best:
                best = h[j]
        else:
            if l[j] < best:
                best = l[j]

        moved = False
        if not armed and fav >= act_r * risk:
            armed = True
            if (d > 0 and entry > cur) or (d < 0 and entry < cur):
                cur = entry
                moved = True
        if armed and trail_r > 0.0:
            lifted = best - d * trail_r * risk
            if (d > 0 and lifted > cur) or (d < 0 and lifted < cur):
                cur = lifted
                moved = True

        if moved:
            inside = (l[j] <= cur) if d > 0 else (h[j] >= cur)
            if inside:
                amb = True
                if stop_first:
                    return j, cur, TRAIL, True, armed, mfe, mae

    return -1, 0.0, OPEN, amb, armed, mfe, mae


@njit(cache=True)
def first_touch(h, l, i0, i1, d, level):
    """Index of the first bar in i0..i1 to trade at `level`, or -1."""
    for j in range(i0, i1 + 1):
        if d > 0:
            if h[j] >= level:
                return j
        else:
            if l[j] <= level:
                return j
    return -1


def warm():
    """
    Compile the kernels on a throwaway call.

    Worth doing once at import in a timing run, so the first strategy
    evaluated is not charged for the JIT.
    """
    z = np.zeros(4, np.float64)
    resolve(z, z, z, 0, 3, 1, 1.0, 0.5, 0.5, 0.0, 0.0, False, 0.0, 0.0)
    resolve_trail(z, z, z, 0, 3, 1, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, True)
    first_touch(z, z, 0, 3, 1, 1.0)
