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
REASONS = {OPEN: "open", STOP: "stop", TARGET: "target", BREAKEVEN: "be"}


@njit(cache=True)
def resolve(o, h, l, i0, i1, d, entry, stop, risk, target, be_mult,
            armed_in=False):
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

    Returns
    -------
    (exit_index, exit_price, reason, ambiguous, cur_stop, armed)
        exit_index is -1 and reason is OPEN if the window ended with the
        position still open; cur_stop and armed carry the latch forward.
    """
    cur_stop = entry if armed_in else stop
    armed = armed_in

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
            return j, cur_stop, (BREAKEVEN if armed else STOP), True, cur_stop, armed

        if hit_stop:
            if d > 0:
                px = o[j] if o[j] < cur_stop else cur_stop
            else:
                px = o[j] if o[j] > cur_stop else cur_stop
            return j, px, (BREAKEVEN if armed else STOP), False, cur_stop, armed

        if hit_tgt:
            if d > 0:
                px = o[j] if o[j] > target else target
            else:
                px = o[j] if o[j] < target else target
            return j, px, TARGET, False, cur_stop, armed

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
                    return j, entry, BREAKEVEN, False, cur_stop, armed
                if d < 0 and h[j] >= entry:
                    return j, entry, BREAKEVEN, False, cur_stop, armed

    return -1, 0.0, OPEN, False, cur_stop, armed


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
    resolve(z, z, z, 0, 3, 1, 1.0, 0.5, 0.5, 0.0, 0.0, False)
    first_touch(z, z, 0, 3, 1, 1.0)
