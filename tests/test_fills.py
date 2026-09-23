"""
Unit tests for the fill engine.

These are the tests worth having. Everything else in a backtest is
arithmetic on top of "did this position survive these bars", and if that
answer is wrong by one bar in a hundred the equity curve is fiction.
"""
from __future__ import annotations

import numpy as np

from bcbt.fills import resolve, first_touch, STOP, TARGET, BREAKEVEN, OPEN


def bars(rows):
    """rows of (open, high, low) -> the three arrays resolve() wants."""
    a = np.array(rows, float)
    return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()


def test_long_stop_before_target():
    o, h, l = bars([(100, 101, 99), (100, 101, 94), (100, 110, 99)])
    i, px, code, amb, _, _, _ = resolve(o, h, l, 0, 2, 1, 100.0, 95.0, 5.0, 110.0, 0.0)
    assert (i, code, amb) == (1, STOP, False)
    assert px == 95.0


def test_long_target_before_stop():
    o, h, l = bars([(100, 101, 99), (100, 110, 99), (100, 101, 94)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 2, 1, 100.0, 95.0, 5.0, 110.0, 0.0)
    assert (i, code, px) == (1, TARGET, 110.0)


def test_tie_inside_one_bar_goes_against_the_trade():
    """A bar spanning both levels must be read as the stop, and flagged."""
    o, h, l = bars([(100, 110, 94)])
    i, px, code, amb, _, _, _ = resolve(o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 110.0, 0.0)
    assert (code, px, amb) == (STOP, 95.0, True)


def test_gap_through_stop_fills_at_the_open_not_the_level():
    o, h, l = bars([(90, 91, 89)])          # opened below a stop at 95
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 0.0, 0.0)
    assert code == STOP and px == 90.0      # worse than the stop, as in life


def test_gap_through_target_fills_at_the_open():
    o, h, l = bars([(115, 116, 114)])       # opened above a target at 110
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 110.0, 0.0)
    assert code == TARGET and px == 115.0   # better than the target


def test_breakeven_arms_then_protects():
    # Bar 1 reaches +1R (105) and holds above entry, so breakeven arms but
    # does not fire. Bar 2 falls back through 100: out at breakeven, not at
    # the original 95 stop.
    o, h, l = bars([(100, 101, 99), (101, 106, 100.5), (100, 101, 99.5)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 2, 1, 100.0, 95.0, 5.0, 0.0, 1.0)
    assert code == BREAKEVEN and px == 100.0 and i == 2


def test_breakeven_does_not_protect_the_bar_that_armed_it():
    """Reaching 1R and returning to entry inside one bar still exits flat."""
    o, h, l = bars([(101, 106, 99.5)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 0.0, 1.0)
    assert code == BREAKEVEN and px == 100.0


def test_without_breakeven_the_same_path_runs_to_the_stop():
    o, h, l = bars([(101, 106, 99.5), (100, 101, 94)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 1, 1, 100.0, 95.0, 5.0, 0.0, 0.0)
    assert code == STOP and i == 1


def test_position_can_survive_the_window():
    o, h, l = bars([(100, 101, 99), (100, 102, 98)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 1, 1, 100.0, 95.0, 5.0, 110.0, 0.0)
    assert (i, code) == (-1, OPEN)


def test_short_mirrors_long():
    o, h, l = bars([(100, 99, 98), (100, 106, 99)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 1, -1, 100.0, 105.0, 5.0, 90.0, 0.0)
    assert code == STOP and i == 1 and px == 105.0


def test_short_target():
    o, h, l = bars([(100, 101, 99), (100, 101, 89)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 1, -1, 100.0, 105.0, 5.0, 90.0, 0.0)
    assert code == TARGET and px == 90.0


def test_no_target_means_stop_or_nothing():
    o, h, l = bars([(100, 200, 99)])
    i, px, code, _, _, _, _ = resolve(o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 0.0, 0.0)
    assert code == OPEN


def test_first_touch():
    o, h, l = bars([(100, 101, 99), (100, 105, 99), (100, 110, 99)])
    assert first_touch(h, l, 0, 2, 1, 104.0) == 1
    assert first_touch(h, l, 0, 2, 1, 999.0) == -1
    assert first_touch(h, l, 0, 2, -1, 99.5) == 0


def test_window_bounds_are_respected():
    o, h, l = bars([(100, 101, 94), (100, 101, 94), (100, 101, 99)])
    # Scanning only bar 2 must not see the stop hits in bars 0 and 1.
    i, px, code, _, _, _, _ = resolve(o, h, l, 2, 2, 1, 100.0, 95.0, 5.0, 0.0, 0.0)
    assert code == OPEN


def test_breakeven_latch_survives_between_windows():
    """
    Resolving a trade in pieces must not make it re-earn breakeven.

    Bar 0 reaches +1R and arms. Scanning bar 0 alone leaves the position
    open but armed; feeding that state back in must protect bar 1, which
    dips to entry. Without the latch the trade would run to the 95 stop.
    """
    o, h, l = bars([(101, 106, 100.5), (100, 101, 99.8), (100, 101, 94)])
    i, px, code, _, cur_stop, armed, _ = resolve(
        o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 0.0, 1.0)
    assert code == OPEN and armed and cur_stop == 100.0

    i, px, code, _, cur_stop, armed, _ = resolve(
        o, h, l, 1, 2, 1, 100.0, cur_stop, 5.0, 0.0, 1.0, armed)
    assert code == BREAKEVEN and px == 100.0 and i == 1


def test_trail_follows_price_up_and_exits_on_the_pullback():
    """
    Price runs to 110, the trail sits 1R (5) behind at 105, and the next bar
    dipping to 104 takes it. Without the trail the position would survive.
    """
    o, h, l = bars([(100, 110, 99.5), (106, 107, 104.0)])
    i, px, code, _, cur_stop, _, best = resolve(
        o, h, l, 0, 1, 1, 100.0, 95.0, 5.0, 0.0, 0.0, False, 1.0, 0.0)
    assert code == STOP and i == 1
    assert px == 105.0 and best == 110.0

    # Same path, trailing switched off: nothing is hit.
    i2, _, code2, _, _, _, _ = resolve(
        o, h, l, 0, 1, 1, 100.0, 95.0, 5.0, 0.0, 0.0, False, 0.0, 0.0)
    assert code2 == OPEN and i2 == -1


def test_trail_never_loosens():
    """A lower high must not drag the stop back down."""
    o, h, l = bars([(100, 112, 99.5), (108, 109, 107.5), (108, 109, 106.5)])
    i, px, code, _, cur_stop, _, best = resolve(
        o, h, l, 0, 2, 1, 100.0, 95.0, 5.0, 0.0, 0.0, False, 1.0, 0.0)
    # Trail locked at 112-5 = 107 on bar 0; bar 2's low of 106.5 takes it.
    assert code == STOP and i == 2 and px == 107.0
    assert best == 112.0


def test_trail_does_not_exit_on_the_bar_that_made_the_high():
    """The bar creating the high is judged before the trail tightens."""
    o, h, l = bars([(100, 110, 100.0)])
    i, px, code, _, cur_stop, _, _ = resolve(
        o, h, l, 0, 0, 1, 100.0, 95.0, 5.0, 0.0, 0.0, False, 1.0, 0.0)
    assert code == OPEN
    assert cur_stop == 105.0          # armed for the next bar


def test_trail_mirrors_for_shorts():
    o, h, l = bars([(100, 100.5, 90.0), (94, 96.0, 93.0)])
    i, px, code, _, _, _, best = resolve(
        o, h, l, 0, 1, -1, 100.0, 105.0, 5.0, 0.0, 0.0, False, 1.0, 0.0)
    assert code == STOP and i == 1 and px == 95.0 and best == 90.0
