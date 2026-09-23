"""
Does the look-ahead detector actually detect look-ahead?

A check that never fires is worse than no check, because it is reassuring.
So it is pointed at a bug whose existence is certain: the zone strategy
originally read breakouts from the hourly bar's LABEL, which is when the
bar STARTS, and therefore allowed trades inside the breakout hour before
that hour had closed. That produced a t-statistic of 15 out of nothing.

Both versions are built here from the same data. The detector must flag the
broken one and clear the fixed one. If it cannot tell them apart it is
useless, however sensible it looks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bcbt.audit import cost_feasibility, future_poison


def synthetic(n=4000, seed=3):
    """A plain random walk on one-minute bars."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    spread = np.abs(rng.normal(0, 2e-4, n)) * close
    return pd.DataFrame(
        {"o": close, "h": close + spread, "l": close - spread, "c": close},
        index=idx)


def honest_signals(df):
    """
    Decide on a bar's close, act at the next bar. Uses only the past.
    """
    c = df["c"].to_numpy()
    idx = df.index
    rows = []
    for i in range(50, len(df) - 1):
        if c[i] > c[i - 50:i].max():
            rows.append(dict(entry_ts=idx[i + 1], entry=float(df["o"].iloc[i + 1])))
    return pd.DataFrame(rows, columns=["entry_ts", "entry"])


def cheating_signals(df):
    """
    The same shape of rule, but the trigger is confirmed by a bar that has
    not happened yet -- exactly the zone bug, in miniature.
    """
    c = df["c"].to_numpy()
    idx = df.index
    rows = []
    for i in range(50, len(df) - 10):
        if c[i] > c[i - 50:i].max() and c[i + 5] > c[i]:
            rows.append(dict(entry_ts=idx[i + 1], entry=float(df["o"].iloc[i + 1])))
    return pd.DataFrame(rows, columns=["entry_ts", "entry"])


@pytest.fixture(scope="module")
def data():
    return synthetic()


def test_detector_clears_an_honest_rule(data):
    r = future_poison(honest_signals, data)
    assert not r["leak"], r["detail"]
    assert r["probes_tried"] >= 5
    assert len(honest_signals(data)) > 50   # it had plenty to examine


def test_detector_catches_a_rule_that_reads_ahead(data):
    r = future_poison(cheating_signals, data)
    assert r["leak"], "look-ahead went undetected"
    assert r["probes_fired"] >= 1


def test_detector_catches_leak_even_when_counts_match(data):
    """
    The easy case is a different number of trades. The hard case is the
    same trades at slightly different prices, which is what a subtle leak
    looks like, so that has to be caught too. Sweeping the cut is what
    makes this findable: only signals near a boundary are affected, so one
    cut would usually miss it.
    """
    def subtle(df):
        s = honest_signals(df)
        if len(s):
            # Price nudged by a value from one bar into the future.
            nxt = df["c"].shift(-3).reindex(s["entry_ts"]).to_numpy()
            s = s.assign(entry=s["entry"].to_numpy() * 0.999
                         + np.nan_to_num(nxt) * 0.001)
        return s

    r = future_poison(subtle, data)
    assert r["leak"]
    assert r["n_clean"] == r["n_poisoned"]      # counts matched; values did not


def test_cost_feasibility_arithmetic():
    # A 20-point stop with a 1-point spread: costs 5% of the risk taken.
    f = cost_feasibility(np.full(100, 20.0), spread=1.0, rr=2.0)
    assert f["cost_r"] == pytest.approx(0.05)
    # Free, a 2:1 payoff breaks even at one win in three.
    assert f["breakeven_win_free_pct"] == pytest.approx(100 / 3)
    # The spread pushes that requirement up.
    assert f["breakeven_win_pct"] == pytest.approx(100 * 1.05 / 3)
    assert f["extra_wins_needed_pct"] > 0


def test_cost_feasibility_flags_a_hopeless_spread():
    """A stop barely wider than the spread cannot be traded at any skill."""
    f = cost_feasibility(np.full(50, 1.5), spread=1.0, rr=2.0)
    assert f["cost_r"] > 0.6
    assert f["breakeven_win_pct"] > 50      # needs to win more than half
