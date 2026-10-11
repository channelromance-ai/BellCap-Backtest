"""
The 1R breakeven-then-trail rule, on bars small enough to check by hand.

The kernel tests pin the management rule bar by bar. The strategy tests run
the whole thing on a synthetic random walk and check the things the rules
promise: one trade a day, entered at 10:00, the stop at the far side of the
range, the side chosen by the close against the EMA, and nothing decided at
10:00 depending on anything after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bcbt import audit, store
from bcbt.fills import resolve_trail, STOP, TRAIL, TARGET, OPEN
from bcbt.strategies import orb_trail


def bars(rows):
    a = np.array(rows, float)
    return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()


def rt(rows, d=1, entry=100.0, stop=90.0, act=1.0, trail=1.0, tgt=0.0,
       stop_first=True):
    o, h, l = bars(rows)
    return resolve_trail(o, h, l, 0, len(o) - 1, d, entry, stop,
                         abs(entry - stop), act, trail, tgt, stop_first)


def test_initial_stop_holds_until_one_r():
    # +0.8R then down to the original stop. A trail measured from entry
    # would have lifted the stop to 98 and exited there; the rule does not.
    i, px, code, *_ = rt([(100, 108, 99), (107, 107, 95), (95, 96, 89)])
    assert (i, code, px) == (2, STOP, 90.0)


def test_breakeven_then_trail_one_r_behind_the_high():
    # Reaches 110 (+1R): stop to entry (= 110 - 10). Runs to 125: stop 115.
    # Falls to 114: out at 115, +1.5R.
    i, px, code, amb, armed, mfe, _ = rt(
        [(100, 110, 101), (110, 125, 116), (124, 124, 114)])
    assert (i, code, px, amb, armed) == (2, TRAIL, 115.0, False, True)
    assert mfe == 25.0


def test_trail_never_loosens():
    i, px, code, *_ = rt([(100, 120, 111), (119, 119, 112), (112, 113, 109)])
    assert code == TRAIL and px == 110.0       # 120 - 10, not 119 - 10


def test_short_mirror():
    i, px, code, *_ = rt([(100, 99, 90), (90, 95, 75), (76, 86, 76)],
                         d=-1, entry=100.0, stop=110.0)
    assert code == TRAIL and px == 85.0


def test_gap_through_trail_fills_at_open():
    i, px, code, *_ = rt([(100, 120, 111), (105, 106, 104)])
    assert code == TRAIL and px == 105.0


def test_bar_that_arms_and_returns_is_ambiguous_both_ways():
    # One bar reaches +1R and trades back to entry. Order unknown.
    row = [(100, 110, 100), (101, 102, 101)]
    i, px, code, amb, *_ = rt(row, stop_first=True)
    assert (i, code, px, amb) == (0, TRAIL, 100.0, True)
    i, px, code, amb, *_ = rt(row, stop_first=False)
    assert (code, amb) == (OPEN, True)


def test_breakeven_only_when_trail_is_zero():
    i, px, code, *_ = rt([(100, 130, 101), (125, 126, 99)], trail=0.0)
    assert code == TRAIL and px == 100.0


def test_no_management_holds_initial_stop_all_day():
    i, px, code, *_ = rt([(100, 130, 101), (125, 126, 99)], act=0.0)
    assert code == OPEN


def test_fixed_target():
    i, px, code, *_ = rt([(100, 115, 99), (114, 125, 113)], act=0.0, tgt=120.0)
    assert (i, code, px) == (1, TARGET, 120.0)


# --------------------------------------------------------------- strategy


def synthetic_m1(n_days=45, seed=7):
    """24-hour random-walk minute bars over real NYSE weekdays."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=n_days * 1440, freq="1min",
                        tz="America/New_York")
    idx = idx[idx.dayofweek < 5]
    c = 18000 + np.cumsum(rng.normal(0, 4.0, len(idx)))
    o = np.r_[c[0], c[:-1]]
    wig = np.abs(rng.normal(0, 2.0, len(idx)))
    h = np.maximum(o, c) + wig
    l = np.minimum(o, c) - wig
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)


@pytest.fixture(scope="module")
def prepared():
    m1 = synthetic_m1()
    days, a = orb_trail.prepare(m1)
    return m1, days, a


def test_one_trade_a_day_entered_at_ten(prepared):
    _, days, a = prepared
    t = orb_trail.run("X", days, a, cost_pts=1.0)
    assert len(t) > 20
    assert t["day"].is_unique
    et = pd.DatetimeIndex(t["entry_ts"])
    assert ((et.hour == 10) & (et.minute == 0)).all()
    assert (pd.DatetimeIndex(t["exit_ts"]).hour < 16).all()


def test_stop_and_side_follow_the_rules(prepared):
    _, days, a = prepared
    t = orb_trail.run("X", days, a, cost_pts=1.0)
    lg, sh = t[t["dir"] == "long"], t[t["dir"] == "short"]
    assert (lg["stop"] == lg["or_lo"]).all()
    assert (sh["stop"] == sh["or_hi"]).all()
    assert (lg["sig_close"] > lg["sig_ema"]).all()
    assert (sh["sig_close"] < sh["sig_ema"]).all()
    # An initial-stop exit loses 1R gross, or more through a gap.
    st = t[t["reason"] == "stop"]
    assert len(st) and (st["r_gross"] <= -1.0 + 1e-9).all()
    # A trailed exit never gives back more than breakeven gross.
    tr = t[t["reason"].isin(["trail", "be"])]
    assert (tr["r_gross"] >= -1e-9).all()


def test_opening_range_is_the_first_thirty_minutes(prepared):
    m1, days, a = prepared
    t = orb_trail.run("X", days, a, cost_pts=0.0)
    row = t.iloc[5]
    d = pd.Timestamp(row["day"]).tz_localize("America/New_York")
    w = m1[(m1.index >= d + pd.Timedelta("9h30min")) &
           (m1.index < d + pd.Timedelta("10h"))]
    assert row["or_hi"] == w["h"].max()
    assert row["or_lo"] == w["l"].min()


def test_signal_ema_matches_a_chart_ema(prepared):
    m1, days, a = prepared
    t = orb_trail.run("X", days, a, cost_pts=0.0)
    m5 = store.resample(m1, "5min")
    e = store.ema(m5["c"], 12)
    row = t.iloc[10]
    d = pd.Timestamp(row["day"]).tz_localize("America/New_York")
    bar = d + pd.Timedelta("9h55min")          # the bar that ends at 10:00
    assert row["sig_ema"] == pytest.approx(e.loc[bar])
    assert row["sig_close"] == pytest.approx(m5.loc[bar, "c"])


def test_forced_sides_and_the_complement(prepared):
    _, days, a = prepared
    rule = orb_trail.run("X", days, a, 1.0)
    anti = orb_trail.run("X", days, a, 1.0, force=-2)
    assert (rule["dir"].to_numpy() != anti["dir"].to_numpy()).all()
    lg = orb_trail.run("X", days, a, 1.0, force=1)
    assert (lg["dir"] == "long").all()


def test_no_look_ahead(prepared):
    m1, _, _ = prepared

    def decide(frame):
        d, arr = orb_trail.prepare(frame)
        t = orb_trail.run("X", d, arr, 1.0)
        return t[["entry_ts", "entry", "stop", "risk", "sig_ema"]].assign(
            side=np.where(t["dir"] == "long", 1.0, -1.0))

    rep = audit.future_poison(decide, m1, probes=6)
    assert not rep["leak"], rep


def test_no_edge_on_a_martingale():
    """
    On a random walk with ticks inside each minute, no side may make money
    before costs. A walk with one step a minute is not good enough for this:
    price jumps over the stop and a fill at the level is a free gift, which
    is how this test was first got wrong.
    """
    from bcbt import metrics
    rng = np.random.default_rng(5)
    idx = pd.date_range("2022-01-03", "2023-12-29", freq="1min",
                        tz="America/New_York")
    idx = idx[idx.dayofweek < 5]
    k = 30
    ticks = 15000 + np.cumsum(
        rng.normal(0, 3.75 / np.sqrt(k), len(idx) * k)).reshape(len(idx), k)
    prev = np.r_[ticks[0, 0], ticks[:-1, -1]]
    m1 = pd.DataFrame(dict(o=prev, h=np.maximum(ticks.max(1), prev),
                           l=np.minimum(ticks.min(1), prev), c=ticks[:, -1]),
                      index=idx)
    days, a = orb_trail.prepare(m1)
    for force in (0, -2, 1, -1):
        t = orb_trail.run("X", days, a, 0.0, force=force, stop_first=False)
        lo, hi = metrics.bootstrap_ci(t["r_gross"])
        assert lo < 0 < hi, (force, lo, hi)
