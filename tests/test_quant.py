"""The quant battery's engine: timing, costs, financing, buffering, gaps."""
import math

import numpy as np
import pandas as pd
import pytest

from bcbt import quant as Q


def _prices(rets, sym="SPX500_USD", start="2020-01-06"):
    idx = pd.bdate_range(start, periods=len(rets) + 1)
    c = 100 * np.cumprod(np.r_[1.0, 1 + np.asarray(rets, float)])
    return pd.DataFrame({sym: c}, index=idx)


def test_signal_earns_the_next_bar_not_its_own():
    C = _prices([0.01, -0.02, 0.03, 0.0])
    sig = pd.DataFrame({"SPX500_USD": [0, 1, 0, 0, 0]}, index=C.index,
                       dtype=float)
    r = Q.run(sig, C, scale=False, fin=False, cost_mult=0)
    # long at the close of bar 1 earns bar 2's -2%, booked at bar 1
    assert r.gross.iloc[1] == pytest.approx(-0.02)
    assert r.gross.drop(r.gross.index[1]).abs().sum() == 0


def test_round_trip_costs_one_spread_plus_two_slippages():
    C = _prices([0.0] * 4)
    sig = pd.DataFrame({"SPX500_USD": [1, 1, 0, 0, 0]}, index=C.index,
                       dtype=float)
    r = Q.run(sig, C, scale=False, fin=False)
    spread = Q.SPREAD_BP["SPX500_USD"] / 1e4
    assert -r.net.sum() == pytest.approx(spread + 2 * Q.SLIP_BP / 1e4)


def test_financing_three_nights_over_a_weekend():
    idx = pd.to_datetime(["2020-01-09", "2020-01-10", "2020-01-13"])
    C = pd.DataFrame({"SPX500_USD": [100.0, 100.0, 100.0]}, index=idx)
    sig = pd.DataFrame({"SPX500_USD": [0.0, 1.0, 0.0]}, index=idx)
    r = Q.run(sig, C, scale=False, cost_mult=0)
    # held Friday close -> Monday close: three nights at 2.5% a year
    assert -r.net.iloc[1] == pytest.approx(3 * 0.025 / 365)


def test_fx_financing_is_the_fx_rate():
    idx = pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-09"])
    C = pd.DataFrame({"EUR_USD": [1.1, 1.1, 1.1]}, index=idx)
    sig = pd.DataFrame({"EUR_USD": [0.0, -1.0, 0.0]}, index=idx)
    r = Q.run(sig, C, scale=False, cost_mult=0)
    assert -r.net.iloc[1] == pytest.approx(0.01 / 365)


def test_buffer_ignores_small_resizes_but_not_flips():
    w = np.array([[1.0], [1.1], [1.2], [1.3], [-1.0], [0.0]])
    out = Q._buffered(w, 0.25)
    assert list(out[:, 0]) == [1.0, 1.0, 1.0, 1.3, -1.0, 0.0]


def test_missing_bar_neither_closes_nor_charges():
    C = _prices([0.01, 0.0, 0.02, 0.0])
    C.iloc[2] = np.nan                              # a holiday
    sig = pd.DataFrame({"SPX500_USD": 1.0}, index=C.index)
    r = Q.run(sig, C, scale=False, fin=False)
    trades = r.w.diff().abs().sum().sum()
    assert trades == 0                              # entered once, held


def test_vol_scaling_targets_ten_percent():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, 3000)                # ~32% vol
    C = _prices(rets)
    sig = pd.DataFrame({"SPX500_USD": 1.0}, index=C.index)
    r = Q.run(sig, C, fin=False, cost_mult=0, buffer=0)
    vol = r.gross.iloc[100:].std() * math.sqrt(252)
    assert vol == pytest.approx(0.10, rel=0.1)


def test_reality_check_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    streams = {f"m{i}": pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
               for i in range(30)}
    _, _, p, _ = Q.reality_check(streams, n_boot=300)
    assert p > 0.05
