"""
Correctness tests for the standalone intraday pullback backtester.

The interesting failures in a backtest are not crashes, they are results.
A look-ahead bug does not raise; it just pays you. So these tests go after
the specific claims the script makes: that signals are lagged, that fills
happen at the next open, that a bar holding both levels resolves against the
position, that the day is flat at 15:45, and that no indicator baseline
contains the bar it is judging.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "standalone"))

import intraday_pullback as ip  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    bars = ip._synthetic_bars(days=90, seed=7)
    return bars, ip.prepare_data(bars, ip.Config())


# ---------------------------------------------------------------- lag rules


def test_entry_signal_is_the_raw_signal_lagged_one_bar(prepared):
    _, df = prepared
    expected = df["raw_signal"].shift(1, fill_value=False)
    assert df["entry_signal"].equals(expected)
    assert not bool(df["entry_signal"].iloc[0])


def test_risk_distances_come_from_the_signal_bar_not_the_fill_bar(prepared):
    _, df = prepared
    cfg = ip.Config()
    shifted = df["atr"].shift(1)
    pd.testing.assert_series_equal(
        df["sl_dist"], cfg.atr_sl_mult * shifted, check_names=False)
    pd.testing.assert_series_equal(
        df["tp_dist"], cfg.atr_tp_mult * shifted, check_names=False)


def test_fills_happen_at_the_open_of_the_entry_bar(prepared):
    _, df = prepared
    trades = ip.run_backtest(df, ip.Config())
    assert not trades.empty
    for _, t in trades.iterrows():
        assert t["entry_price"] == pytest.approx(df.loc[t["entry_time"], "open"])


def test_every_trade_follows_a_signal_on_the_previous_bar(prepared):
    _, df = prepared
    trades = ip.run_backtest(df, ip.Config())
    pos = df.index.get_indexer(trades["entry_time"])
    assert (pos > 0).all()
    assert df["raw_signal"].to_numpy()[pos - 1].all()


# ------------------------------------------------------- no baseline leakage


def test_rvol_baseline_excludes_the_bar_it_measures(prepared):
    """The 10:05 baseline must be the mean of PRIOR sessions' 10:05 bars."""
    _, df = prepared
    slot = df["minute_of_day"].to_numpy()
    target = slot[50]
    same = df[df["minute_of_day"] == target]
    k = 25                                     # well past the 20-day warmup
    expected = same["volume"].iloc[k - 20:k].mean()
    assert same["rvol_baseline"].iloc[k] == pytest.approx(expected)
    # And the warmup emits nothing rather than a partial average.
    assert same["rvol_baseline"].iloc[:20].isna().all()


def test_adr_uses_only_completed_prior_sessions(prepared):
    _, df = prepared
    daily = df.groupby("day_key").agg(hi=("high", "max"), lo=("low", "min"))
    rng = (daily["hi"] - daily["lo"]).to_numpy()
    days = list(daily.index)
    d = days[40]
    expected = rng[40 - 14:40].mean()
    assert df[df["day_key"] == d]["adr"].iloc[0] == pytest.approx(expected)


def test_session_vwap_resets_every_day(prepared):
    _, df = prepared
    first = df.groupby("day_key").head(1)
    typical = (first["high"] + first["low"] + first["close"]) / 3.0
    # The first bar of a session has no history, so VWAP is its own typical
    # price. If yesterday leaked in, this fails.
    np.testing.assert_allclose(
        first["vwap"].to_numpy(), typical.to_numpy(), rtol=1e-12)


# ------------------------------------------------------------ session rules


def test_signals_only_fire_inside_the_entry_window(prepared):
    _, df = prepared
    cfg = ip.Config()
    fired = df[df["raw_signal"]]["minute_of_day"]
    assert fired.between(cfg.minutes("entry_start"),
                         cfg.minutes("entry_end")).all()


def test_nothing_is_held_past_the_flat_time_or_overnight(prepared):
    _, df = prepared
    cfg = ip.Config()
    trades = ip.run_backtest(df, cfg)
    eod = cfg.minutes("eod_flat")
    exits = pd.DatetimeIndex(trades["exit_time"])
    assert ((exits.hour * 60 + exits.minute) <= eod).all()
    assert (exits.normalize() ==
            pd.DatetimeIndex(trades["entry_time"]).normalize()).all()


def test_trades_never_overlap(prepared):
    """One position at a time; a signal fired while live must be ignored."""
    _, df = prepared
    trades = ip.run_backtest(df, ip.Config())
    entries = df.index.get_indexer(trades["entry_time"])
    exits = df.index.get_indexer(trades["exit_time"])
    assert (entries[1:] > exits[:-1]).all()


# ------------------------------------------------- intrabar conflict rule


def _one_bar_case(high, low, sl_dist=1.0, tp_dist=2.0):
    """A two-bar session: bar 0 signals, bar 1 is the crafted outcome."""
    idx = pd.DatetimeIndex(
        ["2024-03-01 10:00", "2024-03-01 10:05"], tz="America/New_York")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, high],
            "low": [100.0, low],
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "day_key": idx.normalize().view("int64"),
            "minute_of_day": (idx.hour * 60 + idx.minute).to_numpy(np.int32),
            "entry_signal": [False, True],
            "sl_dist": [np.nan, sl_dist],
            "tp_dist": [np.nan, tp_dist],
        },
        index=idx,
    )
    return ip.run_backtest(df, ip.Config())


def test_stop_wins_when_one_bar_holds_both_levels():
    """Low reaches 99 and high reaches 102 on the same bar -> stop."""
    t = _one_bar_case(high=102.0, low=99.0)
    assert len(t) == 1
    assert t.loc[0, "reason"] == "stop"
    assert t.loc[0, "exit_price"] == pytest.approx(99.0)
    assert t.loc[0, "r_multiple"] == pytest.approx(-1.0)


def test_target_taken_when_the_stop_is_untouched():
    t = _one_bar_case(high=102.0, low=99.5)
    assert t.loc[0, "reason"] == "target"
    assert t.loc[0, "exit_price"] == pytest.approx(102.0)
    assert t.loc[0, "r_multiple"] == pytest.approx(2.0)


def test_stop_taken_when_the_target_is_untouched():
    t = _one_bar_case(high=101.0, low=99.0)
    assert t.loc[0, "reason"] == "stop"
    assert t.loc[0, "r_multiple"] == pytest.approx(-1.0)


def test_unresolved_position_squares_off_at_the_session_end():
    t = _one_bar_case(high=100.5, low=99.5)
    assert t.loc[0, "reason"] in ("eod_flat", "session_end")


# --------------------------------------------------------------- analytics


def test_performance_report_math():
    trades = pd.DataFrame(
        {
            "return_pct": [10.0, -5.0, 20.0, -10.0],
            "r_multiple": [2.0, -1.0, 2.0, -1.0],
            "reason": ["target", "stop", "target", "stop"],
        }
    )
    s = ip.performance_report(trades, verbose=False)
    assert s["total_trades"] == 4
    assert s["win_rate_pct"] == pytest.approx(50.0)
    assert s["profit_factor"] == pytest.approx(0.30 / 0.15)
    expected = (1.10 * 0.95 * 1.20 * 0.90 - 1.0) * 100.0
    assert s["total_return_pct"] == pytest.approx(expected)
    assert s["expectancy_r"] == pytest.approx(0.5)


def test_empty_trades_do_not_blow_up():
    s = ip.performance_report(pd.DataFrame(), verbose=False)
    assert s["total_trades"] == 0
    assert s["total_return_pct"] == 0.0


def test_costs_reduce_returns():
    bars = ip._synthetic_bars(days=90, seed=7)
    df = ip.prepare_data(bars)
    free = ip.run_backtest(df, ip.Config(cost_bps=0.0))
    paid = ip.run_backtest(df, ip.Config(cost_bps=5.0))
    assert paid["return_pct"].mean() < free["return_pct"].mean()


def test_missing_columns_are_rejected():
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02 10:00"], tz="America/New_York"),
    )
    with pytest.raises(ValueError, match="missing required column"):
        ip.prepare_data(df)
