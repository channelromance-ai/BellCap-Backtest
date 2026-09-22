"""
Intraday EMA-50 pullback backtester with institutional confluence filters.

A five-layer long-only intraday system:

    Layer 1  Trend and value anchors: EMA 50/200 and a session-anchored VWAP.
    Layer 2  Trigger: close crosses up through the EMA 50 while the macro
             filter (close above both the EMA 200 and the session VWAP) holds.
    Layer 3  Confluences: time-of-day relative volume, and an average-daily-
             range exhaustion veto.
    Layer 4  Entry window 10:00-15:30 New York, hard flat at 15:45.
    Layer 5  ATR-based stop and target, simulated bar by bar.

Design notes
------------
Everything that can be vectorised is vectorised. Layers 1, 3 and 4 are pure
pandas/numpy; the only iteration is in the simulator, and it steps from one
trade to the next rather than over every bar. After a position closes at bar
j, the next candidate entry is found with a binary search over the precomputed
signal indices, so bars spent flat are never visited.

No look-ahead, specifically
---------------------------
This is the part that is easy to get wrong and expensive to get wrong.

  * A signal is evaluated on the close of bar t-1 and filled at the OPEN of
    bar t. `entry_signal` is the raw signal shifted forward one bar, so a True
    at row t means "buy this bar's open".
  * Stop and target distances come from the ATR as it stood at bar t-1, not
    bar t, for the same reason.
  * The relative-volume baseline for a given time slot is shifted one day
    before the rolling mean is taken, so a bar is never part of the average it
    is measured against.
  * The average daily range uses the 14 *completed* sessions before today.
    Including today's range would leak the day's outcome into the filter that
    decides whether to trade it.
  * `min_periods` is set to the full window everywhere. Partial windows would
    emit signals from a half-formed baseline rather than no signal at all.

Intrabar conflicts resolve against the position: if one bar's low reaches the
stop and its high reaches the target, the stop is taken. Bar data cannot say
which came first, so the tie is always given away. This is the single most
common way an intraday backtest flatters itself.

Usage
-----
    python standalone/intraday_pullback.py                 # synthetic demo
    python standalone/intraday_pullback.py path/to/bars.csv

Requires only pandas and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "Config",
    "prepare_data",
    "run_backtest",
    "performance_report",
    "load_csv",
]

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Every tunable in one place, so a sweep is a loop over dataclasses."""

    # Layer 1 -- trend and value
    ema_fast: int = 50
    ema_slow: int = 200

    # Layer 3 -- confluences
    rvol_days: int = 20
    rvol_mult: float = 1.3
    adr_days: int = 14
    adr_exhaustion: float = 0.85

    # Layer 4 -- New York session clock
    entry_start: str = "10:00"
    entry_end: str = "15:30"
    eod_flat: str = "15:45"

    # Layer 5 -- risk
    atr_period: int = 14
    atr_sl_mult: float = 2.5
    atr_tp_mult: float = 5.0

    # Round-trip friction in basis points of notional. The default of zero
    # reproduces the specification exactly; set it to something real before
    # believing any of the numbers.
    cost_bps: float = 0.0

    def minutes(self, which: str) -> int:
        hhmm = getattr(self, which)
        hours, mins = hhmm.split(":")
        return int(hours) * 60 + int(mins)


# --------------------------------------------------------------------------
# Indicator helpers (all vectorised)
# --------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average, seeded on the first value."""
    return series.ewm(span=span, adjust=False).mean()


def _wilder_atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's ATR.

    Wilder smoothing is an EWM with alpha = 1/period, which is not the same as
    a span of `period`. Using span here would make the ATR roughly twice as
    responsive and quietly tighten every stop in the backtest.
    """
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def _session_vwap(frame: pd.DataFrame, day_key: np.ndarray) -> pd.Series:
    """
    Volume-weighted average price, reset at every calendar day.

    Both cumulative sums are grouped by day, so the first bar of a session has
    a VWAP equal to its own typical price and nothing carries over from
    yesterday.
    """
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    pv = (typical * frame["volume"]).groupby(day_key).cumsum()
    vol = frame["volume"].groupby(day_key).cumsum()
    return pv / vol.where(vol > 0)


def _rvol_baseline(
    volume: pd.Series, slot: np.ndarray, days: int
) -> pd.Series:
    """
    Rolling average volume for each time-of-day slot.

    Grouping on minute-of-day is equivalent to grouping on `df.index.time` but
    avoids building a column of Python `datetime.time` objects, which is the
    slow part on a large frame.

    The `shift(1)` happens inside the group and before the rolling mean, so
    the 10:05 bar is compared against the previous 20 sessions' 10:05 bars and
    never against itself.
    """
    grouped = volume.groupby(slot, sort=False)
    return grouped.transform(
        lambda s: s.shift(1).rolling(days, min_periods=days).mean()
    )


def _adr(frame: pd.DataFrame, day_key: np.ndarray, days: int) -> pd.Series:
    """
    Average daily range over the `days` sessions *before* the current one.

    The trailing `shift(1)` is what makes this usable as a filter: today's own
    high and low are not known when the day starts, and including them would
    let the ADR veto peek at the range it is meant to be predicting.
    """
    daily = frame.groupby(day_key).agg(hi=("high", "max"), lo=("low", "min"))
    span = (daily["hi"] - daily["lo"]).rolling(days, min_periods=days).mean()
    return pd.Series(day_key, index=frame.index).map(span.shift(1))


# --------------------------------------------------------------------------
# Layer 1-4: data preparation
# --------------------------------------------------------------------------


def prepare_data(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """
    Attach every indicator and emit the execution-ready entry flag.

    Parameters
    ----------
    df : DataFrame
        Multi-day intraday bars indexed by a DatetimeIndex already localised
        to (or naive in) New York time, with columns open/high/low/close/
        volume.
    cfg : Config, optional

    Returns
    -------
    DataFrame
        A copy of `df` with the indicator columns appended. The column that
        matters downstream is `entry_signal`: True at row t means the rule
        fired on the close of t-1 and the position is opened at row t's open.
    """
    cfg = cfg or Config()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex in New York time")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")

    out = df.copy()

    # Session keys. Integer keys group far faster than timestamps or
    # `datetime.time` objects and compare exactly.
    day_key = out.index.normalize().view("int64")
    slot = (out.index.hour * 60 + out.index.minute).to_numpy(np.int32)
    out["day_key"] = day_key
    out["minute_of_day"] = slot

    # ---- Layer 1: trend and value anchors
    out["ema_fast"] = _ema(out["close"], cfg.ema_fast)
    out["ema_slow"] = _ema(out["close"], cfg.ema_slow)
    out["vwap"] = _session_vwap(out, day_key)
    out["macro_bull"] = (out["close"] > out["ema_slow"]) & (
        out["close"] > out["vwap"]
    )

    # ---- Layer 2: the pullback trigger, evaluated on this bar's close
    crossed_up = (out["close"] > out["ema_fast"]) & (
        out["close"].shift(1) <= out["ema_fast"].shift(1)
    )
    out["cross_up"] = crossed_up

    # ---- Layer 3: confluences
    out["rvol_baseline"] = _rvol_baseline(out["volume"], slot, cfg.rvol_days)
    out["rvol"] = out["volume"] / out["rvol_baseline"]
    out["rvol_ok"] = out["rvol"] > cfg.rvol_mult

    out["adr"] = _adr(out, day_key, cfg.adr_days)
    run_hi = out["high"].groupby(day_key).cummax()
    run_lo = out["low"].groupby(day_key).cummin()
    out["session_range"] = run_hi - run_lo
    # Exhausted once the day has already travelled most of a normal range.
    out["adr_ok"] = out["session_range"] <= (cfg.adr_exhaustion * out["adr"])

    # ---- Layer 4: the entry window (signal bar must sit inside it)
    start, end = cfg.minutes("entry_start"), cfg.minutes("entry_end")
    out["in_window"] = (out["minute_of_day"] >= start) & (
        out["minute_of_day"] <= end
    )

    # ---- Raw signal, as known at the close of this bar
    out["raw_signal"] = (
        out["macro_bull"]
        & out["cross_up"]
        & out["rvol_ok"]
        & out["adr_ok"]
        & out["in_window"]
    ).fillna(False)

    # ---- Shift into execution space: fill at the NEXT bar's open.
    out["entry_signal"] = out["raw_signal"].shift(1, fill_value=False)

    # ---- Layer 5 inputs, carried forward with the same one-bar lag so the
    # risk distances are the ones known when the decision was made.
    out["atr"] = _wilder_atr(out, cfg.atr_period)
    atr_at_signal = out["atr"].shift(1)
    out["atr_at_signal"] = atr_at_signal
    out["sl_dist"] = cfg.atr_sl_mult * atr_at_signal
    out["tp_dist"] = cfg.atr_tp_mult * atr_at_signal

    return out


# --------------------------------------------------------------------------
# Layer 5: simulation engine
# --------------------------------------------------------------------------


def _session_exit_bounds(
    day_key: np.ndarray, slot: np.ndarray, eod_minute: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every bar, the index of its session's forced-flat bar and last bar.

    Precomputing these turns the end-of-day rule into an array lookup instead
    of a per-bar time comparison inside the simulation.
    """
    n = len(day_key)
    idx = np.arange(n)
    days = pd.Series(day_key)

    last_idx = pd.Series(idx).groupby(days).transform("max").to_numpy()

    sentinel = np.iinfo(np.int64).max
    eligible = np.where(slot >= eod_minute, idx, sentinel)
    flat_idx = pd.Series(eligible).groupby(days).transform("min").to_numpy()
    has_flat = flat_idx != sentinel
    flat_idx = np.where(has_flat, flat_idx, last_idx)
    return flat_idx, has_flat


def run_backtest(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """
    Simulate the rule and return one row per closed trade.

    The loop is over trades, not bars. For each candidate entry the exit is
    found inside the remainder of that session with vectorised first-touch
    tests, and the next candidate is located with `np.searchsorted`, so bars
    spent flat are never touched and a signal that fires while a position is
    already open is ignored rather than queued.
    """
    cfg = cfg or Config()
    if "entry_signal" not in df.columns:
        raise ValueError("call prepare_data() first")

    op = df["open"].to_numpy(np.float64)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    sl_dist = df["sl_dist"].to_numpy(np.float64)
    tp_dist = df["tp_dist"].to_numpy(np.float64)
    day_key = df["day_key"].to_numpy(np.int64)
    slot = df["minute_of_day"].to_numpy(np.int32)
    stamps = df.index

    eod_minute = cfg.minutes("eod_flat")
    flat_idx, has_flat = _session_exit_bounds(day_key, slot, eod_minute)

    # A candidate needs a signal and a usable risk distance. Rows where the
    # ATR has not warmed up yet are dropped here rather than producing a
    # zero-width stop later.
    valid = (
        df["entry_signal"].to_numpy(bool)
        & np.isfinite(sl_dist)
        & np.isfinite(tp_dist)
        & (sl_dist > 0.0)
    )
    candidates = np.flatnonzero(valid)

    cost = cfg.cost_bps / 10_000.0
    trades: list[dict[str, Any]] = []
    ptr = 0

    while ptr < candidates.size:
        i = int(candidates[ptr])

        # An entry bar at or past the flat time has nowhere to go. This only
        # applies when the session actually has a flat bar: on a shortened
        # session `flat_idx` falls back to the last bar, and an entry there
        # is legitimate -- it simply exits at that bar's close.
        if has_flat[i] and i >= flat_idx[i]:
            ptr += 1
            continue

        entry = op[i]
        stop = entry - sl_dist[i]
        target = entry + tp_dist[i]

        # The forced-flat bar exits at its OPEN, so it is not scanned for
        # stop or target: at 15:45 the position is already gone.
        if has_flat[i]:
            scan_end = int(flat_idx[i]) - 1
            fallback_px = op[int(flat_idx[i])]
            fallback_i = int(flat_idx[i])
            fallback_reason = "eod_flat"
        else:
            scan_end = int(flat_idx[i])
            fallback_px = cl[scan_end]
            fallback_i = scan_end
            fallback_reason = "session_end"

        exit_i, exit_px, reason = fallback_i, fallback_px, fallback_reason

        if scan_end >= i:
            window = slice(i, scan_end + 1)
            sl_hits = lo[window] <= stop
            tp_hits = hi[window] >= target
            sl_at = int(np.argmax(sl_hits)) if sl_hits.any() else -1
            tp_at = int(np.argmax(tp_hits)) if tp_hits.any() else -1

            if sl_at >= 0 and (tp_at < 0 or sl_at <= tp_at):
                # Ties go to the stop: one bar cannot say which came first.
                exit_i, exit_px, reason = i + sl_at, stop, "stop"
            elif tp_at >= 0:
                exit_i, exit_px, reason = i + tp_at, target, "target"

        gross = exit_px / entry - 1.0
        net = gross - 2.0 * cost
        trades.append(
            {
                "entry_time": stamps[i],
                "exit_time": stamps[exit_i],
                "entry_price": entry,
                "exit_price": exit_px,
                "stop": stop,
                "target": target,
                "reason": reason,
                "bars_held": exit_i - i,
                "return_pct": 100.0 * net,
                # R is the move in units of the risk taken, so a 2.5x/5.0x
                # ATR pair caps a winner at +2R against a -1R loser.
                "r_multiple": (exit_px - entry) / sl_dist[i],
            }
        )

        # Skip any signal that fired while this trade was live, including one
        # on the exit bar itself.
        ptr = int(np.searchsorted(candidates, exit_i, side="right"))

    return pd.DataFrame(trades)


# --------------------------------------------------------------------------
# Performance analytics
# --------------------------------------------------------------------------


def performance_report(
    trades: pd.DataFrame, verbose: bool = True
) -> dict[str, float]:
    """
    Headline statistics for a set of closed trades.

    Equity compounds one position at a time at full notional, which is the
    cleanest assumption that needs no extra parameters. Drawdown is measured
    on the trade-by-trade equity curve, so it is the worst peak-to-trough on
    *closed* trades and will understate what the open position went through
    intrabar.
    """
    if trades is None or trades.empty:
        empty = {
            "total_trades": 0,
            "win_rate_pct": float("nan"),
            "total_return_pct": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown_pct": 0.0,
            "expectancy_r": float("nan"),
        }
        if verbose:
            print("No trades generated.")
        return empty

    ret = trades["return_pct"].to_numpy(np.float64) / 100.0
    wins = ret[ret > 0.0]
    losses = ret[ret <= 0.0]

    equity = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (
        gross_win / gross_loss if gross_loss > 0 else float("inf")
    )

    stats = {
        "total_trades": int(len(ret)),
        "win_rate_pct": 100.0 * len(wins) / len(ret),
        "total_return_pct": 100.0 * (equity[-1] - 1.0),
        "profit_factor": profit_factor,
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
        "expectancy_r": float(trades["r_multiple"].mean()),
        "avg_win_pct": 100.0 * float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": 100.0 * float(losses.mean()) if len(losses) else 0.0,
    }

    if verbose:
        print("\n" + "=" * 54)
        print("PERFORMANCE")
        print("=" * 54)
        print(f"  Total trades      : {stats['total_trades']:>12d}")
        print(f"  Win rate          : {stats['win_rate_pct']:>11.2f} %")
        print(f"  Total return      : {stats['total_return_pct']:>11.2f} %")
        print(f"  Profit factor     : {stats['profit_factor']:>11.2f}")
        print(f"  Max drawdown      : {stats['max_drawdown_pct']:>11.2f} %")
        print(f"  Expectancy        : {stats['expectancy_r']:>11.2f} R")
        print(f"  Average win       : {stats['avg_win_pct']:>11.2f} %")
        print(f"  Average loss      : {stats['avg_loss_pct']:>11.2f} %")
        counts = trades["reason"].value_counts()
        print("  Exits             : " + ", ".join(
            f"{k}={v}" for k, v in counts.items()))
        print("=" * 54)

    return stats


# --------------------------------------------------------------------------
# Data plumbing -- plug your own source in here
# --------------------------------------------------------------------------


def load_csv(
    path: str,
    timestamp_col: str = "timestamp",
    tz: str | None = "America/New_York",
) -> pd.DataFrame:
    """
    Load intraday bars from a local CSV.

    Expects a timestamp column plus open/high/low/close/volume. If the stamps
    are naive they are assumed to be New York wall clock and localised as
    such; if they carry a zone they are converted. Getting this wrong shifts
    the entire session clock and silently invalidates Layer 4.

    ------------------------------------------------------------------
    PLUG IN YOUR OWN DATA HERE
    ------------------------------------------------------------------
    Any source works as long as it ends up as a DatetimeIndex in New York
    time with the five required columns, for example:

        # Local file
        df = load_csv("data/spy_5min.csv")

        # Parquet
        df = pd.read_parquet("data/spy_5min.parquet")

        # A vendor API returning UTC stamps
        raw = broker.get_bars("SPY", "5min", start, end)
        df = pd.DataFrame(raw).set_index("t")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert("America/New_York")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                "c": "close", "v": "volume"})

        # A live stream: append each finished bar, then re-run prepare_data
        # on the tail. Never call it on a bar that is still forming -- the
        # rule is defined on completed closes.
    ------------------------------------------------------------------
    """
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(
            f"'{timestamp_col}' not in {list(df.columns)}; pass timestamp_col"
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing column(s): {missing}")
    return df[list(REQUIRED_COLUMNS)]


def _synthetic_bars(
    days: int = 90, freq_min: int = 5, seed: int = 7
) -> pd.DataFrame:
    """
    Generate plausible 5-minute bars so the file runs with no data on hand.

    Prices follow a mild upward drift with intraday mean reversion; volume
    gets the usual U-shaped session profile so the time-of-day RVOL filter
    has something real to measure against.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-02", periods=days, tz="America/New_York")

    frames = []
    price = 100.0
    for day in sessions:
        stamps = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30),
            day + pd.Timedelta(hours=16),
            freq=f"{freq_min}min",
            inclusive="left",
        )
        n = len(stamps)
        shocks = rng.normal(0.0002, 0.0016, n)
        closes = price * np.cumprod(1.0 + shocks)
        opens = np.empty(n)
        opens[0] = price
        opens[1:] = closes[:-1]
        spread = np.abs(rng.normal(0.0009, 0.0005, n)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread

        minute = np.arange(n)
        shape = 1.6 - 1.2 * np.sin(np.pi * minute / max(n - 1, 1))
        volume = rng.lognormal(10.0, 0.35, n) * shape

        frames.append(
            pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volume,
                },
                index=stamps,
            )
        )
        price = closes[-1]

    return pd.concat(frames)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    cfg = Config()

    if argv:
        # ---- Local CSV path supplied on the command line.
        print(f"Loading {argv[0]} ...")
        bars = load_csv(argv[0])
    else:
        print("No data file given; running on synthetic bars.")
        bars = _synthetic_bars()

    print(
        f"{len(bars):,} bars, "
        f"{bars.index.normalize().nunique()} sessions, "
        f"{bars.index[0]} -> {bars.index[-1]}"
    )

    prepared = prepare_data(bars, cfg)
    print(
        f"raw signals: {int(prepared['raw_signal'].sum())}, "
        f"executable: {int(prepared['entry_signal'].sum())}"
    )

    trades = run_backtest(prepared, cfg)
    performance_report(trades)

    if not trades.empty:
        print("\nFirst five trades:")
        cols = [
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "reason",
            "return_pct",
            "r_multiple",
        ]
        print(trades[cols].head().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
