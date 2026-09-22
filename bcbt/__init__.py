"""
BellCap backtest simulator.

A small event-driven harness for intraday index strategies, built around one
conviction: the entry signal is almost never where the money is, and a
backtest is mostly a machine for not fooling yourself.

Three things it does that most quick backtests do not:

  * Signals come off the slow chart, fills off the 1-minute bars underneath,
    so "did the stop or the target come first" is usually answered rather
    than guessed.
  * Sessions come from the exchange calendar, so holiday prints -- a CFD
    feed quotes a flat line through Christmas -- never become trades.
  * Every variant search is scored against White's Reality Check, which
    asks whether the best of N variants beats the best of N variants in
    data with no edge in it. Two strategies have gone through this package
    so far. Both looked fine on their best cell and neither survived.

Layout
------
  bcbt.fetch       Dukascopy downloader, bid and ask
  bcbt.store       Parquet bar store, resampling, indicators, pivots
  bcbt.calendars   Real NYSE/LSE sessions and half-days
  bcbt.fills       numba minute-resolution fill engine
  bcbt.metrics     R statistics, bootstrap CIs, Reality Check, coin flip
  bcbt.strategies  Concrete rules, each returning a trade table
"""

__version__ = "0.1.0"

from . import calendars, fills, metrics, store   # noqa: F401

__all__ = ["calendars", "fills", "metrics", "store", "strategies"]
