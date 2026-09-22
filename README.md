# BellCap Backtest

An intraday backtest simulator for index CFDs, built around one conviction:
**the entry signal is almost never where the money is, and a backtest is mostly
a machine for not fooling yourself.**

Two strategies have been run through it so far. Both looked fine on their best
configuration. Neither survived the tests in `bcbt.metrics`.

## What makes it different

**Fills resolve on 1-minute bars.** Signals come off the slow chart — 5-minute,
hourly — and the position is resolved against the minute bars underneath. A
backtest that asks a single 5-minute candle whether the stop or the target came
first has to guess, and that guess is worth about a third of the result. Where
one *minute* bar still spans both levels, the stop wins and the trade is flagged
`ambiguous` so you can count how often it mattered.

**Sessions come from the exchange calendar, not the feed.** A CFD feed quotes
straight through US market holidays, printing a flat line at the last traded
price — ten such sessions in three years of this data never move at all. Left
in, a holiday hands an opening-range strategy a range of nothing and an R
multiple of anything. Half-days are truncated to the real 13:00 close rather
than running on stale prints until 16:00.

**Costs are measured, not assumed.** `bcbt.fetch --ask` pulls the ask side
alongside the bid. On this data the S&P quotes 0.51 points all day; the Nasdaq
quotes ~0.96 during New York hours and ~1.46 overnight. A flat cost assumption
misprices half the session. (Dukascopy is tighter than most retail brokers —
treat these as a floor.)

**Every search is scored as a search.** With 36 variants, one of them always
looks good. `metrics.reality_check` implements White's Reality Check: it demeans
every variant so none has an edge by construction, resamples whole sessions in
blocks, and asks how high the best of N gets anyway. Both strategies tested so
far produced a best cell *below the median of that null distribution*.

## Install

```bash
pip install -e ".[extras,dev]"
python -m bcbt.fetch --symbols SPX500 NAS100 --start 2023-09-01 --ask
```

The fetcher writes `data/m1.parquet` (~46 MB for three years of two
instruments). It is gitignored — rebuild it rather than committing it. Expect
the first full download to take a while; Dukascopy throttles, so the fetcher
retries and reports any days it could not get. Re-running skips what it already
has.

## Use

```python
from bcbt import metrics, store
from bcbt.strategies import orb_ema

m1 = store.load_m1("NAS100")             # 0.2s for 1.1M bars
days, arr = orb_ema.prepare(m1)
trades = orb_ema.run("NAS100", days, arr, cost_pts=1.5,
                     or_min=15, stop_mode="or", rr=1.5)

print(metrics.summary(trades))
print(metrics.bootstrap_ci(trades["r"]))
print(metrics.coin_flip(trades, cost_pts=1.5))
```

`examples/search.py` runs a full 36-cell sweep and scores it. The order of
operations there is the point: pick the best cell first and you will always
find one; score the search first and you learn whether picking was worth doing.

## Layout

| Module | Does |
|---|---|
| `bcbt.fetch` | Dukascopy downloader, bid and ask, with retries |
| `bcbt.store` | Parquet bar store, resampling, EMA/ATR, confirmed pivots |
| `bcbt.calendars` | Real NYSE/LSE sessions and half-days |
| `bcbt.fills` | numba minute-resolution fill engine |
| `bcbt.metrics` | R stats, bootstrap CIs, Reality Check, coin flip |
| `bcbt.strategies` | Concrete rules, each returning a trade table |

Results are kept in **R** — multiples of the risk taken — so a 32-point S&P stop
and a 162-point Nasdaq stop are the same unit and can be pooled.

## Gotchas worth knowing

- **The month in a Dukascopy URL is zero-indexed.** September is `08`. This is
  the most common way to fetch the wrong month.
- **`DatetimeIndex.to_numpy()` on a tz-aware index gives an object array** of
  Timestamps, which compares unequal to any `datetime64` and silently turns
  every grouping into a miss. Use `store.day_keys()`.
- **Breakeven is a latch.** If you resolve a trade in pieces, thread `armed`
  and the returned stop back in, or the position re-earns its protection every
  window — which quietly changes the strategy and moved mean R by 0.006 the
  first time it was missed here.
- **`numba` pins `numpy<2.6`.** Upgrading numpy past that will break the build.

## Tests

```bash
pytest -q
```

`tests/test_fills.py` covers the engine — ties, gaps, breakeven latching, window
bounds. `tests/test_reproduces.py` pins the numbers from the original studies,
so a refactor that silently changes a fill rule fails loudly instead of quietly
producing a different result. It skips cleanly when the bar store is absent.
