"""
Does a simple stock-momentum strategy beat just holding SPY?

This script answers that one question as honestly as it can. It downloads
daily prices for S&P 500 stocks, runs a monthly momentum strategy with
trading costs, compares it to buying and holding SPY, and prints a clear
PASS or FAIL against a bar that is written down below BEFORE any results
exist.

How to run it
-------------
    pip install yfinance pandas numpy matplotlib pyarrow
    python standalone/momentum_vs_spy.py

The first run downloads about 1,100 tickers and takes a while. Prices are
saved to disk, so later runs start in seconds. Results land in the folder
named by OUTPUT_DIR below.

How the file is laid out
------------------------
    1. SETTINGS        every number the test uses, in one place
    2. LOAD DATA       membership list, prices, caching, dropped tickers
    3. SIGNALS         momentum ranking and the SPY market filter
    4. BACKTEST        month-by-month trading, with costs
    5. STATS           returns, drawdowns, volatility, yearly table
    6. CHARTS          growth of $10,000 and drawdown
    7. CHECKS          look-ahead self-check, bias report
    8. MAIN            runs every test and prints the verdict
"""

from __future__ import annotations

import io
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. SETTINGS
# Everything the test depends on lives here. The PASS/FAIL bar was set before
# the script was ever run on real data. Do not edit the bar after looking at
# the results -- that turns a test into a search for a story.
# =============================================================================

CONFIG = {
    # ---- PASS/FAIL bar (fixed in advance) ----------------------------------
    # The strategy's yearly growth rate (CAGR) after costs must beat SPY's by
    # at least this much. 0.02 means 2 percentage points per year.
    "MIN_EXCESS_CAGR": 0.02,
    # The strategy's worst drop may be at most this much deeper than SPY's.
    # 0.05 means 5 percentage points: if SPY fell 55%, the strategy may fall
    # up to 60% and still pass this part.
    "MAX_EXTRA_DRAWDOWN": 0.05,
    # Out-of-sample rule: the period is split in half by time, each half is
    # run separately, and the strategy must beat SPY by at least this much
    # per year in BOTH halves. Set equal to the main bar on purpose.
    "OOS_MIN_EXCESS_CAGR": 0.02,
    # The verdict uses this cost level. The harsher level is also reported.
    "VERDICT_COST": 0.001,

    # ---- Strategy rules (given, NOT tuned) ---------------------------------
    "LOOKBACK_MONTHS": 12,     # measure the return over 12 months...
    "SKIP_MONTHS": 1,          # ...but skip the most recent month
    "TOP_PCT": 0.10,           # buy the top 10% of ranked stocks
    "SMA_MONTHS": 10,          # SPY must be above its 10-month average
    "CASH_RETURN": 0.0,        # cash earns nothing (as instructed)

    # ---- Costs -------------------------------------------------------------
    # Charged on every buy and every sell, as a share of the amount traded.
    "COST_NORMAL": 0.001,      # 0.1%
    "COST_HARSH": 0.003,       # 0.3%

    # ---- Robustness grid (report only -- never pick the best one) ----------
    "ROBUST_TOP_PCTS": [0.05, 0.10, 0.20],
    "ROBUST_LOOKBACKS": [6, 9, 12],
    # A neighbour of the main setting counts as "wildly different" if its
    # yearly excess return over SPY differs from the main setting's by more
    # than this, or if it lands on the other side of zero.
    "FRAGILE_GAP": 0.03,

    # ---- Data --------------------------------------------------------------
    # Prices start here. The first 12 months are only used to warm up the
    # momentum signal, so trading starts about a year later (early 2000).
    # That gives roughly 26 years: the dot-com bust, 2008, 2020 and 2022.
    "DATA_START": "1999-01-01",
    "DATA_END": None,          # None = up to the latest available day
    "BENCHMARK": "SPY",
    "START_CAPITAL": 10_000,
    # Free list of which stocks were in the S&P 500 on each date since 1996,
    # maintained by an individual on GitHub (github.com/fja05680/sp500).
    "MEMBERSHIP_URL": (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
    ),
    # A ticker with fewer days of data than this is treated as failed.
    "MIN_DAYS_OF_DATA": 60,
    "DOWNLOAD_BATCH": 50,      # tickers per download request
    "REFRESH_DATA": False,     # True = ignore the saved data and re-download

    # ---- Folders -----------------------------------------------------------
    "CACHE_DIR": "data/momentum_cache",
    "OUTPUT_DIR": "out/momentum",
}


def say(msg: str) -> None:
    """Print a progress line straight away (no waiting for buffers)."""
    print(msg, flush=True)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_membership(cfg: dict) -> pd.DataFrame:
    """
    Return a table of S&P 500 members by date.

    Each row is a date, and the 'tickers' column holds the set of stocks in
    the index on that date. Saved to disk after the first download.

    This is the fix for one half of survivorship bias: on each month-end the
    strategy may only pick from stocks that were actually in the index THEN,
    not from today's list.
    """
    os.makedirs(cfg["CACHE_DIR"], exist_ok=True)
    path = os.path.join(cfg["CACHE_DIR"], "sp500_membership.csv")

    if os.path.exists(path) and not cfg["REFRESH_DATA"]:
        say(f"  Membership list: using saved copy ({path})")
        raw = pd.read_csv(path)
    else:
        import requests
        say("  Membership list: downloading historical S&P 500 members...")
        resp = requests.get(cfg["MEMBERSHIP_URL"], timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
        raw.to_csv(path, index=False)

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").set_index("date")
    # Yahoo writes share classes with a dash (BRK-B), the list uses a dot.
    members = raw["tickers"].apply(
        lambda s: frozenset(t.strip().replace(".", "-")
                            for t in str(s).split(",") if t.strip()))
    say(f"  Membership list: {len(members)} snapshots, "
        f"{members.index[0].date()} to {members.index[-1].date()}")
    return members.to_frame("tickers")


def members_on(members: pd.DataFrame, date: pd.Timestamp) -> frozenset:
    """
    Stocks in the index on a given date.

    Uses the most recent snapshot ON OR BEFORE that date -- never a later
    one, so no future index changes leak in.
    """
    past = members.loc[:date]
    if past.empty:
        return frozenset()
    return past["tickers"].iloc[-1]


def _download_batch(tickers: list[str], start: str, end: str | None
                    ) -> pd.DataFrame:
    """Ask Yahoo for one batch of tickers. Returns adjusted closes."""
    import logging
    import yfinance as yf
    # Yahoo prints an error for every delisted ticker; we log those
    # ourselves in dropped_tickers.csv, so keep the screen readable.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data = yf.download(tickers, start=start, end=end, interval="1d",
                       auto_adjust=True,   # adjusted = dividends + splits in
                       progress=False, threads=True)
    if data is None or data.empty:
        return pd.DataFrame()
    # yfinance returns columns like ("Close", "AAPL"); keep only closes.
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    return closes


def load_prices(tickers: list[str], cfg: dict,
                download=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily adjusted closing prices, one column per ticker.

    * Saves to disk after every batch, so an interrupted first run picks up
      where it stopped instead of starting over.
    * Never crashes on a bad ticker: anything that fails or comes back
      (nearly) empty is recorded with a reason and left out.

    Returns (prices, dropped) where 'dropped' lists every ticker left out.
    """
    download = download or _download_batch
    os.makedirs(cfg["CACHE_DIR"], exist_ok=True)
    price_path = os.path.join(cfg["CACHE_DIR"], "prices.parquet")
    tried_path = os.path.join(cfg["CACHE_DIR"], "tickers_tried.csv")

    prices = pd.DataFrame()
    tried: dict[str, str] = {}
    if not cfg["REFRESH_DATA"]:
        if os.path.exists(price_path):
            prices = pd.read_parquet(price_path)
        if os.path.exists(tried_path):
            t = pd.read_csv(tried_path)
            tried = dict(zip(t["ticker"], t["status"]))

    todo = [t for t in tickers if t not in tried]
    if todo:
        say(f"  Prices: {len(tickers) - len(todo)} tickers already saved, "
            f"{len(todo)} to download")
    else:
        say(f"  Prices: all {len(tickers)} tickers loaded from saved copy")

    step = cfg["DOWNLOAD_BATCH"]
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        say(f"    downloading {i + 1}-{i + len(batch)} of {len(todo)}...")
        got = pd.DataFrame()
        for attempt in range(3):           # try each batch up to 3 times
            try:
                got = download(batch, cfg["DATA_START"], cfg["DATA_END"])
            except Exception as exc:       # network hiccup, rate limit...
                say(f"      batch failed ({exc})")
                got = pd.DataFrame()
            if not got.empty and got.notna().any().any():
                break
            time.sleep(5 * (attempt + 1))

        # A whole batch with nothing in it usually means Yahoo blocked us
        # for a while, not that 50 companies vanished. Don't record these
        # as dead -- leave them to be tried again on the next run.
        if got.empty or not got.notna().any().any():
            say("      nothing came back for this batch; will retry these "
                "on the next run")
            continue

        if getattr(got.index, "tz", None) is not None:
            got.index = got.index.tz_localize(None)
        new_cols = {}
        for t in batch:
            col = got[t] if t in got.columns else pd.Series(dtype=float)
            col = col.where(col > 0).dropna()   # prices must be positive
            if len(col) >= cfg["MIN_DAYS_OF_DATA"]:
                new_cols[t] = col
                tried[t] = "ok"
            elif len(col) == 0:
                tried[t] = "no data from Yahoo (likely delisted or renamed)"
            else:
                tried[t] = f"only {len(col)} days of data"
        if new_cols:
            prices = pd.concat([prices, pd.DataFrame(new_cols)], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
            prices = prices.sort_index()
        # Save progress after each batch.
        if not prices.empty:
            prices.to_parquet(price_path)
        pd.DataFrame({"ticker": list(tried), "status": list(tried.values())}
                     ).to_csv(tried_path, index=False)

    missing = [t for t in tickers if t not in tried]
    if missing:
        say(f"  WARNING: {len(missing)} tickers could not be downloaded this "
            "run (Yahoo may be rate-limiting). Run the script again later to "
            "fill them in; they are left out of this run.")

    keep = [t for t in tickers if tried.get(t) == "ok" and t in prices.columns]
    dropped = pd.DataFrame(
        [(t, tried.get(t, "not downloaded this run (retry later)"))
         for t in tickers if t not in keep],
        columns=["ticker", "reason"])
    prices = prices[keep].sort_index()
    prices.index = pd.DatetimeIndex(prices.index)
    start = pd.Timestamp(cfg["DATA_START"])
    prices = prices.loc[start:]
    if cfg["DATA_END"]:
        prices = prices.loc[:pd.Timestamp(cfg["DATA_END"])]
    say(f"  Prices: {len(keep)} tickers usable, {len(dropped)} dropped")
    return prices, dropped


def data_quality_flags(prices: pd.DataFrame) -> pd.DataFrame:
    """
    List any single-day move bigger than +100% or -60%.

    Free data sometimes has bad prints. A bogus spike can push a stock to the
    top of the momentum ranking, so these are logged for you to eyeball. They
    are NOT removed, because some are real (buyouts, bankruptcies).
    """
    rets = prices.pct_change(fill_method=None)
    big = rets.where((rets > 1.0) | (rets < -0.6)).stack().dropna()
    out = big.reset_index()
    out.columns = ["date", "ticker", "daily_return"]
    return out


# =============================================================================
# 3. SIGNALS
# =============================================================================

def month_end_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    The last trading day of each month.

    A day counts only if the NEXT trading day is in a new month. So the most
    recent month, if it is still in progress, is left out -- we can't know
    today is the last day of the month until tomorrow arrives.
    """
    d = pd.DatetimeIndex(trading_days)
    nxt = d[1:]
    is_last = np.array([a.month != b.month for a, b in zip(d[:-1], nxt)])
    return d[:-1][is_last]


def calc_signals(prices: pd.DataFrame, spy: pd.Series,
                 members: pd.DataFrame, lookback: int, skip: int,
                 top_pct: float, sma_months: int) -> pd.DataFrame:
    """
    Decide, at each month-end, what to hold next month.

    Returns one row per month-end with:
      * month_end   the signal date (uses that day's close and earlier only)
      * trade_day   the NEXT trading day, when the trades actually happen
      * risk_on     True if SPY closed above its 10-month average
      * picks       the stocks to buy (empty when the filter says cash)
      * n_eligible  how many index members had enough data to be ranked
      * n_members   how many stocks the index had that day
    """
    days = spy.index
    ends = month_end_dates(days)

    # Month-end prices only. Missing = the stock had no price that day.
    monthly = prices.reindex(ends)
    spy_m = spy.reindex(ends)

    # Momentum: price one month ago divided by price 12 months ago, minus 1.
    # .shift(n) means "the value from n rows (= n months) earlier", so this
    # can only ever look backwards.
    momentum = monthly.shift(skip) / monthly.shift(lookback) - 1

    # Market filter: SPY's month-end close vs the average of the last 10
    # month-end closes (this month included -- it's known at today's close).
    sma = spy_m.rolling(sma_months).mean()
    risk_on = spy_m > sma

    rows = []
    for i, end in enumerate(ends):
        later = days[days > end]
        if len(later) == 0:
            continue
        trade_day = later[0]

        # Not enough history yet to compute the signal or the average.
        if pd.isna(sma.iloc[i]) or i < lookback:
            continue

        in_index = members_on(members, end)
        # Rankable = in the index today, priced today, and with a momentum
        # value (which needs prices 1 and 12 months ago).
        today_px = monthly.iloc[i]
        mom = momentum.iloc[i]
        ok = mom.index.isin(list(in_index)) & today_px.notna() & mom.notna()
        ranked = mom[ok].sort_values(ascending=False)
        n_pick = max(1, int(round(len(ranked) * top_pct))) if len(ranked) else 0

        on = bool(risk_on.iloc[i])
        picks = list(ranked.index[:n_pick]) if on else []
        rows.append({"month_end": end, "trade_day": trade_day,
                     "risk_on": on, "picks": picks,
                     "n_eligible": len(ranked), "n_members": len(in_index)})

    return pd.DataFrame(rows)


# =============================================================================
# 4. BACKTEST
# =============================================================================

def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame,
                 cost: float, start: pd.Timestamp | None = None,
                 end: pd.Timestamp | None = None,
                 capital: float = 10_000, cash_return: float = 0.0) -> dict:
    """
    Trade the signals day by day and track the account value.

    How a rebalance works, on each trade day (the day AFTER the signal):
      1. Value what we hold at that day's closing prices.
      2. Work out the target: equal amounts in each pick, or all cash.
      3. The cost is (cost rate) x (total dollars bought + total sold).
      4. Buy/sell at that day's close, then hold until the next trade day.

    Between trade days, holdings drift with the market. If a held stock
    stops trading (e.g. it was taken over), it is valued at its last price
    and sold at that price on the next trade day.

    Returns a dict with:
      equity    daily account value
      turnover  per rebalance: share of the portfolio swapped (0..1)
      costs     dollars paid in costs at each rebalance
      holdings  what was held after each rebalance
    """
    sig = signals.copy()
    if start is not None:
        sig = sig[sig["trade_day"] >= start]
    if end is not None:
        sig = sig[sig["trade_day"] <= end]
    sig = sig.reset_index(drop=True)
    if sig.empty:
        raise ValueError("No signals in the chosen period.")

    # Forward-fill so a stock with a gap (or that stopped trading) keeps
    # its last known price. Only used for valuing and selling holdings --
    # the ranking above used the raw prices with gaps left in.
    px = prices.ffill()
    first_day = sig["trade_day"].iloc[0]
    last_day = px.index[-1] if end is None else min(end, px.index[-1])
    days = px.loc[first_day:last_day].index

    units = pd.Series(dtype=float)     # number of "shares" per stock
    cash = float(capital)
    daily_cash_growth = (1 + cash_return) ** (1 / 252)

    equity = pd.Series(index=days, dtype=float)
    turnover, costs, holdings = [], [], []

    trade_days = list(sig["trade_day"]) + [None]
    for k, row in sig.iterrows():
        t0 = row["trade_day"]
        t1 = trade_days[k + 1]

        # 1. What we own right now, at today's close.
        p_now = px.loc[t0]
        cur_val = (units * p_now.reindex(units.index)).fillna(0.0)
        total = cash + cur_val.sum()

        # 2. Target dollars in each stock (equal weight) or all cash.
        picks = row["picks"]
        target = pd.Series(total / len(picks), index=picks) if picks \
            else pd.Series(dtype=float)

        # 3. Dollars traded = |target - current| for every stock touched.
        all_names = cur_val.index.union(target.index)
        diff = target.reindex(all_names, fill_value=0.0) \
            - cur_val.reindex(all_names, fill_value=0.0)
        traded = diff.abs().sum()
        fee = traded * cost
        total_after = total - fee

        # Shrink the targets slightly so the fee comes out of the account.
        if picks:
            target = target * (total_after / total)
            units = target / p_now.reindex(picks)
            cash = 0.0
        else:
            units = pd.Series(dtype=float)
            cash = total_after

        # Turnover: half the dollars traded, as a share of the account.
        # 1.0 = the whole portfolio was replaced (sold everything, bought
        # all new). Half because each swap is one sell plus one buy.
        turnover.append((t0, traded / 2 / total if total > 0 else 0.0))
        costs.append((t0, fee))
        holdings.append((row["month_end"], t0, row["risk_on"],
                         len(picks), " ".join(picks)))

        # 4. Hold until the next trade day (or the end of the data).
        stop = t1 if t1 is not None else last_day
        span = px.loc[t0:stop].index
        if t1 is not None:
            span = span[:-1]            # next trade day is valued next loop
        if len(units):
            vals = px.loc[span, units.index].mul(units, axis=1).sum(axis=1)
            equity.loc[span] = vals.values + cash
        else:
            n = np.arange(len(span))
            equity.loc[span] = cash * daily_cash_growth ** n
            cash = cash * daily_cash_growth ** len(span)

    equity = equity.dropna()
    return {
        "equity": equity,
        "turnover": pd.Series(dict(turnover)),
        "costs": pd.Series(dict(costs)),
        "holdings": pd.DataFrame(holdings, columns=[
            "month_end", "trade_day", "risk_on", "n_stocks", "tickers"]),
    }


def buy_and_hold(spy: pd.Series, start: pd.Timestamp,
                 end: pd.Timestamp | None, capital: float) -> pd.Series:
    """SPY bought at the close on the start day and never sold."""
    s = spy.loc[start:end] if end is not None else spy.loc[start:]
    return capital * s / s.iloc[0]


# =============================================================================
# 5. STATS
# =============================================================================

def drawdown(equity: pd.Series) -> pd.Series:
    """How far below its previous high the account is on each day (0 to -1)."""
    return equity / equity.cummax() - 1


def calc_stats(equity: pd.Series) -> dict:
    """The headline numbers for one equity curve."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = drawdown(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    recovered = after[after >= equity.loc[peak]]
    vol = equity.pct_change().std() * np.sqrt(252)
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "years": round(years, 2),
        "final_value": equity.iloc[-1],
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": dd.min(),
        "dd_peak": peak.date(),
        "dd_trough": trough.date(),
        "dd_recovered": recovered.index[0].date() if len(recovered)
        else "not yet",
        "volatility": vol,
    }


def yearly_returns(strat: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Calendar-year returns side by side. Partial years are marked."""
    out = []
    for yr in sorted(set(strat.index.year)):
        s = strat[strat.index.year == yr]
        b = spy[spy.index.year == yr]
        # Use the previous year's last value as the starting point, so each
        # year's return includes its first trading day.
        s_prev = strat[strat.index.year < yr]
        b_prev = spy[spy.index.year < yr]
        s0 = s_prev.iloc[-1] if len(s_prev) else s.iloc[0]
        b0 = b_prev.iloc[-1] if len(b_prev) else b.iloc[0]
        partial = (s.index[0].month > 1 and not len(s_prev)) \
            or s.index[-1].month < 12
        out.append({"year": yr,
                    "strategy": s.iloc[-1] / s0 - 1,
                    "spy": b.iloc[-1] / b0 - 1,
                    "partial_year": partial})
    df = pd.DataFrame(out).set_index("year")
    df["difference"] = df["strategy"] - df["spy"]
    df["beat_spy"] = df["difference"] > 0
    return df


def share_of_years_beating(yearly: pd.DataFrame) -> float:
    """Share of FULL calendar years in which the strategy beat SPY."""
    full = yearly[~yearly["partial_year"]]
    return full["beat_spy"].mean() if len(full) else float("nan")


def yearly_turnover(turnover: pd.Series, equity: pd.Series) -> float:
    """
    Average share of the portfolio replaced per year.

    100% means the equivalent of selling the whole portfolio and buying a
    brand-new one once a year.
    """
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return turnover.sum() / years


# =============================================================================
# 6. CHARTS
# =============================================================================

def make_charts(strat: pd.Series, spy: pd.Series, out_dir: str,
                label: str) -> list[str]:
    """Save the two charts as PNG files and return their paths."""
    import matplotlib
    matplotlib.use("Agg")              # draw to file, no window needed
    import matplotlib.pyplot as plt

    blue, grey = "#2a6fdb", "#7a7a7a"
    paths = []

    # Chart 1: growth of $10,000. Log scale, so a 10% move looks the same
    # size whether the account is at $10k or $100k.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(strat.index, strat.values, color=blue, lw=1.6, label=label)
    ax.plot(spy.index, spy.values, color=grey, lw=1.6,
            label="SPY buy and hold")
    ax.set_yscale("log")
    # Label the money axis in plain dollars at 1-2-5 steps.
    lo = min(strat.min(), spy.min())
    hi = max(strat.max(), spy.max())
    ticks = [m * 10 ** e for e in range(3, 8) for m in (1, 2, 5)
             if lo / 1.2 <= m * 10 ** e <= hi * 1.2]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("Growth of $10,000 (log scale)", loc="left", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "chart1_growth_of_10000.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # Chart 2: drawdown -- how far below its last high each one was.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(spy.index, drawdown(spy).values * 100, 0,
                    color=grey, alpha=0.35, label="SPY buy and hold")
    ax.plot(strat.index, drawdown(strat).values * 100, color=blue, lw=1.2,
            label=label)
    ax.set_ylabel("% below previous high")
    ax.set_title("Drawdown over time", loc="left", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "chart2_drawdown.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


# =============================================================================
# 7. CHECKS
# =============================================================================

def lookahead_self_check(prices: pd.DataFrame, spy: pd.Series,
                         members: pd.DataFrame, signals: pd.DataFrame,
                         cfg: dict, n_checks: int = 12) -> bool:
    """
    Prove the signal never peeks at the future.

    For a spread of month-ends, cut ALL data off at that day's close (as if
    it were really that day), recompute the signal from scratch, and check
    it picks exactly the same stocks and makes the same cash/invested call
    as the full-history run. If any later data were leaking in, the two
    would differ.
    """
    check_rows = signals.iloc[
        np.linspace(0, len(signals) - 1, n_checks).astype(int)]
    all_ok = True
    for _, row in check_rows.iterrows():
        cut = row["month_end"]
        # Keep one extra trading day so the cut-off day is recognised as a
        # month-end -- but blank out every price on that extra day.
        nxt = spy.index[spy.index > cut][:1]
        p_cut = prices.loc[:cut].reindex(prices.loc[:cut].index.append(nxt))
        s_cut = spy.loc[:cut].reindex(spy.loc[:cut].index.append(nxt))
        m_cut = members.loc[:cut]
        redo = calc_signals(p_cut, s_cut, m_cut, cfg["LOOKBACK_MONTHS"],
                            cfg["SKIP_MONTHS"], cfg["TOP_PCT"],
                            cfg["SMA_MONTHS"])
        redo = redo[redo["month_end"] == cut]
        same = (len(redo) == 1
                and redo["picks"].iloc[0] == row["picks"]
                and redo["risk_on"].iloc[0] == row["risk_on"])
        all_ok &= same
        say(f"    {cut.date()}: {'same picks' if same else 'MISMATCH'}")
    return all_ok


# =============================================================================
# 8. MAIN
# =============================================================================

def pct(x: float) -> str:
    return f"{x * 100:+.1f}%" if pd.notna(x) else "n/a"


def run_one(prices, spy, members, cfg, cost, top_pct=None, lookback=None,
            start=None, end=None, signals=None):
    """Signals + backtest + stats for one setting. Returns a small dict."""
    top_pct = top_pct or cfg["TOP_PCT"]
    lookback = lookback or cfg["LOOKBACK_MONTHS"]
    if signals is None:
        signals = calc_signals(prices, spy, members, lookback,
                               cfg["SKIP_MONTHS"], top_pct, cfg["SMA_MONTHS"])
    bt = run_backtest(prices, signals, cost, start, end,
                      cfg["START_CAPITAL"], cfg["CASH_RETURN"])
    eq = bt["equity"]
    bench = buy_and_hold(spy, eq.index[0], eq.index[-1], cfg["START_CAPITAL"])
    return {"bt": bt, "equity": eq, "spy": bench, "signals": signals,
            "s": calc_stats(eq), "b": calc_stats(bench)}


def main(cfg: dict = CONFIG) -> str:
    out = cfg["OUTPUT_DIR"]
    os.makedirs(out, exist_ok=True)

    # ---- Load ---------------------------------------------------------------
    say("\n[1/7] Loading data")
    members = load_membership(cfg)
    start = pd.Timestamp(cfg["DATA_START"])
    # Every stock that was in the index at any point in our period.
    period = members.loc[:start].tail(1).index.append(
        members.loc[start:].index)
    universe = sorted(set().union(*members.loc[period, "tickers"]))
    say(f"  {len(universe)} different stocks were in the S&P 500 at some "
        f"point since {start.date()}")

    bench_px, _ = load_prices([cfg["BENCHMARK"]], cfg)
    if cfg["BENCHMARK"] not in bench_px.columns:
        raise SystemExit("Could not download SPY -- nothing to compare with.")
    spy = bench_px[cfg["BENCHMARK"]].dropna()

    prices, dropped = load_prices(
        [t for t in universe if t != cfg["BENCHMARK"]], cfg)
    prices = prices.reindex(spy.index)   # line everything up on SPY's days
    dropped.to_csv(os.path.join(out, "dropped_tickers.csv"), index=False)
    say(f"  Dropped tickers logged to {out}/dropped_tickers.csv")
    flags = data_quality_flags(prices)
    flags.to_csv(os.path.join(out, "data_quality_big_moves.csv"), index=False)
    say(f"  {len(flags)} suspicious one-day moves logged to "
        f"{out}/data_quality_big_moves.csv")

    # ---- Main test ----------------------------------------------------------
    say("\n[2/7] Calculating signals (12-1 momentum, top 10%, SPY filter)")
    base_sig = calc_signals(prices, spy, members, cfg["LOOKBACK_MONTHS"],
                            cfg["SKIP_MONTHS"], cfg["TOP_PCT"],
                            cfg["SMA_MONTHS"])
    say(f"  {len(base_sig)} month-ends from {base_sig['month_end'].iloc[0].date()}"
        f" to {base_sig['month_end'].iloc[-1].date()}; "
        f"invested {base_sig['risk_on'].mean():.0%} of months, "
        f"in cash {1 - base_sig['risk_on'].mean():.0%}")

    say("\n[3/7] Running the full-period backtest at three cost levels")
    runs = {}
    for name, c in [("no costs", 0.0), ("0.1% per trade", cfg["COST_NORMAL"]),
                    ("0.3% per trade", cfg["COST_HARSH"])]:
        runs[name] = run_one(prices, spy, members, cfg, c, signals=base_sig)
        say(f"  {name:>15}: CAGR {pct(runs[name]['s']['cagr'])}")
    main_name = "0.1% per trade" if cfg["VERDICT_COST"] == cfg["COST_NORMAL"] \
        else "0.3% per trade"
    main_run = runs[main_name]
    s, b = main_run["s"], main_run["b"]

    # ---- Stats tables -------------------------------------------------------
    say("\n[4/7] Calculating stats")
    yearly = yearly_returns(main_run["equity"], main_run["spy"])
    yearly.to_csv(os.path.join(out, "yearly_returns.csv"))
    beat_share = share_of_years_beating(yearly)
    spy_yearly_share = 1 - beat_share if pd.notna(beat_share) else np.nan

    summary_rows = []
    for name, r in runs.items():
        row = {"series": f"Strategy ({name})", **r["s"]}
        row["share_of_years_beating_spy"] = share_of_years_beating(
            yearly_returns(r["equity"], r["spy"]))
        row["yearly_turnover"] = yearly_turnover(r["bt"]["turnover"],
                                                 r["equity"])
        row["total_costs_paid"] = r["bt"]["costs"].sum()
        summary_rows.append(row)
    summary_rows.append({"series": "SPY buy and hold", **b,
                         "share_of_years_beating_spy": np.nan,
                         "yearly_turnover": 0.0, "total_costs_paid": 0.0})
    summary = pd.DataFrame(summary_rows).set_index("series")
    summary.to_csv(os.path.join(out, "summary.csv"))

    curves = pd.DataFrame({f"strategy_{k}": v["equity"]
                           for k, v in runs.items()})
    curves["spy"] = main_run["spy"]
    curves.to_csv(os.path.join(out, "equity_curves.csv"))
    main_run["bt"]["holdings"].to_csv(os.path.join(out, "holdings.csv"),
                                      index=False)

    gross = runs["no costs"]["s"]
    cost_drag_cagr = gross["cagr"] - s["cagr"]
    cost_drag_final = gross["final_value"] - s["final_value"]
    harsh = runs["0.3% per trade"]["s"]
    turn = yearly_turnover(main_run["bt"]["turnover"], main_run["equity"])

    # ---- Out-of-sample ------------------------------------------------------
    say("\n[5/7] Out-of-sample test: each half of the period run separately")
    first, last = main_run["equity"].index[0], main_run["equity"].index[-1]
    mid = first + (last - first) / 2
    halves = []
    for label, a, z in [("First half", first, mid), ("Second half", mid, last)]:
        r = run_one(prices, spy, members, cfg, cfg["VERDICT_COST"],
                    start=a, end=z, signals=base_sig)
        excess = r["s"]["cagr"] - r["b"]["cagr"]
        halves.append({"half": label, "start": r["s"]["start"],
                       "end": r["s"]["end"],
                       "strategy_cagr": r["s"]["cagr"],
                       "spy_cagr": r["b"]["cagr"], "excess_cagr": excess,
                       "strategy_max_dd": r["s"]["max_drawdown"],
                       "spy_max_dd": r["b"]["max_drawdown"],
                       "passes": excess >= cfg["OOS_MIN_EXCESS_CAGR"]})
        say(f"  {label} ({r['s']['start']} to {r['s']['end']}): strategy "
            f"{pct(r['s']['cagr'])}/yr vs SPY {pct(r['b']['cagr'])}/yr")
    oos = pd.DataFrame(halves).set_index("half")
    oos.to_csv(os.path.join(out, "out_of_sample.csv"))

    # ---- Robustness ---------------------------------------------------------
    say("\n[6/7] Robustness grid (report only -- nothing is picked from this)")
    grid = []
    for lb in cfg["ROBUST_LOOKBACKS"]:
        for tp in cfg["ROBUST_TOP_PCTS"]:
            r = run_one(prices, spy, members, cfg, cfg["VERDICT_COST"],
                        top_pct=tp, lookback=lb, start=first)
            grid.append({"lookback_months": lb, "top_pct": tp,
                         "cagr": r["s"]["cagr"],
                         "excess_vs_spy": r["s"]["cagr"] - r["b"]["cagr"],
                         "max_drawdown": r["s"]["max_drawdown"]})
            say(f"  lookback {lb:>2}m, top {tp:.0%}: CAGR "
                f"{pct(r['s']['cagr'])}, vs SPY "
                f"{pct(r['s']['cagr'] - r['b']['cagr'])}")
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(out, "robustness.csv"), index=False)
    base_x = grid[(grid["lookback_months"] == cfg["LOOKBACK_MONTHS"])
                  & (grid["top_pct"] == cfg["TOP_PCT"])]["excess_vs_spy"].iloc[0]
    lbs, tps = cfg["ROBUST_LOOKBACKS"], cfg["ROBUST_TOP_PCTS"]
    li, ti = lbs.index(cfg["LOOKBACK_MONTHS"]), tps.index(cfg["TOP_PCT"])
    neighbours = []
    for dl, dt in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if 0 <= li + dl < len(lbs) and 0 <= ti + dt < len(tps):
            neighbours.append(grid[(grid["lookback_months"] == lbs[li + dl])
                                   & (grid["top_pct"] == tps[ti + dt])]
                              ["excess_vs_spy"].iloc[0])
    fragile = any(abs(n - base_x) > cfg["FRAGILE_GAP"]
                  or np.sign(n) != np.sign(base_x) for n in neighbours)

    # ---- Checks: look-ahead and survivorship --------------------------------
    say("\n[7/7] Checks")
    say("  Look-ahead self-check: re-running the signal with data cut off "
        "at each test date")
    lookahead_ok = lookahead_self_check(prices, spy, members, base_sig, cfg)
    say(f"  Look-ahead self-check: {'PASSED' if lookahead_ok else 'FAILED'}")

    # How many index members on each month-end had usable price data.
    cov = base_sig["n_eligible"] / base_sig["n_members"]
    cov_by_year = cov.groupby(base_sig["month_end"].dt.year).mean()
    cov_by_year.rename("share_of_members_with_data").to_csv(
        os.path.join(out, "data_coverage_by_year.csv"))

    # Survivorship comparison: same rules, but pretending today's members
    # were the members all along (the classic biased way).
    today = members_on(members, members.index[-1])
    biased_members = pd.DataFrame({"tickers": [today]},
                                  index=[members.index[0]])
    biased = run_one(prices, spy, biased_members, cfg, cfg["VERDICT_COST"],
                     start=first)
    say(f"  Using today's member list for all history instead: CAGR "
        f"{pct(biased['s']['cagr'])} (vs {pct(s['cagr'])} point-in-time)")

    # ---- Charts -------------------------------------------------------------
    chart_paths = make_charts(main_run["equity"], main_run["spy"], out,
                              "Momentum strategy (0.1% costs)")

    # ---- Verdict ------------------------------------------------------------
    excess = s["cagr"] - b["cagr"]
    ok_return = excess >= cfg["MIN_EXCESS_CAGR"]
    ok_dd = s["max_drawdown"] >= b["max_drawdown"] - cfg["MAX_EXTRA_DRAWDOWN"]
    ok_oos = bool(oos["passes"].all())
    passed = ok_return and ok_dd and ok_oos

    excess_h = harsh["cagr"] - b["cagr"]
    harsh_dd = harsh["max_drawdown"]
    harsh_pass = (excess_h >= cfg["MIN_EXCESS_CAGR"]
                  and harsh_dd >= b["max_drawdown"] - cfg["MAX_EXTRA_DRAWDOWN"])

    def yes(ok):
        return "yes" if ok else "NO"

    lines = []
    w = lines.append
    w("")
    w("=" * 74)
    w("RESULTS  (strategy at 0.1% per trade vs SPY buy and hold)")
    w("=" * 74)
    w(f"Period: {s['start']} to {s['end']} ({s['years']} years)")
    w("Universe: point-in-time S&P 500 members, but only those Yahoo still "
      "has data for.")
    w("Label: OPTIMISTIC CEILING -- see survivorship notes below.")
    w("")
    w(f"{'':28}{'Strategy':>14}{'SPY':>14}")
    w(f"{'Final value of $10,000':28}{s['final_value']:>14,.0f}"
      f"{b['final_value']:>14,.0f}")
    w(f"{'Total return':28}{pct(s['total_return']):>14}"
      f"{pct(b['total_return']):>14}")
    w(f"{'Yearly average (CAGR)':28}{pct(s['cagr']):>14}{pct(b['cagr']):>14}")
    w(f"{'Worst drop (max drawdown)':28}{pct(s['max_drawdown']):>14}"
      f"{pct(b['max_drawdown']):>14}")
    w(f"{'  drop started':28}{str(s['dd_peak']):>14}{str(b['dd_peak']):>14}")
    w(f"{'  bottom reached':28}{str(s['dd_trough']):>14}"
      f"{str(b['dd_trough']):>14}")
    w(f"{'  recovered':28}{str(s['dd_recovered']):>14}"
      f"{str(b['dd_recovered']):>14}")
    w(f"{'Volatility (per year)':28}{s['volatility'] * 100:>13.1f}%"
      f"{b['volatility'] * 100:>13.1f}%")
    w(f"{'Full years ahead of other':28}{beat_share:>14.0%}"
      f"{spy_yearly_share:>14.0%}")
    w("")
    w("Yearly returns (* = partial year)")
    w(f"{'year':>6}{'Strategy':>11}{'SPY':>9}{'Diff':>9}")
    for yr, r in yearly.iterrows():
        mark = "*" if r["partial_year"] else " "
        w(f"{yr:>5}{mark}{pct(r['strategy']):>11}{pct(r['spy']):>9}"
          f"{pct(r['difference']):>9}")
    w("")
    w("Costs and trading")
    w(f"  CAGR with no costs:          {pct(gross['cagr'])}")
    w(f"  CAGR at 0.1% per trade:      {pct(s['cagr'])}")
    w(f"  CAGR at 0.3% per trade:      {pct(harsh['cagr'])}")
    w(f"  Costs at 0.1% took off {cost_drag_cagr * 100:.2f} points a year, "
      f"or ${cost_drag_final:,.0f} of the final value")
    w(f"  Costs at 0.3% took off "
      f"{(gross['cagr'] - harsh['cagr']) * 100:.2f} points a year, or "
      f"${gross['final_value'] - harsh['final_value']:,.0f}")
    w(f"  Yearly turnover: {turn:.0%} of the portfolio replaced per year")
    w("")
    w("Out-of-sample (each half run separately, 0.1% costs)")
    for h, r in oos.iterrows():
        w(f"  {h:<12} {r['start']} to {r['end']}: strategy "
          f"{pct(r['strategy_cagr'])}/yr, SPY {pct(r['spy_cagr'])}/yr, "
          f"excess {pct(r['excess_cagr'])} -> "
          f"{'clears' if r['passes'] else 'misses'} the 2% bar")
    if oos["passes"].sum() == 1:
        w("  It only works in ONE half. That is a warning sign.")
    w("")
    w("Robustness (yearly excess over SPY; report only, nothing chosen)")
    hdr = "".join(f"{f'top {tp:.0%}':>10}" for tp in tps)
    w(f"  {'lookback':<10}{hdr}")
    for lb in lbs:
        cells = "".join(
            f"{pct(grid[(grid.lookback_months == lb) & (grid.top_pct == tp)]['excess_vs_spy'].iloc[0]):>10}"
            for tp in tps)
        w(f"  {f'{lb} months':<10}{cells}")
    w(f"  Fragile? {'YES -- neighbouring settings give very different answers' if fragile else 'No -- neighbouring settings tell a similar story'}")
    w("")
    w("Bias checks")
    w(f"  Look-ahead self-check (signals rebuilt with future data removed): "
      f"{'PASSED' if lookahead_ok else 'FAILED'}")
    w(f"  Stocks ever in the index in this period: {len(universe)}; "
      f"no usable Yahoo data for {len(dropped)} "
      f"({len(dropped) / len(universe):.0%})")
    w(f"  Share of index members with data, first year: "
      f"{cov_by_year.iloc[0]:.0%}; last year: {cov_by_year.iloc[-1]:.0%}")
    w(f"  Same rules using TODAY's member list for all history: CAGR "
      f"{pct(biased['s']['cagr'])} vs {pct(s['cagr'])} point-in-time "
      f"(difference {pct(biased['s']['cagr'] - s['cagr'])}/yr)")
    w("")
    w("=" * 74)
    w("PASS/FAIL BAR (fixed before running)")
    w(f"  1. Beats SPY by >= {cfg['MIN_EXCESS_CAGR']:.0%}/yr after costs: "
      f"{pct(excess)}/yr -> {yes(ok_return)}")
    w(f"  2. Worst drop no more than {cfg['MAX_EXTRA_DRAWDOWN'] * 100:.0f} "
      f"points deeper than SPY's ({pct(s['max_drawdown'])} vs "
      f"{pct(b['max_drawdown'])}) -> {yes(ok_dd)}")
    w(f"  3. Clears the bar in BOTH halves on its own -> {yes(ok_oos)}")
    w("")
    w(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    w(f"  (At 0.3% per trade it would {'also pass' if harsh_pass else 'fail'}"
      f" bars 1 and 2.)")
    if passed:
        w("  Even a PASS here is an optimistic ceiling: the data is missing "
          "many dead stocks.")
    w("=" * 74)
    w("Not modelled: taxes (TFSA assumed), currency conversion (all in USD),")
    w("dividend withholding tax on US stocks held in a TFSA, bid/ask spread")
    w("beyond the flat cost, and trading at the exact closing price.")
    w(f"Files saved in {out}/ : summary.csv, yearly_returns.csv, "
      "equity_curves.csv,")
    w("  out_of_sample.csv, robustness.csv, holdings.csv, dropped_tickers.csv,")
    w("  data_quality_big_moves.csv, data_coverage_by_year.csv, "
      + ", ".join(os.path.basename(p) for p in chart_paths))

    report = "\n".join(lines)
    print(report, flush=True)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    main()
