"""
Twelve Data intraday equity bars, for the volume the CFD feed cannot give.

The index CFDs carry a broker-side activity proxy in their volume column, not
exchange volume. That is fine for most things and fatal for one: a relative-
volume filter is *entirely* a volume measurement, so a strategy that leans on
one has to be validated against real prints before it means anything.

SPY and QQQ solve this. They track the same two indices, trade on US
exchanges with real consolidated volume, and sit on Twelve Data's free tier
where the indices themselves (SPX, NDX) are plan-gated.

Budget
------
The free plan allows 800 credits a day and 8 requests a minute. One call
returns at most 5000 bars, which is about 64 regular-hours sessions at the
5-minute interval, so three years of one symbol costs roughly a dozen calls.
The throttle here keeps to 7 a minute to stay clear of the ceiling, since the
same key is used by other jobs.

The API key is read from the TWELVEDATA_API_KEY environment variable, or from
a .env file if one is present. It is never logged.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import pandas as pd
import requests

from . import store

BASE = "https://api.twelvedata.com/time_series"
MAX_BARS = 5000
MAX_PER_MINUTE = 7
ET = "America/New_York"

# ETF proxies for the two indices already in the bar store.
PROXY = {"SPX500": "SPY", "NAS100": "QQQ"}

_calls: deque[float] = deque()


def api_key(env_path: str | None = None) -> str:
    """Find the key without ever putting it on stdout."""
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if key:
        return key
    candidates = [env_path] if env_path else []
    candidates += [
        os.path.join(store.ROOT, ".env"),
        os.path.join(os.path.expanduser("~"), "Documents", "GitHub",
                     "BellCap-Terminal", ".env"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.startswith("TWELVEDATA_API_KEY="):
                    got = line.split("=", 1)[1].strip()
                    if got:
                        return got
    raise RuntimeError(
        "no Twelve Data key: set TWELVEDATA_API_KEY or add it to a .env")


def _throttle() -> None:
    """Sliding window, so a burst waits only as long as it must."""
    while True:
        now = time.monotonic()
        while _calls and now - _calls[0] >= 60.0:
            _calls.popleft()
        if len(_calls) < MAX_PER_MINUTE:
            _calls.append(now)
            return
        time.sleep(max(0.2, 60.0 - (now - _calls[0]) + 0.2))


def _chunk(symbol: str, start: str, end: str, interval: str,
           key: str) -> pd.DataFrame:
    _throttle()
    r = requests.get(
        BASE,
        params={
            "symbol": symbol,
            "interval": interval,
            "start_date": start,
            "end_date": end,
            "outputsize": MAX_BARS,
            "timezone": "America/New_York",
            "format": "JSON",
            "apikey": key,
        },
        timeout=60,
    )
    if r.status_code == 429:
        raise RuntimeError("rate limited")
    payload = r.json()
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(str(payload.get("message"))[:160])
    values = (payload or {}).get("values") or []
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    df["ts"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume"]]


def fetch(symbol: str, start: str, end: str, interval: str = "5min",
          key: str | None = None, quiet: bool = False) -> pd.DataFrame:
    """
    Pull one symbol over a date range, paging backwards in month blocks.

    Twelve Data caps a response at 5000 bars and silently truncates rather
    than erroring, so the range is walked in pieces small enough that the cap
    can never bite. Overlapping edges are de-duplicated on the timestamp.
    """
    key = key or api_key()
    bounds = pd.date_range(start, end, freq="MS").tolist()
    if not bounds or pd.Timestamp(start) < bounds[0]:
        bounds.insert(0, pd.Timestamp(start))
    if pd.Timestamp(end) > bounds[-1]:
        bounds.append(pd.Timestamp(end))

    frames = []
    for i in range(len(bounds) - 1):
        a = bounds[i].strftime("%Y-%m-%d")
        b = (bounds[i + 1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            part = _chunk(symbol, a, b, interval, key)
        except Exception as exc:                       # noqa: BLE001
            if not quiet:
                print(f"    {symbol} {a}..{b}: {exc}")
            continue
        if not part.empty:
            frames.append(part)
            if not quiet:
                print(f"    {symbol} {a} -> {len(part):>5} bars", flush=True)

    if not frames:
        raise RuntimeError(f"nothing returned for {symbol}")
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates("ts")
             .sort_values("ts")
             .reset_index(drop=True))
    out["ts"] = out["ts"].dt.tz_localize(ET, nonexistent="shift_forward",
                                         ambiguous="NaT")
    return out.dropna(subset=["ts"])


def build(symbols=("SPY", "QQQ"), start="2023-09-01", end="2026-09-18",
          interval="5min", dest=None) -> str:
    """Fetch each symbol and write one tidy Parquet file."""
    dest = dest or store.path("equity_5min.parquet")
    key = api_key()
    frames = []
    for sym in symbols:
        print(f"  {sym}:")
        df = fetch(sym, start, end, interval, key)
        df["sym"] = sym
        frames.append(df)
        print(f"  {sym}: {len(df):,} bars "
              f"{df['ts'].min()} -> {df['ts'].max()}")
    allb = pd.concat(frames, ignore_index=True)
    allb = allb.rename(columns={"open": "o", "high": "h", "low": "l",
                                "close": "c", "volume": "v"})
    store.write(allb[["sym", "ts", "o", "h", "l", "c", "v"]], dest)
    print(f"\nwrote {dest}")
    return dest


def load(sym: str, src: str | None = None) -> pd.DataFrame:
    """Read one symbol back as open/high/low/close/volume in New York time."""
    src = src or store.path("equity_5min.parquet")
    df = store.load_m1(sym, src=src)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch intraday ETF bars.")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--start", default="2023-09-01")
    p.add_argument("--end", default="2026-09-18")
    p.add_argument("--interval", default="5min")
    a = p.parse_args(argv)
    build(a.symbols, a.start, a.end, a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
