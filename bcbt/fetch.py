"""
Dukascopy minute bars, bid and ask, into the Parquet store.

Dukascopy publishes one LZMA file per instrument per day holding 1440
fixed-width records. Records are 24 bytes big-endian: seconds from midnight
UTC (int32), open/close/low/high as integers scaled by 1000, and volume
(float32). Minutes with no trade are written as zeros and dropped here.

Two things are worth knowing before changing this.

The month in the URL path is zero-indexed. September is `08`. This is not a
typo and it is the single most common way to fetch the wrong month.

The server returns a transient 503 under concurrency often enough that a
naive fetch loses a few percent of days silently. Retries are not optional,
and the run reports what it could not get rather than quietly skipping it.

The ask side is fetched alongside the bid so the spread is an observation
rather than a guess. On this data the S&P quotes 0.51 points all day while
the Nasdaq quotes about 0.96 during New York hours and 1.46 overnight --
a difference big enough that a flat cost assumption misprices half the day.
Dukascopy is tighter than most retail CFD brokers, so treat these as a
floor, not as what you will be charged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import lzma
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import store

# Dukascopy instrument codes.
INSTRUMENTS = {
    "SPX500": "USA500IDXUSD",
    "NAS100": "USATECHIDXUSD",
    "GER40": "DEUIDXEUR",
    "UK100": "GBRIDXGBP",
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",
}

# Prices arrive as integers scaled by the instrument's point value, and the
# point value is NOT the same everywhere. Indices and JPY pairs quote to
# three decimals; the other majors quote to five. Getting this wrong does not
# raise -- it silently moves EURUSD to 103.831 and every ATR with it.
DEFAULT_SCALE = 1000.0
SCALES = {
    "EURUSD": 1e5, "GBPUSD": 1e5, "USDCHF": 1e5,
    "AUDUSD": 1e5, "USDCAD": 1e5, "NZDUSD": 1e5,
    "USDJPY": 1e3,          # JPY pairs quote to three decimals
    "XAUUSD": 1e3,
}


REC = 24
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE = "https://datafeed.dukascopy.com/datafeed"


def scale_for(symbol):
    """Point value for one instrument's integer-encoded prices."""
    return SCALES.get(symbol, DEFAULT_SCALE)


def url(inst, day, side="BID"):
    # The month is zero-indexed in Dukascopy's paths.
    return (f"{BASE}/{inst}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
            f"{side}_candles_min_1.bi5")


def _one(session, inst, day, side, tries=7, timeout=45):
    u = url(inst, day, side)
    for attempt in range(tries):
        try:
            r = session.get(u, timeout=timeout, headers=HEADERS)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""          # nothing published for that day
        except requests.RequestException:
            pass
        import time
        time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(f"gave up on {u}")


def _decode(raw, day, scale=DEFAULT_SCALE):
    if not raw:
        return []
    body = lzma.LZMADecompressor().decompress(raw)
    base = dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc)
    rows = []
    for i in range(len(body) // REC):
        t, o, c, lo, hi, v = struct.unpack(">5if", body[i * REC:(i + 1) * REC])
        if o == 0 and c == 0 and lo == 0 and hi == 0:
            continue
        rows.append(((base + dt.timedelta(seconds=t)),
                     o / scale, hi / scale, lo / scale, c / scale, float(v)))
    return rows


def fetch_range(symbols, start, end, sides=("BID",), workers=8, quiet=False):
    """Pull every weekday in [start, end] for each symbol and side."""
    jobs = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            for s in symbols:
                for side in sides:
                    jobs.append((s, INSTRUMENTS[s], d, side))
        d += dt.timedelta(days=1)

    frames, failed = [], []
    session = requests.Session()

    def work(job):
        sym, inst, day, side = job
        try:
            return sym, side, day, _decode(
                _one(session, inst, day, side), day, scale_for(sym))
        except Exception:
            return sym, side, day, None

    bar = None
    if not quiet:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(jobs), unit="day", desc="dukascopy")
        except ImportError:
            pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, side, day, rows in pool.map(work, jobs):
            if bar is not None:
                bar.update(1)
            if rows is None:
                failed.append((sym, side, day))
                continue
            if rows:
                f = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
                f["sym"] = sym if side == "BID" else f"{sym}.ASK"
                frames.append(f)
    if bar is not None:
        bar.close()

    if not frames:
        raise RuntimeError("nothing downloaded")
    out = pd.concat(frames, ignore_index=True)
    return out, failed


def spread_table(df):
    """Median spread in points per symbol per ET hour, from BID/ASK pairs."""
    rows = []
    for sym in sorted({s for s in df["sym"].unique() if not s.endswith(".ASK")}):
        ask_name = f"{sym}.ASK"
        if ask_name not in set(df["sym"].unique()):
            continue
        b = df[df["sym"] == sym].set_index("ts")["c"]
        a = df[df["sym"] == ask_name].set_index("ts")["c"]
        j = pd.DataFrame({"bid": b, "ask": a}).dropna()
        if j.empty:
            continue
        j["spread"] = j["ask"] - j["bid"]
        j["hour"] = j.index.tz_convert(store.ET).hour
        g = j.groupby("hour")["spread"].median()
        for h, v in g.items():
            rows.append(dict(sym=sym, hour_et=int(h), spread_pts=round(float(v), 4)))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the Parquet bar store.")
    p.add_argument("--symbols", nargs="+", default=["SPX500", "NAS100"],
                   choices=sorted(INSTRUMENTS))
    p.add_argument("--start", default="2023-09-01")
    p.add_argument("--end", default=None, help="default: yesterday")
    p.add_argument("--ask", action="store_true",
                   help="also fetch the ask side, to measure the spread")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=store.PARQUET)
    a = p.parse_args(argv)

    start = dt.date.fromisoformat(a.start)
    end = (dt.date.fromisoformat(a.end) if a.end
           else dt.date.today() - dt.timedelta(days=1))
    sides = ("BID", "ASK") if a.ask else ("BID",)

    df, failed = fetch_range(a.symbols, start, end, sides, a.workers)
    store.write(df, a.out)

    print(f"\nwrote {a.out}")
    for sym, g in df.groupby("sym", observed=True):
        print(f"  {sym:14s} {len(g):>9,} bars  "
              f"{g['ts'].min()} -> {g['ts'].max()}")
    if failed:
        print(f"\n{len(failed)} day(s) failed after retries; re-run to "
              f"pick them up:")
        for f in failed[:10]:
            print("   ", f)
    if a.ask:
        sp = spread_table(df)
        if not sp.empty:
            print("\nmeasured spread (points, median by ET hour):")
            print(sp.pivot(index="hour_et", columns="sym",
                           values="spread_pts").to_string())
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
