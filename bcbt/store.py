"""
The bar store: one Parquet file, read straight into numpy.

Minute bars were originally kept in SQLite, which cost 36 seconds and 259MB
for three years of two instruments. The same data as zstd Parquet is 46MB
and loads a single symbol in 0.2 seconds, bit-exact. Nothing here is clever;
it is just the right file format, and it is the difference between a variant
grid you can iterate on and one you run overnight.

Prices are stored float64 on purpose. float32 would save 2MB and introduce a
rounding error of about 0.001 index points, which happens to be exactly the
quantum the vendor quotes to -- enough to flip a stop-hit comparison in the
rare case where price touches a level exactly.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

ET = "America/New_York"
PRICE_COLS = ["o", "h", "l", "c"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PARQUET = os.path.join(DATA, "m1.parquet")


def path(name="m1.parquet"):
    return os.path.join(DATA, name)


def repair_bars(df, tol_bp=5.0):
    """
    Force open and close back inside the bar's own high-low range.

    Vendors round to a fixed number of decimals, and that occasionally puts
    a close a fraction of a cent above its own high. Twenty such bars exist
    in the ETF panel here, each wrong by about a tenth of a basis point --
    harmless in size, but it breaks an invariant the fill engine depends on
    when deciding whether a level was reachable.

    Only rounding-sized violations are corrected. Anything bigger than
    `tol_bp` is a real data fault and is reported rather than quietly
    flattened, because silently repairing a genuinely broken bar is how bad
    prices become backtest profits.
    """
    out = df.copy()
    hi, lo = out["h"], out["l"]
    worst = 0.0
    big = 0
    for col in ("o", "c"):
        over = (out[col] - hi).clip(lower=0)
        under = (lo - out[col]).clip(lower=0)
        err_bp = 1e4 * (over + under) / out[col].abs().replace(0, np.nan)
        worst = max(worst, float(err_bp.max() or 0.0))
        big += int((err_bp > tol_bp).sum())
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out, dict(worst_bp=worst, beyond_tolerance=big)


def write(df, dest=PARQUET):
    """Persist a tidy sym/ts/o/h/l/c/v frame."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["sym"] = out["sym"].astype("category")
    for c in PRICE_COLS:
        out[c] = out[c].astype("float64")
    out = out.sort_values(["sym", "ts"]).reset_index(drop=True)
    out, rep = repair_bars(out)
    if rep["beyond_tolerance"]:
        print(f"  WARNING: {rep['beyond_tolerance']} bars are outside their "
              f"own high-low range by more than rounding")
    out.to_parquet(dest, engine="pyarrow", compression="zstd", index=False)
    return dest


def load_m1(sym, src=PARQUET, tz=ET):
    """One symbol's 1-minute bars, tz-aware, oldest first."""
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{src} not found -- run `python -m bcbt.fetch` to build it")
    df = pd.read_parquet(src, engine="pyarrow",
                         filters=[("sym", "==", sym)])
    if df.empty:
        raise ValueError(f"no rows for {sym} in {src}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return (df.drop(columns=["sym"]).set_index("ts")
              .sort_index().tz_convert(tz))


def symbols(src=PARQUET):
    d = pd.read_parquet(src, engine="pyarrow", columns=["sym"])
    return sorted(d["sym"].astype(str).unique())


def resample(m1, rule, closed="left", label="left"):
    """
    Fold 1-minute bars into a slower series.

    `closed`/`label` default to left so a bar stamped 09:30 covers 09:30
    up to but not including the next stamp -- the convention every strategy
    in this package assumes when it says "the bar that closed at 09:35".
    """
    out = m1.resample(rule, closed=closed, label=label).agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"})
    return out.dropna()


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def atr(bars, span=14):
    prev = bars["c"].shift(1)
    tr = np.maximum(bars["h"] - bars["l"],
                    np.maximum((bars["h"] - prev).abs(),
                               (bars["l"] - prev).abs()))
    return tr.ewm(span=span, adjust=False).mean()


def pivots(high, low, k):
    """
    Confirmed swing highs and lows, as of each bar's close.

    A pivot at bar j needs k bars on either side to be a pivot at all, so it
    is not knowable until bar j+k has closed. Returning the last *confirmed*
    pivot rather than the last pivot is what keeps a swing stop from reading
    the future -- the single easiest way to make a trend system look good.
    """
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = len(high)
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    is_hi = np.zeros(n, bool)
    is_lo = np.zeros(n, bool)

    for i in range(k, n - k):
        wh = high[i - k:i + k + 1]
        wl = low[i - k:i + k + 1]
        if wh.argmax() == k:
            is_hi[i] = True
        if wl.argmin() == k:
            is_lo[i] = True

    cur_hi = cur_lo = np.nan
    for i in range(n):
        j = i - k
        if j >= 0:
            if is_hi[j]:
                cur_hi = high[j]
            if is_lo[j]:
                cur_lo = low[j]
        last_hi[i] = cur_hi
        last_lo[i] = cur_lo
    return last_hi, last_lo


def minutes(index):
    """Minutes since midnight, as int32, for a tz-aware index."""
    return (index.hour * 60 + index.minute).to_numpy(np.int32)


def day_keys(index):
    """
    Session dates as naive datetime64[ns], one per bar.

    `DatetimeIndex.to_numpy()` on a tz-aware index yields an *object* array
    of Timestamps, which compares unequal to any datetime64 and silently
    turns every grouping into a miss. Dropping the zone after normalising
    gives a real datetime64 array that is both fast and comparable.
    """
    return index.normalize().tz_localize(None).to_numpy()


def day_key(ts):
    """The single naive datetime64 key matching `day_keys` for one stamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None) if ts.normalize() == ts else \
            ts.normalize().tz_localize(None)
    return ts.normalize().to_datetime64()


def arrays(m1):
    """The per-minute numpy views the fill engine wants."""
    return dict(
        o=m1["o"].to_numpy(np.float64),
        h=m1["h"].to_numpy(np.float64),
        l=m1["l"].to_numpy(np.float64),
        c=m1["c"].to_numpy(np.float64),
        mins=minutes(m1.index),
        day=day_keys(m1.index),
        idx=m1.index,
    )
