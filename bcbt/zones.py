"""
Consolidation zones on a higher timeframe, and what price does at them.

The idea being tested is the one most discretionary traders actually use:
price pauses and builds a range on the hourly chart, breaks out of it, and
then comes back to that range later. The claim is that the old range acts as
a floor when approached from above, or a ceiling from below, and that the
reaction there is tradeable on a faster chart.

A zone here is a run of hourly bars whose whole span is small relative to how
far the market has been moving lately. That is a deliberately mechanical
stand-in for what an eye does when it sees "price went sideways here": the
test is only meaningful if the same chart always produces the same zones.

The timing rules exist to stop the test cheating:

  * Hourly bars are labelled with the time they START. A bar labelled
    09:00 is not finished until 10:00, so every label here is converted to
    the moment the information actually existed before it is used. Getting
    this wrong is not a small error: allowing trades inside the breakout
    hour means trading on the knowledge that the hour will close outside
    the zone, which is the whole setup, and it turned a flat strategy into
    one with a t-statistic of 15.
  * It only becomes a level once price has actually left it, because a
    range nobody has broken out of is not support or resistance yet.
  * A zone goes stale after a fixed number of hours, so the test cannot
    quietly wait months for a level to be revisited and call that a
    prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN = 1, -1


def hourly(m1: pd.DataFrame) -> pd.DataFrame:
    """Hourly bars from minute data, with the usual true-range measure."""
    h = m1.resample("1h", closed="left", label="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    prev = h["c"].shift(1)
    tr = pd.concat([h["h"] - h["l"], (h["h"] - prev).abs(),
                    (h["l"] - prev).abs()], axis=1).max(axis=1)
    h["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    h["range"] = h["h"] - h["l"]

    # How big a bar normally is AT THIS HOUR OF THE DAY.
    #
    # Without this the whole test measures the wrong thing. A plain ATR is
    # dominated by the New York session, so the overnight hours -- which
    # move about a quarter as far -- pass any "unusually tight" test
    # automatically. Measured that way, 81% of the zones found were
    # overnight and 161 of 424 came from the 23:00 bar alone. Those are not
    # consolidations, they are the hours when nobody is trading.
    hour = h.index.hour
    h["norm"] = (h["range"].groupby(hour)
                 .transform(lambda s: s.shift(1).rolling(60, min_periods=20)
                            .median()))
    return h


def find_zones(h: pd.DataFrame, bars=8, max_width_atr=1.5, break_atr=0.5,
               expiry_bars=120, hours=(3, 15)) -> pd.DataFrame:
    """
    Every consolidation zone, with when it formed, broke and expired.

    `bars`          how many hourly bars must sit inside the range
    `max_width_atr` how tight the range has to be, measured against what
                    those particular hours of the day normally cover
    `break_atr`     how far price must leave the zone to count as a break
    `expiry_bars`   how long a zone stays worth watching after it breaks
    `hours`         the window, in New York time, during which a zone may
                    form and be traded -- London open to the New York close

    A zone is recorded at the close of its final bar. Its break, and any
    later return to it, are found strictly after that.
    """
    hi = h["h"].to_numpy()
    lo = h["l"].to_numpy()
    cl = h["c"].to_numpy()
    atr = h["atr"].to_numpy()
    norm = h["norm"].to_numpy()
    hr = h.index.hour.to_numpy()
    idx = h.index
    n = len(h)
    h0, h1 = hours

    rows = []
    i = bars
    while i < n:
        win = slice(i - bars, i)
        # Form only inside the liquid window, and only from bars that were
        # themselves in it.
        if not (h0 <= hr[i - 1] <= h1) or not np.all(
                (hr[win] >= h0) & (hr[win] <= h1)):
            i += 1
            continue

        top = hi[win].max()
        bot = lo[win].min()
        a = atr[i - 1]
        # The yardstick is what these hours usually cover between them, not
        # what an average hour covers.
        expected = np.nanmax(norm[win])
        if (not np.isfinite(a) or a <= 0 or not np.isfinite(expected)
                or expected <= 0 or (top - bot) > max_width_atr * expected):
            i += 1
            continue

        # A zone exists as of the close of bar i-1. Walk forward for the
        # break, which is the first close clearly outside it.
        brk_i, side = -1, 0
        for j in range(i, min(i + expiry_bars, n)):
            if cl[j] > top + break_atr * a:
                brk_i, side = j, UP
                break
            if cl[j] < bot - break_atr * a:
                brk_i, side = j, DOWN
                break
        if brk_i < 0:
            i += 1
            continue

        bar = pd.Timedelta(hours=1)
        rows.append(dict(
            formed=idx[i - 1], broke=idx[brk_i], side=side,
            # When each fact could first have been acted on: one bar after
            # the label, because the bar has to finish first.
            formed_known=idx[i - 1] + bar,
            broke_known=idx[brk_i] + bar,
            top=float(top), bot=float(bot), atr=float(a),
            width=float(top - bot),
            expires=idx[min(brk_i + expiry_bars, n - 1)] + bar))
        # Start the next search after the break, so one sideways stretch
        # does not spawn a dozen overlapping zones.
        i = brk_i + 1
    return pd.DataFrame(rows)


def first_retest(m5: pd.DataFrame, zone, max_bars=None):
    """
    The first time price comes back to a zone after breaking out of it.

    Returns the index position in `m5`, or None. Touching means trading
    into the band at all, which is the loosest reading and therefore the
    one least likely to flatter the strategy by cherry-picking depth.
    """
    after = m5.loc[(m5.index > zone.broke) & (m5.index <= zone.expires)]
    if after.empty:
        return None
    if zone.side == UP:
        hit = after["l"] <= zone.top
    else:
        hit = after["h"] >= zone.bot
    if not hit.any():
        return None
    return after.index[hit.argmax()]


def summarise_zones(z: pd.DataFrame, label=""):
    if z is None or z.empty:
        print(f"  {label}: no zones found")
        return
    up = int((z["side"] == UP).sum())
    print(f"  {label}: {len(z)} zones  ({up} broke up, {len(z) - up} down)")
    w = z["width"].median()
    # Currency pairs move in ten-thousandths, indices in whole points.
    fmt = f"{w:.5f}" if w < 1 else f"{w:.1f}"
    print(f"    width: median {fmt} "
          f"({(z['width'] / z['atr']).median():.2f} x ATR)")
    span = (z["broke"] - z["formed"]).dt.total_seconds() / 3600
    print(f"    hours from forming to breaking: median {span.median():.0f}")
