"""
Real sessions, and why the feed's idea of a session is not one.

A CFD feed quotes straight through US market holidays. It prints a flat line
at the last traded price: in three years of this data, ten sessions never
move at all, and another sixteen move less than a quarter of a percent.
Left in, a holiday hands an opening-range strategy a range of nothing and an
R multiple of anything, and hands a crossover strategy a signal off an EMA
of a constant.

Half-days matter for the same reason from the other end. The NYSE closes at
13:00 on seven days in this sample; the feed keeps quoting until 16:00, so a
strategy that holds until the bell holds three hours of stale prints.

Both problems go away by taking the session boundaries from the exchange
calendar rather than from the data.
"""
from __future__ import annotations

import functools
import pandas as pd

ET = "America/New_York"

# London's cash open and New York's close, in ET minutes from midnight.
# The London equity session drives European index flow; the combined window
# is what "London and New York only" means everywhere in this package.
LONDON_OPEN_ET = 3 * 60
NY_CLOSE_ET = 16 * 60


@functools.lru_cache(maxsize=8)
def sessions(start="2023-09-01", end="2026-12-31", cal="NYSE", tz=ET):
    """
    Map each real session date to its real closing minute, in `tz`.

    Cached because building a calendar is slow relative to everything else
    and the answer never changes within a run.
    """
    import pandas_market_calendars as mcal
    sched = mcal.get_calendar(cal).schedule(start_date=start, end_date=end)
    close = sched["market_close"].dt.tz_convert(tz)
    open_ = sched["market_open"].dt.tz_convert(tz)
    out = {}
    for day, c in close.items():
        key = pd.Timestamp(day.date()).tz_localize(tz)
        o = open_.loc[day]
        out[key] = (o.hour * 60 + o.minute, c.hour * 60 + c.minute)
    return out


def session_bounds(day, cal="NYSE", window="cash"):
    """
    (start_minute, end_minute) for one session, or None if it is not one.

    window="cash"   -> the exchange's own hours, 09:30-16:00 for the NYSE.
    window="london" -> London's open through the exchange close, which is
                       the 03:00-16:00 ET band the crossover strategy uses.
    """
    tbl = sessions(cal=cal)
    got = tbl.get(pd.Timestamp(day).normalize().tz_localize(ET)
                  if pd.Timestamp(day).tzinfo is None
                  else pd.Timestamp(day).normalize())
    if got is None:
        return None
    o, c = got
    if window == "london":
        return LONDON_OPEN_ET, c
    return o, c


def valid_days(index, cal="NYSE"):
    """Boolean mask over a tz-aware index: is this a real session day?"""
    tbl = sessions(cal=cal)
    days = index.normalize()
    return pd.Index(days).isin(tbl.keys())
