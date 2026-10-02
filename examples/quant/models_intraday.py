"""
Intraday models on hourly (H) and 15-minute (M) bars: seasonality and
sessions, gaps and the cash open, news events, lead-lag, CFD execution
filters, tick-volume flow proxies, jumps, information-driven bars and
multi-timeframe entries.

Times are New York. A position frame P here means "the position held
during bar t"; `during(P)` turns it into the engine's signal (decided at the
previous bar's close), so P may only use information from before bar t.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bcbt.quant import GROUPS, SPREAD_BP, SLIP_BP, to_ny_day
from examples.quant import lib as L
from examples.quant.core import model

US_IDX = ["SPX500_USD", "NAS100_USD", "US2000_USD"]
EVENT_SYMS = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CAD", "XAU_USD",
              "SPX500_USD", "NAS100_USD", "USB10Y_USD", "USB02Y_USD"]


def _rows(mask):
    """A per-row boolean as a `where` condition broadcast across columns (a
    Series would be aligned on the columns instead, and pandas will not
    broadcast an (n, 1) array in `where`)."""
    m = np.asarray(mask, dtype=bool).reshape(-1, 1)
    return lambda f: np.broadcast_to(m, f.shape)


def _col(mask):
    """A per-row boolean as an (n, 1) array for &, | with a frame."""
    return np.asarray(mask, dtype=bool).reshape(-1, 1)


def during(P):
    """Position held during bar t -> signal at the close of bar t-1."""
    return P.shift(-1).fillna(0)


def nyt(X, freq):
    def f():
        idx = (X.H if freq == "H" else X.M)["C"].index
        ny = idx.tz_convert("America/New_York")
        return pd.DataFrame(dict(
            hour=ny.hour, minute=ny.minute, hm=ny.hour * 100 + ny.minute,
            dow=ny.dayofweek, date=ny.tz_localize(None).normalize(),
            day=to_ny_day(idx), dom=ny.day, year=ny.year), index=idx)
    return X.cached(f"nyt_{freq}", f)


def P_(X, freq):
    return X.H if freq == "H" else X.M


def zeros(X, freq):
    C = P_(X, freq)["C"]
    return pd.DataFrame(0.0, index=C.index, columns=C.columns)


def lr(X, freq):
    return X.cached(f"lr_{freq}", lambda: np.log(P_(X, freq)["C"]).diff())


def window_mask(T, start_hm, end_hm):
    """Bars whose label lies in [start, end) New York time (wraps midnight)."""
    hm = T["hm"].to_numpy()
    if start_hm <= end_hm:
        return (hm >= start_hm) & (hm < end_hm)
    return (hm >= start_hm) | (hm < end_hm)


def window_sum(X, freq, start_hm, end_hm, key):
    """Per symbol, the summed log return of a window, by the NY trading day
    the window ends in; indexed by trading day."""
    def f():
        T = nyt(X, freq)
        m = window_mask(T, start_hm, end_hm)
        r = lr(X, freq)[m]
        return r.groupby(T["day"].to_numpy()[m]).sum(min_count=1)
    return X.cached(key, f)


def map_daily(X, freq, daily, mask):
    """Broadcast a per-trading-day frame onto intraday bars within mask."""
    T = nyt(X, freq)
    out = daily.reindex(T["day"].to_numpy())
    out.index = T.index
    return out.where(_rows(mask), 0.0).fillna(0.0)


def yearly_stats(values, key_series, min_years=3):
    """
    Walk-forward bucket statistics: for each year, mean and t of `values`
    per bucket using only earlier years. values: frame (rows dated),
    key_series: bucket label per row. Returns (mean, t) frames aligned to
    the rows.
    """
    years = values.index.year
    M = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    Tt = M.copy()
    keys = pd.Series(key_series, index=values.index)
    for y in sorted(set(years))[min_years:]:
        tr = values[years < y]
        kt = keys[years < y]
        g = tr.groupby(kt.to_numpy())
        mu, sd, n = g.mean(), g.std(), g.count()
        t = mu / (sd / np.sqrt(n))
        te = years == y
        kk = keys[te].to_numpy()
        M.loc[te] = mu.reindex(kk).to_numpy()
        Tt.loc[te] = t.reindex(kk).to_numpy()
    return M, Tt


# ============================================================ seasonality

@model("tod_season", freq="H")
def tod_season(X):
    """Time-of-day seasonality: each year, the hours whose mean return over
    all earlier years has |t| > 2 are traded in that direction."""
    T = nyt(X, "H")
    r = lr(X, "H")
    M, t = X.cached("tod_stats", lambda: yearly_stats(r, T["hour"]))
    P = np.sign(M).where(t.abs() > 2, 0.0).fillna(0)
    return during(P)


@model("intraday_pattern", freq="H")
def intraday_pattern(X):
    """Intraday seasonal pattern: hold each hour in the sign of its mean
    return over all earlier years (no significance filter)."""
    T = nyt(X, "H")
    r = lr(X, "H")
    M, t = X.cached("tod_stats", lambda: yearly_stats(r, T["hour"]))
    return during(np.sign(M).fillna(0))


@model("dow_season")
def dow_season(X):
    """Day-of-week seasonality: trade the next day's weekday in the sign of
    its mean return over earlier years when |t| > 1.5."""
    nxt_dow = pd.Series(X.C.index.dayofweek, index=X.C.index).shift(-1)
    r = X.LR
    M, t = yearly_stats(r, pd.Series(X.C.index.dayofweek, index=X.C.index))
    # stats are for the day itself; we need them for tomorrow's weekday
    P = np.sign(M).where(t.abs() > 1.5, 0.0).fillna(0)
    return P.shift(-1).fillna(0)


@model("moy_season")
def moy_season(X):
    """Month-of-year seasonality: hold each month in the sign of its mean
    daily return over earlier years when |t| > 1.5."""
    M, t = yearly_stats(X.LR, pd.Series(X.C.index.month, index=X.C.index))
    return (np.sign(M).where(t.abs() > 1.5, 0.0).fillna(0)).shift(-1).fillna(0)


SESS = {"asia": (1900, 300), "london": (300, 800), "ny": (800, 1600)}


@model("session_season", freq="H")
def session_season(X):
    """Session seasonality: each session (Asia 19-03, London 03-08, New York
    08-16 NY time) held in the sign of its earlier-years mean when |t| > 1.5."""
    T = nyt(X, "H")
    lab = np.select([window_mask(T, *SESS["asia"]),
                     window_mask(T, *SESS["london"]),
                     window_mask(T, *SESS["ny"])], [0, 1, 2], 3)
    r = lr(X, "H")
    M, t = yearly_stats(r, pd.Series(lab, index=r.index))
    P = np.sign(M).where((t.abs() > 1.5) & (lab < 3)[:, None], 0.0).fillna(0)
    return during(P)


def _session_predict(X, pred, trade, key, cols=None):
    """Follow or fade the predictor window's return during the trade window,
    by the sign of their correlation over the previous 500 trading days."""
    a = window_sum(X, "H", *pred, key + "_p")
    b = window_sum(X, "H", *trade, key + "_t")
    a, b = a.align(b, join="outer")
    corr = a.rolling(500, min_periods=250).corr(b).shift(1)
    s = (np.sign(corr) * np.sign(a)).where(corr.abs() > 0.05, 0.0)
    if cols is not None:
        k = np.asarray(s.columns.isin(cols)).reshape(1, -1)
        s = s.where(lambda f: np.broadcast_to(k, f.shape), 0.0)
    T = nyt(X, "H")
    return map_daily(X, "H", s, window_mask(T, *trade))


@model("london_session", freq="H")
def london_session(X):
    """London session (03-08 NY) follows or fades the Asia session's return,
    by the sign of their trailing 500-day correlation."""
    return during(_session_predict(X, SESS["asia"], SESS["london"], "s_lon"))


@model("ny_session", freq="H")
def ny_session(X):
    """New York session (08-16) follows or fades the London session."""
    return during(_session_predict(X, SESS["london"], SESS["ny"], "s_ny"))


@model("asia_session", freq="H")
def asia_session(X):
    """Asia session (19-03) follows or fades the prior New York session
    (13-16 window, the part known by the 17:00 roll)."""
    return during(_session_predict(X, (1300, 1600), SESS["asia"], "s_asia"))


@model("session_transition", freq="H")
def session_transition(X):
    """Session transition: first two London hours (03-05) follow or fade the
    last Asia hour (02-03), by trailing correlation."""
    return during(_session_predict(X, (200, 300), (300, 500), "s_tr"))


# ================================================== cash open, gaps, close

def _m15_cash(X):
    """US cash-session anchors on 15-minute bars, per NY calendar date."""
    def f():
        T = nyt(X, "M")
        C = X.M["C"]
        O = X.M["O"]
        d = T["date"].to_numpy()
        def at(hm, frame):
            m = (T["hm"] == hm).to_numpy()
            x = frame[m]
            x.index = d[m]
            return x[~x.index.duplicated()]
        prev_close = at(1545, C).shift(1)          # 16:00 close (bar 15:45)
        open_ = at(930, O)                         # 09:30 open
        first = at(930, C)                         # 09:45
        h1 = at(1015, C)                           # 10:30
        c1000 = at(945, C)                         # 10:00
        c1530 = at(1515, C)                        # 15:30
        close = at(1545, C)
        return dict(prev_close=prev_close, open=open_, first=first, h1=h1,
                    c1000=c1000, c1530=c1530, close=close)
    return X.cached("cash", f)


def _map_date(X, s_daily, mask):
    T = nyt(X, "M")
    out = s_daily.reindex(T["date"].to_numpy())
    out.index = T.index
    return out.where(_rows(mask), 0.0).fillna(0.0)


def _only_cols(frame, cols):
    keep = frame.columns.isin(cols)
    k = np.asarray(keep, dtype=bool).reshape(1, -1)
    return frame.where(lambda f: np.broadcast_to(k, f.shape), 0.0)


@model("overnight_gap_rev", freq="M")
def overnight_gap_rev(X):
    """US index cash gap (09:30 open vs prior 16:00 close) larger than 0.5
    of its 60-day sd: fade it from 09:45 to 16:00."""
    K = _m15_cash(X)
    gap = np.log(K["open"] / K["prev_close"])
    z = gap / gap.rolling(60, min_periods=30).std().shift(1)
    s = -np.sign(gap).where(z.abs() > 0.5, 0.0)
    T = nyt(X, "M")
    P = _map_date(X, s, window_mask(T, 945, 1600))
    return during(_only_cols(P, US_IDX))


@model("gap_return_dist", freq="M")
def gap_return_dist(X):
    """Gap-return distribution: gaps bucketed by size (in 60-day sd); each
    year, the 09:45-16:00 return's mean per bucket over earlier years sets
    the side when |t| > 1.5."""
    K = _m15_cash(X)
    gap = np.log(K["open"] / K["prev_close"])
    z = gap / gap.rolling(60, min_periods=30).std().shift(1)
    rest = np.log(K["close"] / K["first"])
    bucket = pd.DataFrame(np.digitize(z.fillna(0), [-1, -0.25, 0.25, 1]),
                          index=z.index, columns=z.columns)
    out = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for c in US_IDX:
        M, t = yearly_stats(rest[[c]], bucket[c])
        out[c] = np.sign(M[c]).where(t[c].abs() > 1.5, 0.0)
    T = nyt(X, "M")
    return during(_map_date(X, out.fillna(0), window_mask(T, 945, 1600)))


@model("close_to_open", freq="M")
def close_to_open(X):
    """Close-to-open: the overnight move (16:00 -> 09:45) is followed or
    faded to 16:00 by the sign of its trailing 250-day correlation with the
    rest of the day."""
    K = _m15_cash(X)
    on = np.log(K["first"] / K["prev_close"])
    rest = np.log(K["close"] / K["first"])
    corr = on.rolling(250, min_periods=120).corr(rest).shift(1)
    s = (np.sign(corr) * np.sign(on)).where(corr.abs() > 0.05, 0.0)
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s, window_mask(T, 945, 1600)),
                             US_IDX))


@model("opening_auction_proxy", freq="M")
def opening_auction_proxy(X):
    """Opening auction proxy: the first 15 minutes of the cash session
    (09:30-09:45) set the side, held to 16:00."""
    K = _m15_cash(X)
    s = np.sign(np.log(K["first"] / K["open"]))
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s, window_mask(T, 945, 1600)),
                             US_IDX))


@model("open_to_close_mom", freq="M")
def open_to_close_mom(X):
    """Open-to-close momentum: the first hour (09:30-10:30) return sets the
    side, held 10:30 to 16:00."""
    K = _m15_cash(X)
    s = np.sign(np.log(K["h1"] / K["open"]))
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s, window_mask(T, 1030, 1600)),
                             US_IDX))


@model("cash_session_mom", freq="M")
def cash_session_mom(X):
    """Intraday momentum (Gao, Han, Li & Zhou): the first half-hour return
    (prior 16:00 close to 10:00) predicts the last half-hour (15:30-16:00)."""
    K = _m15_cash(X)
    s = np.sign(np.log(K["c1000"] / K["prev_close"]))
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s, window_mask(T, 1530, 1600)),
                             US_IDX))


@model("opening_window", freq="M")
def opening_window(X):
    """Opening-window statistics: hold 09:30-10:30 in the sign of that
    window's earlier-years mean (|t| > 1.5), every CFD."""
    T = nyt(X, "M")
    m = window_mask(T, 930, 1030)
    r = lr(X, "M")
    daily = r[m].groupby(T["date"].to_numpy()[m]).sum(min_count=1)
    M, t = yearly_stats(daily, pd.Series(0, index=daily.index))
    s = np.sign(M).where(t.abs() > 1.5, 0.0)
    return during(_map_date(X, s.fillna(0), m))


@model("closing_window", freq="M")
def closing_window(X):
    """Closing-window statistics: 15:00-16:00, same rule."""
    T = nyt(X, "M")
    m = window_mask(T, 1500, 1600)
    r = lr(X, "M")
    daily = r[m].groupby(T["date"].to_numpy()[m]).sum(min_count=1)
    M, t = yearly_stats(daily, pd.Series(0, index=daily.index))
    s = np.sign(M).where(t.abs() > 1.5, 0.0)
    return during(_map_date(X, s.fillna(0), m))


@model("open_to_hilo", freq="M")
def open_to_hilo(X):
    """Open-to-high/low model: once price has run from the 09:30 open by more
    than the 60-day median open-to-high (open-to-low), fade it to 16:00."""
    T = nyt(X, "M")
    K = _m15_cash(X)
    cash = window_mask(T, 930, 1600)
    H, Lo, C = X.M["H"], X.M["L"], X.M["C"]
    d = T["date"].to_numpy()
    hi = H.where(_rows(cash)).groupby(d).max()
    lo = Lo.where(_rows(cash)).groupby(d).min()
    oh = np.log(hi / K["open"]).rolling(60, min_periods=30).median().shift(1)
    ol = np.log(K["open"] / lo).rolling(60, min_periods=30).median().shift(1)
    op = K["open"].reindex(d)
    op.index = T.index
    oh_b, ol_b = oh.reindex(d), ol.reindex(d)
    oh_b.index = ol_b.index = T.index
    x = np.log(C / op)
    up = (x.gt(oh_b, axis=0)).where(_rows(cash), False)
    dn = (x.lt(-ol_b, axis=0)).where(_rows(cash), False)
    # once triggered, hold the fade to the close of that day
    trig = (dn.astype(float) - up.astype(float)).replace(0, np.nan)
    trig = trig.groupby(d).ffill().fillna(0)
    P = trig.shift(1).where(_rows(cash), 0.0).fillna(0)
    return during(_only_cols(P, US_IDX))


@model("overnight_return", freq="H")
def overnight_return(X):
    """Overnight return: hold the US index CFDs long from the 16:00 close to
    the 09:30 open (which crosses the 17:00 financing roll)."""
    T = nyt(X, "H")
    m = window_mask(T, 1600, 900)
    P = zeros(X, "H")
    for c in US_IDX:
        P[c] = m.astype(float)
    return during(P)


@model("weekend_gap", freq="H")
def weekend_gap(X):
    """Weekend gap: fade the move from Friday's close to the close of the
    week's first hour, for the next 6 hours."""
    C = X.H["C"]
    T = nyt(X, "H")
    first = (T["dow"].to_numpy() == 6) | ((T["dow"].to_numpy() == 0) &
                                           (T["hour"].to_numpy() < 1))
    newwk = pd.Series(first, index=C.index) & ~pd.Series(first, index=C.index
                                                         ).shift(1, fill_value=False)
    gap = np.log(C / C.ffill().shift(1)).where(newwk, np.nan)
    ev = (-np.sign(gap)).where(gap.abs() > 0, np.nan)
    P = ev.ffill(limit=6).shift(1).where(_rows(~newwk), 0.0).fillna(0)
    return during(P)


@model("gap_adjusted", freq="D")
def gap_adjusted(X):
    """Gap-adjusted momentum for US indices: trend on the cash-session
    (09:30-16:00) return only, summed over 20 days, ignoring overnight gaps."""
    T = nyt(X, "M")
    m = window_mask(T, 930, 1600)
    r = lr(X, "M")
    d = pd.Series(to_ny_day(r.index), index=r.index)[m]
    day = r[m].groupby(d.to_numpy()).sum(min_count=1).reindex(X.C.index)
    s = np.sign(day.rolling(20, min_periods=15).sum())
    return _only_cols(s.fillna(0), US_IDX)


# ============================================================== news events

def _event_days(X, hm, syms, mult, dow=None, key=None):
    """Days whose bar at `hm` (NY) had a range above `mult` x the median of
    that bar over the prior 60 days in the average of `syms`. Detected after
    the fact -- usable only for trades that start after the bar closes."""
    def f():
        T = nyt(X, "M")
        m = (T["hm"] == hm).to_numpy()
        rng = np.log(X.M["H"] / X.M["L"])[m][syms]
        rng.index = T["date"].to_numpy()[m]
        rng = rng[~rng.index.duplicated()]
        rel = rng / rng.rolling(60, min_periods=20).median().shift(1)
        ev = rel.mean(axis=1) > mult
        if dow is not None:
            ev &= pd.Series(rng.index.dayofweek, index=rng.index).isin(dow)
        return ev
    return X.cached(key or f"ev_{hm}_{mult}", f)


def _nfp_days(X):
    """First Friday of each month (the usual payrolls date), known ahead."""
    T = nyt(X, "M")
    d = pd.DatetimeIndex(sorted(set(T["date"])))
    return pd.Series((d.dayofweek == 4) & (d.day <= 7), index=d)


def _bar_move(X, hm):
    T = nyt(X, "M")
    m = (T["hm"] == hm).to_numpy()
    mv = np.log(X.M["C"] / X.M["O"])[m]
    mv.index = T["date"].to_numpy()[m]
    return mv[~mv.index.duplicated()]


def _post_event(X, days, hm, end_hm, side=1, cols=EVENT_SYMS, after=None):
    """Follow (side=1) or fade (-1) the event bar's move from its close to
    end_hm on event days."""
    mv = _bar_move(X, hm)
    s = side * np.sign(mv).where(_rows(days.reindex(mv.index).fillna(False)),
                                 0.0)
    T = nyt(X, "M")
    start = after or (hm + 15 if hm % 100 < 45 else hm + 55)
    P = _map_date(X, s, window_mask(T, start, end_hm))
    return during(_only_cols(P, cols))


def _ev830(X):
    return _event_days(X, 830, ["EUR_USD", "USB10Y_USD", "SPX500_USD"], 2.5,
                       key="ev830")


@model("post_news_mom", freq="M")
def post_news_mom(X):
    """Post-news momentum: on days with a large 08:30 release bar, follow
    that bar's direction from 08:45 to 11:00."""
    return _post_event(X, _ev830(X), 830, 1100, +1)


@model("post_news_mr", freq="M")
def post_news_mr(X):
    """Post-news mean reversion: fade the 08:30 release bar, 08:45-11:00."""
    return _post_event(X, _ev830(X), 830, 1100, -1)


@model("event_drift", freq="M")
def event_drift(X):
    """Event drift: follow the 08:30 release bar to the 16:00 close."""
    return _post_event(X, _ev830(X), 830, 1600, +1)


@model("news_event_stat", freq="M")
def news_event_stat(X):
    """News-event statistical model: on 08:30 event days, follow or fade the
    release bar (08:45-11:00) per symbol by the sign of the mean continuation
    on earlier years' event days (|t| > 1.5)."""
    mv = _bar_move(X, 830)
    days = _ev830(X).reindex(mv.index).fillna(False)
    T = nyt(X, "M")
    m = window_mask(T, 845, 1100)
    r = lr(X, "M")
    after = r[m].groupby(T["date"].to_numpy()[m]).sum(min_count=1)
    cont = (np.sign(mv) * after.reindex(mv.index))[days]
    M, t = yearly_stats(cont, pd.Series(0, index=cont.index))
    side = np.sign(M).where(t.abs() > 1.5, 0.0).reindex(mv.index).ffill()
    s = (side * np.sign(mv)).where(_rows(days), 0.0)
    return during(_only_cols(_map_date(X, s.fillna(0), m), EVENT_SYMS))


@model("nfp_reaction", freq="M")
def nfp_reaction(X):
    """NFP reaction: on first-Friday payrolls days, follow the 08:30 bar from
    08:45 to 11:00."""
    return _post_event(X, _nfp_days(X), 830, 1100, +1)


@model("cpi_reaction_proxy", freq="M")
def cpi_reaction(X):
    """CPI reaction (proxy calendar: large 08:30 bars on the 10th-17th, not a
    Friday -- mostly CPI and retail sales): follow 08:45-11:00."""
    ev = _ev830(X)
    d = ev.index
    cpi = ev & (d.day >= 10) & (d.day <= 17) & (d.dayofweek != 4)
    return _post_event(X, cpi, 830, 1100, +1)


def _fomc_days(X):
    a = _event_days(X, 1400, ["SPX500_USD", "EUR_USD", "USB10Y_USD"], 3.5,
                    dow=[2], key="ev1400")
    b = _event_days(X, 1415, ["SPX500_USD", "EUR_USD", "USB10Y_USD"], 3.5,
                    dow=[2], key="ev1415")
    a, b = a.align(b, fill_value=False)
    return a, b


@model("fomc_reaction", freq="M")
def fomc_reaction(X):
    """FOMC reaction (statement days detected as a 3.5x range spike in the
    14:00 or 14:15 bar on a Wednesday): follow the statement bar to 16:00."""
    a, b = _fomc_days(X)
    pa = _post_event(X, a, 1400, 1600, +1, after=1415)
    pb = _post_event(X, b & ~a, 1415, 1600, +1, after=1430)
    return pa + pb


@model("cb_decision", freq="M")
def cb_decision(X):
    """Central-bank decision model: FOMC as above plus ECB days (07:45 NY bar
    spike on a Thursday in EUR/USD and Bund): follow the decision bar to
    11:00 (ECB) or 16:00 (FOMC)."""
    ecb = _event_days(X, 745, ["EUR_USD", "DE10YB_EUR"], 3.0, dow=[3],
                      key="ev745")
    pe = _post_event(X, ecb, 745, 1100, +1, after=800,
                     cols=["EUR_USD", "DE10YB_EUR", "FR40_EUR", "NL25_EUR",
                           "EUR_JPY"])
    return pe + fomc_reaction(X)


@model("pre_news", freq="M")
def pre_news(X):
    """Pre-news positioning: on payrolls Fridays (known in advance) hold
    07:30-08:30 in the sign of the earlier-years mean of that window on
    payrolls days (|t| > 1.5)."""
    T = nyt(X, "M")
    m = window_mask(T, 730, 830)
    r = lr(X, "M")
    w = r[m].groupby(T["date"].to_numpy()[m]).sum(min_count=1)
    nfp = _nfp_days(X).reindex(w.index).fillna(False)
    M, t = yearly_stats(w[nfp], pd.Series(0, index=w.index[nfp]))
    side = np.sign(M).where(t.abs() > 1.5, 0.0).reindex(w.index).ffill()
    s = side.where(_rows(nfp), 0.0).fillna(0)
    return during(_only_cols(_map_date(X, s, m), EVENT_SYMS))


@model("surprise_magnitude_proxy", freq="M")
def surprise_magnitude(X):
    """Surprise-magnitude proxy (no consensus data): on payrolls days a
    release bar larger than 2x the median payrolls bar of earlier releases
    is followed, a smaller one faded, 08:45-11:00."""
    mv = _bar_move(X, 830)
    nfp = _nfp_days(X).reindex(mv.index).fillna(False)
    a = mv.abs().where(_rows(nfp))
    med = a.expanding(min_periods=12).median().shift(1).ffill()
    big = a > 2 * med
    s = np.sign(mv).where(big, -np.sign(mv)).where(_rows(nfp), 0.0)
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s.fillna(0),
                                       window_mask(T, 845, 1100)),
                             EVENT_SYMS))


@model("econ_surprise_proxy", freq="M")
def econ_surprise(X):
    """Economic-surprise proxy: the sign of the 08:30 bar on any large
    release day is taken as the surprise and held to the close only when it
    agrees with the sign of the previous release day's (surprise momentum)."""
    mv = _bar_move(X, 830)
    ev = _ev830(X).reindex(mv.index).fillna(False)
    s0 = np.sign(mv).where(ev)
    prev = s0.ffill().shift(1)
    s = s0.where(s0 == prev, 0.0).fillna(0)
    T = nyt(X, "M")
    return during(_only_cols(_map_date(X, s, window_mask(T, 845, 1600)),
                             EVENT_SYMS))


@model("event_vol", freq="M")
def event_vol(X):
    """Event volatility breakout: on 08:30 event days, the first 15-minute
    close beyond the 07:30-08:30 range sets the side until 11:00."""
    T = nyt(X, "M")
    pre = window_mask(T, 730, 830)
    d = T["date"].to_numpy()
    hi = X.M["H"].where(_rows(pre)).groupby(d).max()
    lo = X.M["L"].where(_rows(pre)).groupby(d).min()
    ev = _ev830(X).reindex(hi.index).fillna(False)
    hb, lb, eb = hi.reindex(d), lo.reindex(d), ev.reindex(d).to_numpy()
    hb.index = lb.index = T.index
    post = window_mask(T, 830, 1100) & eb
    C = X.M["C"]
    sig = ((C > hb).astype(float) - (C < lb).astype(float)).where(
        _rows(post), 0.0)
    first = sig.replace(0, np.nan).groupby(d).ffill().fillna(0)
    P = first.shift(1).where(_rows(window_mask(T, 845, 1100) & eb),
                             0.0).fillna(0)
    return during(_only_cols(P, EVENT_SYMS))


@model("macro_event_regime")
def macro_event_regime(X):
    """Macro-event regime: the multi-speed trend model, but flat across
    payrolls Fridays (the day's risk is event risk, not trend)."""
    from examples.quant.models_daily import trend_follow
    s = trend_follow(X)
    nfp = (X.C.index.dayofweek == 4) & (X.C.index.day <= 7)
    # flat over the payrolls day: the position held from Thursday's close
    nxt = pd.Series(nfp, index=X.C.index).shift(-1, fill_value=False)
    return s.where(_rows(~nxt.to_numpy()), 0.0)


# =============================================================== lead-lag

def _lead_follow(X, freq, leader, followers, k=1.0, sign=1.0, lead_series=None):
    r = lr(X, freq)
    lead = r[leader] if lead_series is None else lead_series
    z = lead / lead.rolling(500, min_periods=200).std().shift(1)
    s = sign * (np.sign(z).where(z.abs() > k, 0.0))
    P = zeros(X, freq)
    for f in followers:
        P[f] = s
    return P.fillna(0)


@model("cross_market_ll", freq="H")
def cross_market_ll(X):
    """Cross-market lead-lag: a 1-sd S&P hour is followed the next hour in
    every other index CFD."""
    return _lead_follow(X, "H", "SPX500_USD",
                        [c for c in GROUPS["index"] if c != "SPX500_USD"])


@model("fx_ll", freq="M")
def fx_ll(X):
    """FX lead-lag: a 1-sd EUR/USD 15-minute bar is followed the next bar in
    GBP/USD and AUD/USD."""
    return _lead_follow(X, "M", "EUR_USD", ["GBP_USD", "AUD_USD"])


def _usd_h(X, freq):
    r = lr(X, freq)
    return (-r["EUR_USD"] - r["GBP_USD"] - r["AUD_USD"] + r["USD_CAD"]) / 4


@model("gold_usd_ll", freq="H")
def gold_usd_ll(X):
    """Gold/USD lead-lag: a 1-sd synthetic-dollar hour -> gold the opposite
    way next hour."""
    return _lead_follow(X, "H", None, ["XAU_USD"], sign=-1.0,
                        lead_series=_usd_h(X, "H"))


@model("oil_usd_ll", freq="H")
def oil_usd_ll(X):
    """Oil/USD lead-lag: a 1-sd WTI hour -> USD/CAD the opposite way next
    hour."""
    return _lead_follow(X, "H", "WTICO_USD", ["USD_CAD"], sign=-1.0)


@model("bond_yield_ll", freq="H")
def bond_yield_ll(X):
    """Bond/yield lead-lag: a 1-sd Treasury CFD hour (yields down) -> short
    the dollar next hour (long EUR/USD, GBP/USD, AUD/USD; short USD/CAD)."""
    P = _lead_follow(X, "H", "USB10Y_USD", ["EUR_USD", "GBP_USD", "AUD_USD"])
    P["USD_CAD"] = -P["EUR_USD"]
    return P


@model("eq_fx_ll", freq="H")
def eq_fx_ll(X):
    """Equity-index/FX lead-lag: a 1-sd S&P hour -> AUD/JPY and EUR/JPY the
    same way next hour."""
    return _lead_follow(X, "H", "SPX500_USD", ["AUD_JPY", "EUR_JPY"])


def _stat_ll(X, freq, tthr=4.0):
    r = lr(X, freq).fillna(0)
    cols = list(r.columns)
    years = r.index.year
    pred = pd.DataFrame(0.0, index=r.index, columns=cols)
    for y in sorted(set(years))[2:]:
        tr = r[(years < y) & (years >= y - 2)]
        Xl = tr.shift(1).fillna(0).to_numpy()
        Y = tr.to_numpy()
        n = len(Y)
        sx = Xl.std(axis=0) + 1e-12
        sy = Y.std(axis=0) + 1e-12
        cc = ((Xl - Xl.mean(0)).T @ (Y - Y.mean(0))) / n / np.outer(sx, sy)
        t = cc * math.sqrt(n)
        np.fill_diagonal(t, 0)              # own-lag is not lead-lag
        B = np.where(np.abs(t) > tthr, cc * sy[None, :] / sx[:, None], 0.0)
        te = years == y
        pred.loc[te] = r[te].to_numpy() @ B
    cost = pd.Series({c: (SPREAD_BP.get(c, 5) / 2 + SLIP_BP) / 1e4 * 2
                      for c in cols})
    return np.sign(pred).where(pred.abs() > cost, 0.0)


@model("stat_ll", freq="H")
def stat_ll(X):
    """Statistical lead-lag: each year, lagged hourly cross-correlations
    (prior 2 years, |t| > 4, own lag excluded) form a next-hour forecast;
    trade it when it exceeds the round-trip cost."""
    return X.cached("stat_ll_H", lambda: _stat_ll(X, "H"))


@model("lead_lag_arb", freq="M")
def lead_lag_arb(X):
    """Lead-lag arbitrage at 15 minutes: the same estimator on 15-minute
    bars."""
    return X.cached("stat_ll_M", lambda: _stat_ll(X, "M"))


# =================================================== CFD execution filters

def _h1_mr_entries(X):
    z = L.zscore(X.H["C"], 20)
    return z


def _h1_mr(X, entry_ok=None):
    """Base hourly mean reversion: fade |z20| > 2, exit at 0, flat over the
    17:00 roll."""
    z = _h1_mr_entries(X)
    ok = True if entry_ok is None else entry_ok
    le, se = (z < -2) & ok, (z > 2) & ok
    T = nyt(X, "H")
    roll = pd.Series(window_mask(T, 1600, 1800), index=z.index)
    pos = L.machine(le & ~roll.to_numpy()[:, None],
                    se & ~roll.to_numpy()[:, None],
                    (z >= 0) | roll.to_numpy()[:, None],
                    (z <= 0) | roll.to_numpy()[:, None], z)
    return pos


def _spread_frac(cols):
    return pd.Series({c: SPREAD_BP.get(c, 5.0) / 1e4 for c in cols})


@model("h1_mr_base", freq="H")
def h1_mr_base(X):
    """Reference for the CFD filters: hourly Bollinger(20,2) fade to the
    mean, flat over the 17:00 roll."""
    return _h1_mr(X)


@model("spread_norm_entry", freq="H")
def spread_norm_entry(X):
    """Spread-normalised entry: the hourly fade only when hourly ATR is more
    than 15 spreads."""
    C = X.H["C"]
    a = L.atr(X.H, 14) / C
    return _h1_mr(X, a.gt(15 * _spread_frac(C.columns), axis=1))


@model("cost_adj_entry", freq="H")
def cost_adj_entry(X):
    """Cost-adjusted entry: the hourly fade only when the distance back to the
    mean is more than 4x the round-trip cost."""
    C = X.H["C"]
    dist = (C / L.sma(C, 20) - 1).abs()
    cost = _spread_frac(C.columns) + 2 * SLIP_BP / 1e4
    return _h1_mr(X, dist.gt(4 * cost, axis=1))


@model("spread_exp_filter", freq="H")
def spread_exp_filter(X):
    """Spread-expansion filter (no quote data, so the known wide-spread
    hours): no entries 16:00-20:00 New York or in the first two hours of the
    week."""
    T = nyt(X, "H")
    bad = window_mask(T, 1600, 2000) | ((T["dow"] == 6).to_numpy())
    return _h1_mr(X, pd.Series(~bad, index=T.index).to_numpy()[:, None])


@model("slippage_filter", freq="H")
def slippage_filter(X):
    """Execution-slippage filter: no entry on a bar whose range is more than
    3 hourly ATRs (fast market)."""
    rng = X.H["H"] - X.H["L"]
    return _h1_mr(X, rng < 3 * L.atr(X.H, 14).shift(1))


@model("liquidity_adj", freq="H")
def liquidity_adj(X):
    """Liquidity-adjusted entry: only when tick volume is at least the median
    for that hour over the last 20 days."""
    T = nyt(X, "H")
    V = X.H["V"]
    med = V.groupby(T["hour"].to_numpy()).transform(
        lambda s: s.rolling(20, min_periods=10).median().shift(1))
    return _h1_mr(X, V >= med)


@model("vol_spread_ratio", freq="H")
def vol_spread_ratio(X):
    """Volatility/spread ratio: each day trade the hourly fade only in the
    half of CFDs with the highest daily ATR per unit of spread."""
    ratio = (L.atr(X.D, 14) / X.C).div(_spread_frac(X.C.columns), axis=1)
    top = (ratio.rank(axis=1, pct=True) > 0.5).shift(1)   # yesterday's
    T = nyt(X, "H")
    tb = top.reindex(T["day"].to_numpy())
    tb.index = T.index
    return _h1_mr(X, tb.fillna(False).astype(bool))


@model("expected_move_model", freq="H")
def expected_move_model(X):
    """CFD expected-move model: hourly fades only while the day's remaining
    expected range (daily ATR minus range so far) exceeds 5x round-trip
    cost."""
    T = nyt(X, "H")
    d = T["day"].to_numpy()
    hi = X.H["H"].groupby(d).cummax()
    lo = X.H["L"].groupby(d).cummin()
    datr = (L.atr(X.D, 14).shift(1)).reindex(d)
    datr.index = T.index
    remain = (datr - (hi - lo)) / X.H["C"]
    cost = _spread_frac(X.H["C"].columns) + 2 * SLIP_BP / 1e4
    return _h1_mr(X, remain.gt(5 * cost, axis=1))


@model("session_adj_stat", freq="H")
def session_adj_stat(X):
    """Session-adjusted statistical model: hourly returns standardised by
    their hour-of-day volatility (expanding, prior data); fade |z| > 2.5 for
    3 hours."""
    T = nyt(X, "H")
    r = lr(X, "H")
    sd = (r ** 2).groupby(T["hour"].to_numpy()).transform(
        lambda s: s.expanding(min_periods=100).mean().shift(1)) ** 0.5
    z = r / sd
    ev = (z < -2.5).astype(float) - (z > 2.5).astype(float)
    return L.hold_for(ev, 3)


@model("financing_aware")
def financing_aware(X):
    """Overnight-financing-aware entry: the multi-speed trend position is held
    only when the trailing 12-month move is worth more than twice the yearly
    financing markup."""
    from examples.quant.models_daily import trend_follow
    from bcbt.quant import FIN_YR, GROUP_OF
    s = trend_follow(X)
    fin = pd.Series({c: FIN_YR["fx" if GROUP_OF[c] == "fx" else "other"]
                     for c in X.C.columns})
    big = L.ret(X.C, 252).abs().gt(2 * fin, axis=1)
    return s.where(big, 0.0)


# =================================================== tick-volume flow

def _signed_vol(P):
    return np.sign(P["C"] - P["O"]) * P["V"]


@model("flow_model")
def flow_model(X):
    """Flow-based model: 5-day signed tick volume (sign of each day's candle
    x volume) relative to total volume; follow when |imbalance| > 0.3."""
    sv = _signed_vol(X.D).rolling(5).sum() / X.D["V"].rolling(5).sum()
    return (sv > 0.3).astype(float) - (sv < -0.3).astype(float)


@model("vw_flow")
def vw_flow(X):
    """Volume-weighted flow: sign of the tick-volume-weighted mean daily
    return over 20 days."""
    vw = (X.R * X.D["V"]).rolling(20).sum() / X.D["V"].rolling(20).sum()
    return L.sign(vw)


@model("cvd", freq="H")
def cvd(X):
    """Cumulative volume delta (signed hourly tick volume): follow the
    4-hour CVD change, standardised, beyond 1 sd."""
    sv = _signed_vol(X.H).rolling(4).sum()
    z = sv / sv.rolling(500, min_periods=200).std().shift(1)
    return (z > 1).astype(float) - (z < -1).astype(float)


@model("tick_volume")
def tick_volume(X):
    """Tick-volume model: a day with volume > 1.5x its 20-day average is
    followed the next day in its candle's direction."""
    hi = X.D["V"] > 1.5 * X.D["V"].rolling(20).mean().shift(1)
    return L.sign(X.C - X.D["O"]).where(hi, 0.0)


@model("tick_vol_anomaly", freq="H")
def tick_vol_anomaly(X):
    """Tick-volume anomaly: hourly volume z (500h) > 3, follow the bar's
    direction for 4 hours."""
    z = L.zscore(X.H["V"], 500)
    ev = L.sign(X.H["C"] - X.H["O"]).where(z > 3, 0.0)
    return L.hold_for(ev, 4)


@model("rel_vol_anomaly", freq="H")
def rel_vol_anomaly(X):
    """Relative-volume anomaly: volume > 2.5x the same hour's 20-day average,
    follow the bar for 4 hours."""
    T = nyt(X, "H")
    V = X.H["V"]
    avg = V.groupby(T["hour"].to_numpy()).transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    ev = L.sign(X.H["C"] - X.H["O"]).where(V > 2.5 * avg, 0.0)
    return L.hold_for(ev, 4)


@model("volume_z")
def volume_z(X):
    """Volume z-score: daily volume z (60d) > 2 -> fade the day's candle
    next day (climax)."""
    z = L.zscore(X.D["V"], 60)
    return (-L.sign(X.C - X.D["O"])).where(z > 2, 0.0)


@model("vol_price_div")
def vol_price_div(X):
    """Volume/price divergence: a new 20-day high on below-average volume ->
    short 5 days; a new low on below-average volume -> long 5 days."""
    lowv = X.D["V"] < X.D["V"].rolling(20).mean().shift(1)
    hi = X.C >= X.C.rolling(20).max()
    lo = X.C <= X.C.rolling(20).min()
    return L.hold_for((lo & lowv).astype(float) - (hi & lowv).astype(float), 5)


@model("price_vol_reg")
def price_vol_reg(X):
    """Price/volume regression: rolling 250-day regression of tomorrow's
    return on today's return x relative volume; trade the forecast sign."""
    rv = X.D["V"] / X.D["V"].rolling(20).mean().shift(1)
    x = X.R * rv
    y = X.R.shift(-1)
    beta = (x.rolling(250).cov(y.shift(1)) / x.shift(1).rolling(250).var())
    # beta from (x_{t-1}, r_t) pairs only -> known at t
    return L.sign(beta * x)


@model("vol_volatility")
def vol_volatility(X):
    """Volume-volatility relationship: high volume (z > 1) on a narrow range
    (range z < -0.5) = absorption; trade toward the close location for 3
    days."""
    vz = L.zscore(X.D["V"], 60)
    rz = L.zscore(X.D["H"] - X.D["L"], 60)
    loc = (X.C - X.D["L"]) / (X.D["H"] - X.D["L"]).replace(0, np.nan)
    ev = L.sign(loc - 0.5).where((vz > 1) & (rz < -0.5), 0.0)
    return L.hold_for(ev, 3)


@model("ofi_proxy", freq="M")
def ofi_proxy(X):
    """Order-flow imbalance proxy: signed tick volume over the last 4
    15-minute bars / total; follow the next bar when |imbalance| > 0.5."""
    sv = _signed_vol(X.M).rolling(4).sum() / X.M["V"].rolling(4).sum()
    return (sv > 0.5).astype(float) - (sv < -0.5).astype(float)


@model("micro_signal", freq="M")
def micro_signal(X):
    """Microstructure proxy: close-location value x relative volume of the
    15-minute bar; follow the next bar beyond +-1."""
    rng = (X.M["H"] - X.M["L"]).replace(0, np.nan)
    clv = ((X.M["C"] - X.M["L"]) - (X.M["H"] - X.M["C"])) / rng
    rv = X.M["V"] / X.M["V"].rolling(96, min_periods=48).mean().shift(1)
    s = clv * rv
    return (s > 1).astype(float) - (s < -1).astype(float)


@model("tick_imbalance", freq="M")
def tick_imbalance(X):
    """Tick imbalance: of the last 8 15-minute bars, 6+ up (down) -> follow
    the next bar."""
    u = np.sign(X.M["C"] - X.M["O"]).rolling(8).sum()
    return (u >= 4).astype(float) - (u <= -4).astype(float)


@model("trade_intensity", freq="M")
def trade_intensity(X):
    """Trade intensity: 15-minute tick volume z (960 bars) > 3, follow the
    bar for 4 bars."""
    z = L.zscore(X.M["V"], 960)
    return L.hold_for(L.sign(X.M["C"] - X.M["O"]).where(z > 3, 0.0), 4)


@model("arrival_rate", freq="H")
def arrival_rate(X):
    """Arrival-rate model: hourly volume more than double the previous hour
    and above its 500-hour mean -> follow the bar for 2 hours."""
    V = X.H["V"]
    acc = (V > 2 * V.shift(1)) & (V > V.rolling(500, min_periods=200).mean())
    return L.hold_for(L.sign(X.H["C"] - X.H["O"]).where(acc, 0.0), 2)


@model("hawkes", freq="H")
def hawkes(X):
    """Hawkes self-excitation: intensity of 2.5-sd hourly moves with an
    exponential kernel (half-life 6h); while intensity > 2 events, follow
    the direction of the latest event."""
    r = lr(X, "H")
    z = r / r.rolling(500, min_periods=200).std().shift(1)
    ev = (z.abs() > 2.5).astype(float)
    lam = ev.ewm(halflife=6, adjust=False).mean() * 8.66   # ~ sum of kernel
    last = np.sign(z).where(ev > 0).ffill(limit=24)
    return last.where(lam > 2, 0.0).fillna(0)


# =================================================================== jumps

def _jumps(X):
    """Lee-Mykland: hourly return / bipower-variation vol of the prior 120
    hours; |stat| > 4.6 is a jump."""
    def f():
        r = lr(X, "H")
        bv = (r.abs() * r.abs().shift(1)).rolling(120, min_periods=60).mean(
        ).shift(1) * (math.pi / 2)
        return r / np.sqrt(bv)
    return X.cached("lm_stat", f)


@model("jump_rev", freq="H")
def jump_rev(X):
    """Price-jump reversion: fade a Lee-Mykland jump for 4 hours."""
    s = _jumps(X)
    return L.hold_for((s < -4.6).astype(float) - (s > 4.6).astype(float), 4)


@model("jump_cont", freq="H")
def jump_cont(X):
    """Price-jump continuation: follow the jump for 4 hours."""
    return -jump_rev(X)


@model("jump_diffusion", freq="H")
def jump_diffusion(X):
    """Intraday jump-diffusion: follow the sign of the last 24 hours' drift
    with jump returns removed."""
    r = lr(X, "H")
    s = _jumps(X)
    diff = r.where(s.abs() <= 4.6, 0.0).rolling(24).sum()
    z = diff / (r.rolling(500, min_periods=200).std() * math.sqrt(24))
    return (z > 0.5).astype(float) - (z < -0.5).astype(float)


@model("jump_detection")
def jump_detection(X):
    """Jump-filtered momentum: 20-day trend of daily returns rebuilt from
    hourly returns with the jumps removed."""
    def f():
        r = lr(X, "H")
        s = _jumps(X)
        clean = r.where(s.abs() <= 4.6, 0.0)
        return clean.groupby(to_ny_day(clean.index)).sum().reindex(X.C.index)
    d = X.cached("clean_daily", f)
    return L.sign(d.rolling(20).sum())


# =========================================== information-driven bars (H1)

def _event_bars(values, thresh):
    """Indices where the running sum of `values` since the last bar exceeds
    `thresh` (per column). Returns +1/-1 at bar ends (sign of the sum)."""
    v = values.to_numpy()
    th = thresh.to_numpy()
    out = np.zeros_like(v)
    for j in range(v.shape[1]):
        acc = 0.0
        for i in range(v.shape[0]):
            x = v[i, j]
            if not np.isfinite(x) or not np.isfinite(th[i, j]):
                continue
            acc += x
            if abs(acc) >= th[i, j]:
                out[i, j] = np.sign(acc)
                acc = 0.0
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def _bar_momentum(events_val, n=5):
    """Momentum on information bars: sign of the sum of the last n bars'
    returns, held until the next bar forms."""
    ev = events_val.replace(0, np.nan)
    out = ev.copy()
    for c in ev.columns:
        s = ev[c].dropna()
        m = np.sign(s.rolling(n).sum())
        out[c] = m.reindex(ev.index)
    return out.ffill().fillna(0)


def _bar_returns(X, ends):
    C = X.H["C"]
    last = C.where(ends != 0).ffill().shift(1)
    return np.log(C / last).where(ends != 0, 0.0)


@model("dollar_bars", freq="H")
def dollar_bars(X):
    """Dollar bars (price x tick volume, about 6 a day): momentum of the
    last 5 bars."""
    C = X.H["C"]
    dv = C * X.H["V"]
    th = dv.rolling(24 * 20, min_periods=200).sum().shift(1) / 20 / 6
    ends = _event_bars(dv, th).abs()
    return _bar_momentum(_bar_returns(X, ends))


@model("vol_bars", freq="H")
def vol_bars(X):
    """Volatility bars (a new bar after 1 daily-sd of cumulative absolute
    hourly return): momentum of the last 5 bars."""
    r = lr(X, "H")
    th = r.abs().rolling(24 * 20, min_periods=200).sum().shift(1) / 20
    ends = _event_bars(r.abs(), th).abs()
    return _bar_momentum(_bar_returns(X, ends))


@model("info_bars", freq="H")
def info_bars(X):
    """Tick-imbalance information bars: a bar closes when cumulative signed
    tick volume exceeds 3 hours of average volume; follow that imbalance's
    sign until the next bar."""
    sv = _signed_vol(X.H)
    th = 3 * X.H["V"].rolling(500, min_periods=200).mean().shift(1)
    ev = _event_bars(sv, th)
    return ev.replace(0, np.nan).ffill().fillna(0)


# ===================================================== multi-timeframe

def _daily_to_h(X, frame):
    """Daily frame known at day d's close -> hourly bars of day d+1."""
    T = nyt(X, "H")
    out = frame.shift(1).reindex(T["day"].to_numpy())
    out.index = T.index
    return out


@model("mtf_align")
def mtf_align(X):
    """Multi-timeframe alignment: weekly trend (price vs 26-week EMA) and
    daily trend (EMA20 vs EMA50) agree -> trade it."""
    w = L.sign(X.C - L.ema(X.C, 130))
    d = L.sign(L.ema(X.C, 20) - L.ema(X.C, 50))
    return w.where(w == d, 0.0)


@model("mtf_prob", freq="H")
def mtf_prob(X):
    """Multi-timeframe probability: expanding P(next 4 hours up | daily trend
    sign, hourly trend sign); trade when it leaves 0.48-0.52."""
    dtr = _daily_to_h(X, L.sign(L.ema(X.C, 20) - L.ema(X.C, 50)))
    htr = L.sign(L.ema(X.H["C"], 24) - L.ema(X.H["C"], 96))
    state = (dtr + 1) * 3 + (htr + 1)
    fwd = np.log(X.H["C"]).shift(-4) - np.log(X.H["C"])
    out = zeros(X, "H")
    years = state.index.year
    for y in sorted(set(years))[2:]:
        tr = years < y
        te = years == y
        for c in state.columns:
            st, f = state.loc[tr, c], fwd.loc[tr, c].copy()
            f.iloc[-4:] = np.nan              # outcomes not known by year end
            ok = f.notna()
            g = (f[ok] > 0).groupby(st[ok]).mean()
            p = state.loc[te, c].map(g)
            out.loc[te, c] = np.where(p > 0.52, 1.0, np.where(p < 0.48, -1.0,
                                                              0.0))
    return out


@model("mtf_regime", freq="H")
def mtf_regime(X):
    """Multi-timeframe regime: daily efficiency ratio(20) > 0.4 -> hourly
    trend (EMA24/96); < 0.2 -> hourly z20 fade; otherwise flat."""
    er = _daily_to_h(X, L.efficiency_ratio(X.C, 20))
    tr = L.sign(L.ema(X.H["C"], 24) - L.ema(X.H["C"], 96))
    mr = L.fade_band(L.zscore(X.H["C"], 20), 2.0, 0.0)
    return tr.where(er > 0.4, mr.where(er < 0.2, 0.0)).fillna(0)


@model("htf_trend_ltf", freq="H")
def htf_trend_ltf(X):
    """HTF trend + LTF trigger: daily close above (below) EMA50, hourly
    RSI(2) < 10 (> 90) -> long (short) until hourly RSI(2) > 70 (< 30)."""
    dt = _daily_to_h(X, L.sign(X.C - L.ema(X.C, 50)))
    r2 = L.rsi(X.H["C"], 2)
    return L.machine((dt > 0) & (r2 < 10), (dt < 0) & (r2 > 90), r2 > 70,
                     r2 < 30, r2)


@model("htf_mr_ltf", freq="H")
def htf_mr_ltf(X):
    """HTF mean reversion + LTF trigger: daily z20 < -2 (> 2) and an hourly
    close above the previous hour's high (below its low) -> long (short)
    until the daily z crosses 0."""
    dz = _daily_to_h(X, L.zscore(X.C, 20))
    C, H, Lo = X.H["C"], X.H["H"], X.H["L"]
    le = (dz < -2) & (C > H.shift(1))
    se = (dz > 2) & (C < Lo.shift(1))
    return L.machine(le, se, dz >= 0, dz <= 0, C)


@model("mtf_ensemble", freq="H")
def mtf_ensemble(X):
    """Ensemble multi-timeframe: mean of hourly (EMA24/96), daily (20-day
    return) and weekly (12-week return) trend signs."""
    h = L.sign(L.ema(X.H["C"], 24) - L.ema(X.H["C"], 96))
    d = _daily_to_h(X, L.sign(L.ret(X.C, 20)))
    w = _daily_to_h(X, L.sign(L.ret(X.C, 60)))
    return ((h + d + w) / 3).fillna(0)


# ======================================= breakouts and levels, intraday

@model("expected_move_bo", freq="H")
def expected_move_bo(X):
    """Expected-move breakout: once price leaves the day's open by 0.7 x
    daily ATR, follow until the 17:00 roll."""
    T = nyt(X, "H")
    d = T["day"].to_numpy()
    op = X.H["O"].groupby(d).transform("first")
    a = (L.atr(X.D, 14).shift(1)).reindex(d)
    a.index = T.index
    C = X.H["C"]
    ev = ((C > op + 0.7 * a).astype(float) - (C < op - 0.7 * a).astype(float))
    first = ev.replace(0, np.nan).groupby(d).ffill().fillna(0)
    roll = window_mask(T, 1600, 1700)
    return first.where(_rows(~roll), 0.0)


@model("implied_move_breach", freq="H")
def implied_move_breach(X):
    """Implied-move breach (S&P): VIX/sqrt(252) is the implied daily move; a
    break of the 09:30 open by that much is followed to 16:00."""
    T = nyt(X, "H")
    d = T["date"].to_numpy()
    C = X.H["C"]["SPX500_USD"]
    cash = window_mask(T, 900, 1600)
    op = X.H["O"]["SPX500_USD"].where(T["hour"].to_numpy() == 9)
    op = op.groupby(d).transform("first")
    vix = X.vix.shift(1).reindex(T["day"].to_numpy())
    vix.index = T.index
    mv = op * vix / 100 / math.sqrt(252)
    ev = ((C > op + mv).astype(float) - (C < op - mv).astype(float)).where(
        cash, 0.0)
    first = ev.replace(0, np.nan).groupby(d).ffill().fillna(0)
    P = zeros(X, "H")
    P["SPX500_USD"] = first.where(window_mask(T, 900, 1500), 0.0)
    return P


@model("false_breakout_stat", freq="H")
def false_breakout_stat(X):
    """Statistical false breakout: an hour trades above the prior day's high
    and closes back below it -> short until the roll (mirror at the low)."""
    T = nyt(X, "H")
    d = T["day"].to_numpy()
    pdh = X.D["H"].shift(1).reindex(d)
    pdl = X.D["L"].shift(1).reindex(d)
    pdh.index = pdl.index = T.index
    H, Lo, C = X.H["H"], X.H["L"], X.H["C"]
    ev = ((Lo < pdl) & (C > pdl)).astype(float) - ((H > pdh) & (C < pdh)
                                                   ).astype(float)
    first = ev.replace(0, np.nan).groupby(d).ffill().fillna(0)
    roll = window_mask(T, 1600, 1700)
    return first.where(_rows(~roll), 0.0)
