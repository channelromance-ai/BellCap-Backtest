"""
Gold entry-model battery: the user's list of retail entry models, one
textbook version each, judged the same way.

The list has ~500 entries. Testing them all on one history would hand back
~25 "winners" by luck alone, so the battery is built to resist that:

  1. Duplicates merged; models that cannot be traded on a gold CFD (options,
     crypto funding/OI, earnings, pre-market, exchange volume) or cannot be
     defined without fitting (Elliott, harmonics, Wyckoff phases, loose chart
     patterns) are left out.
  2. One standard version of each, textbook parameters, no tuning.
  3. One risk system for all, so only the ENTRY differs: signal on an M15
     close, entry at the next M5 open, stop 1.0 x ATR14(M15), target 1.5R,
     one position at a time, 3 a day, out by the 17:00 New York roll, stop
     first on ties, each bar's own spread plus $8.22 a lot commission.
  4. Develop on Aug 2019 - Dec 2023; one look at Jan 2024 - Sep 2026 for
     every finalist (dev PF >= 1.2 on >= 100 trades). The same battery with
     random directions, five times, measures how many finalists luck makes.

Volume here is the broker's TICK volume (all a CFD has); VWAP is built from
it. Server time is New York + 7; the server day starts at the 17:00 NY roll.
Sessions by server hour: Asia 02-08, London 09-14, New York 15-23.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_zones as G                        # noqa: E402

DEV_END = np.datetime64("2024-01-01")
RR = 1.5
STOP_ATR = 1.0


# ------------------------------------------------------------- indicators

def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def indicators(m15, h4):
    d = {}
    o, h, l, c = (m15[x].to_numpy(np.float64) for x in ("open", "high",
                                                          "low", "close"))
    v = m15["tick_volume"].to_numpy(np.float64)
    t = m15["server"]
    d.update(o=o, h=h, l=l, c=c)
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    d["atr"] = pd.Series(tr).rolling(14).mean().to_numpy()
    for n in (9, 20, 21, 50, 200):
        d[f"ema{n}"] = ema(c, n)
    delta = np.diff(c, prepend=c[0])
    up, dn = wilder(np.clip(delta, 0, None), 14), wilder(np.clip(-delta, 0,
                                                                 None), 14)
    d["rsi"] = 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))
    macd = ema(c, 12) - ema(c, 26)
    d["macd"], d["macd_sig"] = macd, ema(macd, 9)
    s = pd.Series(c)
    mid, sd = s.rolling(20).mean().to_numpy(), s.rolling(20).std().to_numpy()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    d["bb_w"] = (d["bb_up"] - d["bb_lo"]) / mid
    d["bb_w_q10"] = pd.Series(d["bb_w"]).rolling(100).quantile(0.1).to_numpy()
    # ADX / DI
    upm = np.r_[0, np.diff(h)]
    dnm = np.r_[0, -np.diff(l)]
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atrw = wilder(tr, 14)
    d["pdi"], d["ndi"] = 100 * wilder(pdm, 14) / atrw, 100 * wilder(ndm,
                                                                    14) / atrw
    dx = 100 * abs(d["pdi"] - d["ndi"]) / (d["pdi"] + d["ndi"] + 1e-12)
    d["adx"] = wilder(dx, 14)
    # Donchian of the PREVIOUS n bars
    for n in (20, 55):
        d[f"dh{n}"] = pd.Series(h).rolling(n).max().shift(1).to_numpy()
        d[f"dl{n}"] = pd.Series(l).rolling(n).min().shift(1).to_numpy()
    # Server day / week levels from COMPLETED periods
    day = t.dt.normalize()
    daily = m15.groupby(day).agg(H=("high", "max"), L=("low", "min"),
                                 C=("close", "last"))
    prevd = daily.shift(1)
    for k in ("H", "L", "C"):
        d[f"pd{k}"] = day.map(prevd[k]).to_numpy()
    wk = t.dt.to_period("W-SAT")
    weekly = m15.groupby(wk).agg(H=("high", "max"), L=("low", "min"))
    prevw = weekly.shift(1)
    d["pwH"], d["pwL"] = wk.map(prevw["H"]).to_numpy(), \
        wk.map(prevw["L"]).to_numpy()
    P = (d["pdH"] + d["pdL"] + d["pdC"]) / 3
    d["R1"], d["S1"] = 2 * P - d["pdL"], 2 * P - d["pdH"]
    rng = d["pdH"] - d["pdL"]
    d["H3"], d["L3"] = d["pdC"] + 1.1 * rng / 4, d["pdC"] - 1.1 * rng / 4
    d["H4"], d["L4"] = d["pdC"] + 1.1 * rng / 2, d["pdC"] - 1.1 * rng / 2
    dch = daily["C"]
    d1_trend = np.sign(dch - dch.ewm(span=20, adjust=False).mean()).shift(1)
    d["d1_trend"] = day.map(d1_trend).to_numpy()
    # Session VWAP (server day) from tick volume
    tp = (h + l + c) / 3
    g = pd.DataFrame(dict(day=day, pv=tp * v, v=v, pv2=tp * tp * v))
    cs = g.groupby("day")[["pv", "v", "pv2"]].cumsum()
    vw = cs["pv"] / cs["v"]
    d["vwap"] = vw.to_numpy()
    d["vwap_sd"] = np.sqrt(np.clip(cs["pv2"] / cs["v"] - vw * vw, 0,
                                   None)).to_numpy()
    # Sessions
    hour = t.dt.hour.to_numpy()
    minute = t.dt.minute.to_numpy()
    d["hour"], d["minute"], d["day"] = hour, minute, day.to_numpy()

    def sess_range(h0, h1):
        m = (hour >= h0) & (hour < h1)
        hi = pd.Series(np.where(m, h, np.nan)).groupby(day.to_numpy()) \
            .transform("max").to_numpy()
        lo = pd.Series(np.where(m, l, np.nan)).groupby(day.to_numpy()) \
            .transform("min").to_numpy()
        return hi, lo
    d["asia_hi"], d["asia_lo"] = sess_range(2, 9)
    d["lon1_hi"], d["lon1_lo"] = sess_range(9, 10)
    nyor = (hour == 15) & (minute >= 30)
    d["nyor_hi"] = pd.Series(np.where(nyor, h, np.nan)).groupby(
        day.to_numpy()).transform("max").to_numpy()
    d["nyor_lo"] = pd.Series(np.where(nyor, l, np.nan)).groupby(
        day.to_numpy()).transform("min").to_numpy()
    # Confirmed fractal swings, N = 3 (known 3 bars after the swing bar)
    N = 3
    hh = pd.Series(h).rolling(2 * N + 1, center=True).max().to_numpy()
    ll = pd.Series(l).rolling(2 * N + 1, center=True).min().to_numpy()
    is_sh, is_sl = h == hh, l == ll
    sh_val = pd.Series(np.where(is_sh, h, np.nan)).shift(N)
    sl_val = pd.Series(np.where(is_sl, l, np.nan)).shift(N)
    rsi_sh = pd.Series(np.where(is_sh, d["rsi"], np.nan)).shift(N)
    rsi_sl = pd.Series(np.where(is_sl, d["rsi"], np.nan)).shift(N)
    d["new_sh"], d["new_sl"] = sh_val.notna().to_numpy(), \
        sl_val.notna().to_numpy()
    d["sh1"] = sh_val.ffill().to_numpy()                     # last swing high
    d["sl1"] = sl_val.ffill().to_numpy()
    shs = sh_val.dropna()
    sls = sl_val.dropna()
    d["sh2"] = shs.shift(1).reindex(range(len(c))).ffill().to_numpy()
    d["sl2"] = sls.shift(1).reindex(range(len(c))).ffill().to_numpy()
    d["rsi_sh1"] = rsi_sh.dropna().reindex(range(len(c))).ffill().to_numpy()
    d["rsi_sh2"] = rsi_sh.dropna().shift(1).reindex(range(len(c))).ffill() \
        .to_numpy()
    d["rsi_sl1"] = rsi_sl.dropna().reindex(range(len(c))).ffill().to_numpy()
    d["rsi_sl2"] = rsi_sl.dropna().shift(1).reindex(range(len(c))).ffill() \
        .to_numpy()
    close_t = (t + pd.Timedelta(minutes=15)).to_numpy()
    d["h4"] = G.h4_bias(h4, close_t)
    d["close_t"] = close_t
    return d


def prev(x, k=1):
    return np.r_[np.full(k, np.nan), x[:-k]]


def cross_up(a, b):
    return (a > b) & (prev(a) <= prev(b))


def cross_dn(a, b):
    return (a < b) & (prev(a) >= prev(b))


def first_per_day(mask, day):
    """Keep only the first True of each server day."""
    s = pd.Series(np.nan_to_num(mask, nan=0).astype(bool))
    cs = s.groupby(np.asarray(day)).cumsum()
    return (s & (cs == 1)).to_numpy()


# ------------------------------------------------------------------ models

def models(d):
    o, h, l, c, a = d["o"], d["h"], d["l"], d["c"], d["atr"]
    hr, day = d["hour"], d["day"]
    lon = (hr >= 9) & (hr < 15)
    body = abs(c - o)
    rngb = h - l
    up_tr = (d["ema50"] > d["ema200"]) & (c > d["ema50"])
    dn_tr = (d["ema50"] < d["ema200"]) & (c < d["ema50"])
    M = {}

    def add(name, family, long, short):
        M[name] = (family, np.nan_to_num(long, nan=0).astype(bool),
                   np.nan_to_num(short, nan=0).astype(bool))

    # ---- breakouts
    add("Donchian 20 breakout", "Breakout", cross_up(c, d["dh20"]) & (prev(c) <= prev(d["dh20"])), cross_dn(c, d["dl20"]))
    add("Donchian 55 breakout", "Breakout", cross_up(c, d["dh55"]), cross_dn(c, d["dl55"]))
    add("Previous day high/low breakout", "Breakout", cross_up(c, d["pdH"]), cross_dn(c, d["pdL"]))
    add("Previous week high/low breakout", "Breakout", cross_up(c, d["pwH"]), cross_dn(c, d["pwL"]))
    add("Asian range breakout (London)", "Session", first_per_day(lon & cross_up(c, d["asia_hi"]), day),
        first_per_day(lon & cross_dn(c, d["asia_lo"]), day))
    lon_after = (hr >= 10) & (hr < 15)
    add("London opening-hour breakout", "Session", first_per_day(lon_after & cross_up(c, d["lon1_hi"]), day),
        first_per_day(lon_after & cross_dn(c, d["lon1_lo"]), day))
    ny_after = (hr >= 16) & (hr < 20)
    add("NY opening range breakout (08:30-09:00 NY)", "Session",
        first_per_day(ny_after & cross_up(c, d["nyor_hi"]), day),
        first_per_day(ny_after & cross_dn(c, d["nyor_lo"]), day))
    add("Bollinger band breakout", "Bollinger", cross_up(c, d["bb_up"]), cross_dn(c, d["bb_lo"]))
    add("ATR breakout (close-to-close > 1 ATR)", "Breakout", (c - prev(c)) > prev(a), (prev(c) - c) > prev(a))
    ib = (prev(h) < prev(h, 2)) & (prev(l) > prev(l, 2))
    add("Inside bar breakout", "Candlestick", ib & (c > prev(h)), ib & (c < prev(l)))
    sq = prev(d["bb_w"]) <= prev(d["bb_w_q10"])
    add("Bollinger squeeze breakout", "Bollinger", sq & (c > d["bb_up"]), sq & (c < d["bb_lo"]))
    maru = (body >= 0.8 * rngb) & (rngb >= a)
    add("Marubozu continuation", "Candlestick", maru & (c > o), maru & (c < o))
    add("Daily trend + Donchian 20 breakout", "Multi-timeframe",
        (d["d1_trend"] > 0) & cross_up(c, d["dh20"]), (d["d1_trend"] < 0) & cross_dn(c, d["dl20"]))
    add("H4 trend + Donchian 20 breakout", "Multi-timeframe",
        (d["h4"] > 0) & cross_up(c, d["dh20"]), (d["h4"] < 0) & cross_dn(c, d["dl20"]))

    # ---- break-and-retest / failed breaks / sweeps
    above_pdh = pd.Series(c > d["pdH"]).groupby(day).cummax().to_numpy()
    below_pdl = pd.Series(c < d["pdL"]).groupby(day).cummax().to_numpy()
    add("Previous day high/low break-and-retest", "Break-and-retest",
        np.r_[False, above_pdh[:-1]] & (l <= d["pdH"]) & (c > d["pdH"]),
        np.r_[False, below_pdl[:-1]] & (h >= d["pdL"]) & (c < d["pdL"]))
    brk_up = pd.Series(cross_up(c, d["dh20"]).astype(float)).rolling(12).max().shift(1).to_numpy() > 0
    brk_dn = pd.Series(cross_dn(c, d["dl20"]).astype(float)).rolling(12).max().shift(1).to_numpy() > 0
    lvl_up = pd.Series(np.where(cross_up(c, d["dh20"]), d["dh20"], np.nan)).ffill().shift(1).to_numpy()
    lvl_dn = pd.Series(np.where(cross_dn(c, d["dl20"]), d["dl20"], np.nan)).ffill().shift(1).to_numpy()
    add("Donchian break-and-retest", "Break-and-retest", brk_up & (l <= lvl_up) & (c > lvl_up),
        brk_dn & (h >= lvl_dn) & (c < lvl_dn))
    add("Previous day high/low sweep-and-reclaim (fade)", "Liquidity",
        (l < d["pdL"]) & (c > d["pdL"]), (h > d["pdH"]) & (c < d["pdH"]))
    add("Asian range sweep-and-reclaim in London (Judas)", "Liquidity",
        lon & (l < d["asia_lo"]) & (c > d["asia_lo"]), lon & (h > d["asia_hi"]) & (c < d["asia_hi"]))
    eqh = abs(d["sh1"] - d["sh2"]) <= 0.1 * a
    eql = abs(d["sl1"] - d["sl2"]) <= 0.1 * a
    add("Equal highs/lows sweep-and-reclaim", "Liquidity",
        eql & (l < np.fmin(d["sl1"], d["sl2"])) & (c > np.fmin(d["sl1"], d["sl2"])),
        eqh & (h > np.fmax(d["sh1"], d["sh2"])) & (c < np.fmax(d["sh1"], d["sh2"])))

    # ---- pullbacks / moving averages
    add("EMA20 pullback in trend", "Pullback", up_tr & (l <= d["ema20"]) & (c > d["ema20"]),
        dn_tr & (h >= d["ema20"]) & (c < d["ema20"]))
    up50 = (d["ema50"] > d["ema200"])
    add("EMA50 pullback in trend", "Pullback", up50 & (l <= d["ema50"]) & (c > d["ema50"]),
        ~up50 & (h >= d["ema50"]) & (c < d["ema50"]))
    add("EMA200 bounce", "Moving average", (l <= d["ema200"]) & (c > d["ema200"]) & (prev(c) > prev(d["ema200"])),
        (h >= d["ema200"]) & (c < d["ema200"]) & (prev(c) < prev(d["ema200"])))
    add("H4 trend + M15 EMA20 pullback", "Multi-timeframe", (d["h4"] > 0) & (l <= d["ema20"]) & (c > d["ema20"]),
        (d["h4"] < 0) & (h >= d["ema20"]) & (c < d["ema20"]))
    add("EMA 9/21 cross", "Moving average", cross_up(d["ema9"], d["ema21"]), cross_dn(d["ema9"], d["ema21"]))
    add("EMA 20/50 cross", "Moving average", cross_up(d["ema20"], d["ema50"]), cross_dn(d["ema20"], d["ema50"]))
    add("Golden/death cross 50/200", "Moving average", cross_up(d["ema50"], d["ema200"]), cross_dn(d["ema50"], d["ema200"]))
    add("EMA50 reclaim/rejection", "Moving average", cross_up(c, d["ema50"]), cross_dn(c, d["ema50"]))
    add("Overextension fade (2.5 ATR from EMA20)", "Mean reversion",
        c < d["ema20"] - 2.5 * a, c > d["ema20"] + 2.5 * a)
    leg_up = d["sh1"] > d["sl1"]
    fib_up = d["sh1"] - 0.618 * (d["sh1"] - d["sl1"])
    fib_dn = d["sl1"] + 0.618 * (d["sh1"] - d["sl1"])
    add("Fibonacci 61.8% pullback", "Fibonacci",
        leg_up & (l <= fib_up) & (c > fib_up) & (c < d["sh1"]),
        ~leg_up & (h >= fib_dn) & (c < fib_dn) & (c > d["sl1"]))

    # ---- VWAP (tick volume)
    add("VWAP cross", "VWAP", cross_up(c, d["vwap"]), cross_dn(c, d["vwap"]))
    add("VWAP pullback in trend", "VWAP", up_tr & (l <= d["vwap"]) & (c > d["vwap"]),
        dn_tr & (h >= d["vwap"]) & (c < d["vwap"]))
    add("VWAP 2-sigma fade", "VWAP", c < d["vwap"] - 2 * d["vwap_sd"], c > d["vwap"] + 2 * d["vwap_sd"])

    # ---- support / resistance / pivots
    lvl25_dn = np.floor(c / 25) * 25
    lvl25_up = np.ceil(c / 25) * 25
    add("Round number ($25) bounce/rejection", "Support/resistance",
        (l <= lvl25_dn) & (c > lvl25_dn) & (c - lvl25_dn < 0.5 * a),
        (h >= lvl25_up) & (c < lvl25_up) & (lvl25_up - c < 0.5 * a))
    add("Round number ($25) breakout", "Support/resistance",
        np.floor(c / 25) > np.floor(prev(c) / 25), np.floor(c / 25) < np.floor(prev(c) / 25))
    add("Pivot S1 bounce / R1 rejection", "Support/resistance",
        (l <= d["S1"]) & (c > d["S1"]), (h >= d["R1"]) & (c < d["R1"]))
    add("Camarilla L3/H3 fade", "Support/resistance", (l <= d["L3"]) & (c > d["L3"]), (h >= d["H3"]) & (c < d["H3"]))
    add("Camarilla H4/L4 breakout", "Support/resistance", cross_up(c, d["H4"]), cross_dn(c, d["L4"]))
    add("Previous day high/low rejection (touch)", "Support/resistance",
        (l <= d["pdL"]) & (c > d["pdL"]) & (l > d["pdL"] - 0.3 * a),
        (h >= d["pdH"]) & (c < d["pdH"]) & (h < d["pdH"] + 0.3 * a))

    # ---- candlesticks
    eng_b = (c > o) & (prev(c) < prev(o)) & (c >= prev(o)) & (o <= prev(c))
    eng_s = (c < o) & (prev(c) > prev(o)) & (c <= prev(o)) & (o >= prev(c))
    add("Engulfing", "Candlestick", eng_b, eng_s)
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    add("Pin bar (hammer / shooting star)", "Candlestick",
        (lw >= 2 * body) & (lw >= 0.6 * rngb) & (rngb >= 0.8 * a),
        (uw >= 2 * body) & (uw >= 0.6 * rngb) & (rngb >= 0.8 * a))
    ms = (prev(c, 2) < prev(o, 2)) & (abs(prev(c) - prev(o)) < 0.3 * prev(h - l)) & (c > o) & \
         (c > (prev(o, 2) + prev(c, 2)) / 2)
    es = (prev(c, 2) > prev(o, 2)) & (abs(prev(c) - prev(o)) < 0.3 * prev(h - l)) & (c < o) & \
         (c < (prev(o, 2) + prev(c, 2)) / 2)
    add("Morning/evening star", "Candlestick", ms, es)
    add("Three-bar reversal", "Candlestick",
        (prev(l) < prev(l, 2)) & (prev(l) < l) & (c > prev(h)),
        (prev(h) > prev(h, 2)) & (prev(h) > h) & (c < prev(l)))
    near_pdl = abs(l - d["pdL"]) < 0.5 * a
    near_pdh = abs(h - d["pdH"]) < 0.5 * a
    add("Engulfing at previous day high/low", "Candlestick + level", eng_b & near_pdl, eng_s & near_pdh)
    add("Engulfing with EMA trend", "Candlestick + level", eng_b & up_tr, eng_s & dn_tr)

    # ---- market structure / reversals
    add("Break of structure (close beyond last swing)", "Market structure",
        cross_up(c, d["sh1"]), cross_dn(c, d["sl1"]))
    down_struct = d["sh1"] < d["sh2"]
    up_struct = d["sl1"] > d["sl2"]
    add("Change of character", "Market structure", down_struct & cross_up(c, d["sh1"]),
        up_struct & cross_dn(c, d["sl1"]))
    add("Higher-low / lower-high entry", "Market structure",
        d["new_sl"] & (d["sl1"] > d["sl2"]) & (d["sh1"] > d["sh2"]),
        d["new_sh"] & (d["sh1"] < d["sh2"]) & (d["sl1"] < d["sl2"]))
    add("Double bottom / top (neckline break)", "Reversal",
        eql & cross_up(c, d["sh1"]), eqh & cross_dn(c, d["sl1"]))
    add("RSI divergence at a new swing", "Reversal",
        d["new_sl"] & (d["sl1"] < d["sl2"]) & (d["rsi_sl1"] > d["rsi_sl2"]),
        d["new_sh"] & (d["sh1"] > d["sh2"]) & (d["rsi_sh1"] < d["rsi_sh2"]))
    climax = prev(rngb) > 3 * prev(a)
    add("Climax bar reversal", "Reversal",
        climax & (prev(c) < prev(o)) & (c > (prev(h) + prev(l)) / 2),
        climax & (prev(c) > prev(o)) & (c < (prev(h) + prev(l)) / 2))

    # ---- momentum / RSI / MACD / ADX / Bollinger
    add("RSI 50 cross", "Momentum", cross_up(d["rsi"], np.full_like(c, 50)), cross_dn(d["rsi"], np.full_like(c, 50)))
    add("RSI 30 reclaim / 70 rejection", "RSI", cross_up(d["rsi"], np.full_like(c, 30)),
        cross_dn(d["rsi"], np.full_like(c, 70)))
    add("RSI 70 breakout / 30 breakdown (momentum)", "RSI", cross_up(d["rsi"], np.full_like(c, 70)),
        cross_dn(d["rsi"], np.full_like(c, 30)))
    add("MACD signal-line cross", "Momentum", cross_up(d["macd"], d["macd_sig"]), cross_dn(d["macd"], d["macd_sig"]))
    add("MACD zero-line cross", "Momentum", cross_up(d["macd"], np.zeros_like(c)), cross_dn(d["macd"], np.zeros_like(c)))
    add("ADX > 25 + DI cross", "Momentum", (d["adx"] > 25) & cross_up(d["pdi"], d["ndi"]),
        (d["adx"] > 25) & cross_dn(d["pdi"], d["ndi"]))
    add("Bollinger mean reversion (close back inside)", "Bollinger",
        (prev(c) < prev(d["bb_lo"])) & (c > d["bb_lo"]), (prev(c) > prev(d["bb_up"])) & (c < d["bb_up"]))
    add("Bollinger + RSI reversal", "Bollinger", (c < d["bb_lo"]) & (d["rsi"] < 30), (c > d["bb_up"]) & (d["rsi"] > 70))
    add("Bollinger midline reclaim in trend", "Bollinger", up_tr & cross_up(c, d["bb_mid"]),
        dn_tr & cross_dn(c, d["bb_mid"]))

    # ---- sessions / gaps
    first_ny = (hr == 16) & (d["minute"] == 0)
    ny_move = c - pd.Series(np.where((hr == 15) & (d["minute"] == 30), o, np.nan)).groupby(day).transform("max").to_numpy()
    add("New York open reversal (fade 08:30-09:00 move)", "Session", first_ny & (ny_move < -0.5 * a),
        first_ny & (ny_move > 0.5 * a))
    first_lon = (hr == 10) & (d["minute"] == 0)
    lon_move = c - pd.Series(np.where((hr == 9) & (d["minute"] == 0), o, np.nan)).groupby(day).transform("max").to_numpy()
    add("London open reversal (fade first hour)", "Session", first_lon & (lon_move < -0.5 * a),
        first_lon & (lon_move > 0.5 * a))
    add("London open continuation (follow first hour)", "Session", first_lon & (lon_move > 0.5 * a),
        first_lon & (lon_move < -0.5 * a))
    week_first = pd.Series(d["day"]).dt.weekday.to_numpy() == 0
    gap = o - prev(c)
    mon_open = week_first & (hr == 1) & (d["minute"] == 0)
    add("Weekend gap fade", "Gap", mon_open & (gap < -0.5 * a), mon_open & (gap > 0.5 * a))

    # ---- ICT-style
    fvg_b = l > prev(h, 2)
    fvg_s = h < prev(l, 2)
    fb_lo = pd.Series(np.where(fvg_b, prev(h, 2), np.nan)).ffill(limit=20).shift(1).to_numpy()
    fb_hi = pd.Series(np.where(fvg_b, l, np.nan)).ffill(limit=20).shift(1).to_numpy()
    fs_hi = pd.Series(np.where(fvg_s, prev(l, 2), np.nan)).ffill(limit=20).shift(1).to_numpy()
    fs_lo = pd.Series(np.where(fvg_s, h, np.nan)).ffill(limit=20).shift(1).to_numpy()
    add("Fair value gap retrace", "ICT", (l <= fb_hi) & (c > fb_lo), (h >= fs_lo) & (c < fs_hi))
    disp_b = (c > o) & (rngb > 2 * a)
    disp_s = (c < o) & (rngb > 2 * a)
    ob_hi = pd.Series(np.where(disp_b & (prev(c) < prev(o)), prev(h), np.nan)).ffill(limit=30).shift(1).to_numpy()
    ob_lo = pd.Series(np.where(disp_b & (prev(c) < prev(o)), prev(l), np.nan)).ffill(limit=30).shift(1).to_numpy()
    os_hi = pd.Series(np.where(disp_s & (prev(c) > prev(o)), prev(h), np.nan)).ffill(limit=30).shift(1).to_numpy()
    os_lo = pd.Series(np.where(disp_s & (prev(c) > prev(o)), prev(l), np.nan)).ffill(limit=30).shift(1).to_numpy()
    add("Order block retest", "ICT", (l <= ob_hi) & (c > ob_lo), (h >= os_lo) & (c < os_hi))
    kill = (hr >= 9) & (hr < 12)
    add("London killzone break of structure", "ICT", kill & cross_up(c, d["sh1"]), kill & cross_dn(c, d["sl1"]))
    add("Daily bias + intraday EMA20 pullback", "Multi-timeframe",
        (d["d1_trend"] > 0) & (l <= d["ema20"]) & (c > d["ema20"]),
        (d["d1_trend"] < 0) & (h >= d["ema20"]) & (c < d["ema20"]))
    return M


# -------------------------------------------------------------- simulation

@njit(cache=True)
def simulate(si, sd, sdist, o, h, l, c, spread, day, rr, commission):
    n = len(o)
    out_i = np.full(len(si), -1)
    out_r = np.zeros(len(si))
    busy, cur_day, count, m = -1, -1, 0, 0
    for q in range(len(si)):
        j, d, dist = si[q], sd[q], sdist[q]
        if j >= n - 1 or j <= busy or not dist > 0:
            continue
        if day[j] != cur_day:
            cur_day, count = day[j], 0
        if count >= 3:
            continue
        entry = o[j]
        cost = spread[j] + commission
        if dist <= 2 * cost:
            continue
        stop = entry - d * dist
        tgt = entry + d * rr * dist
        k, px = j, np.nan
        while k < n and day[k] == day[j]:
            if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
                px = min(stop, o[k]) if d > 0 else max(stop, o[k])
                break
            if (d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt):
                px = max(tgt, o[k]) if d > 0 else min(tgt, o[k])
                break
            k += 1
        if np.isnan(px):
            k = min(k, n) - 1
            px = c[k]
        out_i[m] = j
        out_r[m] = (d * (px - entry) - cost) / dist
        m += 1
        busy = k
        count += 1
    return out_i[:m], out_r[:m]


def pf(x):
    w, lo = x[x > 0].sum(), -x[x <= 0].sum()
    return w / lo if lo > 0 else np.inf


def main() -> int:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    m5, m15, h4 = G.load()
    m15 = m15[m15["server"] >= m5["server"].iloc[0]].reset_index(drop=True)
    d = indicators(m15, h4)
    M = models(d)
    print(f"{len(M)} models built [{time.time() - t0:.0f}s]")

    t5 = m5["server"].to_numpy()
    o5, h5, l5, c5 = (m5[x].to_numpy(np.float64) for x in ("open", "high",
                                                          "low", "close"))
    sp5 = (m5["spread"] * G.POINT).to_numpy(np.float64)
    day5 = m5["server"].dt.normalize().to_numpy().astype("int64")
    entry_idx = np.searchsorted(t5, d["close_t"])     # first M5 bar after
    dist_all = STOP_ATR * d["atr"]

    def run(long, short, rng=None):
        idx = np.flatnonzero(long | short)
        dirs = np.where(long[idx], 1, -1)
        if rng is not None:
            dirs = np.where(rng.random(len(idx)) < 0.5, 1, -1)
        si = entry_idx[idx]
        ok = (si < len(o5) - 1) & np.isfinite(dist_all[idx])
        order = np.argsort(si[ok], kind="stable")
        ti, r = simulate(si[ok][order].astype(np.int64),
                         dirs[ok][order].astype(np.int64),
                         dist_all[idx][ok][order], o5, h5, l5, c5, sp5, day5,
                         RR, G.COMMISSION)
        return t5[ti], r

    rows = []
    for name, (fam, lg, sh) in M.items():
        when, r = run(lg, sh)
        dev, val = r[when < DEV_END], r[when >= DEV_END]
        rows.append(dict(family=fam, model=name, dev_n=len(dev),
                         dev_pf=pf(dev), dev_win=100 * (dev > 0).mean()
                         if len(dev) else np.nan, val_n=len(val),
                         val_pf=pf(val), all_pf=pf(r)))
    res = pd.DataFrame(rows).sort_values("dev_pf", ascending=False)
    print(f"real battery done [{time.time() - t0:.0f}s]")

    luck_counts = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n_fin = 0
        for _name, (_fam, lg, sh) in M.items():
            when, r = run(lg, sh, rng)
            dev = r[when < DEV_END]
            n_fin += (len(dev) >= 100) and pf(dev) >= 1.2
        luck_counts.append(n_fin)
    print(f"luck runs done [{time.time() - t0:.0f}s]")

    print("\n" + "=" * 110)
    print(f"ALL {len(res)} MODELS (1.5:1 needs 45.5% wins for PF 1.25; coin "
          f"toss 40%), sorted by development PF")
    print("=" * 110)
    print(res.round(3).to_string(index=False))
    fin = res[(res["dev_pf"] >= 1.2) & (res["dev_n"] >= 100)]
    print(f"\nreal finalists: {len(fin)}   finalists from RANDOM directions "
          f"(5 runs): {luck_counts} -> luck makes about "
          f"{np.mean(luck_counts):.1f}")
    print("\nfinalists and their one look at 2024-26:")
    print(fin[["family", "model", "dev_n", "dev_pf", "val_n", "val_pf"]]
          .round(3).to_string(index=False) if len(fin) else "  none")
    print(f"\nmodels with PF >= 1.25 on BOTH periods: "
          f"{((res['dev_pf'] >= 1.25) & (res['val_pf'] >= 1.25) & (res['dev_n'] >= 100)).sum()}")
    print(f"correlation of development PF with 2024-26 PF across models: "
          f"{res['dev_pf'].replace(np.inf, np.nan).corr(res['val_pf'].replace(np.inf, np.nan)):+.2f}")
    res.to_csv("data/gold_battery_results.csv", index=False)
    print(f"done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
