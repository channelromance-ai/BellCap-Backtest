"""
Consolidation zone on the hourly chart, reaction entry on a faster one.

The strategy as described: find where price went sideways on the 1-hour
chart, wait for it to break out, and when it later comes back to that area
take the reaction on a 5- or 15-minute chart, with the stop on the far side
of the zone.

Run on index CFDs and on the seven major currency pairs, because the
question "does structure hold" should not be answered on one asset class.

What counts as a trade, in order:

  1. An hourly consolidation forms and is recorded at its last bar's close.
  2. Price closes clearly outside it, which is what turns a range into a
     level worth watching.
  3. Price later comes back and touches the zone.
  4. On the faster chart, a bar closes back OUT of the zone in the breakout
     direction -- the reaction. Entry is the next bar's open.
  5. Stop on the far side of the zone plus a buffer. Target a multiple of
     that risk. Flat if neither happens within a time limit.

Stops and targets resolve on 1-minute bars, so a bar containing both is
usually settled in the order it actually happened; where one minute holds
both, the stop is taken.

Nothing may look forward: the zone, the break, the retest and the reaction
are each confirmed strictly before the next is looked for, and the fill is
always the bar after the signal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bcbt import store, zones as Z
from bcbt.fills import OPEN, REASONS, resolve

warnings.filterwarnings("ignore")

INDEX_PANEL = "data/m1.parquet"
FX_PANEL = "data/fx_m1.parquet"

# Round-trip dealing cost in the instrument's own points, from the ask-side
# measurements made earlier for the indices and a realistic retail one pip
# for the currency majors.
COST_PTS = {
    "SPX500": 0.6, "NAS100": 1.2,
    "EURUSD": 0.00010, "GBPUSD": 0.00012, "AUDUSD": 0.00010,
    "NZDUSD": 0.00014, "USDCAD": 0.00012, "USDCHF": 0.00012,
    "USDJPY": 0.010,
}
FX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
INDICES = ["SPX500", "NAS100"]


def load(sym):
    src = FX_PANEL if sym in FX else INDEX_PANEL
    return store.load_m1(sym, src=src)


def bars(m1, minutes):
    return m1.resample(f"{minutes}min", closed="left", label="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def signals(sym, m1, entry_tf=5, zone_bars=4, width=1.5, buffer_frac=0.1,
            break_atr=0.5, bounce="close_out", min_cost_r=0.0):
    """
    Every entry the rule produces: when, which way, at what price, with
    what stop. Finding these is the slow part and none of it depends on how
    the trade is then managed, so it is done once and reused.

    The sequence is fixed, and every step must complete before the next is
    looked for:

      1. An hourly candle CLOSES outside the zone, by `break_atr` ATRs, so
         a one-tick poke through does not count as a break.
      2. On the faster chart, a candle CLOSES back INSIDE the zone. Price
         has to actually return and trade there -- a wick through is not
         enough, because a wick is a price that was rejected immediately.
      3. A later candle CLOSES back OUTSIDE the zone again, on the same
         side as the original break. That close is the signal.
      4. Entry is the open of the bar after it.

    If instead price closes clean through the far side, the level has
    failed and the zone is abandoned rather than waited on.

    `min_cost_r` drops trades where the zone is so narrow that the spread
    would be more than this share of the risk. Those cannot pay for
    themselves whatever happens next.
    """
    h = Z.hourly(m1)
    z = Z.find_zones(h, bars=zone_bars, max_width_atr=width,
                     break_atr=break_atr)
    if z.empty:
        return pd.DataFrame()
    lo_tf = bars(m1, entry_tf)

    o1 = m1["o"].to_numpy(np.float64)
    m_idx = m1.index

    rows = []
    for _, zn in z.iterrows():
        # From the moment the breakout hour CLOSED, not from when it began.
        seg = lo_tf.loc[(lo_tf.index >= zn.broke_known)
                        & (lo_tf.index <= zn.expires)]
        if len(seg) < 3:
            continue

        # Steps 2 and 3: a close back inside the zone, then a close back
        # outside it. Closes only -- a wick into the zone is a price that
        # was refused, not one the market accepted.
        sig = None
        been_inside = False
        prev_out = False
        for k in range(len(seg) - 1):
            c = seg["c"].iloc[k]
            inside = zn.bot <= c <= zn.top
            if inside:
                been_inside = True
                prev_out = False
                continue

            back_out = (c > zn.top) if zn.side == Z.UP else (c < zn.bot)
            if been_inside and back_out:
                if bounce != "two_bar" or prev_out:
                    sig = k
                    break
            prev_out = back_out

            # Closed clean through the far side: the level did not hold.
            if zn.side == Z.UP and c < zn.bot:
                break
            if zn.side == Z.DOWN and c > zn.top:
                break
        if sig is None:
            continue

        fill_ts = seg.index[sig] + pd.Timedelta(minutes=entry_tf)
        # pandas' own searchsorted keeps the timezone; numpy's turns a
        # tz-aware index into objects and refuses to compare them.
        j = int(m_idx.searchsorted(fill_ts))
        if j >= len(o1) - 2:
            continue
        entry = float(o1[j])

        d = 1 if zn.side == Z.UP else -1
        pad = buffer_frac * zn.width
        stop = (zn.bot - pad) if d > 0 else (zn.top + pad)
        risk = abs(entry - stop)
        if risk <= 0 or (d > 0 and stop >= entry) or (d < 0 and stop <= entry):
            continue
        # Skip trades the spread cannot be paid out of.
        if min_cost_r > 0 and (COST_PTS[sym] / risk) > min_cost_r:
            continue

        rows.append(dict(sym=sym, i=j, entry_ts=m_idx[j], d=d, entry=entry,
                         stop=stop, risk=risk))
    # Always the same columns, even with nothing in it: a strict setting can
    # legitimately produce no trades, and a bare empty frame has no columns
    # for the caller to filter on.
    return pd.DataFrame(rows, columns=["sym", "i", "entry_ts", "d", "entry",
                                       "stop", "risk"])


def evaluate(sym, m1, sigs, rr=2.0, be_at=0.0, max_hold_min=240):
    """Resolve each entry under one way of managing the trade."""
    if sigs is None or sigs.empty:
        return pd.DataFrame()
    o1 = m1["o"].to_numpy(np.float64)
    h1 = m1["h"].to_numpy(np.float64)
    l1 = m1["l"].to_numpy(np.float64)
    c1 = m1["c"].to_numpy(np.float64)
    m_idx = m1.index
    cost = COST_PTS[sym]
    n = len(o1)

    rows = []
    for _, s in sigs.iterrows():
        j, d, entry, stop, risk = int(s.i), int(s.d), s.entry, s.stop, s.risk
        target = entry + d * rr * risk if rr else 0.0
        last = min(j + max_hold_min, n - 1)
        xi, px, code, amb, _, _, _ = resolve(
            o1, h1, l1, j, last, d, entry, stop, risk, target, be_at)
        if code == OPEN:
            xi, px, why = last, float(c1[last]), "time"
        else:
            why = REASONS[code]
        rows.append(dict(
            sym=sym, entry_ts=m_idx[j], exit_ts=m_idx[xi],
            dir="long" if d > 0 else "short", entry=entry, stop=stop,
            risk=risk, exit=float(px), reason=why,
            r=(d * (float(px) - entry) - cost) / risk,
            held_min=xi - j, ambiguous=bool(amb)))
    return pd.DataFrame(rows)


def run(sym, m1, entry_tf=5, zone_bars=4, width=1.5, rr=2.0, buffer_frac=0.1,
        max_hold_min=240, be_at=0.0, start=None, end=None):
    """Convenience wrapper: find the entries, then manage them one way."""
    sg = signals(sym, m1, entry_tf, zone_bars, width, buffer_frac)
    if sg.empty:
        return sg
    if start:
        sg = sg[sg.entry_ts >= pd.Timestamp(start, tz=sg.entry_ts.iloc[0].tz)]
    if end:
        sg = sg[sg.entry_ts < pd.Timestamp(end, tz=sg.entry_ts.iloc[0].tz)]
    return evaluate(sym, m1, sg, rr, be_at, max_hold_min)


def score(t, label):
    if t is None or len(t) < 25:
        return None
    r = t["r"].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    se = r.std(ddof=1) / np.sqrt(len(r))
    wins, losses = r[r > 0], r[r <= 0]
    return dict(label=label, trades=len(r), win_pct=100 * len(wins) / len(r),
                avg_r=r.mean(), t=r.mean() / se, total_r=r.sum(),
                max_dd_r=dd,
                pf=(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
                held_min=t["held_min"].mean())


def split_date(m1, frac=0.6):
    days = pd.DatetimeIndex(sorted(set(m1.index.normalize())))
    return days[int(len(days) * frac)]


def main() -> int:
    pd.set_option("display.width", 210)
    data = {}
    for s in INDICES + FX:
        try:
            data[s] = load(s)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {s}: {exc}")

    print("=" * 96)
    print("ZONES FOUND (liquid hours only, tightness judged per hour of day)")
    print("=" * 96)
    for s, m1 in data.items():
        z = Z.find_zones(Z.hourly(m1), bars=4, max_width_atr=1.5)
        Z.summarise_zones(z, s)

    print("\n" + "=" * 96)
    print("DESIGN PERIOD: first 60% of each instrument's history")
    print("=" * 96)
    rows = []
    for s, m1 in data.items():
        cut = split_date(m1)
        t = run(s, m1, end=str(cut.date()))
        sc = score(t, s)
        if sc:
            rows.append(sc)
    if rows:
        d = pd.DataFrame(rows).set_index("label")
        print(d.round(3).to_string())
        allr = pd.concat([run(s, m1, end=str(split_date(m1).date()))
                          for s, m1 in data.items()])
        pooled = score(allr, "POOLED")
        if pooled:
            print(f"\n  pooled: {pooled['trades']} trades, "
                  f"win {pooled['win_pct']:.1f}%, "
                  f"avg {pooled['avg_r']:+.4f}R, t={pooled['t']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
