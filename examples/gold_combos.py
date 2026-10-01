"""
Gold: the battery's models working together, with no risk system.

The user's request: drop the stop/target rules and the daily cap, and test
models in combination rather than alone. Fixed before running:

  Scoring   No stops, targets or caps. Every signal is scored by the move in
            its direction after costs, entered at the next M15 open and held
            for 1 hour, 4 hours, or to the 17:00 New York roll. Costs: that
            bar's spread plus $8.22 a lot commission.
  Pairs     All 2,211 pairs of the 67 battery models: a signal when both
            have fired the same way within the last 4 M15 bars (one hour).
  Vote      At each M15 close, models that fired long within the last 4
            bars count +1, short -1. A signal when the vote first reaches
            3, 5, 8 or 12 in one direction.
  Unit      Days, not signals: signals on one day move together, so means
            and t-statistics are computed from daily averages.
  Pass      Development (Aug 2019 - Dec 2023): mean > 0 after cost, >= 200
            signals, t >= 3 -- strict because ~6,600 tests are being run.
            Every passer gets ONE look at Jan 2024 - Sep 2026.
  Luck      The identical search with each signal's direction randomised.
            How many pass there is what luck alone produces.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import examples.gold_battery as B                      # noqa: E402
import examples.gold_zones as G                        # noqa: E402

DEV_END = np.datetime64("2024-01-01")
WINDOW = 4                 # bars a model's signal stays "recent"
HORIZONS = {"1h": 4, "4h": 16, "to_roll": None}
T_PASS = 3.0
MIN_N = 200


def forward_returns(m15):
    """Long-side net move in bp per horizon, entering at the next open.
    Short-side net = -(gross) - cost, so both are returned."""
    o = m15["open"].to_numpy(np.float64)
    c = m15["close"].to_numpy(np.float64)
    spread = (m15["spread"] * G.POINT).to_numpy(np.float64)
    day = m15["server"].dt.normalize().to_numpy()
    n = len(c)
    # Index of the last bar of each server day.
    last = pd.Series(np.arange(n)).groupby(day).transform("max").to_numpy()
    entry_i = np.arange(n) + 1
    ok = entry_i < n
    entry_i = np.clip(entry_i, 0, n - 1)
    same_day = last[entry_i] == last
    out = {}
    for name, hb in HORIZONS.items():
        exit_i = last if hb is None else np.minimum(entry_i + hb - 1, last)
        entry = o[entry_i]
        gross = 1e4 * (c[exit_i] / entry - 1)
        cost = 1e4 * (spread[entry_i] + G.COMMISSION) / entry
        valid = ok & same_day & (exit_i >= entry_i)
        out[name] = (np.where(valid, gross - cost, np.nan),
                     np.where(valid, -gross - cost, np.nan))
    return out


def day_t(vals, dayidx, n_days):
    """
    Mean per TRADE, with the t-statistic taken across days.

    Every trade counts equally, which is what trading them gives. Days are
    the unit only for the uncertainty: each day's trades are SUMMED (its
    real P&L) and the t-statistic is over those daily sums.

    The first version averaged each day's trades before averaging days. That
    weights a trade by 1/(signals that day) -- a number not known until the
    day is over -- and it manufactured the whole of this script's original
    "edge": the fade signals fire repeatedly and lose on trending days and
    fire once and win on quiet ones. Trade-weighted, every top combination
    lost money.
    """
    s = np.bincount(dayidx, weights=vals, minlength=n_days)
    k = np.bincount(dayidx, minlength=n_days)
    m = k > 0
    ds = s[m]
    if len(ds) < 3 or k[m].sum() == 0:
        return np.nan, np.nan, len(ds)
    per_trade = ds.sum() / k[m].sum()
    t = ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))
    return per_trade, t, len(ds)


def score(idx_long, idx_short, fwd, dayidx, n_days, is_dev, rng=None):
    """Stats for one signal set on one horizon; optional random directions."""
    L, S = fwd
    idx = np.r_[idx_long, idx_short]
    d = np.r_[np.ones(len(idx_long)), -np.ones(len(idx_short))]
    if rng is not None:
        d = np.where(rng.random(len(idx)) < 0.5, 1.0, -1.0)
    v = np.where(d > 0, L[idx], S[idx])
    good = np.isfinite(v)
    idx, v = idx[good], v[good]
    dv = is_dev[idx]
    res = {}
    for part, sel in (("dev", dv), ("val", ~dv)):
        mean, t, nd = day_t(v[sel], dayidx[idx[sel]], n_days)
        res[f"{part}_n"] = int(sel.sum())
        res[f"{part}_days"] = nd
        res[f"{part}_bp"] = mean
        res[f"{part}_t"] = t
    return res


def main() -> int:
    pd.set_option("display.width", 240)
    t0 = time.time()
    m5, m15, h4 = G.load()
    m15 = m15[m15["server"] >= m5["server"].iloc[0]].reset_index(drop=True)
    d = B.indicators(m15, h4)
    M = B.models(d)
    names = list(M)
    fwd = forward_returns(m15)
    day = m15["server"].dt.normalize()
    dayidx = pd.factorize(day)[0]
    n_days = dayidx.max() + 1
    is_dev = m15["server"].to_numpy() < DEV_END
    # "Recent" long/short state per model.
    rec_l = {k: pd.Series(M[k][1]).rolling(WINDOW, min_periods=1).max()
             .to_numpy().astype(bool) for k in names}
    rec_s = {k: pd.Series(M[k][2]).rolling(WINDOW, min_periods=1).max()
             .to_numpy().astype(bool) for k in names}
    print(f"{len(names)} models, {len(m15):,} M15 bars [{time.time() - t0:.0f}s]")

    rows_real, rows_luck = [], []
    rng = np.random.default_rng(7)
    for a, b in itertools.combinations(names, 2):
        lg = (M[a][1] & rec_l[b]) | (M[b][1] & rec_l[a])
        sh = (M[a][2] & rec_s[b]) | (M[b][2] & rec_s[a])
        conflict = lg & sh
        il, is_ = np.flatnonzero(lg & ~conflict), np.flatnonzero(sh & ~conflict)
        if len(il) + len(is_) < MIN_N:
            continue
        # Variant knowable in advance: only the first signal of each day.
        allidx = np.sort(np.r_[il, is_])          # time order first
        first = allidx[np.unique(dayidx[allidx], return_index=True)[1]]
        first = np.sort(first)
        fl, fs = first[np.isin(first, il)], first[np.isin(first, is_)]
        for variant, (xl, xs) in (("every signal", (il, is_)),
                                  ("first of day", (fl, fs))):
            for hz, f in fwd.items():
                base = dict(combo=f"{a}  +  {b}", variant=variant, horizon=hz)
                rows_real.append({**base, **score(xl, xs, f, dayidx, n_days,
                                                  is_dev)})
                rows_luck.append({**base, **score(xl, xs, f, dayidx, n_days,
                                                  is_dev, rng)})
    print(f"pairs scored [{time.time() - t0:.0f}s]")

    # The 67 models alone, scored the same way, for reference.
    for k in names:
        il, is_ = np.flatnonzero(M[k][1]), np.flatnonzero(M[k][2])
        allidx = np.sort(np.r_[il, is_])
        first = np.sort(allidx[np.unique(dayidx[allidx],
                                         return_index=True)[1]])
        for variant, (xl, xs) in (
                ("every signal", (il, is_)),
                ("first of day", (first[np.isin(first, il)],
                                  first[np.isin(first, is_)]))):
            for hz, f in fwd.items():
                base = dict(combo=f"SINGLE: {k}", variant=variant,
                            horizon=hz)
                rows_real.append({**base, **score(xl, xs, f, dayidx, n_days,
                                                  is_dev)})
                rows_luck.append({**base, **score(xl, xs, f, dayidx, n_days,
                                                  is_dev, rng)})

    vote = sum(rec_l[k].astype(int) - rec_s[k].astype(int) for k in names)
    for k in (3, 5, 8, 12):
        lg = (vote >= k) & (B.prev(vote.astype(float)) < k)
        sh = (vote <= -k) & (B.prev(vote.astype(float)) > -k)
        il, is_ = np.flatnonzero(lg), np.flatnonzero(sh)
        for hz, f in fwd.items():
            base = dict(combo=f"VOTE >= {k} of 67", variant="every signal",
                        horizon=hz)
            rows_real.append({**base, **score(il, is_, f, dayidx, n_days,
                                              is_dev)})
            rows_luck.append({**base, **score(il, is_, f, dayidx, n_days,
                                              is_dev, rng)})

    real, luck = pd.DataFrame(rows_real), pd.DataFrame(rows_luck)

    def passers(df):
        return df[(df["dev_bp"] > 0) & (df["dev_t"] >= T_PASS)
                  & (df["dev_n"] >= MIN_N)]
    pr, pl = passers(real), passers(luck)
    print("\n" + "=" * 110)
    print(f"TESTS RUN: {len(real)}  (pairs x 3 horizons, plus 4 vote levels x 3)")
    print("=" * 110)
    print(f"  passed development (t >= {T_PASS}):  real {len(pr)}   "
          f"random directions {len(pl)}")
    print(f"  development t >= 2:  real {(real['dev_t'] >= 2).sum()}   "
          f"random {(luck['dev_t'] >= 2).sum()}")
    print(f"  development mean > 0 after cost:  real "
          f"{(real['dev_bp'] > 0).mean():.1%}   random "
          f"{(luck['dev_bp'] > 0).mean():.1%}")
    print("\nEVERY REAL PASSER and its one look at 2024-26:")
    cols = ["combo", "variant", "horizon", "dev_n", "dev_bp", "dev_t", "val_n",
            "val_bp", "val_t"]
    print(pr.sort_values("dev_t", ascending=False)[cols].round(2)
          .to_string(index=False) if len(pr) else "  none")
    if len(pr):
        print(f"\n  passers still positive on 2024-26: "
              f"{(pr['val_bp'] > 0).sum()} of {len(pr)};  with t >= 2: "
              f"{(pr['val_t'] >= 2).sum()}")
    print("\nVOTING ENSEMBLE (all 67 models together):")
    print(real[real["combo"].str.startswith("VOTE")][cols].round(2)
          .to_string(index=False))
    print(f"\ncorrelation of development t with 2024-26 t across all tests: "
          f"{real['dev_t'].corr(real['val_t']):+.2f}")
    real.to_csv("data/gold_combos_real.csv", index=False)
    luck.to_csv("data/gold_combos_random.csv", index=False)
    print(f"done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
