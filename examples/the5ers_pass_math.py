"""
The5ers one-step evaluation: what actually decides whether it passes.

Rules, from the5ers.com/hyper-growth: +10% target, 3% daily loss, 6% overall
loss measured from the starting balance (it does not trail), no time limit.
Assumed where the page is silent: the daily loss is measured from the
start-of-day balance and counts open losses.

The account is 40-45% of the way to target. The question is which way of
trading gives the best chance of reaching +10% by 31 December, given what
the rest of this repo found: no setup tested here has a reliable edge, and
the one real effect (EUR/USD drifting down in the European morning) is
small and has shrunk.

That makes the arithmetic of the account more important than the entry.
For a trader with no edge and no costs, the chance of reaching the target
before the floor is simply

        distance to floor / (distance to target + distance to floor)

whatever the strategy -- about 64% from +4.25%. Costs pull it down, and
more trades mean more cost. A deadline pulls it down too, because a trader
who risks too little never gets anywhere. Everything below is Monte Carlo
over those three forces, using The5ers' measured EUR/USD cost.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

TARGET, FLOOR, DAILY = 10.0, -6.0, 3.0
DAYS = len(pd.bdate_range("2026-09-24", "2026-12-31")) - 1   # less 25 Dec
COST_BP = 0.392                                            # EUR/USD
N = 20000


def simulate(start, draw, risk_pct, per_day, rng):
    """
    Walk N accounts forward. `draw(rng, shape)` returns trade results in R
    (after cost). Returns pass/fail/open shares and median days to pass.
    """
    bal = np.full(N, start)
    state = np.zeros(N, int)                  # 0 open, 1 passed, -1 failed
    day_passed = np.full(N, np.nan)
    for d in range(DAYS):
        live = state == 0
        if not live.any():
            break
        day_start = bal.copy()
        for _ in range(per_day):
            live = state == 0
            r = draw(rng, live.sum())
            bal[live] += risk_pct * r
            done = live & (bal >= TARGET)
            state[done] = 1
            day_passed[done] = d + 1
            dead = live & ((bal <= FLOOR) | (day_start - bal >= DAILY))
            state[dead] = -1
    return dict(pass_pct=100 * (state == 1).mean(),
                fail_pct=100 * (state == -1).mean(),
                still_open_pct=100 * (state == 0).mean(),
                median_days_to_pass=np.nanmedian(day_passed))


def coin(rr, stop_bp):
    """A trade with no edge: wins at the coin-toss rate for its payoff."""
    cost_r = COST_BP / stop_bp
    p = 1 / (1 + rr)

    def draw(rng, n):
        win = rng.random(n) < p
        return np.where(win, rr, -1.0) - cost_r
    return draw


def european_morning(stop_bp=30.0, since=2021):
    """Short EUR/USD 03:00-08:00 New York, stop `stop_bp` above, from history."""
    a = pd.read_parquet("data/the5ers_h1.parquet")
    e = a[a["sym"] == "EURUSD"].copy()
    e["ny"] = e["server"] - pd.Timedelta(hours=7)
    e = e[(e["ny"].dt.year >= since) & (e["ny"].dt.weekday < 5)]
    e["day"] = e["ny"].dt.normalize()
    e["h"] = e["ny"].dt.hour
    out = []
    for _, g in e[(e["h"] >= 3) & (e["h"] <= 7)].groupby("day"):
        if len(g) != 5 or g["h"].iloc[0] != 3:
            continue
        entry = g["open"].iloc[0]
        stop = entry * (1 + stop_bp / 1e4)
        px = g["close"].iloc[-1]
        for _, bar in g.iterrows():
            if bar["high"] >= stop:
                # A bar that opens through the stop fills at its open.
                px = max(stop, bar["open"])
                break
        out.append((entry - px) / (entry * stop_bp / 1e4)
                   - COST_BP / stop_bp)
    r = np.array(out)

    def draw(rng, n):
        return rng.choice(r, size=n)
    draw.mean_r = r.mean()
    draw.n = len(r)
    return draw


def main() -> int:
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(7)
    print(f"trading days left this year: {DAYS}")
    rows = []
    for start in (4.0, 4.5):
        base = 100 * (start - FLOOR) / (TARGET - FLOOR)
        print(f"\nfrom +{start}%: no-edge, no-cost, no-deadline odds = "
              f"{base:.1f}%")
        for rr in (1.0, 2.0):
            for risk in (0.5, 1.0, 1.5, 2.0, 2.5):
                for per_day in (1, 2):
                    s = simulate(start, coin(rr, 20.0), risk, per_day, rng)
                    rows.append(dict(start=start, system=f"no edge {rr:.0f}:1",
                                     risk_pct=risk, trades_per_day=per_day,
                                     **s))
        for since in (2021, 2000):
            draw = european_morning(since=since)
            for risk in (0.5, 1.0, 1.5, 2.0, 2.5):
                s = simulate(start, draw, risk, 1, rng)
                rows.append(dict(start=start,
                                 system=f"EUR morning ({since}+, "
                                        f"{draw.mean_r:+.3f}R)",
                                 risk_pct=risk, trades_per_day=1, **s))
    d = pd.DataFrame(rows)
    print(d.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
