"""
The port has to agree with the runs it was ported from.

These numbers were produced by the original standalone scripts against the
same bars. They are pinned here not because they are interesting -- both
strategies turned out to be worth nothing -- but because a refactor that
silently changes a fill rule is the easiest way to lose a result, and the
only way to notice is to have written the old answer down.

Skipped when the Parquet store is absent, so a fresh clone still passes.
"""
from __future__ import annotations

import os
import pytest

from bcbt import store
from bcbt.strategies import ema_cross, orb_ema

COST = {"SPX500": 0.6, "NAS100": 2.0}

pytestmark = pytest.mark.skipif(
    not os.path.exists(store.PARQUET),
    reason="no bar store; run `python -m bcbt.fetch` first")


@pytest.fixture(scope="module")
def m1():
    return {s: store.load_m1(s) for s in ("SPX500", "NAS100")}


# (symbol, expected trades, expected mean R)
ORB_EXPECTED = [("SPX500", 762, 0.0571), ("NAS100", 764, 0.0563)]


@pytest.mark.parametrize("sym,n_exp,r_exp", ORB_EXPECTED)
def test_orb_15m_or_stop_1_5r(m1, sym, n_exp, r_exp):
    """The best cell of the ORB search, on both instruments."""
    days, arr = orb_ema.prepare(m1[sym])
    t = orb_ema.run(sym, days, arr, COST[sym], or_min=15,
                    stop_mode="or", rr=1.5)
    assert len(t) == n_exp
    assert t["r"].mean() == pytest.approx(r_exp, abs=5e-4)


EMA_EXPECTED = [("SPX500", 506, 0.0072), ("NAS100", 499, -0.0051)]


@pytest.mark.parametrize("sym,n_exp,r_exp", EMA_EXPECTED)
def test_ema_cross_as_described(m1, sym, n_exp, r_exp):
    """9/21 hourly, 2-bar swing stop, breakeven at 1R, flat outside RTH."""
    h = ema_cross.prepare(m1[sym], k=2, ema_basis="continuous")
    t = ema_cross.run(sym, m1[sym], h, COST[sym], use_be=True,
                      hold_overnight=False)
    assert len(t) == n_exp
    assert t["r"].mean() == pytest.approx(r_exp, abs=5e-4)


def test_breakeven_still_costs_money(m1):
    """A finding worth protecting: BE at 1R hurt on both instruments."""
    for sym in ("SPX500", "NAS100"):
        h = ema_cross.prepare(m1[sym], k=2)
        on = ema_cross.run(sym, m1[sym], h, COST[sym], use_be=True)
        off = ema_cross.run(sym, m1[sym], h, COST[sym], use_be=False)
        assert on["r"].mean() < off["r"].mean()
        assert on.attrs.get("x") is None      # table shape unchanged


def test_holidays_never_become_trades(m1):
    """The feed quotes a flat line through Christmas; nothing may trade."""
    days, arr = orb_ema.prepare(m1["SPX500"])
    got = {str(d.date()) for d in days}
    for holiday in ("2023-12-25", "2024-01-01", "2024-12-25",
                    "2025-01-01", "2025-12-25"):
        assert holiday not in got


def test_sessions_are_full_length(m1):
    """Every kept session must carry a full 78 five-minute cash bars."""
    days, arr = orb_ema.prepare(m1["NAS100"])
    lens = {len(v["t5"]) for v in days.values()}
    # 09:30-16:00 is 78 bars; the 13:00 half-day close is 42.
    assert lens <= {78, 42}
