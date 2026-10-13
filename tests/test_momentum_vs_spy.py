"""
Correctness tests for the momentum-vs-SPY backtest.

None of these touch the internet. They build a small made-up market where
the right answer is known in advance, and check the claims the script makes:
trades happen the day after the signal, the skipped month really is skipped,
the SPY filter moves to cash, costs come out of the account, no signal
changes when future data is altered, and failed downloads are logged rather
than crashing the run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "standalone"))

import momentum_vs_spy as mm  # noqa: E402


DAYS = pd.bdate_range("2005-01-03", "2014-12-31")


def fake_market(n=40, seed=3):
    """Random-walk prices for n stocks plus a rising SPY with one crash."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, size=(len(DAYS), n))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=DAYS,
                          columns=[f"S{i:02d}" for i in range(n)])
    spy_r = np.full(len(DAYS), 0.0004)
    crash = (DAYS >= "2008-06-01") & (DAYS < "2009-03-01")
    spy_r[crash] = -0.004
    spy = pd.Series(100 * np.exp(np.cumsum(spy_r)), index=DAYS, name="SPY")
    members = pd.DataFrame({"tickers": [frozenset(prices.columns)]},
                           index=[DAYS[0]])
    return prices, spy, members


def signals(prices, spy, members, lookback=12, top=0.10):
    return mm.calc_signals(prices, spy, members, lookback, 1, top, 10)


def test_month_ends_skip_the_unfinished_month():
    days = pd.bdate_range("2020-01-01", "2020-03-17")
    ends = mm.month_end_dates(days)
    assert list(ends) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]


def test_trades_happen_the_next_trading_day():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    for _, r in sig.iterrows():
        assert r["trade_day"] > r["month_end"]
        assert r["trade_day"] == spy.index[spy.index > r["month_end"]][0]


def test_momentum_window_is_t12_to_t1():
    prices, spy, members = fake_market()
    ends = mm.month_end_dates(DAYS)
    i = 60
    end = ends[i]
    # S00 doubles in the most recent month only (should be ignored by the
    # skip month). S01 doubles between t-12 and t-1 (should be ranked top).
    prices.loc[ends[i - 1] + pd.Timedelta(days=1):, "S00"] *= 2
    prices.loc[ends[i - 12] + pd.Timedelta(days=1):, "S01"] *= 2
    prices.loc[ends[i - 1] + pd.Timedelta(days=1):, "S01"] /= 2
    sig = signals(prices, spy, members, top=0.05)    # 2 picks of 40
    row = sig[sig["month_end"] == end].iloc[0]
    assert row["risk_on"], "pick a month where the filter is on"
    assert "S01" in row["picks"]
    mom = (prices.reindex(ends).shift(1) / prices.reindex(ends).shift(12)
           - 1).loc[end]
    assert row["picks"] == list(mom.sort_values(ascending=False).index[:2])


def test_filter_goes_to_cash_and_cash_is_flat():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    off = sig[~sig["risk_on"]]
    assert len(off) > 0, "the fake crash should trip the filter"
    assert all(len(p) == 0 for p in off["picks"])
    bt = mm.run_backtest(prices, sig, cost=0.0)
    eq = bt["equity"]
    k = sig.index[~sig["risk_on"]][0]
    t0, t1 = sig.loc[k, "trade_day"], sig.loc[k + 1, "trade_day"]
    held = eq.loc[t0:t1].iloc[:-1]
    assert held.nunique() == 1


def test_filter_uses_ten_month_average_of_month_end_closes():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    ends = mm.month_end_dates(DAYS)
    spy_m = spy.reindex(ends)
    for _, r in sig.iterrows():
        i = ends.get_loc(r["month_end"])
        expect = spy_m.iloc[i] > spy_m.iloc[i - 9:i + 1].mean()
        assert r["risk_on"] == expect


def test_cost_on_first_buy_and_cost_drag():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    sig = sig[sig["risk_on"]].reset_index(drop=True)
    free = mm.run_backtest(prices, sig, cost=0.0)
    paid = mm.run_backtest(prices, sig, cost=0.001)
    assert paid["equity"].iloc[0] == pytest.approx(10_000 * 0.999)
    assert paid["costs"].iloc[0] == pytest.approx(10.0)
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]
    # First rebalance buys from cash: half of 100% in + 0% out = 0.5.
    assert paid["turnover"].iloc[0] == pytest.approx(0.5)


def test_equal_weight_and_drift_valuation():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    sig = sig[sig["risk_on"]].reset_index(drop=True)
    bt = mm.run_backtest(prices, sig, cost=0.0)
    t0, t1 = sig.loc[0, "trade_day"], sig.loc[1, "trade_day"]
    picks = sig.loc[0, "picks"]
    day = prices.loc[t0:t1].index[3]
    growth = (prices.loc[day, picks] / prices.loc[t0, picks]).mean()
    assert bt["equity"].loc[day] == pytest.approx(10_000 * growth)


def test_changing_the_future_does_not_change_past_signals():
    prices, spy, members = fake_market()
    base = signals(prices, spy, members)
    cut = pd.Timestamp("2010-06-30")
    p2, s2 = prices.copy(), spy.copy()
    rng = np.random.default_rng(99)
    p2.loc[cut + pd.Timedelta(days=1):] *= rng.uniform(
        0.2, 5, size=p2.loc[cut + pd.Timedelta(days=1):].shape)
    s2.loc[cut + pd.Timedelta(days=1):] *= 0.3
    alt = signals(p2, s2, members)
    before = base["month_end"] <= cut
    assert base[before]["picks"].tolist() == alt[before]["picks"].tolist()
    assert base[before]["risk_on"].tolist() == alt[before]["risk_on"].tolist()


def test_lookahead_self_check_passes():
    prices, spy, members = fake_market()
    sig = signals(prices, spy, members)
    cfg = dict(mm.CONFIG)
    assert mm.lookahead_self_check(prices, spy, members, sig, cfg, n_checks=5)


def test_only_point_in_time_members_are_picked():
    prices, spy, _ = fake_market()
    all_t = frozenset(prices.columns)
    # S00-S09 leave the index at the start of 2009; S39 only joins in 2011.
    members = pd.DataFrame(
        {"tickers": [all_t - {"S39"},
                     all_t - {f"S0{i}" for i in range(10)} - {"S39"},
                     all_t - {f"S0{i}" for i in range(10)}]},
        index=pd.to_datetime(["2005-01-03", "2009-01-02", "2011-01-03"]))
    sig = signals(prices, spy, members, top=0.5)
    for _, r in sig.iterrows():
        allowed = mm.members_on(members, r["month_end"])
        assert set(r["picks"]) <= allowed


def test_failed_downloads_are_logged_not_fatal(tmp_path):
    prices, _, _ = fake_market(n=8)
    calls = {"n": 0}

    def flaky(batch, start, end):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated network error")
        out = prices[[t for t in batch if t not in ("S02", "S05")]].copy()
        out["S05"] = np.nan
        out.loc[out.index[:10], "S05"] = 50.0     # only 10 days of data
        return out

    cfg = dict(mm.CONFIG, CACHE_DIR=str(tmp_path), DOWNLOAD_BATCH=8,
               DATA_START="2005-01-01")
    mm.time.sleep = lambda s: None
    px, dropped = mm.load_prices(list(prices.columns), cfg, download=flaky)
    assert set(dropped["ticker"]) == {"S02", "S05"}
    assert "S00" in px.columns and "S02" not in px.columns

    # Second run reads from disk and never calls the downloader.
    def boom(*a):
        raise AssertionError("should have used the saved copy")
    px2, dropped2 = mm.load_prices(list(prices.columns), cfg, download=boom)
    assert list(px2.columns) == list(px.columns)
    assert set(dropped2["ticker"]) == {"S02", "S05"}


def test_blocked_batch_is_retried_next_run(tmp_path):
    prices, _, _ = fake_market(n=4)
    cfg = dict(mm.CONFIG, CACHE_DIR=str(tmp_path), DOWNLOAD_BATCH=4,
               DATA_START="2005-01-01")
    mm.time.sleep = lambda s: None
    px, dropped = mm.load_prices(list(prices.columns), cfg,
                                 download=lambda *a: pd.DataFrame())
    assert px.empty and len(dropped) == 4
    px, dropped = mm.load_prices(list(prices.columns), cfg,
                                 download=lambda b, s, e: prices[b])
    assert len(px.columns) == 4 and dropped.empty


def test_full_run_end_to_end(tmp_path, monkeypatch):
    """The whole script on the fake market: runs, verdicts, writes files."""
    prices, spy, _ = fake_market(n=40)
    # The membership list writes share classes with a dot (like BRK.B);
    # Yahoo uses a dash (BRK-B). S-01 checks that conversion.
    world = prices.rename(columns={"S01": "S-01"}).join(spy)
    cache = tmp_path / "cache"
    cache.mkdir()
    names = ",".join("S.01" if c == "S01" else c for c in prices.columns)
    pd.DataFrame({"date": ["2004-01-02", "2010-01-04"],
                  "tickers": [names, names.replace(",S39", "") + ",GONE"]}
                 ).to_csv(cache / "sp500_membership.csv", index=False)

    def fake_dl(batch, start, end):
        return world.reindex(columns=batch)     # GONE comes back empty

    monkeypatch.setattr(mm, "_download_batch", fake_dl)
    mm.time.sleep = lambda s: None
    cfg = dict(mm.CONFIG, CACHE_DIR=str(cache),
               OUTPUT_DIR=str(tmp_path / "out"), DATA_START="2005-01-01")
    verdict = mm.main(cfg)
    assert verdict in ("PASS", "FAIL")
    out = tmp_path / "out"
    for f in ["summary.csv", "yearly_returns.csv", "equity_curves.csv",
              "out_of_sample.csv", "robustness.csv", "dropped_tickers.csv",
              "holdings.csv", "chart1_growth_of_10000.png",
              "chart2_drawdown.png", "report.txt"]:
        assert (out / f).exists(), f
    dropped = pd.read_csv(out / "dropped_tickers.csv")
    assert set(dropped["ticker"]) == {"GONE"}
    # S39 left the index in January 2010, so it can't be bought after that.
    h = pd.read_csv(out / "holdings.csv", parse_dates=["month_end"])
    later = h[h["month_end"] >= "2010-01-04"]["tickers"].fillna("")
    assert not any("S39" in t.split() for t in later)
    assert any("S-01" in t.split() for t in h["tickers"].fillna(""))
    oos = pd.read_csv(out / "out_of_sample.csv")
    assert len(oos) == 2
    assert pd.to_datetime(oos["start"]).iloc[1] > pd.to_datetime(
        oos["start"]).iloc[0]
