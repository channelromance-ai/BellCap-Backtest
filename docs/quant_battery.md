# Quant battery: 385 retail CFD models, one test


**Data.** OANDA's own CFD quotes (1-minute bars with tick volume) for 25
instruments: S&P 500, Nasdaq 100, Russell 2000, FTSE 100, CAC 40, Nikkei
225, ASX 200, AEX; EUR/USD, GBP/USD, AUD/USD, USD/CAD, EUR/JPY, AUD/JPY;
gold, WTI, natural gas, corn, wheat, soybeans, sugar; US 2-year and
10-year Treasury, Bund and Gilt CFDs. January 2005 to May 2020, folded to
15-minute, hourly and daily (17:00 New York roll) bars, plus daily VIX.
Fetched from the public FutureSharks/financial-data repository by
`bcbt/oanda_data.py` (Dukascopy and Yahoo are blocked in this environment).

**One engine for every model** (`bcbt/quant.py`). A model only states the
position it wants at each bar's close; the engine holds it to the next
close, sizes every instrument to the same 10% annual volatility (capped at
3x leverage, resized only on a 25% drift), and charges:

* the spread plus 1bp slippage on every change of position, per instrument
  (1bp S&P, 1.2bp EUR/USD, 3bp gold, 5bp WTI, 10-20bp grains, 2-4bp bonds:
  typical retail quotes, not the tightest advertised);
* overnight financing on everything held over the 17:00 New York roll, three
  nights at weekends: a 2.5% a year markup on indices, commodities and
  bonds and 1% on FX, charged long and short. The benchmark rate itself is
  left out (near zero for USD, EUR and JPY most of this sample).

**Every model is judged the same way.** Net Sharpe; the same before
financing and before any cost; 2005-2012 vs 2013-2020 (an edge must show
in both); the Newey-West t of daily net returns; and an alpha t after
regressing on a long-only holding of the same instruments, so that a
model which is mostly long the S&P in a bull market is not credited with
skill. Then the multiple-testing correction for having tried 359 distinct
models. Parameters are textbook values fixed before any result was seen;
everything that learns (ML, walk-forward, Bayesian, seasonal) is fitted
only on data before the year it trades. A look-ahead audit
(`examples/quant_audit.py`) re-runs every model on data cut at mid-2014 and
checks every signal before the cut is identical.

**Result.** 16 of 359 models have a positive net Sharpe ratio and
231 are positive before costs. 1 has
a net t-statistic of 2 or more (1 positive in both halves). The
best net Sharpe is `moy_season` at 0.55, below the
0.75 that the best of 359 skill-less models is
expected to reach by luck.

## Bottom line

* **No model on the list has a demonstrable edge on retail CFDs after
  costs.** The best result is what luck alone produces when this many
  models are tried (Reality Check p = 0.96).
* **Many signals are real before costs and die on them.**
  30 models have
  a gross t of 2 or more and lose money net: hourly seasonality, lead-lag,
  session patterns and daily machine-learning direction calls all predict
  something, but by less than the spread they pay to act on it, and the
  more often a model trades the bigger the gap.
* **Overnight financing is the second tax.** For the daily models that are
  in the market more than half the time, financing alone costs a median
  0.45 of Sharpe; pairs pay it on both legs (up to 1.3).
  Low-volatility CFDs (bonds, FX) need the most notional per unit of risk
  and pay the most.
* **Some of the few positive models are long equity risk in disguise.**
  The volatility-risk-premium, equity-bond-regime and VIX-fear models move
  0.7-0.9 with simply holding the same CFDs long, and their alpha t is
  near zero. The others (month-of-year seasonality, the rates
  differential, VIX vs realised volatility, the VIX-implied-move breach)
  keep an alpha t of 1.3-2.2 after removing the long exposure, which is
  not significant once 359 tries are accounted for.
* **Sizing and risk rules cannot fix a signal without an edge.** At equal
  volatility no sizing scheme turns the trend model positive in both
  halves; drawdown and loss-limit rules change which days are lost, not
  whether money is made.


## Verdict count

| Verdict | Models |
|---|---|
| no edge (positive before costs, not significant) | 185 |
| no edge (negative even before costs) | 128 |
| real gross edge, eaten by costs | 30 |
| positive both halves, not significant | 9 |
| positive overall, fails one half | 6 |
| significant alone, not after 359 trials | 1 |
| not testable on this data | 22 |


## Multiple testing

With 359 models and no edge anywhere, about 9 would still show a
one-sided t of 2 by luck (2.5% each). 1 did. White's Reality Check on
the best daily net stream (`moy_season`, Sharpe 0.46)
against the best of 359 demeaned, block-bootstrapped streams: p =
0.965. The deflated Sharpe ratio (Bailey and Lopez de Prado)
asks how likely a Sharpe ratio is to be real once it is known to be the
best of 359 tries: with ~15 years of daily data and no skill anywhere, the
best of 359 would still reach a Sharpe of about 0.75
by chance. A model needs dsr > 0.95 to be called an edge here.


## Before costs: the 20 best gross Sharpe ratios

What the signal earns before the broker is paid, and what spread and overnight financing take back.

| Model | Gross | − spread | − financing | Net | Turnover/day | Verdict |
|---|---|---|---|---|---|---|
| `tod_season` | 2.99 | 24.87 | 0.11 | -21.99 | 5.56 | real gross edge, eaten by costs |
| `lead_lag_arb` | 2.42 | 8.37 | 0.01 | -5.96 | 0.30 | real gross edge, eaten by costs |
| `intraday_pattern` | 2.19 | 26.05 | 0.23 | -24.09 | 14.00 | real gross edge, eaten by costs |
| `session_season` | 1.35 | 6.91 | 0.09 | -5.66 | 1.42 | real gross edge, eaten by costs |
| `ml_knn` | 1.28 | 1.46 | 0.58 | -0.76 | 13.16 | real gross edge, eaten by costs |
| `ml_svm` | 1.22 | 1.21 | 0.55 | -0.54 | 11.95 | real gross edge, eaten by costs |
| `stat_ll` | 1.18 | 4.42 | 0.02 | -3.26 | 0.52 | real gross edge, eaten by costs |
| `overnight_return` | 1.06 | 1.14 | 0.24 | -0.32 | 0.17 | real gross edge, eaten by costs |
| `htf_trend_ltf` | 0.99 | 5.99 | 0.23 | -5.23 | 1.96 | real gross edge, eaten by costs |
| `eq_fx_ll` | 0.98 | 7.14 | 0.03 | -6.19 | 0.61 | real gross edge, eaten by costs |
| `ml_stacked` | 0.90 | 1.61 | 0.62 | -1.33 | 12.45 | real gross edge, eaten by costs |
| `expected_move_bo` | 0.90 | 2.71 | 0.17 | -1.97 | 0.72 | real gross edge, eaten by costs |
| `moy_season` | 0.88 | 0.08 | 0.25 | 0.55 | 0.21 | significant alone, not after 359 trials |
| `london_session` | 0.86 | 6.50 | 0.00 | -5.64 | 0.75 | real gross edge, eaten by costs |
| `ml_mlp` | 0.81 | 1.55 | 0.59 | -1.34 | 14.14 | real gross edge, eaten by costs |
| `cross_market_ll` | 0.79 | 4.23 | 0.08 | -3.52 | 1.23 | real gross edge, eaten by costs |
| `ml_voting` | 0.75 | 1.13 | 0.51 | -0.89 | 11.70 | real gross edge, eaten by costs |
| `rates_diff` | 0.75 | 0.17 | 0.15 | 0.43 | 0.64 | positive both halves, not significant |
| `opening_window` | 0.73 | 9.45 | 0.00 | -8.72 | 0.16 | real gross edge, eaten by costs |
| `mkt_neutral_rank` | 0.72 | 0.49 | 0.82 | -0.58 | 2.78 | real gross edge, eaten by costs |

## After costs: the 20 best net Sharpe ratios

β and α t are against a long-only, vol-scaled holding of the same instruments; dsr is the deflated Sharpe ratio.

| Model | Net | Dev | Hold | t | β long | α t | dsr | Verdict |
|---|---|---|---|---|---|---|---|---|
| `moy_season` | 0.55 | 0.67 | 0.46 | 2.01 | 0.05 | 2.23 | 0.16 | significant alone, not after 359 trials |
| `iv_rv` | 0.46 | 0.49 | 0.45 | 1.99 | 0.26 | 1.37 | 0.12 | positive both halves, not significant |
| `rates_diff` | 0.43 | 0.66 | 0.16 | 1.66 | 0.00 | 1.66 | 0.11 | positive both halves, not significant |
| `vrp` | 0.33 | 0.32 | 0.35 | 1.39 | 0.86 | 0.21 | 0.05 | positive both halves, not significant |
| `implied_move_breach` | 0.31 | 0.43 | 0.16 | 1.36 | 0.01 | 1.32 | 0.05 | positive both halves, not significant |
| `sentiment_extreme_proxy` | 0.28 | 0.03 | 0.49 | 1.09 | 0.48 | -0.33 | 0.03 | positive both halves, not significant |
| `yield_curve` | 0.16 | 0.12 | 0.20 | 0.61 | 0.45 | 0.40 | 0.01 | positive both halves, not significant |
| `dollar_gold` | 0.14 | 0.38 | -0.06 | 0.52 | 0.04 | 0.44 | 0.01 | positive overall, fails one half |
| `eq_bond_corr` | 0.11 | 0.03 | 0.20 | 0.45 | 0.90 | -0.14 | 0.01 | positive both halves, not significant |
| `evt_breakout` | 0.11 | 0.31 | -0.04 | 0.46 | -0.11 | 0.23 | 0.01 | positive overall, fails one half |
| `meta_label` | 0.07 | 0.14 | 0.04 | 0.26 | -0.10 | 0.09 | 0.00 | positive both halves, not significant |
| `hv_breakout` | 0.07 | 0.05 | 0.09 | 0.29 | -0.13 | 0.33 | 0.00 | positive both halves, not significant |
| `sentiment_mom_div_proxy` | 0.04 | 0.20 | -0.09 | 0.15 | -0.05 | 0.43 | 0.00 | positive overall, fails one half |
| `vol_targeting` | 0.01 | 0.31 | -0.34 | 0.04 | 2.45 | -1.81 | 0.00 | positive overall, fails one half |
| `vol_fcst_rev` | 0.01 | -0.27 | 0.21 | 0.04 | 0.33 | -0.28 | 0.00 | positive overall, fails one half |
| `gap_return_dist` | 0.00 | -0.66 | 0.24 | 0.00 | -0.01 | 0.08 | 0.00 | positive overall, fails one half |
| `pf_filter` | -0.02 | -0.02 | -0.02 | -0.07 | -0.03 | -0.07 | 0.00 | no edge (positive before costs, not significant) |
| `ccy_strength` | -0.03 | 0.13 | -0.21 | -0.10 | -0.10 | -0.25 | 0.00 | no edge (positive before costs, not significant) |
| `evt_entry` | -0.05 | 0.45 | -0.34 | -0.21 | 0.04 | -0.10 | 0.00 | no edge (positive before costs, not significant) |
| `expectancy_filter` | -0.06 | -0.07 | -0.11 | -0.28 | -0.04 | -0.27 | 0.00 | no edge (positive before costs, not significant) |

## By family

| Family | Models | Median net | Median gross | Best net | Net > 0 |
|---|---|---|---|---|---|
| Cross-asset: pairs, stat arb, ranking, relative strength, macro | 48 | -0.46 | 0.14 | 0.46 | 8 |
| Daily single-market (mean reversion, trend, breakout, volatility, regime, forecasting, technical, patterns) | 127 | -0.52 | 0.03 | 0.11 | 3 |
| Intraday: sessions, seasonality, gaps, news, lead-lag, CFD execution, tick-volume flow, jumps, multi-timeframe | 83 | -1.80 | 0.13 | 0.55 | 3 |
| Machine learning and walk-forward | 34 | -0.76 | 0.16 | 0.07 | 1 |
| Risk, sizing, trade filters, portfolio blends | 47 | -0.31 | 0.16 | 0.01 | 1 |
| Volatility estimators, regime-conditional, support/resistance and others | 20 | -0.46 | 0.03 | -0.26 | 0 |

## Sizing and risk schemes on the same trend signals

Each scheme sizes the multi-speed trend model's positions (or, for the Kelly/utility entries and the long baskets, its own) and is rescaled to 10% realised volatility, so the comparison is at equal risk. *Prop pass / bust* is the share of 60-day windows, one starting every month, that reach +8% before a 10% loss or a 5% losing day. At 10% volatility a simulated stream with no edge passes about 10% of windows (4-16% across 20 simulated histories), so every scheme here, at 3-7%, does no better than luck and mostly worse.

| Scheme | Net | Dev | Hold | Max DD | Worst day | Prop pass | Prop bust |
|---|---|---|---|---|---|---|---|
| `long_basket_fixed` | 0.160 | 0.360 | -0.155 | 0.380 | -0.050 | 0.037 | 0.037 |
| `vol_targeting` | 0.012 | 0.314 | -0.338 | 0.366 | -0.043 | 0.027 | 0.053 |
| `prop_firm_aware` | -0.087 | -0.042 | -0.187 | 0.409 | -0.096 | 0.064 | 0.037 |
| `fixed_fractional` | -0.127 | 0.116 | -0.542 | 0.634 | -0.086 | 0.048 | 0.000 |
| `corr_adj_sizing` | -0.173 | 0.243 | -0.626 | 0.699 | -0.051 | 0.059 | 0.016 |
| `es_sizing` | -0.207 | 0.083 | -0.560 | 0.697 | -0.066 | 0.037 | 0.005 |
| `volscaled_prop` | -0.241 | 0.255 | -0.793 | 0.772 | -0.067 | 0.059 | 0.027 |
| `kelly_opt_entry` | -0.246 | -0.287 | -0.199 | 0.568 | -0.065 | 0.059 | 0.043 |
| `vol_target_sizing` | -0.257 | 0.077 | -0.676 | 0.782 | -0.054 | 0.037 | 0.000 |
| `gk_sizing` | -0.275 | 0.051 | -0.701 | 0.834 | -0.055 | 0.032 | 0.016 |
| `range_vol_sizing` | -0.282 | 0.059 | -0.714 | 0.849 | -0.056 | 0.032 | 0.016 |
| `vol_targeting_portfolio` | -0.287 | 0.042 | -0.668 | 0.809 | -0.052 | 0.043 | 0.016 |
| `parkinson_sizing` | -0.288 | 0.059 | -0.740 | 0.842 | -0.056 | 0.037 | 0.016 |
| `yz_sizing` | -0.292 | 0.040 | -0.721 | 0.839 | -0.054 | 0.032 | 0.032 |
| `utility_max` | -0.294 | -0.374 | -0.205 | 0.651 | -0.052 | 0.064 | 0.043 |
| `egarch_sizing` | -0.300 | 0.095 | -0.642 | 0.864 | -0.052 | 0.043 | 0.005 |
| `atr_sizing` | -0.306 | 0.047 | -0.757 | 0.864 | -0.056 | 0.032 | 0.016 |
| `frac_kelly_sizing` | -0.324 | -0.174 | -0.542 | 0.630 | -0.069 | 0.053 | 0.027 |
| `kelly_sizing` | -0.333 | -0.241 | -0.482 | 0.587 | -0.072 | 0.059 | 0.027 |
| `vol_clustering` | -0.348 | 0.040 | -0.853 | 0.862 | -0.062 | 0.027 | 0.016 |
| `dd_adj_sizing` | -0.349 | -0.220 | -0.747 | 0.785 | -0.082 | 0.037 | 0.032 |
| `rp_sizing` | -0.367 | 0.175 | -0.973 | 1.067 | -0.050 | 0.070 | 0.000 |
| `dynamic_risk_sizing` | -0.385 | -0.211 | -0.611 | 0.864 | -0.062 | 0.043 | 0.032 |
| `capital_preservation` | -0.409 | -0.463 | -0.453 | 0.691 | -0.095 | 0.043 | 0.048 |
| `garch_sizing` | -0.411 | 0.027 | -0.810 | 0.923 | -0.050 | 0.048 | 0.005 |
| `daily_loss_aware` | -0.439 | -0.159 | -0.791 | 0.951 | -0.055 | 0.027 | 0.016 |
| `max_dd_aware` | -0.505 | -0.388 | -0.654 | 0.946 | -0.073 | 0.037 | 0.032 |

## Filters and overlays against the model they modify

Δ is the change in net Sharpe versus the unfiltered base, overall and in each half. A filter that helps should help in both. The trade filters' base, a 20-day Donchian breakout with a 2-ATR stop, nets -0.44; the filters that stop taking trades after a run of losses (profit factor, expectancy, risk of ruin) improve it in both halves, largely by trading less, and still leave it below zero.

| Filter | Base | Net | Δ | Δ dev | Δ hold |
|---|---|---|---|---|---|
| `pf_filter` | trade_base | -0.02 | 0.42 | 0.18 | 0.72 |
| `expectancy_filter` | trade_base | -0.06 | 0.38 | 0.13 | 0.63 |
| `risk_of_ruin` | trade_base | -0.13 | 0.31 | 0.08 | 0.24 |
| `r_multiple_dist` | trade_base | -0.19 | 0.25 | 0.02 | 0.32 |
| `cond_expectancy` | trade_base | -0.23 | 0.21 | 0.01 | 0.29 |
| `cond_r_multiple` | trade_base | -0.31 | 0.13 | -0.02 | 0.17 |
| `prob_payoff` | trade_base | -0.34 | 0.10 | -0.04 | 0.08 |
| `mae_model` | trade_base | -0.36 | 0.08 | 0.03 | 0.11 |
| `inverse_vol_selection` | trend_follow | -0.19 | 0.06 | -0.07 | 0.23 |
| `es_filter` | trend_follow | -0.25 | 0.01 | 0.10 | -0.16 |
| `var_filter` | trend_follow | -0.26 | -0.01 | 0.02 | -0.11 |
| `drawdown_regime` | trend_follow | -0.30 | -0.05 | -0.53 | 0.67 |
| `frac_kelly_filter` | trend_follow | -0.32 | -0.07 | -0.31 | 0.21 |
| `mfe_model` | trade_base | -0.51 | -0.07 | -0.10 | -0.09 |
| `kelly_filter` | trend_follow | -0.34 | -0.08 | -0.27 | 0.11 |
| `drawdown_aware_signal` | trend_follow | -0.39 | -0.14 | -0.61 | 0.50 |

## Not testable on this data

- **CFD Futures-to-CFD Lead/Lag Model**: needs exchange futures quotes alongside the CFD's; a CFD is priced off the future, so the lag is the broker's, sub-second
- **CFD Spot-to-Futures Lead/Lag Model**: needs both spot and futures series
- **CFD Index-to-Component Lead/Lag Model**: needs index constituents' prices
- **Sector Relative Strength**: no sector CFDs in the data
- **Dispersion Trading**: needs index and single-stock options or constituents
- **Correlation Trading**: needs options or correlation swaps; the CFD-tradable forms are the correlation spread/breakout/reversion rows
- **Latency-Based Lead-Lag**: sub-second; not reachable through retail CFD execution or 1-minute data
- **Earnings CFD Reaction Model**: no single-stock CFDs or earnings calendar in the data
- **Earnings Drift Model**: as above
- **Post-Earnings Announcement Drift**: as above
- **News Sentiment Quant Model**: no news feed or text
- **NLP Sentiment Model**: no news feed or text
- **Positioning-Based Entry**: no positioning data reachable (CFTC and broker feeds blocked here)
- **COT Positioning Model**: CFTC COT not reachable from this environment
- **Retail Positioning Contrarian Model**: broker client-positioning history not available
- **Broker Sentiment Model**: as above
- **Long/Short Ratio Model**: as above
- **Commitment-of-Traders Extremes**: CFTC COT not reachable
- **Positioning Z-Score**: no positioning data
- **Positioning Momentum**: no positioning data
- **Bid/Ask Imbalance Model**: no order book
- **Spread-Change Model**: no bid/ask history in this data

## Caveats


* OANDA's quotes are one broker's mid prices. Costs are charged as typical
  retail spreads per instrument, not the spread actually quoted at each
  minute, so the cost of trading through news and the rollover hour is
  understated.
* Commodity CFDs roll between futures contracts; any roll gaps in OANDA's
  series are in the returns.
* Stops and targets in the trade-level filters (MAE/MFE, expectancy,
  R-multiple) are tracked on daily highs and lows, but the engine books the
  exit at that day's close.
* Proxies stand in where the real input is not in the data (no consensus
  forecasts, positioning, order book, TIPS or sector CFDs); those rows say
  so, and a proxy failing is weaker evidence than the real input failing.
* Intraday Sharpe ratios are very negative for many hourly and 15-minute
  models because they trade 25 instruments every bar against a spread;
  the gross column shows what the signal itself was worth.


## Every model on the list

Sharpe ratios are annualised from daily net returns. *Gross* is before any cost, *no-fin* after spread and slippage but before overnight financing. *Dev* is 2005-2012, *hold* 2013-May 2020. *α t* is the Newey-West t of the return left after regressing on a long-only holding of the same instruments.

| # | Model | Test | Net | Gross | No-fin | Dev | Hold | t | α t | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Statistical Mean Reversion | `stat_mr` | -0.87 | -0.05 | -0.49 | -0.75 | -1.02 | -3.8 | -4.1 | no edge (negative even before costs) |
| 2 | Z-Score Mean Reversion | `z_mr` | -0.51 | -0.06 | -0.20 | -0.58 | -0.43 | -2.2 | -2.4 | no edge (negative even before costs) |
| 3 | Bollinger Z-Score Reversion | `boll_z` | -0.48 | -0.02 | -0.18 | -0.51 | -0.44 | -2.1 | -2.2 | no edge (negative even before costs) |
| 4 | VWAP Deviation Reversion | `vwap_dev` | -11.49 | 0.28 | -11.35 | -8.83 | -15.98 | -48.1 | -48.3 | no edge (positive before costs, not significant) |
| 5 | Standard Deviation Band Reversion | `sd_band` | -0.44 | -0.18 | -0.24 | -0.30 | -0.60 | -2.0 | -2.1 | no edge (negative even before costs) |
| 6 | Ornstein-Uhlenbeck Mean Reversion | `ou_mr` | -0.72 | -0.35 | -0.42 | -0.60 | -0.88 | -3.1 | -3.1 | no edge (negative even before costs) |
| 7 | Half-Life Mean Reversion | `halflife_mr` | -0.57 | -0.03 | -0.26 | -0.51 | -0.64 | -2.4 | -2.4 | no edge (negative even before costs) |
| 8 | Cointegration Pair Trading | `coint_pairs` | -0.78 | -0.09 | -0.17 | -0.52 | -1.07 | -3.0 | -3.1 | no edge (negative even before costs) |
| 9 | Statistical Arbitrage | `stat_arb` | -0.78 | 0.21 | -0.15 | -0.98 | -0.59 | -3.1 | -3.1 | no edge (positive before costs, not significant) |
| 10 | Pairs Trading | `pairs_trading` | -1.08 | -0.38 | -0.46 | -0.79 | -1.36 | -4.6 | -4.6 | no edge (negative even before costs) |
| 11 | Relative-Value Trading | `relative_value` | -1.31 | 0.18 | -0.05 | -0.94 | -1.70 | -5.4 | -5.4 | no edge (positive before costs, not significant) |
| 12 | Cross-Asset Relative Value | `cross_asset_rv` | -1.12 | -0.69 | -0.74 | -0.64 | -1.66 | -4.3 | -4.3 | no edge (negative even before costs) |
| 13 | Beta-Neutral Pairs Trading | `beta_neutral_pairs` | -1.11 | -0.04 | -0.27 | -0.48 | -1.69 | -4.5 | -4.5 | no edge (negative even before costs) |
| 14 | Correlation Breakdown/Reversion | `corr_reversion` | -0.25 | 0.31 | 0.05 | 0.07 | -0.51 | -1.0 | -1.0 | no edge (positive before costs, not significant) |
| 15 | Spread Z-Score Entry | `spread_z` | -1.53 | -0.21 | -0.33 | -0.92 | -2.15 | -6.1 | -6.4 | no edge (negative even before costs) |
| 16 | Residual Z-Score Entry | `resid_z` | -0.52 | 0.07 | -0.07 | -0.36 | -0.67 | -2.0 | -2.0 | no edge (positive before costs, not significant) |
| 17 | Kalman Filter Pairs Trading | `kalman_pairs` | -0.39 | -0.20 | -0.35 | -0.26 | -0.70 | -1.4 | -1.4 | no edge (negative even before costs) |
| 18 | Dynamic Hedge-Ratio Pairs Trading | `dyn_hedge_pairs` | -1.65 | -0.24 | -0.46 | -1.74 | -1.55 | -6.6 | -7.2 | no edge (negative even before costs) |
| 19 | PCA Factor-Neutral Trading | `pca_neutral` | -1.09 | -0.15 | -0.48 | -1.07 | -1.11 | -4.2 | -4.2 | no edge (negative even before costs) |
| 20 | Factor Residual Mean Reversion | `factor_resid_mr` | -1.52 | 0.65 | -0.81 | -0.65 | -2.23 | -5.7 | -5.8 | real gross edge, eaten by costs |
| 21 | Cross-Sectional Mean Reversion | `cs_mr` | -1.10 | 0.34 | -0.31 | -0.84 | -1.35 | -4.5 | -5.0 | no edge (positive before costs, not significant) |
| 22 | Time-Series Mean Reversion | `ts_mr` | -0.87 | 0.03 | -0.56 | -0.52 | -1.22 | -3.7 | -4.0 | no edge (positive before costs, not significant) |
| 23 | Momentum Factor | `momentum_factor` | -0.17 | 0.32 | 0.29 | -0.38 | 0.01 | -0.7 | -0.7 | no edge (positive before costs, not significant) |
| 24 | Cross-Sectional Momentum | `cs_mom` | -0.61 | 0.40 | 0.19 | -0.40 | -0.84 | -2.5 | -2.6 | no edge (positive before costs, not significant) |
| 25 | Time-Series Momentum | `ts_mom` | -0.33 | 0.23 | 0.15 | -0.41 | -0.23 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 26 | Trend Following | `trend_follow` | -0.25 | 0.19 | 0.11 | 0.08 | -0.68 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 27 | Quantitative Breakout | `quant_breakout` | -0.36 | 0.07 | 0.02 | -0.22 | -0.54 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 28 | Volatility Breakout | `vol_breakout` | -0.39 | 0.10 | -0.18 | -0.32 | -0.47 | -1.7 | -1.8 | no edge (positive before costs, not significant) |
| 29 | Donchian Breakout | `donchian` | -0.53 | -0.02 | -0.13 | -0.29 | -0.83 | -2.3 | -2.4 | no edge (negative even before costs) |
| 30 | ATR Breakout | `atr_breakout` | -0.14 | 0.23 | 0.02 | -0.03 | -0.26 | -0.6 | -0.6 | no edge (positive before costs, not significant) |
| 31 | Volatility-Adjusted Breakout | `voladj_breakout` | -0.15 | 0.21 | 0.10 | -0.05 | -0.27 | -0.7 | -0.7 | no edge (positive before costs, not significant) |
| 32 | Range Expansion Breakout | `range_exp_breakout` | -0.07 | 0.29 | 0.09 | 0.10 | -0.26 | -0.3 | -0.3 | no edge (positive before costs, not significant) |
| 33 | Statistical Range Breakout | `stat_range_breakout` | -0.10 | 0.30 | 0.25 | 0.17 | -0.44 | -0.4 | -0.4 | no edge (positive before costs, not significant) |
| 34 | Volatility Compression → Expansion | `compress_expand` | -0.94 | -0.11 | -0.51 | -0.97 | -0.91 | -3.5 | -3.5 | no edge (negative even before costs) |
| 35 | Bollinger Band Squeeze | `bb_squeeze` | -0.57 | -0.16 | -0.30 | -0.37 | -0.79 | -2.4 | -2.5 | no edge (negative even before costs) |
| 36 | Keltner Channel Breakout | `keltner` | -0.24 | 0.13 | 0.02 | -0.16 | -0.34 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 37 | ATR Channel Breakout | `atr_channel` | -0.18 | 0.19 | 0.13 | -0.08 | -0.30 | -0.8 | -0.8 | no edge (positive before costs, not significant) |
| 38 | Historical Volatility Breakout | `hv_breakout` | 0.07 | 0.24 | 0.13 | 0.05 | 0.09 | 0.3 | 0.3 | positive both halves, not significant |
| 39 | Realized Volatility Breakout | `rv_breakout` | -0.59 | -0.10 | -0.39 | -0.65 | -0.53 | -2.6 | -2.6 | no edge (negative even before costs) |
| 40 | Volatility Regime Switching | `vol_regime_switch` | -0.37 | 0.46 | 0.18 | -0.12 | -0.71 | -1.5 | -1.6 | no edge (positive before costs, not significant) |
| 41 | Volatility Targeting | `vol_targeting` | 0.01 | 0.48 | 0.48 | 0.31 | -0.34 | 0.0 | -1.8 | positive overall, fails one half |
| 42 | Regime-Based Trend Following | `regime_trend` | -0.35 | 0.04 | -0.16 | -0.52 | -0.16 | -1.4 | -1.4 | no edge (positive before costs, not significant) |
| 43 | Markov Regime-Switching | `markov_switch` | -0.32 | 0.30 | 0.19 | -0.13 | -0.47 | -1.2 | -1.1 | no edge (positive before costs, not significant) |
| 44 | Hidden Markov Model Entry | `hmm_entry` | -0.30 | 0.16 | -0.03 | -0.48 | -0.15 | -1.2 | -1.1 | no edge (positive before costs, not significant) |
| 45 | Bayesian Regime Detection | `bayes_regime` | -0.64 | 0.02 | -0.23 | -0.62 | -0.66 | -2.7 | -2.8 | no edge (positive before costs, not significant) |
| 46 | Change-Point Detection | `change_point` | -0.11 | 0.02 | -0.01 | -0.24 | 0.01 | -0.5 | -0.6 | no edge (positive before costs, not significant) |
| 47 | Structural Break Trading | `structural_break` | -0.72 | -0.67 | -0.71 | -0.68 | -0.79 | -2.8 | -2.9 | no edge (negative even before costs) |
| 48 | Autocorrelation-Based Entry | `autocorr_entry` | -2.14 | 0.08 | -1.67 | -1.68 | -2.68 | -8.5 | -8.5 | no edge (positive before costs, not significant) |
| 49 | Serial-Correlation Momentum | `serial_mom` | -1.45 | -0.31 | -1.22 | -1.49 | -1.43 | -5.7 | -5.7 | no edge (negative even before costs) |
| 50 | Serial-Correlation Reversion | `serial_rev` | -1.44 | 0.28 | -1.07 | -1.00 | -1.97 | -5.9 | -5.9 | no edge (positive before costs, not significant) |
| 51 | AR(1) Forecast Entry | `ar1_fcst` | -0.86 | 0.47 | -0.44 | -0.60 | -1.17 | -3.2 | -3.4 | no edge (positive before costs, not significant) |
| 52 | ARIMA Forecast Entry | `arima_fcst` | -1.70 | 0.34 | -1.11 | -0.63 | -2.55 | -6.0 | -6.3 | no edge (positive before costs, not significant) |
| 53 | Exponential Smoothing Forecast | `exp_smooth` | -0.52 | 0.05 | -0.06 | -0.19 | -0.95 | -2.2 | -2.3 | no edge (positive before costs, not significant) |
| 54 | Kalman Filter Trend | `kalman_trend` | -0.39 | 0.17 | 0.07 | -0.06 | -0.82 | -1.6 | -1.7 | no edge (positive before costs, not significant) |
| 55 | Kalman Filter Mean Reversion | `kalman_mr` | -0.35 | -0.07 | -0.19 | 0.12 | -0.81 | -1.4 | -1.6 | no edge (negative even before costs) |
| 56 | State-Space Trend Model | `state_space_trend` | -0.93 | -0.31 | -0.37 | -1.14 | -0.78 | -3.2 | -3.4 | no edge (negative even before costs) |
| 57 | Linear Regression Forecast | `linreg_fcst` | -1.10 | -0.02 | -0.63 | -0.59 | -1.71 | -4.7 | -4.8 | no edge (negative even before costs) |
| 58 | Polynomial Regression Forecast | `poly_fcst` | -1.30 | -0.18 | -0.81 | -1.02 | -1.61 | -5.2 | -5.2 | no edge (negative even before costs) |
| 59 | Rolling Regression Entry | `roll_reg_entry` | -1.33 | 0.20 | -0.76 | -0.58 | -2.07 | -4.8 | -5.1 | no edge (positive before costs, not significant) |
| 60 | Regression-to-Mean Entry | `reg_to_mean` | -0.52 | -0.18 | -0.25 | -0.39 | -0.66 | -2.1 | -2.4 | no edge (negative even before costs) |
| 61 | Regression Channel Breakout | `regression_channel_bo` | -0.44 | 0.00 | -0.14 | -0.32 | -0.58 | -1.9 | -2.0 | no edge (positive before costs, not significant) |
| 62 | Residual Reversion | `resid_rev` | -0.47 | 0.18 | -0.15 | -0.00 | -1.00 | -2.0 | -2.1 | no edge (positive before costs, not significant) |
| 63 | Forecast Error Reversion | `fcst_err_rev` | -0.39 | -0.11 | -0.23 | -0.03 | -0.75 | -1.7 | -1.9 | no edge (negative even before costs) |
| 64 | Prediction Interval Breakout | `pred_interval_bo` | -0.50 | -0.07 | -0.31 | -0.44 | -0.55 | -2.1 | -2.4 | no edge (negative even before costs) |
| 65 | Quantile Regression Entry | `quantile_reg` | -0.44 | -0.21 | -0.34 | -0.73 | -0.14 | -1.6 | -1.6 | no edge (negative even before costs) |
| 66 | Percentile-Based Entry | `pct_entry` | -0.35 | 0.48 | -0.23 | -0.15 | -0.51 | -1.5 | -1.5 | real gross edge, eaten by costs |
| 67 | Empirical Distribution Entry | `empirical_dist` | -1.42 | -0.24 | -1.09 | -1.54 | -1.35 | -5.5 | -7.4 | no edge (negative even before costs) |
| 68 | Historical Percentile Reversion | `hist_pct_rev` | -0.58 | -0.27 | -0.35 | -0.67 | -0.47 | -2.4 | -2.4 | no edge (negative even before costs) |
| 69 | Volatility-Scaled Entry | `vol_scaled_entry` | -0.30 | 0.14 | 0.04 | -0.12 | -0.53 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 70 | ATR-Normalized Entry | `atr_norm_entry` | -0.30 | 0.13 | 0.06 | -0.06 | -0.60 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 71 | Risk-Adjusted Momentum Entry | `risk_adj_mom` | -0.30 | 0.15 | 0.08 | -0.42 | -0.17 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 72 | Sharpe-Ratio Momentum Filter | `sharpe_mom_filter` | -0.22 | 0.30 | 0.16 | 0.05 | -0.54 | -0.9 | -0.9 | no edge (positive before costs, not significant) |
| 73 | Sortino-Ratio Momentum Filter | `sortino_mom_filter` | -0.24 | 0.28 | 0.14 | 0.04 | -0.58 | -0.9 | -1.0 | no edge (positive before costs, not significant) |
| 74 | Momentum + Volatility Filter | `mom_vol_filter` | -0.40 | 0.18 | 0.04 | -0.45 | -0.34 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 75 | Trend + Volatility Regime Filter | `trend_volregime` | -0.48 | 0.08 | 0.00 | -0.48 | -0.48 | -1.8 | -1.7 | no edge (positive before costs, not significant) |
| 76 | Trend Strength Quant Model | `trend_strength` | -0.11 | 0.26 | 0.20 | -0.05 | -0.18 | -0.4 | -0.4 | no edge (positive before costs, not significant) |
| 77 | ADX Quant Filter | `adx_filter` | -0.19 | 0.18 | 0.08 | 0.11 | -0.52 | -0.8 | -0.9 | no edge (positive before costs, not significant) |
| 78 | Slope-Based Trend Entry | `slope_entry` | -0.42 | 0.08 | -0.12 | -0.20 | -0.69 | -1.9 | -1.9 | no edge (positive before costs, not significant) |
| 79 | Moving-Average Slope Model | `ma_slope` | -0.59 | -0.04 | -0.11 | -0.33 | -0.91 | -2.4 | -2.4 | no edge (negative even before costs) |
| 80 | Linear Regression Slope Entry | `linreg_slope` | -0.38 | 0.10 | 0.03 | -0.21 | -0.60 | -1.6 | -1.6 | no edge (positive before costs, not significant) |
| 81 | Multi-Moving-Average Quant Model | `multi_ma` | -0.46 | 0.20 | -0.06 | -0.16 | -0.82 | -1.9 | -2.0 | no edge (positive before costs, not significant) |
| 82 | Moving-Average Distance Model | `ma_dist` | -0.55 | -0.19 | -0.25 | -0.55 | -0.55 | -2.3 | -2.4 | no edge (negative even before costs) |
| 83 | EMA Distance Z-Score | `ema_dist_z` | -0.53 | -0.27 | -0.34 | -0.41 | -0.66 | -2.3 | -2.5 | no edge (negative even before costs) |
| 84 | Price-to-VWAP Z-Score | `price_vwap_z` | -0.55 | -0.29 | -0.37 | -0.43 | -0.67 | -2.3 | -2.6 | no edge (negative even before costs) |
| 85 | Price-to-MA Z-Score | `price_ma_z` | -0.35 | -0.15 | -0.21 | -0.34 | -0.37 | -1.5 | -1.5 | no edge (negative even before costs) |
| 86 | MA Spread Z-Score | `ma_spread_z` | -0.18 | 0.19 | 0.15 | 0.23 | -0.66 | -0.7 | -0.8 | no edge (positive before costs, not significant) |
| 87 | MACD Quantitative Signal | `macd` | -0.49 | 0.24 | -0.01 | -0.32 | -0.68 | -2.0 | -2.1 | no edge (positive before costs, not significant) |
| 88 | RSI Quantitative Signal | `rsi_quant` | -0.75 | 0.09 | -0.45 | -0.29 | -1.25 | -3.3 | -3.4 | no edge (positive before costs, not significant) |
| 89 | Stochastic Quantitative Signal | `stoch_quant` | -0.57 | 0.03 | -0.11 | -0.68 | -0.43 | -2.4 | -2.4 | no edge (positive before costs, not significant) |
| 90 | Oscillator Percentile Entry | `osc_pct` | -0.48 | -0.16 | -0.24 | -0.57 | -0.39 | -2.0 | -2.1 | no edge (negative even before costs) |
| 91 | Momentum Oscillator Composite | `osc_composite` | -0.39 | -0.08 | -0.17 | -0.46 | -0.32 | -1.6 | -1.8 | no edge (negative even before costs) |
| 92 | Multi-Factor Technical Model | `multi_factor_tech` | -0.94 | 0.60 | -0.38 | -0.09 | -1.93 | -3.6 | -3.7 | real gross edge, eaten by costs |
| 93 | Technical Factor Model | `tech_factor` | -0.35 | 0.35 | 0.13 | 0.02 | -0.69 | -1.4 | -1.5 | no edge (positive before costs, not significant) |
| 94 | Factor Score Model | `factor_score` | -0.60 | -0.01 | -0.27 | -0.71 | -0.54 | -2.2 | -2.2 | no edge (negative even before costs) |
| 95 | Weighted Confluence Score | `confluence` | -0.61 | -0.01 | -0.22 | -0.37 | -0.90 | -2.6 | -2.7 | no edge (negative even before costs) |
| 96 | Bayesian Signal Aggregation | `bayes_agg` | -1.12 | -0.00 | -0.69 | -0.95 | -1.29 | -4.3 | -6.0 | no edge (negative even before costs) |
| 97 | Logistic Regression Direction Model | `ml_logit` | -1.49 | 0.19 | -0.99 | -0.81 | -1.99 | -5.4 | -5.9 | no edge (positive before costs, not significant) |
| 98 | Linear Classification Model | `ml_lda` | -1.54 | 0.13 | -1.04 | -0.96 | -1.97 | -5.6 | -6.1 | no edge (positive before costs, not significant) |
| 99 | Random Forest Direction Model | `ml_rf` | -1.00 | 0.47 | -0.51 | -0.52 | -1.33 | -3.7 | -4.3 | no edge (positive before costs, not significant) |
| 100 | Gradient Boosting Direction Model | `ml_gbm` | -1.31 | 0.64 | -0.76 | -0.41 | -1.93 | -4.8 | -5.0 | real gross edge, eaten by costs |
| 101 | XGBoost Signal Model | `ml_xgb` | -1.39 | 0.63 | -0.82 | -0.67 | -1.87 | -5.2 | -5.2 | real gross edge, eaten by costs |
| 102 | Support Vector Machine Entry | `ml_svm` | -0.54 | 1.22 | 0.01 | 0.11 | -1.02 | -1.9 | -1.7 | real gross edge, eaten by costs |
| 103 | k-Nearest Neighbors Entry | `ml_knn` | -0.76 | 1.28 | -0.18 | -0.33 | -1.11 | -2.8 | -2.8 | real gross edge, eaten by costs |
| 104 | Neural-Network Direction Model | `ml_mlp` | -1.34 | 0.81 | -0.75 | -0.64 | -1.89 | -4.7 | -4.8 | real gross edge, eaten by costs |
| 105 | LSTM Sequence Model | `ml_lstm` | -1.03 | 0.14 | -0.56 | -0.96 | -1.06 | -3.7 | -4.4 | no edge (positive before costs, not significant) |
| 106 | Transformer Time-Series Model | `ml_transformer` | -1.28 | 0.38 | -0.75 | -1.05 | -1.41 | -4.4 | -4.8 | no edge (positive before costs, not significant) |
| 107 | Reinforcement-Learning Entry | `rl_qlearn` | -0.31 | 0.08 | -0.22 | -0.14 | -0.76 | -1.2 | -1.0 | no edge (positive before costs, not significant) |
| 108 | Ensemble Model Entry | `ml_ensemble` | -1.21 | 0.48 | -0.70 | -0.56 | -1.67 | -4.4 | -4.8 | no edge (positive before costs, not significant) |
| 109 | Stacked Ensemble Entry | `ml_stacked` | -1.33 | 0.90 | -0.71 | -0.95 | -1.48 | -4.6 | -4.7 | real gross edge, eaten by costs |
| 110 | Voting Classifier Entry | `ml_voting` | -0.89 | 0.75 | -0.38 | -0.19 | -1.38 | -3.2 | -3.4 | real gross edge, eaten by costs |
| 111 | Probability Threshold Entry | `ml_prob_threshold` | -0.46 | 0.41 | -0.30 | -0.55 | -0.39 | -1.8 | -1.6 | no edge (positive before costs, not significant) |
| 112 | Expected-Value Threshold Entry | `ml_ev_threshold` | -0.90 | 0.24 | -0.64 | -0.63 | -1.33 | -3.3 | -3.2 | no edge (positive before costs, not significant) |
| 113 | Conditional Probability Entry | `cond_prob` | -1.04 | 0.15 | -0.71 | -0.81 | -1.25 | -4.1 | -5.3 | no edge (positive before costs, not significant) |
| 114 | Bayesian Probability Update | `bayes_prob_update` | -0.35 | 0.05 | -0.06 | -0.39 | -0.30 | -1.4 | -2.0 | no edge (positive before costs, not significant) |
| 115 | Monte Carlo Entry Model | `monte_carlo_entry` | -0.13 | 0.29 | 0.22 | -0.28 | 0.07 | -0.5 | -0.5 | no edge (positive before costs, not significant) |
| 116 | Monte Carlo Path Forecast | `mc_path_fcst` | -0.34 | 0.26 | 0.07 | -0.28 | -0.41 | -1.4 | -1.4 | no edge (positive before costs, not significant) |
| 117 | Bootstrap Distribution Entry | `bootstrap_dist` | -0.11 | 0.15 | 0.06 | -0.02 | -0.22 | -0.5 | -0.5 | no edge (positive before costs, not significant) |
| 118 | Historical Scenario Matching | `scenario_match` | -1.54 | -0.31 | -0.74 | -1.77 | -1.41 | -5.6 | -5.9 | no edge (negative even before costs) |
| 119 | Analog Pattern Matching | `analog_match` | -1.15 | 0.12 | -0.37 | -0.78 | -1.42 | -4.5 | -4.4 | no edge (positive before costs, not significant) |
| 120 | Nearest-Neighbor Pattern Matching | `nn_pattern` | -1.11 | 0.05 | -0.36 | -0.80 | -1.33 | -4.2 | -4.2 | no edge (positive before costs, not significant) |
| 121 | DTW Pattern Matching | `dtw_pattern` | -1.53 | -0.48 | -0.74 | -1.18 | -1.76 | -5.6 | -5.7 | no edge (negative even before costs) |
| 122 | Fractal Pattern Model | `fractal_breakout` | -0.85 | -0.17 | -0.34 | -0.64 | -1.09 | -3.6 | -3.8 | no edge (negative even before costs) |
| 123 | Hurst Exponent Trend/Reversion Model | `hurst` | -1.08 | -0.32 | -0.72 | -0.88 | -1.30 | -4.3 | -4.5 | no edge (negative even before costs) |
| 124 | Entropy-Based Market Regime Model | `entropy_regime` | -0.19 | 0.49 | 0.18 | 0.14 | -0.47 | -0.7 | -0.8 | no edge (positive before costs, not significant) |
| 125 | Market Efficiency Ratio Model | `efficiency_ratio` | -0.56 | -0.03 | -0.35 | -0.56 | -0.56 | -2.5 | -2.5 | no edge (negative even before costs) |
| 126 | Variance-Ratio Model | `variance_ratio` | -1.46 | -0.39 | -1.01 | -0.77 | -2.42 | -5.7 | -5.7 | no edge (negative even before costs) |
| 127 | Variance-Ratio Mean Reversion | `vr_mr` | -1.12 | -0.24 | -0.74 | -0.54 | -2.00 | -4.4 | -4.5 | no edge (negative even before costs) |
| 128 | Fractal Dimension Model | `fractal_dim` | -0.39 | 0.36 | 0.07 | -0.19 | -0.64 | -1.6 | -1.7 | no edge (positive before costs, not significant) |
| 129 | Long-Memory Model | `long_memory` | -1.27 | -0.30 | -0.71 | -0.98 | -1.51 | -4.7 | -4.7 | no edge (negative even before costs) |
| 130 | Volatility Clustering Model | `vol_clustering` | -0.35 | 0.15 | 0.04 | 0.04 | -0.85 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 131 | GARCH Volatility Model | `garch_sizing` | -0.41 | 0.02 | -0.06 | 0.03 | -0.81 | -1.6 | -1.7 | no edge (positive before costs, not significant) |
| 132 | EGARCH Volatility Model | `egarch_sizing` | -0.30 | 0.11 | 0.03 | 0.09 | -0.64 | -1.1 | -1.2 | no edge (positive before costs, not significant) |
| 133 | GARCH Forecast Breakout | `garch_breakout` | -0.45 | 0.00 | -0.26 | -0.32 | -0.54 | -1.7 | -2.1 | no edge (positive before costs, not significant) |
| 134 | Volatility Forecast Reversion | `vol_fcst_rev` | 0.01 | 0.08 | 0.07 | -0.27 | 0.21 | 0.0 | -0.3 | positive overall, fails one half |
| 135 | Volatility Shock Model | `vol_shock` | -0.38 | -0.26 | -0.33 | -0.21 | -0.52 | -1.6 | -1.8 | no edge (negative even before costs) |
| 136 | Volatility-of-Volatility Model | `volofvol` | -0.51 | 0.05 | -0.06 | -0.90 | -0.18 | -1.9 | -1.9 | no edge (positive before costs, not significant) |
| 137 | Range-Based Volatility Model | `range_vol_sizing` | -0.28 | 0.16 | 0.09 | 0.06 | -0.71 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 138 | Parkinson Volatility Model | `parkinson_sizing` | -0.29 | 0.16 | 0.08 | 0.06 | -0.74 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 139 | Garman-Klass Volatility Model | `gk_sizing` | -0.28 | 0.17 | 0.10 | 0.05 | -0.70 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 140 | Yang-Zhang Volatility Model | `yz_sizing` | -0.29 | 0.16 | 0.08 | 0.04 | -0.72 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 141 | Volatility Risk Premium Model | `vrp` | 0.33 | 0.51 | 0.49 | 0.32 | 0.35 | 1.4 | 0.2 | positive both halves, not significant (VIX vs realised S&P vol) |
| 142 | ATR Regime Model | `atr_regime` | -0.35 | 0.07 | -0.15 | -0.24 | -0.46 | -1.5 | -1.6 | no edge (positive before costs, not significant) |
| 143 | Implied-vs-Realized Volatility Model | `iv_rv` | 0.46 | 0.53 | 0.52 | 0.49 | 0.45 | 2.0 | 1.4 | positive both halves, not significant (VIX is the only implied vol available) |
| 144 | CFD Spread-Normalized Entry | `spread_norm_entry` | -1.16 | -0.31 | -1.16 | -1.17 | -1.19 | -5.4 | -5.4 | no edge (negative even before costs) |
| 145 | CFD Overnight-Financing-Aware Entry | `financing_aware` | -0.30 | 0.05 | -0.06 | -0.01 | -0.65 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 146 | CFD Cost-Adjusted Entry | `cost_adj_entry` | -3.09 | -0.23 | -3.08 | -2.80 | -3.61 | -14.0 | -14.0 | no edge (negative even before costs) |
| 147 | CFD Spread Expansion Filter | `spread_exp_filter` | -3.44 | -0.35 | -3.43 | -2.96 | -4.26 | -15.4 | -15.5 | no edge (negative even before costs) (no quote history: known wide-spread hours used instead) |
| 148 | CFD Execution-Slippage Filter | `slippage_filter` | -3.44 | -0.25 | -3.43 | -3.13 | -3.98 | -15.5 | -15.6 | no edge (negative even before costs) |
| 149 | CFD Liquidity-Adjusted Entry | `liquidity_adj` | -2.82 | -0.14 | -2.82 | -2.51 | -3.35 | -12.7 | -12.8 | no edge (negative even before costs) |
| 150 | CFD Volatility/Spread Ratio Model | `vol_spread_ratio` | -2.49 | -0.24 | -2.48 | -2.20 | -2.97 | -11.2 | -11.2 | no edge (negative even before costs) |
| 151 | CFD Expected-Move Model | `expected_move_model` | -2.71 | -0.17 | -2.70 | -2.38 | -3.41 | -12.1 | -12.2 | no edge (negative even before costs) |
| 152 | CFD Session-Adjusted Statistical Model | `session_adj_stat` | -2.44 | 0.44 | -2.37 | -2.53 | -2.34 | -10.4 | -10.4 | no edge (positive before costs, not significant) |
| 153 | CFD Gap-Adjusted Model | `gap_adjusted` | -0.43 | -0.18 | -0.26 | -0.46 | -0.41 | -1.8 | -1.5 | no edge (negative even before costs) |
| 154 | CFD Overnight Gap Reversion | `overnight_gap_rev` | -1.04 | -0.50 | -1.04 | -0.93 | -1.19 | -4.5 | -4.5 | no edge (negative even before costs) |
| 155 | CFD Weekend Gap Model | `weekend_gap` | -1.20 | 0.57 | -1.14 | -0.97 | -1.49 | -5.3 | -5.4 | real gross edge, eaten by costs |
| 156 | CFD Opening Auction Proxy Model | `opening_auction_proxy` | -0.24 | 0.68 | -0.23 | -0.17 | -0.35 | -1.1 | -1.1 | real gross edge, eaten by costs |
| 157 | CFD Cash-Session Momentum | `cash_session_mom` | -1.93 | 0.63 | -1.93 | -0.95 | -3.30 | -7.6 | -7.6 | real gross edge, eaten by costs |
| 158 | CFD Futures-to-CFD Lead/Lag Model | — | | | | | | | | not testable: needs exchange futures quotes alongside the CFD's; a CFD is priced off the future, so the lag is the broker's, sub-second |
| 159 | CFD Spot-to-Futures Lead/Lag Model | — | | | | | | | | not testable: needs both spot and futures series |
| 160 | CFD Index-to-Component Lead/Lag Model | — | | | | | | | | not testable: needs index constituents' prices |
| 161 | CFD Cross-Market Lead/Lag | `cross_market_ll` | -3.52 | 0.79 | -3.44 | -2.89 | -4.29 | -14.8 | -14.8 | real gross edge, eaten by costs |
| 162 | FX Lead/Lag Model | `fx_ll` | -21.51 | -0.62 | -21.50 | -18.41 | -27.55 | -70.1 | -70.1 | no edge (negative even before costs) |
| 163 | Gold/USD Lead-Lag Model | `gold_usd_ll` | -6.16 | 0.01 | -6.14 | -4.73 | -7.87 | -23.1 | -23.1 | no edge (positive before costs, not significant) |
| 164 | Oil/USD Lead-Lag Model | `oil_usd_ll` | -6.86 | 0.72 | -6.85 | -5.28 | -8.99 | -26.7 | -26.7 | real gross edge, eaten by costs |
| 165 | Bond/Yield Lead-Lag Model | `bond_yield_ll` | -7.90 | 0.10 | -7.88 | -6.85 | -9.33 | -31.6 | -31.6 | no edge (positive before costs, not significant) |
| 166 | Equity Index/FX Lead-Lag Model | `eq_fx_ll` | -6.19 | 0.98 | -6.16 | -5.36 | -7.27 | -23.6 | -23.6 | real gross edge, eaten by costs |
| 167 | Intermarket Confirmation Model | `intermarket_confirm` | -0.35 | -0.03 | -0.07 | -0.08 | -0.65 | -1.3 | -1.3 | no edge (negative even before costs) |
| 168 | Macro Factor Model | `macro_factor` | -0.25 | 0.30 | 0.13 | 0.06 | -0.54 | -0.9 | -1.0 | no edge (positive before costs, not significant) |
| 169 | Dollar-Index Factor Model | `dollar_factor` | -0.31 | 0.22 | 0.06 | -0.18 | -0.48 | -1.2 | -1.3 | no edge (positive before costs, not significant) (synthetic dollar from four USD pairs) |
| 170 | Rates-Differential Model | `rates_diff` | 0.43 | 0.75 | 0.58 | 0.66 | 0.16 | 1.7 | 1.7 | positive both halves, not significant (proxy: bond CFD prices, not yields) |
| 171 | Yield-Curve Signal Model | `yield_curve` | 0.16 | 0.30 | 0.26 | 0.12 | 0.20 | 0.6 | 0.4 | positive both halves, not significant (proxy: 2s10s from bond CFD prices) |
| 172 | Real-Yield Signal Model | `real_yield_proxy` | -0.12 | 0.13 | 0.04 | -0.26 | 0.01 | -0.4 | -0.7 | no edge (positive before costs, not significant) (proxy: no inflation-linked data; nominal Treasury prices used) |
| 173 | Commodity-Currency Model | `commodity_ccy` | -0.27 | 0.02 | -0.12 | -0.13 | -0.42 | -1.1 | -1.0 | no edge (positive before costs, not significant) |
| 174 | Risk-On/Risk-Off Factor Model | `risk_on_off` | -0.31 | 0.25 | 0.03 | 0.34 | -0.96 | -1.2 | -1.3 | no edge (positive before costs, not significant) |
| 175 | Equity-Bond Correlation Regime Model | `eq_bond_corr` | 0.11 | 0.70 | 0.68 | 0.03 | 0.20 | 0.5 | -0.1 | positive both halves, not significant |
| 176 | Dollar-Gold Relationship Model | `dollar_gold` | 0.14 | 0.30 | 0.26 | 0.38 | -0.06 | 0.5 | 0.4 | positive overall, fails one half |
| 177 | Oil-Currency Relationship Model | `oil_ccy` | -0.11 | -0.02 | -0.05 | 0.03 | -0.26 | -0.4 | -0.4 | no edge (negative even before costs) |
| 178 | Relative Strength Quant Model | `rs_quant` | -2.02 | -0.41 | -1.11 | -2.26 | -1.78 | -8.2 | -8.2 | no edge (negative even before costs) |
| 179 | Cross-Asset Relative Strength | `cross_asset_rs` | -0.15 | 0.21 | 0.18 | 0.06 | -0.35 | -0.6 | -0.6 | no edge (positive before costs, not significant) |
| 180 | Currency Strength Model | `ccy_strength` | -0.03 | 0.21 | 0.12 | 0.13 | -0.21 | -0.1 | -0.3 | no edge (positive before costs, not significant) |
| 181 | Sector Relative Strength | — | | | | | | | | not testable: no sector CFDs in the data |
| 182 | Index Relative Strength | `index_rs` | -0.42 | 0.10 | 0.01 | -0.23 | -0.63 | -1.8 | -1.9 | no edge (positive before costs, not significant) |
| 183 | Asset-Cluster Relative Strength | `cluster_rs` | -0.08 | 0.13 | 0.10 | 0.28 | -0.36 | -0.3 | -0.2 | no edge (positive before costs, not significant) |
| 184 | Dispersion Trading | — | | | | | | | | not testable: needs index and single-stock options or constituents |
| 185 | Correlation Trading | — | | | | | | | | not testable: needs options or correlation swaps; the CFD-tradable forms are the correlation spread/breakout/reversion rows |
| 186 | Correlation Spread Trading | `corr_spread` | -1.44 | -0.12 | -0.34 | -0.94 | -1.99 | -5.6 | -5.9 | no edge (negative even before costs) |
| 187 | Correlation Breakout | `corr_breakout` | -0.86 | -0.31 | -0.57 | -0.91 | -0.83 | -3.4 | -3.4 | no edge (negative even before costs) |
| 188 | Correlation Reversion | `corr_reversion` | -0.25 | 0.31 | 0.05 | 0.07 | -0.51 | -1.0 | -1.0 | no edge (positive before costs, not significant) (same test as Correlation Breakdown/Reversion) |
| 189 | Lead-Lag Arbitrage | `lead_lag_arb` | -5.96 | 2.42 | -5.95 | -8.42 | -3.59 | -17.7 | -17.7 | real gross edge, eaten by costs |
| 190 | Latency-Based Lead-Lag | — | | | | | | | | not testable: sub-second; not reachable through retail CFD execution or 1-minute data |
| 191 | Statistical Lead-Lag | `stat_ll` | -3.26 | 1.18 | -3.24 | -3.84 | -2.87 | -12.4 | -12.4 | real gross edge, eaten by costs |
| 192 | Cross-Sectional Ranking | `cs_rank` | -0.49 | 0.14 | -0.05 | -0.29 | -0.70 | -2.1 | -2.1 | no edge (positive before costs, not significant) |
| 193 | Percentile Ranking | `pct_rank` | -0.51 | 0.04 | -0.06 | -0.27 | -0.66 | -1.8 | -1.9 | no edge (positive before costs, not significant) |
| 194 | Momentum Ranking | `mom_rank` | -0.24 | 0.23 | 0.19 | -0.10 | -0.39 | -0.9 | -1.0 | no edge (positive before costs, not significant) |
| 195 | Volatility Ranking | `vol_rank` | -0.68 | -0.07 | -0.09 | -0.79 | -0.56 | -2.7 | -2.9 | no edge (negative even before costs) |
| 196 | Mean-Reversion Ranking | `mr_rank` | -0.76 | 0.06 | -0.33 | -0.57 | -0.97 | -3.3 | -3.4 | no edge (positive before costs, not significant) |
| 197 | Composite Ranking Model | `composite_rank` | -0.83 | -0.06 | -0.30 | -0.79 | -0.86 | -3.3 | -3.3 | no edge (negative even before costs) |
| 198 | Long/Short Ranking Model | `ls_rank` | -0.63 | 0.08 | -0.19 | -0.63 | -0.63 | -2.5 | -2.5 | no edge (positive before costs, not significant) |
| 199 | Market-Neutral Ranking Model | `mkt_neutral_rank` | -0.58 | 0.72 | 0.23 | -0.23 | -0.94 | -2.3 | -2.5 | real gross edge, eaten by costs |
| 200 | Beta-Adjusted Ranking | `beta_adj_rank` | -0.61 | 0.27 | 0.19 | -0.32 | -0.85 | -2.3 | -2.4 | no edge (positive before costs, not significant) |
| 201 | Volatility-Adjusted Ranking | `voladj_rank` | -0.28 | 0.23 | 0.18 | -0.13 | -0.43 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 202 | Risk-Parity Signal Allocation | `risk_parity_alloc` | -0.28 | 0.30 | 0.09 | 0.01 | -0.60 | -1.2 | -1.2 | no edge (positive before costs, not significant) |
| 203 | Inverse-Volatility Position Selection | `inverse_vol_selection` | -0.19 | 0.37 | 0.27 | 0.01 | -0.45 | -0.8 | -0.8 | no edge (positive before costs, not significant) |
| 204 | Kelly-Criterion Signal Filter | `kelly_filter` | -0.34 | 0.04 | -0.05 | -0.19 | -0.57 | -1.4 | -1.4 | no edge (positive before costs, not significant) |
| 205 | Fractional-Kelly Entry Filter | `frac_kelly_filter` | -0.32 | 0.05 | -0.05 | -0.23 | -0.46 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 206 | Expected-Shortfall Filter | `es_filter` | -0.25 | 0.24 | 0.15 | 0.18 | -0.84 | -1.0 | -1.0 | no edge (positive before costs, not significant) |
| 207 | VaR-Based Signal Filter | `var_filter` | -0.26 | 0.21 | 0.13 | 0.11 | -0.78 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 208 | Drawdown-Regime Filter | `drawdown_regime` | -0.30 | -0.09 | -0.14 | -0.45 | -0.01 | -1.3 | -1.3 | no edge (negative even before costs) |
| 209 | Maximum-Adverse-Excursion Model | `mae_model` | -0.36 | 0.14 | -0.01 | -0.17 | -0.64 | -1.6 | -1.6 | no edge (positive before costs, not significant) |
| 210 | Maximum-Favorable-Excursion Model | `mfe_model` | -0.51 | 0.16 | -0.12 | -0.30 | -0.83 | -2.2 | -2.2 | no edge (positive before costs, not significant) |
| 211 | R-Multiple Distribution Model | `r_multiple_dist` | -0.19 | 0.15 | 0.06 | -0.18 | -0.43 | -0.8 | -0.8 | no edge (positive before costs, not significant) |
| 212 | Expectancy-Based Entry Filter | `expectancy_filter` | -0.06 | 0.28 | 0.20 | -0.07 | -0.11 | -0.3 | -0.3 | no edge (positive before costs, not significant) |
| 213 | Profit-Factor Signal Filter | `pf_filter` | -0.02 | 0.32 | 0.23 | -0.02 | -0.02 | -0.1 | -0.1 | no edge (positive before costs, not significant) |
| 214 | Conditional Expectancy Model | `cond_expectancy` | -0.23 | 0.15 | 0.05 | -0.19 | -0.45 | -1.0 | -1.0 | no edge (positive before costs, not significant) |
| 215 | Time-of-Day Seasonality Model | `tod_season` | -21.99 | 2.99 | -21.88 | -16.08 | -33.82 | -49.5 | -49.5 | real gross edge, eaten by costs |
| 216 | Day-of-Week Seasonality | `dow_season` | -1.39 | 0.21 | -1.10 | -1.39 | -1.40 | -4.5 | -4.5 | no edge (positive before costs, not significant) |
| 217 | Month-of-Year Seasonality | `moy_season` | 0.55 | 0.88 | 0.80 | 0.67 | 0.46 | 2.0 | 2.2 | significant alone, not after 359 trials |
| 218 | Session Seasonality | `session_season` | -5.66 | 1.35 | -5.57 | -3.30 | -7.59 | -20.3 | -20.3 | real gross edge, eaten by costs |
| 219 | Intraday Seasonal Pattern | `intraday_pattern` | -24.09 | 2.19 | -23.86 | -18.62 | -32.64 | -66.1 | -66.1 | real gross edge, eaten by costs |
| 220 | Opening-Window Statistical Model | `opening_window` | -8.72 | 0.73 | -8.72 | -5.47 | -11.51 | -27.9 | -27.9 | real gross edge, eaten by costs |
| 221 | Closing-Window Statistical Model | `closing_window` | -7.06 | 0.45 | -7.05 | -6.01 | -8.35 | -22.3 | -22.3 | no edge (positive before costs, not significant) |
| 222 | London Session Statistical Model | `london_session` | -5.64 | 0.86 | -5.64 | -3.88 | -8.47 | -22.3 | -22.3 | real gross edge, eaten by costs |
| 223 | New York Session Statistical Model | `ny_session` | -3.66 | 0.08 | -3.50 | -3.00 | -4.46 | -15.0 | -15.0 | no edge (positive before costs, not significant) |
| 224 | Asia Session Statistical Model | `asia_session` | -4.80 | 0.46 | -4.80 | -4.29 | -5.40 | -18.3 | -18.4 | no edge (positive before costs, not significant) |
| 225 | Session-Transition Model | `session_transition` | -7.12 | 0.36 | -7.12 | -5.35 | -9.81 | -27.9 | -27.9 | no edge (positive before costs, not significant) |
| 226 | News-Event Statistical Model | `news_event_stat` | -0.29 | 0.34 | -0.28 | -0.01 | -0.52 | -1.1 | -1.1 | no edge (positive before costs, not significant) (release days found from the 08:30 volatility spike) |
| 227 | Pre-News Positioning Model | `pre_news` | -1.35 | 0.30 | -1.35 | -1.32 | -1.42 | -5.2 | -5.2 | no edge (positive before costs, not significant) (payrolls Fridays, known in advance) |
| 228 | Post-News Mean Reversion | `post_news_mr` | -0.77 | 0.52 | -0.76 | -0.24 | -1.40 | -3.2 | -3.2 | real gross edge, eaten by costs |
| 229 | Post-News Momentum | `post_news_mom` | -1.69 | -0.52 | -1.69 | -1.53 | -1.90 | -7.2 | -7.3 | no edge (negative even before costs) |
| 230 | Economic Surprise Model | `econ_surprise_proxy` | -1.34 | -0.71 | -1.33 | -1.13 | -1.66 | -5.8 | -5.8 | no edge (negative even before costs) (proxy: no consensus data; the release bar's sign is the surprise) |
| 231 | Surprise-Magnitude Model | `surprise_magnitude_proxy` | -0.56 | 0.44 | -0.56 | -0.34 | -0.92 | -2.4 | -2.4 | no edge (positive before costs, not significant) (proxy: release-bar size as the surprise size) |
| 232 | Event Volatility Model | `event_vol` | -1.80 | -0.38 | -1.80 | -1.58 | -2.08 | -7.6 | -7.6 | no edge (negative even before costs) |
| 233 | Event Drift Model | `event_drift` | -1.46 | -0.57 | -1.45 | -1.25 | -1.73 | -6.4 | -6.4 | no edge (negative even before costs) |
| 234 | CPI Reaction Model | `cpi_reaction_proxy` | -0.99 | -0.54 | -0.99 | -0.76 | -1.24 | -4.1 | -4.1 | no edge (negative even before costs) (proxy calendar: large 08:30 bars on the 10th-17th) |
| 235 | NFP Reaction Model | `nfp_reaction` | -0.89 | -0.12 | -0.89 | -0.86 | -0.96 | -3.9 | -3.9 | no edge (negative even before costs) (first-Friday calendar) |
| 236 | FOMC Reaction Model | `fomc_reaction` | -0.47 | 0.20 | -0.47 | -0.65 | -0.38 | -1.9 | -1.9 | no edge (positive before costs, not significant) (statement days found from the 14:00/14:15 spike) |
| 237 | Central-Bank Decision Model | `cb_decision` | -0.58 | 0.02 | -0.58 | -0.56 | -0.66 | -2.5 | -2.5 | no edge (positive before costs, not significant) (FOMC + ECB days found from their spikes) |
| 238 | Earnings CFD Reaction Model | — | | | | | | | | not testable: no single-stock CFDs or earnings calendar in the data |
| 239 | Earnings Drift Model | — | | | | | | | | not testable: as above |
| 240 | Post-Earnings Announcement Drift | — | | | | | | | | not testable: as above |
| 241 | Macro Event Regime Model | `macro_event_regime` | -0.40 | 0.14 | -0.04 | -0.03 | -0.89 | -1.7 | -1.7 | no edge (positive before costs, not significant) |
| 242 | News Sentiment Quant Model | — | | | | | | | | not testable: no news feed or text |
| 243 | NLP Sentiment Model | — | | | | | | | | not testable: no news feed or text |
| 244 | News Momentum Model | `post_news_mom` | -1.69 | -0.52 | -1.69 | -1.53 | -1.90 | -7.2 | -7.3 | no edge (negative even before costs) (news = scheduled releases; same test as Post-News Momentum) |
| 245 | News Reversion Model | `post_news_mr` | -0.77 | 0.52 | -0.76 | -0.24 | -1.40 | -3.2 | -3.2 | real gross edge, eaten by costs (same test as Post-News Mean Reversion) |
| 246 | Sentiment Extreme Reversion | `sentiment_extreme_proxy` | 0.28 | 0.36 | 0.35 | 0.03 | 0.49 | 1.1 | -0.3 | positive both halves, not significant (VIX as the sentiment gauge) |
| 247 | Sentiment-Momentum Divergence | `sentiment_mom_div_proxy` | 0.04 | 0.12 | 0.09 | 0.20 | -0.09 | 0.1 | 0.4 | positive overall, fails one half (VIX as the sentiment gauge) |
| 248 | Positioning-Based Entry | — | | | | | | | | not testable: no positioning data reachable (CFTC and broker feeds blocked here) |
| 249 | COT Positioning Model | — | | | | | | | | not testable: CFTC COT not reachable from this environment |
| 250 | Retail Positioning Contrarian Model | — | | | | | | | | not testable: broker client-positioning history not available |
| 251 | Broker Sentiment Model | — | | | | | | | | not testable: as above |
| 252 | Long/Short Ratio Model | — | | | | | | | | not testable: as above |
| 253 | Commitment-of-Traders Extremes | — | | | | | | | | not testable: CFTC COT not reachable |
| 254 | Positioning Z-Score | — | | | | | | | | not testable: no positioning data |
| 255 | Positioning Momentum | — | | | | | | | | not testable: no positioning data |
| 256 | Flow-Based Model | `flow_model` | -1.22 | -0.06 | -0.85 | -1.22 | -1.23 | -5.2 | -5.3 | no edge (negative even before costs) (tick volume x candle sign as the flow proxy) |
| 257 | Volume-Weighted Flow Model | `vw_flow` | -0.50 | 0.29 | -0.02 | -0.34 | -0.69 | -2.1 | -2.2 | no edge (positive before costs, not significant) |
| 258 | Cumulative Volume Delta Model | `cvd` | -9.46 | -0.39 | -9.32 | -8.15 | -11.32 | -39.7 | -39.7 | no edge (negative even before costs) (proxy: signed tick volume per hour (no trade-side data)) |
| 259 | Tick-Volume Model | `tick_volume` | -1.10 | -0.28 | -0.96 | -1.24 | -1.05 | -4.7 | -4.7 | no edge (negative even before costs) |
| 260 | Tick-Volume Anomaly | `tick_vol_anomaly` | -1.37 | 0.11 | -1.35 | -1.33 | -1.57 | -5.9 | -5.9 | no edge (positive before costs, not significant) |
| 261 | Relative-Volume Anomaly | `rel_vol_anomaly` | -4.63 | -0.05 | -4.51 | -4.50 | -5.02 | -17.9 | -17.9 | no edge (negative even before costs) |
| 262 | Volume Z-Score | `volume_z` | -0.43 | 0.13 | -0.34 | -0.14 | -0.65 | -1.9 | -2.0 | no edge (positive before costs, not significant) |
| 263 | Volume/Price Divergence | `vol_price_div` | -0.61 | 0.13 | -0.21 | -0.66 | -0.56 | -2.6 | -2.6 | no edge (positive before costs, not significant) |
| 264 | Price/Volume Regression | `price_vol_reg` | -2.61 | -0.44 | -2.12 | -2.43 | -2.79 | -10.1 | -10.2 | no edge (negative even before costs) |
| 265 | Volume-Volatility Relationship | `vol_volatility` | -0.57 | -0.07 | -0.42 | -0.44 | -0.72 | -2.2 | -2.2 | no edge (negative even before costs) |
| 266 | Order-Flow Imbalance Model | `ofi_proxy` | -30.42 | -3.04 | -30.34 | -24.65 | -55.42 | -81.6 | -81.6 | no edge (negative even before costs) (proxy: signed 15-minute tick volume) |
| 267 | Market-Microstructure Signal | `micro_signal` | -27.52 | -4.55 | -27.47 | -22.80 | -43.01 | -81.8 | -81.7 | no edge (negative even before costs) (proxy: close location x relative volume) |
| 268 | Bid/Ask Imbalance Model | — | | | | | | | | not testable: no order book |
| 269 | Spread-Change Model | — | | | | | | | | not testable: no bid/ask history in this data |
| 270 | Tick Imbalance Model | `tick_imbalance` | -25.47 | -2.56 | -25.36 | -21.11 | -38.95 | -75.4 | -75.4 | no edge (negative even before costs) (15-minute bars stand in for ticks) |
| 271 | Trade-Intensity Model | `trade_intensity` | -4.51 | 0.21 | -4.50 | -3.47 | -5.78 | -16.9 | -16.9 | no edge (positive before costs, not significant) |
| 272 | Arrival-Rate Model | `arrival_rate` | -8.64 | -0.01 | -8.59 | -6.63 | -11.43 | -32.6 | -32.6 | no edge (negative even before costs) |
| 273 | Hawkes Process Model | `hawkes` | -0.63 | 0.07 | -0.61 | -0.97 | -0.41 | -2.5 | -2.5 | no edge (positive before costs, not significant) |
| 274 | Jump Detection Model | `jump_detection` | -0.46 | 0.32 | 0.02 | -0.42 | -0.50 | -1.9 | -1.9 | no edge (positive before costs, not significant) |
| 275 | Price-Jump Reversion | `jump_rev` | -2.06 | -0.40 | -2.02 | -1.93 | -2.35 | -8.9 | -9.0 | no edge (negative even before costs) |
| 276 | Price-Jump Continuation | `jump_cont` | -1.31 | 0.40 | -1.26 | -1.67 | -1.12 | -5.7 | -5.7 | no edge (positive before costs, not significant) |
| 277 | Intraday Jump-Diffusion Model | `jump_diffusion` | -5.01 | 0.37 | -4.74 | -4.05 | -6.20 | -22.8 | -22.9 | no edge (positive before costs, not significant) |
| 278 | Extreme-Value Theory Entry | `evt_entry` | -0.05 | 0.42 | 0.02 | 0.45 | -0.34 | -0.2 | -0.1 | no edge (positive before costs, not significant) |
| 279 | Tail-Event Reversion | `evt_tail_rev` | -0.08 | 0.16 | -0.00 | 0.17 | -0.25 | -0.4 | -0.2 | no edge (positive before costs, not significant) |
| 280 | Tail-Event Momentum | `evt_tail_mom` | -0.40 | -0.16 | -0.32 | -0.60 | -0.27 | -1.8 | -1.9 | no edge (negative even before costs) |
| 281 | EVT Breakout Model | `evt_breakout` | 0.11 | 0.19 | 0.17 | 0.31 | -0.04 | 0.5 | 0.2 | positive overall, fails one half |
| 282 | Expected-Move Breakout | `expected_move_bo` | -1.97 | 0.90 | -1.81 | -1.28 | -2.82 | -8.6 | -8.7 | real gross edge, eaten by costs |
| 283 | Implied-Move Breach | `implied_move_breach` | 0.31 | 0.54 | 0.31 | 0.43 | 0.16 | 1.4 | 1.3 | positive both halves, not significant (S&P only, VIX-implied move) |
| 284 | Range-Expansion Probability Model | `range_exp_prob` | -1.09 | -0.18 | -0.82 | -1.09 | -1.10 | -4.2 | -4.2 | no edge (negative even before costs) |
| 285 | Bayesian Breakout Probability | `bayes_breakout` | -0.51 | -0.01 | -0.16 | -0.64 | -0.32 | -2.2 | -2.2 | no edge (negative even before costs) |
| 286 | Breakout Success-Rate Model | `breakout_success` | -0.61 | -0.19 | -0.32 | -0.44 | -0.81 | -2.6 | -2.6 | no edge (negative even before costs) |
| 287 | Conditional Breakout Model | `cond_breakout` | -0.52 | -0.02 | -0.20 | -0.22 | -0.83 | -2.2 | -2.3 | no edge (negative even before costs) |
| 288 | Failed-Breakout Probability Model | `failed_breakout` | -0.61 | -0.02 | -0.32 | -0.77 | -0.42 | -2.6 | -2.6 | no edge (negative even before costs) |
| 289 | Statistical False-Breakout Model | `false_breakout_stat` | -3.38 | 0.62 | -3.18 | -2.78 | -4.24 | -15.1 | -15.1 | real gross edge, eaten by costs |
| 290 | Pattern-Conditional Probability | `pattern_cond_prob` | -0.74 | 0.13 | -0.45 | -0.97 | -0.47 | -3.0 | -3.9 | no edge (positive before costs, not significant) |
| 291 | Candle-Sequence Model | `candle_sequence` | -1.31 | 0.31 | -0.99 | -1.05 | -1.52 | -5.0 | -6.2 | no edge (positive before costs, not significant) |
| 292 | Markov Candle-State Model | `markov_candle` | -0.67 | 0.34 | -0.42 | -0.59 | -0.78 | -2.7 | -3.7 | no edge (positive before costs, not significant) |
| 293 | Multi-Candle Return Model | `multi_candle` | -1.34 | -0.00 | -0.86 | -0.75 | -2.03 | -5.6 | -5.8 | no edge (negative even before costs) |
| 294 | Return Autocorrelation Model | `ret_autocorr_weekly` | -0.74 | -0.07 | -0.34 | -0.51 | -1.01 | -2.9 | -2.9 | no edge (negative even before costs) |
| 295 | Overnight Return Model | `overnight_return` | -0.32 | 1.06 | -0.08 | -0.17 | -0.50 | -1.5 | -1.5 | real gross edge, eaten by costs |
| 296 | Intraday Return Seasonality | `tod_season` | -21.99 | 2.99 | -21.88 | -16.08 | -33.82 | -49.5 | -49.5 | real gross edge, eaten by costs (same test as Time-of-Day Seasonality) |
| 297 | Gap-Return Distribution Model | `gap_return_dist` | 0.00 | 0.38 | 0.00 | -0.66 | 0.24 | 0.0 | 0.1 | positive overall, fails one half |
| 298 | Open-to-High/Low Model | `open_to_hilo` | -1.53 | -0.58 | -1.53 | -1.38 | -1.73 | -7.0 | -7.0 | no edge (negative even before costs) |
| 299 | Close-to-Open Model | `close_to_open` | -0.64 | 0.02 | -0.64 | -0.77 | -0.52 | -2.7 | -2.6 | no edge (positive before costs, not significant) |
| 300 | Open-to-Close Momentum | `open_to_close_mom` | -0.64 | 0.36 | -0.63 | -0.36 | -1.02 | -2.9 | -2.9 | no edge (positive before costs, not significant) |
| 301 | High-Low Range Model | `hl_range` | -1.37 | -0.42 | -1.23 | -1.42 | -1.33 | -5.4 | -5.3 | no edge (negative even before costs) |
| 302 | Range Compression Model | `range_compression` | -1.54 | -0.44 | -1.04 | -1.53 | -1.55 | -6.2 | -6.4 | no edge (negative even before costs) |
| 303 | Range Expansion Model | `range_expansion` | -1.55 | -0.02 | -1.33 | -1.36 | -1.75 | -6.2 | -6.2 | no edge (negative even before costs) |
| 304 | ATR Compression Model | `atr_compression` | -0.53 | -0.30 | -0.39 | -0.18 | -0.99 | -2.0 | -2.0 | no edge (negative even before costs) |
| 305 | Volatility Contraction Pattern | `vcp` | -0.28 | 0.03 | -0.03 | -0.00 | -0.63 | -1.1 | -1.3 | no edge (positive before costs, not significant) |
| 306 | Volatility Expansion Continuation | `vol_exp_cont` | -0.23 | -0.12 | -0.19 | -0.37 | -0.11 | -1.0 | -1.1 | no edge (negative even before costs) |
| 307 | Volatility Mean Reversion | `vol_mr_fade` | -0.21 | 0.14 | -0.09 | -0.03 | -0.39 | -0.9 | -0.9 | no edge (positive before costs, not significant) |
| 308 | Volatility Regime + Trend | `vol_regime_trend` | -0.66 | -0.09 | -0.22 | -0.48 | -0.83 | -2.5 | -2.5 | no edge (negative even before costs) |
| 309 | Volatility Regime + Mean Reversion | `vol_regime_mr` | -0.39 | -0.11 | -0.23 | -0.39 | -0.39 | -1.6 | -1.6 | no edge (negative even before costs) |
| 310 | Regime-Conditional Momentum | `regime_cond_mom` | -0.55 | 0.01 | -0.25 | -0.33 | -0.82 | -2.2 | -2.2 | no edge (positive before costs, not significant) |
| 311 | Regime-Conditional Mean Reversion | `regime_cond_mr` | -0.59 | -0.17 | -0.30 | -0.55 | -0.63 | -2.4 | -2.6 | no edge (negative even before costs) |
| 312 | Regime-Conditional Breakout | `regime_cond_bo` | -0.51 | 0.03 | -0.12 | -0.27 | -0.75 | -2.1 | -2.1 | no edge (positive before costs, not significant) |
| 313 | Regime-Conditional Trend Following | `regime_cond_trend` | -0.26 | 0.14 | 0.07 | 0.04 | -0.62 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 314 | Adaptive Moving-Average Model | `adaptive_ma` | -0.56 | 0.16 | -0.09 | -0.34 | -0.84 | -2.4 | -2.4 | no edge (positive before costs, not significant) |
| 315 | Kaufman Adaptive Moving Average Model | `kama` | -1.18 | -0.23 | -0.70 | -1.35 | -0.99 | -4.9 | -5.1 | no edge (negative even before costs) |
| 316 | Hull Moving Average Quant Model | `hma` | -0.68 | -0.05 | -0.21 | -0.41 | -1.03 | -2.9 | -3.0 | no edge (negative even before costs) |
| 317 | Zero-Lag Moving Average Model | `zlema` | -0.66 | 0.22 | -0.19 | -0.76 | -0.54 | -2.6 | -2.8 | no edge (positive before costs, not significant) |
| 318 | Adaptive Channel Model | `adaptive_channel` | -0.63 | -0.12 | -0.23 | -0.37 | -0.92 | -2.6 | -2.8 | no edge (negative even before costs) |
| 319 | Dynamic Support/Resistance Model | `dyn_sr` | -0.79 | 0.01 | -0.39 | -0.84 | -0.72 | -3.2 | -3.3 | no edge (positive before costs, not significant) |
| 320 | Machine-Learned Support/Resistance | `ml_sr` | -1.81 | -0.50 | -1.20 | -1.54 | -2.06 | -6.9 | -6.9 | no edge (negative even before costs) |
| 321 | Quantified Swing-Level Model | `swing_level` | -0.54 | 0.04 | -0.04 | -0.43 | -0.68 | -2.3 | -2.4 | no edge (positive before costs, not significant) |
| 322 | Price-Level Reaction Probability | `level_reaction_prob` | -0.50 | -0.35 | -0.43 | -0.70 |  | -1.8 | -1.8 | no edge (negative even before costs) |
| 323 | Level-Strength Scoring Model | `level_strength` | -0.72 | -0.07 | -0.37 | -0.82 | -0.60 | -3.1 | -3.1 | no edge (negative even before costs) |
| 324 | Bayesian Support/Resistance Model | `bayes_sr` | -0.47 | -0.29 | -0.39 | -0.66 |  | -1.7 | -1.7 | no edge (negative even before costs) |
| 325 | Multi-Timeframe Statistical Alignment | `mtf_align` | -0.36 | 0.18 | 0.06 | -0.04 | -0.75 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 326 | Multi-Timeframe Probability Model | `mtf_prob` | -0.70 | 0.09 | -0.33 | -0.80 | -0.58 | -2.8 | -2.8 | no edge (positive before costs, not significant) |
| 327 | Multi-Timeframe Regime Model | `mtf_regime` | -2.23 | -0.23 | -1.92 | -2.07 | -2.43 | -10.2 | -10.2 | no edge (negative even before costs) |
| 328 | HTF Trend Probability + LTF Trigger | `htf_trend_ltf` | -5.23 | 0.99 | -5.00 | -3.85 | -7.29 | -22.8 | -22.8 | real gross edge, eaten by costs |
| 329 | HTF Mean Reversion + LTF Trigger | `htf_mr_ltf` | -0.58 | -0.13 | -0.30 | -0.60 | -0.56 | -2.7 | -2.7 | no edge (negative even before costs) |
| 330 | Ensemble Multi-Timeframe Model | `mtf_ensemble` | -0.54 | 0.43 | -0.18 | -0.48 | -0.61 | -2.4 | -2.4 | no edge (positive before costs, not significant) |
| 331 | Walk-Forward Optimized Model | `wf_optimized` | -0.48 | 0.06 | -0.02 | -0.17 | -0.76 | -1.8 | -1.9 | no edge (positive before costs, not significant) |
| 332 | Rolling-Parameter Model | `rolling_param` | -0.38 | 0.32 | 0.14 | -0.28 | -0.48 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 333 | Adaptive Parameter Model | `adaptive_param` | -0.44 | 0.26 | 0.03 | -0.21 | -0.72 | -1.7 | -1.7 | no edge (positive before costs, not significant) |
| 334 | Online Learning Model | `online_learning` | -1.82 | -0.52 | -1.32 | -1.65 | -1.94 | -6.5 | -6.5 | no edge (negative even before costs) |
| 335 | Expanding-Window Model | `expanding_window` | -1.87 | -0.01 | -1.35 | -0.99 | -2.51 | -6.7 | -6.9 | no edge (negative even before costs) |
| 336 | Rolling-Window Model | `rolling_window` | -1.68 | -0.01 | -1.16 | -0.88 | -2.30 | -6.0 | -5.9 | no edge (negative even before costs) |
| 337 | Walk-Forward Momentum | `wf_momentum` | -0.72 | -0.01 | -0.20 | -0.53 | -0.87 | -2.6 | -2.7 | no edge (negative even before costs) |
| 338 | Walk-Forward Mean Reversion | `wf_mr` | -0.51 | -0.17 | -0.28 | -0.17 | -0.74 | -1.9 | -1.8 | no edge (negative even before costs) |
| 339 | Walk-Forward Breakout | `wf_breakout` | -0.54 | -0.06 | -0.13 | -0.28 | -0.77 | -2.0 | -2.1 | no edge (negative even before costs) |
| 340 | Portfolio-Level Signal Model | `portfolio_signal` | -0.41 | 0.38 | 0.04 | -0.08 | -0.80 | -1.8 | -1.8 | no edge (positive before costs, not significant) |
| 341 | Multi-Asset Signal Ensemble | `multi_asset_ensemble` | -0.17 | 0.35 | 0.14 | 0.31 | -0.72 | -0.7 | -0.7 | no edge (positive before costs, not significant) |
| 342 | Risk-Adjusted Signal Selection | `risk_adj_selection` | -0.69 | -0.14 | -0.31 | -0.29 | -1.12 | -2.9 | -2.9 | no edge (negative even before costs) |
| 343 | Correlation-Adjusted Signal Selection | `corr_adj_selection` | -0.41 | 0.42 | 0.05 | -0.09 | -0.79 | -1.8 | -1.8 | no edge (positive before costs, not significant) |
| 344 | Volatility-Adjusted Signal Selection | `voladj_selection` | -0.41 | 0.43 | 0.12 | -0.03 | -0.83 | -1.7 | -1.7 | no edge (positive before costs, not significant) |
| 345 | Dynamic Signal Weighting | `dynamic_weighting` | -0.50 | -0.03 | -0.18 | -0.35 | -0.77 | -2.1 | -2.1 | no edge (negative even before costs) |
| 346 | Regime-Dependent Signal Weighting | `regime_weighting` | -0.47 | 0.22 | -0.06 | -0.35 | -0.61 | -2.0 | -2.3 | no edge (positive before costs, not significant) |
| 347 | Bayesian Model Averaging | `bma` | -0.55 | 0.16 | -0.15 | -0.07 | -1.43 | -2.4 | -2.4 | no edge (positive before costs, not significant) |
| 348 | Ensemble Technical Model | `ensemble_tech` | -0.64 | 0.08 | -0.26 | -0.39 | -0.92 | -2.7 | -2.9 | no edge (positive before costs, not significant) |
| 349 | Ensemble Macro Model | `ensemble_macro` | -0.10 | 0.40 | 0.18 | 0.25 | -0.57 | -0.4 | -0.4 | no edge (positive before costs, not significant) |
| 350 | Ensemble Cross-Asset Model | `ensemble_xasset` | -0.15 | 0.43 | 0.28 | -0.05 | -0.25 | -0.6 | -0.6 | no edge (positive before costs, not significant) |
| 351 | Meta-Labeling | `meta_label` | 0.07 | 0.44 | 0.29 | 0.14 | 0.04 | 0.3 | 0.1 | positive both halves, not significant |
| 352 | Triple-Barrier Entry Model | `triple_barrier` | -0.70 | -0.18 | -0.41 | -1.18 | -0.22 | -2.5 | -2.4 | no edge (negative even before costs) |
| 353 | Event-Based Sampling Model | `cusum_event` | -1.06 | -0.17 | -0.65 | -1.21 | -0.90 | -4.5 | -4.6 | no edge (negative even before costs) |
| 354 | Dollar-Bar Model | `dollar_bars` | -2.89 | 0.45 | -2.41 | -2.43 | -3.51 | -14.1 | -14.1 | real gross edge, eaten by costs (price x tick volume) |
| 355 | Volatility-Bar Model | `vol_bars` | -0.99 | 0.03 | -0.57 | -0.94 | -1.06 | -4.5 | -4.5 | no edge (positive before costs, not significant) |
| 356 | Information-Bar Model | `info_bars` | -4.72 | 0.20 | -4.11 | -4.14 | -5.52 | -21.3 | -21.3 | no edge (positive before costs, not significant) |
| 357 | Fractional Differentiation Model | `frac_diff` | -0.42 | -0.15 | -0.22 | -0.35 | -0.47 | -1.6 | -1.6 | no edge (negative even before costs) |
| 358 | Feature-Engineered Return Model | `ml_feature_return` | -0.91 | 0.52 | -0.39 | -0.82 | -0.97 | -3.3 | -3.2 | no edge (positive before costs, not significant) |
| 359 | Machine-Learned Probability-of-Return Model | `ml_prob_return` | -0.69 | 0.03 | -0.22 | -0.92 | -0.52 | -2.5 | -2.6 | no edge (positive before costs, not significant) |
| 360 | Machine-Learned Probability-of-Breakout Model | `ml_prob_breakout` | -0.35 | 0.14 | -0.03 | -0.25 | -0.43 | -1.4 | -1.5 | no edge (positive before costs, not significant) |
| 361 | Machine-Learned Probability-of-Reversal Model | `ml_prob_reversal` | -0.40 | -0.08 | -0.25 | -0.35 | -0.44 | -1.6 | -1.4 | no edge (negative even before costs) |
| 362 | Machine-Learned Expected-Return Model | `ml_expected_return` | -0.76 | 0.10 | -0.17 | -0.55 | -0.92 | -2.8 | -2.7 | no edge (positive before costs, not significant) |
| 363 | Machine-Learned Risk/Reward Model | `ml_risk_reward` | -0.55 | -0.23 | -0.37 | -1.14 | 0.01 | -2.0 | -2.3 | no edge (negative even before costs) |
| 364 | Expected-Value Optimization | `ev_optimization` | -0.65 | -0.02 | -0.43 | -0.57 | -0.72 | -2.5 | -3.0 | no edge (negative even before costs) |
| 365 | Utility-Maximizing Entry | `utility_max` | -0.29 | 0.14 | 0.07 | -0.37 | -0.21 | -1.1 | -1.2 | no edge (positive before costs, not significant) |
| 366 | Probability × Payoff Model | `prob_payoff` | -0.34 | 0.12 | -0.00 | -0.24 | -0.66 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 367 | Conditional R-Multiple Model | `cond_r_multiple` | -0.31 | 0.10 | -0.01 | -0.22 | -0.57 | -1.4 | -1.4 | no edge (positive before costs, not significant) |
| 368 | Kelly-Optimized Entry | `kelly_opt_entry` | -0.25 | 0.11 | 0.06 | -0.29 | -0.20 | -1.0 | -1.0 | no edge (positive before costs, not significant) |
| 369 | Risk-of-Ruin Filter | `risk_of_ruin` | -0.13 | 0.18 | 0.10 | -0.12 | -0.50 | -0.6 | -0.5 | no edge (positive before costs, not significant) |
| 370 | Drawdown-Aware Signal Model | `drawdown_aware_signal` | -0.39 | -0.12 | -0.22 | -0.53 | -0.17 | -1.7 | -1.7 | no edge (negative even before costs) |
| 371 | Capital-Preservation Quant Model | `capital_preservation` | -0.41 | -0.20 | -0.25 | -0.46 | -0.45 | -1.7 | -1.7 | no edge (negative even before costs) |
| 372 | Prop-Firm Constraint-Aware Quant Model | `prop_firm_aware` | -0.09 | 0.30 | 0.21 | -0.04 | -0.19 | -0.4 | -0.4 | no edge (positive before costs, not significant) |
| 373 | Daily-Loss-Limit-Aware Model | `daily_loss_aware` | -0.44 | 0.07 | -0.06 | -0.16 | -0.79 | -1.9 | -1.9 | no edge (positive before costs, not significant) |
| 374 | Maximum-Drawdown-Aware Model | `max_dd_aware` | -0.50 | -0.15 | -0.24 | -0.39 | -0.65 | -2.2 | -2.2 | no edge (negative even before costs) |
| 375 | Volatility-Scaled Prop-Firm Model | `volscaled_prop` | -0.24 | 0.31 | 0.20 | 0.25 | -0.79 | -1.0 | -1.0 | no edge (positive before costs, not significant) |
| 376 | Dynamic Risk-Sizing Model | `dynamic_risk_sizing` | -0.38 | 0.01 | -0.09 | -0.21 | -0.61 | -1.6 | -1.6 | no edge (positive before costs, not significant) |
| 377 | Fixed-Fractional Quant Model | `fixed_fractional` | -0.13 | 0.08 | 0.04 | 0.12 | -0.54 | -0.6 | -0.6 | no edge (positive before costs, not significant) |
| 378 | ATR Position-Sizing Model | `atr_sizing` | -0.31 | 0.20 | 0.12 | 0.05 | -0.76 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 379 | Volatility Target Position Sizing | `vol_target_sizing` | -0.26 | 0.18 | 0.11 | 0.08 | -0.68 | -1.1 | -1.1 | no edge (positive before costs, not significant) |
| 380 | Kelly Position Sizing | `kelly_sizing` | -0.33 | 0.03 | -0.05 | -0.24 | -0.48 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 381 | Fractional-Kelly Position Sizing | `frac_kelly_sizing` | -0.32 | 0.07 | -0.02 | -0.17 | -0.54 | -1.3 | -1.3 | no edge (positive before costs, not significant) |
| 382 | Risk-Parity Position Sizing | `rp_sizing` | -0.37 | 0.27 | 0.15 | 0.17 | -0.97 | -1.5 | -1.5 | no edge (positive before costs, not significant) |
| 383 | Correlation-Adjusted Position Sizing | `corr_adj_sizing` | -0.17 | 0.31 | 0.22 | 0.24 | -0.63 | -0.7 | -0.7 | no edge (positive before costs, not significant) |
| 384 | Expected-Shortfall Position Sizing | `es_sizing` | -0.21 | 0.19 | 0.13 | 0.08 | -0.56 | -0.9 | -0.9 | no edge (positive before costs, not significant) |
| 385 | Drawdown-Adjusted Position Sizing | `dd_adj_sizing` | -0.35 | -0.01 | -0.07 | -0.22 | -0.75 | -1.5 | -1.5 | no edge (negative even before costs) |

## How each model is defined

One line per test, from its docstring. Code: `examples/quant/`.

- `adaptive_channel` (D): Donchian whose lookback adapts to vol: 20 x (60d vol / 1y vol), clipped 10-60; exit at the half-length opposite channel.
- `adaptive_ma` (D): VIDYA: EMA(20) whose speed scales with |CMO(9)|; sign(price - VIDYA).
- `adaptive_param` (D): Adaptive parameter: the trend lookback scales inversely with the volatility regime (short when vol is high): 20-120 days.
- `adx_filter` (D): EMA20/50 trend taken only while ADX(14) > 25.
- `analog_match` (D): Analog pattern matching: 20 nearest past 20-day normalised paths (Euclidean); trade the sign of their mean next-5-day return.
- `ar1_fcst` (D): Rolling 250-day AR(1) on returns: trade the forecast sign when it exceeds the one-way cost.
- `arima_fcst` (D): ARIMA(1,0,1) on daily returns, refit yearly on the prior 3 years; sign of the one-step forecast.
- `arrival_rate` (H): Arrival-rate model: hourly volume more than double the previous hour and above its 500-hour mean -> follow the bar for 2 hours.
- `asia_session` (H): Asia session (19-03) follows or fades the prior New York session (13-16 window, the part known by the 17:00 roll).
- `atr_breakout` (D): Close more than 1.5 ATR beyond the prior close: follow 5 days.
- `atr_channel` (D): SMA50 +-3 ATR channel: in beyond it, out at SMA50.
- `atr_compression` (D): ATR5/ATR50 < 0.6, then a close beyond the 10-day range: follow 10 days.
- `atr_norm_entry` (D): (price - SMA50)/ATR > 2: long; < -2: short; exit at 0.
- `atr_regime` (D): ATR14/ATR100 > 1.2: fade z20; < 0.8: follow 20-day breakouts.
- `atr_sizing` (D): ATR position sizing: risk 1% per position with a 2 x ATR(14) stop.
- `autocorr_entry` (D): sign(autocorr) x sign(last return), when |autocorr(60)| > 0.1.
- `bayes_agg` (D): Naive-Bayes aggregation of 5 binary signals: each signal's expanding likelihood ratio P(s|up)/P(s|down) on next-day direction; trade when the posterior P(up) leaves 0.48-0.52.
- `bayes_breakout` (D): 20-day breakouts taken only when the Beta(5,5)-prior posterior mean of past breakout success (10-day follow-through) exceeds 0.55.
- `bayes_prob_update` (D): Beta-binomial posterior of P(up next | trend state) with a Beta(50,50) prior and exponential forgetting (half-life 500 obs); trade when the posterior mean leaves 0.48-0.52.
- `bayes_regime` (D): Bayesian online change-point detection on standardised returns: trade the sign of the posterior mean of the current regime, when it is larger than 0.05 sd.
- `bayes_sr` (D): Bayesian support/resistance: the same rejections, taken while the Beta(5,5)-prior posterior mean of the fade's success exceeds 0.55.
- `bb_squeeze` (D): BB width at a 120-day low (within 10%): trade the band break, exit mid.
- `beta_adj_rank` (D): Beta-adjusted ranking: 6-month residual momentum vs the group factor (250d beta), within groups, monthly.
- `beta_neutral_pairs` (D): Return-beta hedged pairs: 60-day return regression beta; fade the 20-day cumulative spread return when its 120-day z exceeds 2.
- `bma` (D): Bayesian model averaging: weights proportional to exp(t/2) of each strategy's trailing 2-year mean return (a BIC-style posterior).
- `boll_z` (D): Bollinger(20,2): fade on the close back inside the band, exit mid.
- `bond_yield_ll` (H): Bond/yield lead-lag: a 1-sd Treasury CFD hour (yields down) -> short the dollar next hour (long EUR/USD, GBP/USD, AUD/USD; short USD/CAD).
- `bootstrap_dist` (D): Stationary bootstrap of the last 60 daily returns: trade the sign of the mean when its 90% interval excludes zero (weekly decisions).
- `breakout_plain` (D): Reference: every 20-day breakout, held 10 days.
- `breakout_success` (D): 20-day breakouts taken only when > 55% of the last 30 succeeded.
- `candle_sequence` (D): Last 3 daily candles up/down (8 states): expanding conditional probability of an up day next.
- `capital_preservation` (D): Capital preservation: half size beyond a 5% drawdown, flat beyond 8%, back to full only once the shadow strategy regains half the loss.
- `cash_session_mom` (M): Intraday momentum (Gao, Han, Li & Zhou): the first half-hour return (prior 16:00 close to 10:00) predicts the last half-hour (15:30-16:00).
- `cb_decision` (M): Central-bank decision model: FOMC as above plus ECB days (07:45 NY bar spike on a Thursday in EUR/USD and Bund): follow the decision bar to 11:00 (ECB) or 16:00 (FOMC).
- `ccy_strength` (D): Currency strength meter (20-day, least squares over six pairs): trade every pair in the direction base strength - quote strength, scaled by the gap in strength-sd units; weekly.
- `change_point` (D): Page CUSUM on standardised returns (k=0.5, h=5): after an upward (downward) shift in mean is detected, hold that direction 20 days.
- `close_to_open` (M): Close-to-open: the overnight move (16:00 -> 09:45) is followed or faded to 16:00 by the sign of its trailing 250-day correlation with the rest of the day.
- `closing_window` (M): Closing-window statistics: 15:00-16:00, same rule.
- `cluster_rs` (D): Asset-cluster relative strength: each year cluster the CFDs (average linkage on 1 - correlation of the prior 2 years) into 6 clusters; long the cluster with the best 6-month return, short the worst; monthly.
- `coint_pairs` (D): Engle-Granger: trade the 250-day OLS spread (|z| > 2, exit 0) only in months when the residual ADF p-value on the prior 250 days is < 0.05.
- `commodity_ccy` (D): Commodity currencies follow their commodity: AUD/USD by gold's 20-day trend, USD/CAD against oil's 20-day trend.
- `composite_rank` (D): Composite rank (6m momentum, 1w reversal, low vol), terciles, weekly.
- `compress_expand` (D): 10d vol in its lowest decile of the year, then the first 1-sd day sets the direction for 10 days.
- `cond_breakout` (D): 20-day breakouts only after compression (BB width in the lowest quintile of 250 days within the last 5 days); hold 10 days.
- `cond_expectancy` (D): Conditional expectancy: expectancy of earlier trades taken in the same volatility regime (above/below median) must be positive.
- `cond_prob` (D): Conditional probability on (trend sign, vol regime, last-day sign): expanding P(up next).
- `cond_r_multiple` (D): Conditional R-multiple: mean R of earlier trades in the same ADX regime (> 25 or not) must be positive.
- `confluence` (D): Five binary trend votes (SMA200, MACD>0, RSI>50, +DI>-DI with ADX>20, 20-day breakout state): long on 4+, short on 1 or fewer.
- `corr_adj_selection` (D): Correlation-adjusted selection: strategy weights proportional to 1 / (1 + average correlation with the others), trailing year.
- `corr_adj_sizing` (D): Correlation-adjusted sizing: each 10%-vol position divided by sqrt(1 + (N-1) x its average 1-year correlation with the other assets' strategy returns), monthly.
- `corr_breakout` (D): Same breakdown, but bet the divergence continues for 10 days.
- `corr_reversion` (D): Correlation breakdown (20d corr 0.4 below its 250d level, normally > 0.5): bet the 20-day relative move reverts, 10 days.
- `corr_spread` (D): Spread z trading (120d OLS, |z| > 2) only while the 60-day return correlation of the pair is above 0.7.
- `cost_adj_entry` (H): Cost-adjusted entry: the hourly fade only when the distance back to the mean is more than 4x the round-trip cost.
- `cpi_reaction_proxy` (M): CPI reaction (proxy calendar: large 08:30 bars on the 10th-17th, not a Friday -- mostly CPI and retail sales): follow 08:45-11:00.
- `cross_asset_rs` (D): Asset-class relative strength: average 6-month return per group; long every asset in the best group, short every asset in the worst; monthly.
- `cross_asset_rv` (D): Cross-asset relative value on economically linked pairs (AUD/gold, CAD/oil, AUDJPY/S&P, Nikkei/EURJPY, gold/Treasuries, ASX/AUD): fade the 120-day OLS spread z beyond 2.
- `cross_market_ll` (H): Cross-market lead-lag: a 1-sd S&P hour is followed the next hour in every other index CFD.
- `cs_mom` (D): Cross-sectional momentum within groups, 3-month return, weekly.
- `cs_mr` (D): Cross-sectional reversal: within each group, long the weakest third and short the strongest third of 5-day returns; weekly.
- `cs_rank` (D): Cross-sectional ranking on 1-month return, all assets, weekly.
- `cusum_event` (D): Event sampling: symmetric CUSUM filter (h = 2 x daily sd) marks events; follow the event direction for 5 days.
- `cvd` (H): Cumulative volume delta (signed hourly tick volume): follow the 4-hour CVD change, standardised, beyond 1 sd.
- `daily_loss_aware` (D): Daily-loss-limit-aware: after a day losing more than 1.5% (book at 10% vol), stand aside the next day.
- `dd_adj_sizing` (D): Drawdown-adjusted sizing: book size x (1 - drawdown / 15%), floored at a quarter, using the full-size book's own drawdown (at 10% vol).
- `dollar_bars` (H): Dollar bars (price x tick volume, about 6 a day): momentum of the last 5 bars.
- `dollar_factor` (D): Dollar-index factor: synthetic DXY from four USD pairs; every asset takes sign(250d beta to the dollar) x sign(dollar 3-month trend).
- `dollar_gold` (D): Dollar-gold relationship: gold's EMA50/200 trend taken only when the synthetic dollar's 50-day trend points the other way.
- `donchian` (D): Turtle: 20-day channel breakout, exit on the 10-day opposite.
- `dow_season` (D): Day-of-week seasonality: trade the next day's weekday in the sign of its mean return over earlier years when |t| > 1.5.
- `drawdown_aware_signal` (D): Drawdown-aware signal: each asset's trend weight scaled by 1 - its strategy drawdown / 3 vol-units (floor 0).
- `drawdown_regime` (D): Drawdown-regime filter: an asset's trend position is switched off while that asset's own strategy is more than 2 vol-units under water, and back on when it recovers to half that.
- `dtw_pattern` (D): Dynamic-time-warping nearest neighbours (k = 20) of the last 15 days' normalised path among 1,500 sampled past windows; next-5-day sign.
- `dyn_hedge_pairs` (D): Dynamic hedge ratio: 60-day rolling OLS, enter |z| > 2, exit 0.
- `dyn_sr` (D): Dynamic support/resistance: the last confirmed swing high/low (5 bars each side) as levels; fade rejections for 5 days.
- `dynamic_risk_sizing` (D): Dynamic risk sizing: 1.5x after a positive trailing 60-day strategy Sharpe, 0.5x after a negative one.
- `dynamic_weighting` (D): Dynamic signal weighting: weights proportional to the positive part of each strategy's trailing 1-year Sharpe.
- `econ_surprise_proxy` (M): Economic-surprise proxy: the sign of the 08:30 bar on any large release day is taken as the surprise and held to the close only when it agrees with the sign of the previous release day's (surprise momentum).
- `efficiency_ratio` (D): Kaufman efficiency ratio(20) > 0.4: follow the 20-day direction.
- `egarch_sizing` (D): EGARCH(1,1,1) (asymmetric) volatility model as the sizing input.
- `ema_dist_z` (D): z (100d) of the distance from EMA20, fade +-2.
- `empirical_dist` (D): Expanding empirical P(next day up | z20 decile); trade when > 53% or < 47% with 100+ observations in the bucket.
- `ensemble_macro` (D): Ensemble macro model: mean of dollar factor, risk-on/off, four-factor macro, commodity currencies, rates differential and yield curve.
- `ensemble_tech` (D): Ensemble technical model: mean of 10 technical signals (MACD, RSI2, stochastic, Bollinger, Donchian, Keltner, multi-MA, ADX trend, KAMA, Hull).
- `ensemble_xasset` (D): Ensemble cross-asset model: mean of cross-sectional momentum, the momentum factor, PCA stat arb, asset-class relative strength and the currency-strength model.
- `entropy_regime` (D): Shannon entropy of 3-day up/down patterns over 60 days: in the lowest third of its 2-year range (ordered market) follow the 20-day trend; otherwise flat.
- `eq_bond_corr` (D): Equity-bond correlation regime: 60-day corr(S&P, Treasuries) < 0 -> long both (they hedge); otherwise hold only whichever has a positive 3-month return.
- `eq_fx_ll` (H): Equity-index/FX lead-lag: a 1-sd S&P hour -> AUD/JPY and EUR/JPY the same way next hour.
- `es_filter` (D): Expected-shortfall filter: skip an asset whose 97.5% ES of vol-standardised returns (1 year) is above 1.25x the cross-sectional median -- fat tails relative to its own volatility.
- `es_sizing` (D): Expected-shortfall sizing: notional inversely proportional to the asset's historical 97.5% one-day expected shortfall (1 year).
- `ev_optimization` (D): Expected-value optimisation: per asset, long / short / flat by which has the higher expanding-window expected next-day return (conditional on trend state x z20 bucket) after the round-trip cost.
- `event_drift` (M): Event drift: follow the 08:30 release bar to the 16:00 close.
- `event_vol` (M): Event volatility breakout: on 08:30 event days, the first 15-minute close beyond the 07:30-08:30 range sets the side until 11:00.
- `evt_breakout` (D): 20-day range beyond the GPD 99% quantile of past 20-day ranges: follow the 20-day direction for 10 days.
- `evt_entry` (D): EVT entry: fade a 1-day move beyond the 97.5% GPD quantile, 1 day.
- `evt_tail_mom` (D): As tail-event reversion, but follow the tail move for 3 days.
- `evt_tail_rev` (D): Daily return beyond the GPD 99% tail of the last 3 years: fade 3 days.
- `exp_smooth` (D): Holt linear-trend exponential smoothing (alpha .2, beta .05) of log price: sign of the smoothed trend.
- `expanding_window` (D): Expanding-window ridge regression of next-day return on the features, refit yearly; trade the sign.
- `expectancy_filter` (D): Expectancy filter: take a new breakout only while the asset's last 30 trades have positive mean R.
- `expected_move_bo` (H): Expected-move breakout: once price leaves the day's open by 0.7 x daily ATR, follow until the 17:00 roll.
- `expected_move_model` (H): CFD expected-move model: hourly fades only while the day's remaining expected range (daily ATR minus range so far) exceeds 5x round-trip cost.
- `factor_resid_mr` (D): Fade the 5-day residual return after a rolling 60-day beta to the asset's group (equal-weight) factor.
- `factor_score` (D): Time-series factor score: z(12m momentum) - z(5d reversal) - z(vol), each standardised over 3 years; weight = score/2 clipped.
- `failed_breakout` (D): Failed breakout: a close above the 20-day high that closes back inside within 2 days -> short for 5 days (and the mirror).
- `false_breakout_stat` (H): Statistical false breakout: an hour trades above the prior day's high and closes back below it -> short until the roll (mirror at the low).
- `fcst_err_rev` (D): Fade large one-step errors of an EMA(10) forecast (z over 100d).
- `financing_aware` (D): Overnight-financing-aware entry: the multi-speed trend position is held only when the trailing 12-month move is worth more than twice the yearly financing markup.
- `fixed_fractional` (D): Fixed-fractional with a fixed 2% stop: every position the same notional (risk 1% of equity per trade at a 2% stop), no vol scaling.
- `flow_model` (D): Flow-based model: 5-day signed tick volume (sign of each day's candle x volume) relative to total volume; follow when |imbalance| > 0.3.
- `fomc_reaction` (M): FOMC reaction (statement days detected as a 3.5x range spike in the 14:00 or 14:15 bar on a Wednesday): follow the statement bar to 16:00.
- `frac_diff` (D): Fractionally differentiated price (d = 0.4, fixed window): stationary yet memory-preserving; fade its 250-day z beyond 2.
- `frac_kelly_filter` (D): Fractional-Kelly filter: only while half-Kelly would justify at least the 10%-vol size the engine takes.
- `frac_kelly_sizing` (D): Half-Kelly: as Kelly, halved and floored at a quarter size so a negative estimate does not switch the asset off entirely.
- `fractal_breakout` (D): Close beyond the last confirmed 5-bar fractal high/low; exit on the opposite fractal.
- `fractal_dim` (D): Sevcik fractal dimension (30d): < 1.4 trending (follow 20d), > 1.6 choppy (fade z20).
- `fx_ll` (M): FX lead-lag: a 1-sd EUR/USD 15-minute bar is followed the next bar in GBP/USD and AUD/USD.
- `gap_adjusted` (D): Gap-adjusted momentum for US indices: trend on the cash-session (09:30-16:00) return only, summed over 20 days, ignoring overnight gaps.
- `gap_return_dist` (M): Gap-return distribution: gaps bucketed by size (in 60-day sd); each year, the 09:45-16:00 return's mean per bucket over earlier years sets the side when |t| > 1.5.
- `garch_breakout` (D): GARCH forecast breakout: a day moving more than 2x its GARCH forecast vol is followed for 5 days.
- `garch_sizing` (D): GARCH(1,1) volatility model: the trend model sized by the GARCH one-day-ahead vol forecast (yearly walk-forward fit).
- `gk_sizing` (D): Garman-Klass OHLC volatility (20 days) as the sizing vol.
- `gold_usd_ll` (H): Gold/USD lead-lag: a 1-sd synthetic-dollar hour -> gold the opposite way next hour.
- `h1_mr_base` (H): Reference for the CFD filters: hourly Bollinger(20,2) fade to the mean, flat over the 17:00 roll.
- `halflife_mr` (D): z-score over a lookback equal to the estimated half-life (2-60d).
- `hawkes` (H): Hawkes self-excitation: intensity of 2.5-sd hourly moves with an exponential kernel (half-life 6h); while intensity > 2 events, follow the direction of the latest event.
- `hist_pct_rev` (D): 20-day return below its 5th / above its 95th 250-day percentile: fade for 10 days.
- `hl_range` (D): Range > 1.5 ATR with the close in the top/bottom 20%: follow 1 day.
- `hma` (D): Hull MA(55) slope sign.
- `hmm_entry` (D): 3-state HMM, same walk-forward; trade only when the expected next-day return is beyond 0.05 sd.
- `htf_mr_ltf` (H): HTF mean reversion + LTF trigger: daily z20 < -2 (> 2) and an hourly close above the previous hour's high (below its low) -> long (short) until the daily z crosses 0.
- `htf_trend_ltf` (H): HTF trend + LTF trigger: daily close above (below) EMA50, hourly RSI(2) < 10 (> 90) -> long (short) until hourly RSI(2) > 70 (< 30).
- `hurst` (D): Rolling 250-day Hurst exponent: > 0.55 follow the 20-day return, < 0.45 fade z20.
- `hv_breakout` (D): 10d vol / 100d vol > 1.5: follow the 10-day return.
- `implied_move_breach` (H): Implied-move breach (S&P): VIX/sqrt(252) is the implied daily move; a break of the 09:30 open by that much is followed to 16:00.
- `index_rs` (D): Index relative strength: equity indices ranked on 3-month return, long top 2, short bottom 2, weekly.
- `info_bars` (H): Tick-imbalance information bars: a bar closes when cumulative signed tick volume exceeds 3 hours of average volume; follow that imbalance's sign until the next bar.
- `intermarket_confirm` (D): Intermarket confirmation: each asset's EMA50/200 trend is taken only when it agrees with the risk-on/off factor's trend through the asset's beta to that factor.
- `intraday_pattern` (H): Intraday seasonal pattern: hold each hour in the sign of its mean return over all earlier years (no significance filter).
- `inverse_vol_selection` (D): Inverse-volatility position selection: hold the trend signal only in the 12 CFDs with the lowest 60-day volatility.
- `iv_rv` (D): Implied-vs-realised: z (250d) of VIX - 20d realised vol; long the S&P for 10 days when z > 1.5 (fear priced well above what is happening).
- `jump_cont` (H): Price-jump continuation: follow the jump for 4 hours.
- `jump_detection` (D): Jump-filtered momentum: 20-day trend of daily returns rebuilt from hourly returns with the jumps removed.
- `jump_diffusion` (H): Intraday jump-diffusion: follow the sign of the last 24 hours' drift with jump returns removed.
- `jump_rev` (H): Price-jump reversion: fade a Lee-Mykland jump for 4 hours.
- `kalman_mr` (D): Kalman local-level filter of log price; fade the deviation of price from the filtered level beyond 2 sd (100d).
- `kalman_pairs` (D): Kalman-filter hedge ratio and intercept (Chan, delta 1e-4); trade the standardised one-step forecast error beyond 1, exit at 0.
- `kalman_trend` (D): Local-linear-trend Kalman filter on log price (noise scaled to each asset's trailing 1-year variance); trade the sign of the filtered slope.
- `kama` (D): Price vs Kaufman adaptive MA (10,2,30): sign.
- `kelly_filter` (D): Kelly-criterion filter: take an asset's trend signal only while its trailing 1-year Kelly fraction is positive.
- `kelly_opt_entry` (D): Kelly-optimised entry: hold each asset at its own full-Kelly weight mu/sigma^2 from its trailing 1-year return (direction and size).
- `kelly_sizing` (D): Kelly sizing: each asset's 10%-vol position scaled by its trailing 1-year Kelly fraction mu/sigma^2 of the strategy's own returns (0-3).
- `keltner` (D): Keltner EMA20 +-2 ATR10 breakout, exit at EMA20.
- `lead_lag_arb` (M): Lead-lag arbitrage at 15 minutes: the same estimator on 15-minute bars.
- `level_reaction_prob` (D): Price-level reaction probability: fade rejections of the 20-day high/low only while > 55% of the asset's earlier rejections paid over 5 days (at least 20 seen).
- `level_strength` (D): Level-strength scoring: fade a rejection of the 20-day high/low only when that level has been tested (within 0.25 ATR) on 3+ days of the last 60.
- `linreg_fcst` (D): Next value of the 20-day regression line vs price: sign.
- `linreg_slope` (D): 50-day OLS slope of log price with |t| > 2 sets the direction.
- `liquidity_adj` (H): Liquidity-adjusted entry: only when tick volume is at least the median for that hour over the last 20 days.
- `london_session` (H): London session (03-08 NY) follows or fades the Asia session's return, by the sign of their trailing 500-day correlation.
- `long_basket_fixed` (D): Reference: every CFD held long at the same fixed notional.
- `long_memory` (D): GPH estimate of d on 500 days of returns: d > 0.1 follow the 20-day return, d < -0.1 fade it.
- `ls_rank` (D): Long/short ranking: composite score, top and bottom quintile only.
- `ma_dist` (D): Fade price more than 3 ATR from SMA50; exit at the average.
- `ma_slope` (D): Sign of the 10-day change of SMA50.
- `ma_spread_z` (D): z (250d) of EMA10 - EMA50 spread: long > 1, short < -1, exit at 0.
- `macd` (D): Sign of the MACD(12,26,9) histogram.
- `macro_event_regime` (D): Macro-event regime: the multi-speed trend model, but flat across payrolls Fridays (the day's risk is event risk, not trend).
- `macro_factor` (D): Four-factor macro model (dollar, risk-on/off, rates = Treasury CFD, commodities = average commodity CFD): 250-day betas times each factor's 3-month trend, summed; trade the sign.
- `mae_model` (D): Maximum-adverse-excursion model: the stop is tightened to the 80th percentile MAE of the asset's earlier winning trades.
- `markov_candle` (D): 3-state candle Markov chain (big up / small / big down by 0.5 sd): expanding transition probabilities, trade expected direction.
- `markov_switch` (D): 2-state Markov-switching (Gaussian HMM) on daily returns, refit each year on prior data; trade the sign of the expected next-day return.
- `max_dd_aware` (D): Maximum-drawdown-aware: size shrinks linearly to zero as the book's drawdown within the calendar year approaches a 10% limit (each year a fresh account, as a funded account resets after a breach).
- `mc_path_fcst` (D): GBM paths with 60-day drift and vol: probability of touching +1 sd before -1 sd over 10 days; trade when > 0.55 / < 0.45.
- `meta_label` (D): Meta-labeling (Lopez de Prado): primary = 20-day Donchian side on entry days; a boosted secondary model decides whether to act (10-day outcome).
- `mfe_model` (D): Maximum-favourable-excursion model: take profit at the median MFE of the asset's earlier trades.
- `micro_signal` (M): Microstructure proxy: close-location value x relative volume of the 15-minute bar; follow the next bar beyond +-1.
- `mkt_neutral_rank` (D): Market-neutral ranking: composite score ranked within each group.
- `ml_ensemble` (D): Average probability of logistic, random forest and boosting.
- `ml_ev_threshold` (D): Expected-value threshold: boosted regression of next-day return; trade only when |forecast| exceeds the round-trip cost.
- `ml_expected_return` (D): ML expected return: boosted regression of the 5-day return, weekly.
- `ml_feature_return` (D): Feature-engineered return model: boosted regression of the next-day return; trade its sign.
- `ml_gbm` (D): Histogram gradient boosting (200 rounds, depth 3).
- `ml_knn` (D): k-nearest neighbours (k=250) on standardised features.
- `ml_lda` (D): Linear discriminant analysis on the same features.
- `ml_logit` (D): Logistic regression (L2, C=0.05) on 30 features; next-day direction.
- `ml_lstm` (D): LSTM (16 units) over 20-day sequences of 6 inputs; next-day direction, yearly walk-forward refit on 40,000 sampled sequences.
- `ml_mlp` (D): Feed-forward neural network (32-16, early stopping).
- `ml_prob_breakout` (D): ML probability of breakout success: boosting on 20-day breakout days predicts 10-day follow-through; take breakouts with p > 0.5.
- `ml_prob_return` (D): ML probability of return: boosting classifier of the 5-day direction, rebalanced weekly.
- `ml_prob_reversal` (D): ML probability of reversal: on |z20| > 2 days, boosting predicts whether a fade pays over 5 days; take fades with p > 0.5.
- `ml_prob_threshold` (D): Probability-threshold entry: trade only when the logistic + boosting average probability is beyond 0.55 / 0.45.
- `ml_rf` (D): Random forest (200 trees, depth 6, leaf 200).
- `ml_risk_reward` (D): ML risk/reward: two boosted regressions predict the 5-day maximum favourable up move and down move (in vol units); long when up/down > 1.3, short when < 1/1.3; weekly.
- `ml_sr` (D): Machine-learned support/resistance: each month, k-means (k=6) on the prices of the last year's confirmed swing points gives the levels; fade rejections of the nearest level above/below for 5 days.
- `ml_stacked` (D): Stacked ensemble: a logistic meta-model, refit yearly, on the base learners' out-of-sample probabilities from earlier years.
- `ml_svm` (D): RBF support-vector machine on a 12,000-row subsample per fit; sign of the decision function.
- `ml_transformer` (D): One-layer Transformer encoder (2 heads, d=16) over the same 20-day sequences.
- `ml_voting` (D): Majority vote of logistic, RF, boosting, kNN and SVM.
- `ml_xgb` (D): XGBoost (300 rounds, depth 3, subsampled).
- `mom_rank` (D): Momentum ranking: 6-month return, all assets, monthly.
- `mom_vol_filter` (D): TSMOM only while 60-day vol is below its 1-year median.
- `momentum_factor` (D): 12-1 month momentum factor across all 25 CFDs (vol-adjusted returns), long top / short bottom third, monthly.
- `monte_carlo_entry` (D): Block-bootstrap 5-day paths from the last 250 days; long when P(5-day return > 0) > 0.55, short when < 0.45 (weekly).
- `moy_season` (D): Month-of-year seasonality: hold each month in the sign of its mean daily return over earlier years when |t| > 1.5.
- `mr_rank` (D): Mean-reversion ranking: 1-week return across all assets, weekly.
- `mtf_align` (D): Multi-timeframe alignment: weekly trend (price vs 26-week EMA) and daily trend (EMA20 vs EMA50) agree -> trade it.
- `mtf_ensemble` (H): Ensemble multi-timeframe: mean of hourly (EMA24/96), daily (20-day return) and weekly (12-week return) trend signs.
- `mtf_prob` (H): Multi-timeframe probability: expanding P(next 4 hours up | daily trend sign, hourly trend sign); trade when it leaves 0.48-0.52.
- `mtf_regime` (H): Multi-timeframe regime: daily efficiency ratio(20) > 0.4 -> hourly trend (EMA24/96); < 0.2 -> hourly z20 fade; otherwise flat.
- `multi_asset_ensemble` (D): Multi-asset signal ensemble: trend, cross-sectional momentum, the macro factor model and risk-on/off, equally blended.
- `multi_candle` (D): Three-day return sign: fade it for one day.
- `multi_factor_tech` (D): Equal blend of trend (EWMA multi-speed), MACD, RSI(2) reversion and Bollinger fade signals.
- `multi_ma` (D): Average of sign(price - SMA n) for n in 20, 50, 100, 200.
- `news_event_stat` (M): News-event statistical model: on 08:30 event days, follow or fade the release bar (08:45-11:00) per symbol by the sign of the mean continuation on earlier years' event days (|t| > 1.5).
- `nfp_reaction` (M): NFP reaction: on first-Friday payrolls days, follow the 08:30 bar from 08:45 to 11:00.
- `nn_pattern` (D): Nearest-neighbour pattern matching by correlation of 10-day return shapes (k = 30), next-5-day mean sign.
- `ny_session` (H): New York session (08-16) follows or fades the London session.
- `ofi_proxy` (M): Order-flow imbalance proxy: signed tick volume over the last 4 15-minute bars / total; follow the next bar when |imbalance| > 0.5.
- `oil_ccy` (D): Oil-currency relationship: USD/CAD's 20-day residual vs WTI (60d return beta), fade beyond 2 sd (120d), exit at 0.
- `oil_usd_ll` (H): Oil/USD lead-lag: a 1-sd WTI hour -> USD/CAD the opposite way next hour.
- `online_learning` (D): Online learning: SGD logistic regression updated every day with the day's realised labels (predict, then learn), from 2007 on.
- `open_to_close_mom` (M): Open-to-close momentum: the first hour (09:30-10:30) return sets the side, held 10:30 to 16:00.
- `open_to_hilo` (M): Open-to-high/low model: once price has run from the 09:30 open by more than the 60-day median open-to-high (open-to-low), fade it to 16:00.
- `opening_auction_proxy` (M): Opening auction proxy: the first 15 minutes of the cash session (09:30-09:45) set the side, held to 16:00.
- `opening_window` (M): Opening-window statistics: hold 09:30-10:30 in the sign of that window's earlier-years mean (|t| > 1.5), every CFD.
- `osc_composite` (D): Mean of z-scored RSI14, Stoch14, CCI20, Williams %R (250d z); fade beyond +-1.5, out at 0.
- `osc_pct` (D): RSI14 at its 250-day 5th percentile: long; 95th: short; out at 50th.
- `ou_mr` (D): Rolling 120-day OU fit; fade |(x-mu)/sigma_eq| > 1.5 when half-life under 30 days, exit at mu.
- `overnight_gap_rev` (M): US index cash gap (09:30 open vs prior 16:00 close) larger than 0.5 of its 60-day sd: fade it from 09:45 to 16:00.
- `overnight_return` (H): Overnight return: hold the US index CFDs long from the 16:00 close to the 09:30 open (which crosses the 17:00 financing roll).
- `pairs_trading` (D): Gatev-Goetzmann-Rouwenhorst distance pairs: each half-year pick the 5 closest normalised-price pairs per asset group over the past year; open at a 2 formation-sd divergence, close at the crossing.
- `parkinson_sizing` (D): Parkinson high-low volatility (20 days) as the sizing vol.
- `pattern_cond_prob` (D): Candle patterns (inside, outside-up, outside-down, other) x trend sign: expanding conditional next-day probability.
- `pca_neutral` (D): PCA factor-neutral: residuals of all 25 CFDs after 3 principal components, s-score entry 1.25, exit 0.5.
- `pct_entry` (D): One-day return beyond its 250-day 5th/95th percentile: fade 1 day.
- `pct_rank` (D): Rank assets by where today's 6-month return sits in its own 3-year history; long top third, short bottom third, weekly.
- `pf_filter` (D): Profit-factor filter: only while the last 30 trades' profit factor is above 1.2.
- `poly_fcst` (D): Quadratic fit to 20 days of log price, extrapolated one day: sign.
- `portfolio_signal` (D): Portfolio-level signal model: equal blend of trend, z-score mean reversion, cross-sectional momentum, breakout and MACD.
- `post_news_mom` (M): Post-news momentum: on days with a large 08:30 release bar, follow that bar's direction from 08:45 to 11:00.
- `post_news_mr` (M): Post-news mean reversion: fade the 08:30 release bar, 08:45-11:00.
- `pre_news` (M): Pre-news positioning: on payrolls Fridays (known in advance) hold 07:30-08:30 in the sign of the earlier-years mean of that window on payrolls days (|t| > 1.5).
- `pred_interval_bo` (D): Close outside the 95% prediction interval of an AR(1) on returns (250d): follow for 5 days.
- `price_ma_z` (D): z (250d) of price / SMA20 - 1, fade +-2.
- `price_vol_reg` (D): Price/volume regression: rolling 250-day regression of tomorrow's return on today's return x relative volume; trade the forecast sign.
- `price_vwap_z` (D): z (100d) of price vs its 20-day tick-volume-weighted average.
- `prob_payoff` (D): Probability x payoff: p(win) x average win - p(loss) x average loss over all earlier trades of the asset must be positive.
- `prop_firm_aware` (D): Prop-firm constraint-aware: VaR-scaled book (as above), size cut linearly with drawdown toward a 10% limit, flat the day after a 2% loss.
- `quant_breakout` (D): Close beyond the prior 50-day high/low; exit on the 25-day opposite.
- `quantile_reg` (D): Quantile regression (yearly refit, prior 3 years) of the next 5-day return on z20 and the 20-day return: long when the 30th-percentile forecast is above 0, short when the 70th is below 0.
- `r_multiple_dist` (D): R-multiple distribution model: take trades only while the last 50 R-multiples have a positive mean AND a 25th percentile above -1 (the stop is doing its job, no gap-through losses).
- `range_compression` (D): NR7 day: follow the next close outside its range for 5 days.
- `range_exp_breakout` (D): Range > 2 ATR, close in the outer quarter: follow for 5 days.
- `range_exp_prob` (D): Range-expansion probability: expanding P(next day's range > 1.2 ATR | NR7 / inside-day state); when a compressed state's probability is above the asset's unconditional rate, the next close beyond today's range is followed for 3 days.
- `range_expansion` (D): Widest range in 7 days: fade its direction next day (exhaustion).
- `range_vol_sizing` (D): Range-based volatility: mean daily log range x 0.627 (20 days) as the sizing vol.
- `rates_diff` (D): Rates-differential proxy: EUR/USD follows the 20-day return of the Bund CFD minus the Treasury CFD (US yields rising relatively -> USD up); GBP/USD likewise with Gilts.
- `real_yield_proxy` (D): Real-yield proxy (no TIPS data): gold follows the 20-day direction of 10-year Treasury prices (falling yields -> long gold).
- `reg_to_mean` (D): Price > 2 residual sd from its 100-day regression line: fade to it.
- `regime_cond_bo` (D): Regime-conditional breakout: 20-day Donchian only in the low half of the 2-year volatility range.
- `regime_cond_mom` (D): Regime-conditional momentum: 3-month time-series momentum only while the cross-asset average 60-day efficiency ratio is above its 2-year median (a trending market overall).
- `regime_cond_mr` (D): Regime-conditional mean reversion: z20 fade only where the 250-day Hurst exponent is below 0.5.
- `regime_cond_trend` (D): Regime-conditional trend following: the multi-speed trend model, long only while the 200-day average rises and short only while it falls.
- `regime_trend` (D): Trend (EMA50/200) only when the 60-day efficiency ratio > 0.25.
- `regime_weighting` (D): Regime-dependent weighting: when the average asset's 20-day vol percentile is above 0.5, 60% mean reversion / 10% each other; otherwise 60% trend / 10% each other.
- `regression_channel_bo` (D): Close outside the 50-day regression line +-2 residual sd: follow; exit back at the line.
- `rel_vol_anomaly` (H): Relative-volume anomaly: volume > 2.5x the same hour's 20-day average, follow the bar for 4 hours.
- `relative_value` (D): Price-ratio relative value (hedge 1:1 in value): fade the 60-day z of log(a/b) beyond 2, exit at 0.
- `resid_rev` (D): Residual of the 20-day regression line, fade |z| > 1.5.
- `resid_z` (D): Residual z-score entry: 20-day cumulative residual vs the group factor (60d beta), z over 120 days, fade beyond 2, exit 0.
- `ret_autocorr_weekly` (D): Weekly-return autocorrelation (52 weeks) sets follow/fade of last week.
- `risk_adj_mom` (D): 12-month return over 12-month vol (a Sharpe), clipped, as weight.
- `risk_adj_selection` (D): Risk-adjusted signal selection: each month hold only the strategy with the best trailing 1-year Sharpe.
- `risk_of_ruin` (D): Risk-of-ruin filter: from the last 50 trades' win rate and payoff, the probability of a 20R drawdown (1% risk per trade, 20% ruin) must be under 5%; otherwise skip new trades.
- `risk_on_off` (D): Risk-on/off factor (20d z of S&P, AUDJPY, EURJPY, minus Treasuries): each asset takes sign(its 250d beta to the factor) x sign(factor).
- `risk_parity_alloc` (D): Risk-parity signal allocation: the five strategies weighted for equal risk contribution (1-year covariance), monthly.
- `rl_qlearn` (D): Tabular Q-learning: state = (trend sign, z20 bucket, vol regime, current position), actions short/flat/long, reward = next-day return minus cost on changes; trained each year on earlier data (10 passes, epsilon-greedy), then acted greedily on the test year.
- `roll_reg_entry` (D): Rolling 500-day regression of next-day return on 1, 5 and 20-day returns; trade the forecast sign.
- `rolling_param` (D): Rolling-parameter model: every month choose the momentum lookback (1/3/6/12 months) with the best trailing 1-year net Sharpe per CFD.
- `rolling_window` (D): Rolling-window (3 years) ridge regression, refit yearly.
- `rp_sizing` (D): Risk-parity (equal risk contribution) sizing across assets, using the covariance of the strategy's per-asset returns over the past year; rebalanced monthly.
- `rs_quant` (D): Relative-strength line (asset / its group index) above its 50-day average: long; below: short; demeaned within the group.
- `rsi_quant` (D): Connors RSI(2): long < 10, short > 90, exit through 50.
- `rv_breakout` (D): Hourly realized vol of the day (sum of squares) z > 2 (100d): follow the day's direction for 5 days.
- `scenario_match` (D): Historical scenario matching on a state vector (5d, 20d, 60d return z-scores and vol percentile): 50 nearest past states, mean next 5-day return sign.
- `sd_band` (D): 50-day mean +-2.5 sd band; fade outside, exit inside +-0.5 sd.
- `sentiment_extreme_proxy` (D): Sentiment extreme reversion, VIX as the fear gauge: VIX in the top 10% of its 1-year range -> long S&P and Nasdaq for 20 days.
- `sentiment_mom_div_proxy` (D): Sentiment-momentum divergence (VIX proxy): S&P at a 20-day high while VIX is also above its 20-day average -> short 5 days; S&P at a 20-day low with VIX below average -> long 5 days.
- `serial_mom` (D): Follow yesterday's return when the 60-day lag-1 autocorr > 0.1.
- `serial_rev` (D): Fade yesterday's return when the 60-day lag-1 autocorr < -0.1.
- `session_adj_stat` (H): Session-adjusted statistical model: hourly returns standardised by their hour-of-day volatility (expanding, prior data); fade |z| > 2.5 for 3 hours.
- `session_season` (H): Session seasonality: each session (Asia 19-03, London 03-08, New York 08-16 NY time) held in the sign of its earlier-years mean when |t| > 1.5.
- `session_transition` (H): Session transition: first two London hours (03-05) follow or fade the last Asia hour (02-03), by trailing correlation.
- `sharpe_mom_filter` (D): TSMOM taken only when the trailing 6-month |Sharpe| > 0.5.
- `slippage_filter` (H): Execution-slippage filter: no entry on a bar whose range is more than 3 hourly ATRs (fast market).
- `slope_entry` (D): EMA20 5-day slope in ATR units: long > 0.1, short < -0.1.
- `sortino_mom_filter` (D): As the Sharpe filter but with the Sortino ratio (downside dev).
- `spread_exp_filter` (H): Spread-expansion filter (no quote data, so the known wide-spread hours): no entries 16:00-20:00 New York or in the first two hours of the week.
- `spread_norm_entry` (H): Spread-normalised entry: the hourly fade only when hourly ATR is more than 15 spreads.
- `spread_z` (D): Log-price spread on a rolling 120-day OLS hedge; enter |z| > 2, exit at 0.
- `stat_arb` (D): Avellaneda-Lee stat arb within each asset group: residuals after the first principal component, s-score entry 1.25, exit 0.5.
- `stat_ll` (H): Statistical lead-lag: each year, lagged hourly cross-correlations (prior 2 years, |t| > 4, own lag excluded) form a next-hour forecast; trade it when it exceeds the round-trip cost.
- `stat_mr` (D): Continuous fade of the 20-day z-score of log price.
- `stat_range_breakout` (D): Close above the 95th / below the 5th percentile of 100 closes; exit at the median.
- `state_space_trend` (D): Unobserved-components local linear trend with noise variances fitted by maximum likelihood each year (statsmodels); sign of filtered slope.
- `stoch_quant` (D): Stochastic(14,3) crosses up through 20: long until 80; mirror.
- `structural_break` (D): Mean of the last 60 days vs the 250 before differs with |t| > 3: trade the new direction.
- `surprise_magnitude_proxy` (M): Surprise-magnitude proxy (no consensus data): on payrolls days a release bar larger than 2x the median payrolls bar of earlier releases is followed, a smaller one faded, 08:45-11:00.
- `swing_level` (D): Close beyond the last confirmed swing (5 bars each side); exit on the opposite swing.
- `tech_factor` (D): Technical factor model: cross-sectional composite of 12-1 momentum, MACD histogram (vol-normalised) and 1-month trend ranks; long top third, short bottom third, weekly.
- `tick_imbalance` (M): Tick imbalance: of the last 8 15-minute bars, 6+ up (down) -> follow the next bar.
- `tick_vol_anomaly` (H): Tick-volume anomaly: hourly volume z (500h) > 3, follow the bar's direction for 4 hours.
- `tick_volume` (D): Tick-volume model: a day with volume > 1.5x its 20-day average is followed the next day in its candle's direction.
- `tod_season` (H): Time-of-day seasonality: each year, the hours whose mean return over all earlier years has |t| > 2 are traded in that direction.
- `trade_base` (D): Reference trade system for the filters: 20-day Donchian breakout, 2-ATR stop, 10-day opposite-channel exit.
- `trade_intensity` (M): Trade intensity: 15-minute tick volume z (960 bars) > 3, follow the bar for 4 bars.
- `trend_follow` (D): Multi-speed EWMA crossover trend (8/24, 16/48, 32/96), averaged.
- `trend_strength` (D): R^2 of the 60-day regression > 0.5: trade the slope's direction.
- `trend_volregime` (D): EMA50/200 trend unless 20d vol is above its 80th 2-year percentile.
- `triple_barrier` (D): Triple-barrier model: labels from +-1.5 x 20-day sd barriers with a 10-day vertical barrier (first touch, from daily highs/lows); a boosting classifier predicts the upper-vs-lower outcome; trade p > 0.55 / < 0.45 weekly.
- `ts_mom` (D): Time-series momentum: sign of the 12-month return (Moskowitz et al.).
- `ts_mr` (D): Time-series reversal: fade the 5-day return in units of its vol.
- `utility_max` (D): Utility-maximising entry: mean-variance weight mu/(gamma sigma^2) with gamma 5 and the 12-month mean shrunk by half; only re-traded when the target moves by more than 25%.
- `var_filter` (D): VaR filter: the same with the 99% VaR of vol-standardised returns.
- `variance_ratio` (D): Lo-MacKinlay VR(5) over 250 days: > 1.1 follow the 5-day return, < 0.9 fade it.
- `vcp` (D): Volatility contraction: 10d vol < 0.7 x 50d vol, close at a 20-day high (long only, Minervini-style); exit at the 10-day low.
- `vol_bars` (H): Volatility bars (a new bar after 1 daily-sd of cumulative absolute hourly return): momentum of the last 5 bars.
- `vol_breakout` (D): A daily move beyond 2 sd (60d): follow it for 5 days.
- `vol_clustering` (D): Volatility clustering: size by a fast EWMA vol (span 10) that reacts within days to vol bursts, instead of the 60-day default.
- `vol_exp_cont` (D): ATR5/ATR20 > 1.3 in the direction of the EMA20/50 trend: follow.
- `vol_fcst_rev` (D): Index CFDs only: 20d vol > 1.5x the 3-year median (vol expected to fall) -> long for 20 days. Buy fear.
- `vol_mr_fade` (D): When 10d vol > 1.5x its 1-year median, fade the 5-day move.
- `vol_price_div` (D): Volume/price divergence: a new 20-day high on below-average volume -> short 5 days; a new low on below-average volume -> long 5 days.
- `vol_rank` (D): Low-volatility ranking: long lowest-vol third, short highest-vol third (each risk-scaled), monthly.
- `vol_regime_mr` (D): z20 fade only in the high-vol half of the 2-year range.
- `vol_regime_switch` (D): High-vol regime (60d vol in the top 3rd of 3 years): fade z20; otherwise 3-month momentum.
- `vol_regime_trend` (D): Trend (EMA50/200) only in the low-vol half of the 2-year range.
- `vol_scaled_entry` (D): 20-day return in sd units > 1: long; < -1: short; out inside 0.
- `vol_shock` (D): Vol shock (5d vol > 2x 60d vol): fade the 5-day move for 5 days.
- `vol_spread_ratio` (H): Volatility/spread ratio: each day trade the hourly fade only in the half of CFDs with the highest daily ATR per unit of spread.
- `vol_target_sizing` (D): Volatility-target sizing: each position at 10% annual vol (EWMA 60).
- `vol_targeting` (D): Volatility targeting as a strategy: hold every CFD long at 10% vol (the risk premium, vol-timed), compared with the same basket held at fixed notional ("long_basket_fixed").
- `vol_targeting_portfolio` (D): Portfolio volatility targeting: asset-level vol sizing, then the whole book scaled so its trailing 60-day realised vol is 10%.
- `vol_volatility` (D): Volume-volatility relationship: high volume (z > 1) on a narrow range (range z < -0.5) = absorption; trade toward the close location for 3 days.
- `voladj_breakout` (D): Close beyond the 20-day high/low by at least 0.5 ATR; exit at SMA20.
- `voladj_rank` (D): Volatility-adjusted ranking: 6-month return / 6-month vol, monthly.
- `voladj_selection` (D): Volatility-adjusted selection: strategy weights proportional to 1 / trailing-year volatility of each strategy's returns.
- `volofvol` (D): TSMOM, flat while vol-of-vol (sd of 20d vol over 60d) is in its top quintile of 2 years.
- `volscaled_prop` (D): Volatility-scaled prop-firm model: the book is scaled so its 99% one-day VaR (2.33 x trailing 20-day vol) is 1% -- half a typical 2% daily-loss allowance.
- `volume_z` (D): Volume z-score: daily volume z (60d) > 2 -> fade the day's candle next day (climax).
- `vr_mr` (D): Variance-ratio mean reversion: only the VR(5) < 0.9 side, fading the 5-day return.
- `vrp` (D): Volatility risk premium: long the S&P while VIX^2 exceeds 20-day realised variance (premium positive); flat when it inverts.
- `vw_flow` (D): Volume-weighted flow: sign of the tick-volume-weighted mean daily return over 20 days.
- `vwap_dev` (M): VWAP deviation reversion: session VWAP (tick volume, from the 17:00 roll) and its volume-weighted sd; fade |z| > 2 back to VWAP; flat from 16:00 to the roll.
- `weekend_gap` (H): Weekend gap: fade the move from Friday's close to the close of the week's first hour, for the next 6 hours.
- `wf_breakout` (D): Walk-forward breakout: Donchian 10/20/55/100 (exit at half) chosen per CFD each year.
- `wf_momentum` (D): Walk-forward momentum: lookback 1, 3, 6 or 12 months chosen per CFD each year on the prior 3 years.
- `wf_mr` (D): Walk-forward mean reversion: z lookback 10/20/50 x entry 1.5/2/2.5 chosen per CFD each year.
- `wf_optimized` (D): Walk-forward optimised moving-average crossover: each year pick the best of 9 fast/slow EMA pairs per CFD on the prior 3 years.
- `yield_curve` (D): Yield-curve signal: 2s10s slope change over 60 days from the bond CFDs (dy ~ -return/duration, durations 1.9 and 8.5); hold equity indices long while the curve is steepening, flat while it flattens.
- `yz_sizing` (D): Yang-Zhang volatility (overnight + open-close + Rogers-Satchell, 20 days) as the sizing vol.
- `z_mr` (D): Enter against |z20| > 2, exit when z crosses 0.
- `zlema` (D): Price vs zero-lag EMA(50): sign.
