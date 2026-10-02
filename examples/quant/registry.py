"""
The user's list, in its original order, mapped to what was tested.

Each line: name | model key (blank = not testable here) | note.
A key used by more than one name is the same test reported under each
name: the list repeats ideas under different labels, and testing the same
thing twice with different parameters would only add another lottery
ticket to the multiple-testing count.
"""

RAW = """
Statistical Mean Reversion | stat_mr |
Z-Score Mean Reversion | z_mr |
Bollinger Z-Score Reversion | boll_z |
VWAP Deviation Reversion | vwap_dev |
Standard Deviation Band Reversion | sd_band |
Ornstein-Uhlenbeck Mean Reversion | ou_mr |
Half-Life Mean Reversion | halflife_mr |
Cointegration Pair Trading | coint_pairs |
Statistical Arbitrage | stat_arb |
Pairs Trading | pairs_trading |
Relative-Value Trading | relative_value |
Cross-Asset Relative Value | cross_asset_rv |
Beta-Neutral Pairs Trading | beta_neutral_pairs |
Correlation Breakdown/Reversion | corr_reversion |
Spread Z-Score Entry | spread_z |
Residual Z-Score Entry | resid_z |
Kalman Filter Pairs Trading | kalman_pairs |
Dynamic Hedge-Ratio Pairs Trading | dyn_hedge_pairs |
PCA Factor-Neutral Trading | pca_neutral |
Factor Residual Mean Reversion | factor_resid_mr |
Cross-Sectional Mean Reversion | cs_mr |
Time-Series Mean Reversion | ts_mr |
Momentum Factor | momentum_factor |
Cross-Sectional Momentum | cs_mom |
Time-Series Momentum | ts_mom |
Trend Following | trend_follow |
Quantitative Breakout | quant_breakout |
Volatility Breakout | vol_breakout |
Donchian Breakout | donchian |
ATR Breakout | atr_breakout |
Volatility-Adjusted Breakout | voladj_breakout |
Range Expansion Breakout | range_exp_breakout |
Statistical Range Breakout | stat_range_breakout |
Volatility Compression → Expansion | compress_expand |
Bollinger Band Squeeze | bb_squeeze |
Keltner Channel Breakout | keltner |
ATR Channel Breakout | atr_channel |
Historical Volatility Breakout | hv_breakout |
Realized Volatility Breakout | rv_breakout |
Volatility Regime Switching | vol_regime_switch |
Volatility Targeting | vol_targeting |
Regime-Based Trend Following | regime_trend |
Markov Regime-Switching | markov_switch |
Hidden Markov Model Entry | hmm_entry |
Bayesian Regime Detection | bayes_regime |
Change-Point Detection | change_point |
Structural Break Trading | structural_break |
Autocorrelation-Based Entry | autocorr_entry |
Serial-Correlation Momentum | serial_mom |
Serial-Correlation Reversion | serial_rev |
AR(1) Forecast Entry | ar1_fcst |
ARIMA Forecast Entry | arima_fcst |
Exponential Smoothing Forecast | exp_smooth |
Kalman Filter Trend | kalman_trend |
Kalman Filter Mean Reversion | kalman_mr |
State-Space Trend Model | state_space_trend |
Linear Regression Forecast | linreg_fcst |
Polynomial Regression Forecast | poly_fcst |
Rolling Regression Entry | roll_reg_entry |
Regression-to-Mean Entry | reg_to_mean |
Regression Channel Breakout | regression_channel_bo |
Residual Reversion | resid_rev |
Forecast Error Reversion | fcst_err_rev |
Prediction Interval Breakout | pred_interval_bo |
Quantile Regression Entry | quantile_reg |
Percentile-Based Entry | pct_entry |
Empirical Distribution Entry | empirical_dist |
Historical Percentile Reversion | hist_pct_rev |
Volatility-Scaled Entry | vol_scaled_entry |
ATR-Normalized Entry | atr_norm_entry |
Risk-Adjusted Momentum Entry | risk_adj_mom |
Sharpe-Ratio Momentum Filter | sharpe_mom_filter |
Sortino-Ratio Momentum Filter | sortino_mom_filter |
Momentum + Volatility Filter | mom_vol_filter |
Trend + Volatility Regime Filter | trend_volregime |
Trend Strength Quant Model | trend_strength |
ADX Quant Filter | adx_filter |
Slope-Based Trend Entry | slope_entry |
Moving-Average Slope Model | ma_slope |
Linear Regression Slope Entry | linreg_slope |
Multi-Moving-Average Quant Model | multi_ma |
Moving-Average Distance Model | ma_dist |
EMA Distance Z-Score | ema_dist_z |
Price-to-VWAP Z-Score | price_vwap_z |
Price-to-MA Z-Score | price_ma_z |
MA Spread Z-Score | ma_spread_z |
MACD Quantitative Signal | macd |
RSI Quantitative Signal | rsi_quant |
Stochastic Quantitative Signal | stoch_quant |
Oscillator Percentile Entry | osc_pct |
Momentum Oscillator Composite | osc_composite |
Multi-Factor Technical Model | multi_factor_tech |
Technical Factor Model | tech_factor |
Factor Score Model | factor_score |
Weighted Confluence Score | confluence |
Bayesian Signal Aggregation | bayes_agg |
Logistic Regression Direction Model | ml_logit |
Linear Classification Model | ml_lda |
Random Forest Direction Model | ml_rf |
Gradient Boosting Direction Model | ml_gbm |
XGBoost Signal Model | ml_xgb |
Support Vector Machine Entry | ml_svm |
k-Nearest Neighbors Entry | ml_knn |
Neural-Network Direction Model | ml_mlp |
LSTM Sequence Model | ml_lstm |
Transformer Time-Series Model | ml_transformer |
Reinforcement-Learning Entry | rl_qlearn |
Ensemble Model Entry | ml_ensemble |
Stacked Ensemble Entry | ml_stacked |
Voting Classifier Entry | ml_voting |
Probability Threshold Entry | ml_prob_threshold |
Expected-Value Threshold Entry | ml_ev_threshold |
Conditional Probability Entry | cond_prob |
Bayesian Probability Update | bayes_prob_update |
Monte Carlo Entry Model | monte_carlo_entry |
Monte Carlo Path Forecast | mc_path_fcst |
Bootstrap Distribution Entry | bootstrap_dist |
Historical Scenario Matching | scenario_match |
Analog Pattern Matching | analog_match |
Nearest-Neighbor Pattern Matching | nn_pattern |
DTW Pattern Matching | dtw_pattern |
Fractal Pattern Model | fractal_breakout |
Hurst Exponent Trend/Reversion Model | hurst |
Entropy-Based Market Regime Model | entropy_regime |
Market Efficiency Ratio Model | efficiency_ratio |
Variance-Ratio Model | variance_ratio |
Variance-Ratio Mean Reversion | vr_mr |
Fractal Dimension Model | fractal_dim |
Long-Memory Model | long_memory |
Volatility Clustering Model | vol_clustering |
GARCH Volatility Model | garch_sizing |
EGARCH Volatility Model | egarch_sizing |
GARCH Forecast Breakout | garch_breakout |
Volatility Forecast Reversion | vol_fcst_rev |
Volatility Shock Model | vol_shock |
Volatility-of-Volatility Model | volofvol |
Range-Based Volatility Model | range_vol_sizing |
Parkinson Volatility Model | parkinson_sizing |
Garman-Klass Volatility Model | gk_sizing |
Yang-Zhang Volatility Model | yz_sizing |
Volatility Risk Premium Model | vrp | VIX vs realised S&P vol
ATR Regime Model | atr_regime |
Implied-vs-Realized Volatility Model | iv_rv | VIX is the only implied vol available
CFD Spread-Normalized Entry | spread_norm_entry |
CFD Overnight-Financing-Aware Entry | financing_aware |
CFD Cost-Adjusted Entry | cost_adj_entry |
CFD Spread Expansion Filter | spread_exp_filter | no quote history: known wide-spread hours used instead
CFD Execution-Slippage Filter | slippage_filter |
CFD Liquidity-Adjusted Entry | liquidity_adj |
CFD Volatility/Spread Ratio Model | vol_spread_ratio |
CFD Expected-Move Model | expected_move_model |
CFD Session-Adjusted Statistical Model | session_adj_stat |
CFD Gap-Adjusted Model | gap_adjusted |
CFD Overnight Gap Reversion | overnight_gap_rev |
CFD Weekend Gap Model | weekend_gap |
CFD Opening Auction Proxy Model | opening_auction_proxy |
CFD Cash-Session Momentum | cash_session_mom |
CFD Futures-to-CFD Lead/Lag Model | | needs exchange futures quotes alongside the CFD's; a CFD is priced off the future, so the lag is the broker's, sub-second
CFD Spot-to-Futures Lead/Lag Model | | needs both spot and futures series
CFD Index-to-Component Lead/Lag Model | | needs index constituents' prices
CFD Cross-Market Lead/Lag | cross_market_ll |
FX Lead/Lag Model | fx_ll |
Gold/USD Lead-Lag Model | gold_usd_ll |
Oil/USD Lead-Lag Model | oil_usd_ll |
Bond/Yield Lead-Lag Model | bond_yield_ll |
Equity Index/FX Lead-Lag Model | eq_fx_ll |
Intermarket Confirmation Model | intermarket_confirm |
Macro Factor Model | macro_factor |
Dollar-Index Factor Model | dollar_factor | synthetic dollar from four USD pairs
Rates-Differential Model | rates_diff | proxy: bond CFD prices, not yields
Yield-Curve Signal Model | yield_curve | proxy: 2s10s from bond CFD prices
Real-Yield Signal Model | real_yield_proxy | proxy: no inflation-linked data; nominal Treasury prices used
Commodity-Currency Model | commodity_ccy |
Risk-On/Risk-Off Factor Model | risk_on_off |
Equity-Bond Correlation Regime Model | eq_bond_corr |
Dollar-Gold Relationship Model | dollar_gold |
Oil-Currency Relationship Model | oil_ccy |
Relative Strength Quant Model | rs_quant |
Cross-Asset Relative Strength | cross_asset_rs |
Currency Strength Model | ccy_strength |
Sector Relative Strength | | no sector CFDs in the data
Index Relative Strength | index_rs |
Asset-Cluster Relative Strength | cluster_rs |
Dispersion Trading | | needs index and single-stock options or constituents
Correlation Trading | | needs options or correlation swaps; the CFD-tradable forms are the correlation spread/breakout/reversion rows
Correlation Spread Trading | corr_spread |
Correlation Breakout | corr_breakout |
Correlation Reversion | corr_reversion | same test as Correlation Breakdown/Reversion
Lead-Lag Arbitrage | lead_lag_arb |
Latency-Based Lead-Lag | | sub-second; not reachable through retail CFD execution or 1-minute data
Statistical Lead-Lag | stat_ll |
Cross-Sectional Ranking | cs_rank |
Percentile Ranking | pct_rank |
Momentum Ranking | mom_rank |
Volatility Ranking | vol_rank |
Mean-Reversion Ranking | mr_rank |
Composite Ranking Model | composite_rank |
Long/Short Ranking Model | ls_rank |
Market-Neutral Ranking Model | mkt_neutral_rank |
Beta-Adjusted Ranking | beta_adj_rank |
Volatility-Adjusted Ranking | voladj_rank |
Risk-Parity Signal Allocation | risk_parity_alloc |
Inverse-Volatility Position Selection | inverse_vol_selection |
Kelly-Criterion Signal Filter | kelly_filter |
Fractional-Kelly Entry Filter | frac_kelly_filter |
Expected-Shortfall Filter | es_filter |
VaR-Based Signal Filter | var_filter |
Drawdown-Regime Filter | drawdown_regime |
Maximum-Adverse-Excursion Model | mae_model |
Maximum-Favorable-Excursion Model | mfe_model |
R-Multiple Distribution Model | r_multiple_dist |
Expectancy-Based Entry Filter | expectancy_filter |
Profit-Factor Signal Filter | pf_filter |
Conditional Expectancy Model | cond_expectancy |
Time-of-Day Seasonality Model | tod_season |
Day-of-Week Seasonality | dow_season |
Month-of-Year Seasonality | moy_season |
Session Seasonality | session_season |
Intraday Seasonal Pattern | intraday_pattern |
Opening-Window Statistical Model | opening_window |
Closing-Window Statistical Model | closing_window |
London Session Statistical Model | london_session |
New York Session Statistical Model | ny_session |
Asia Session Statistical Model | asia_session |
Session-Transition Model | session_transition |
News-Event Statistical Model | news_event_stat | release days found from the 08:30 volatility spike
Pre-News Positioning Model | pre_news | payrolls Fridays, known in advance
Post-News Mean Reversion | post_news_mr |
Post-News Momentum | post_news_mom |
Economic Surprise Model | econ_surprise_proxy | proxy: no consensus data; the release bar's sign is the surprise
Surprise-Magnitude Model | surprise_magnitude_proxy | proxy: release-bar size as the surprise size
Event Volatility Model | event_vol |
Event Drift Model | event_drift |
CPI Reaction Model | cpi_reaction_proxy | proxy calendar: large 08:30 bars on the 10th-17th
NFP Reaction Model | nfp_reaction | first-Friday calendar
FOMC Reaction Model | fomc_reaction | statement days found from the 14:00/14:15 spike
Central-Bank Decision Model | cb_decision | FOMC + ECB days found from their spikes
Earnings CFD Reaction Model | | no single-stock CFDs or earnings calendar in the data
Earnings Drift Model | | as above
Post-Earnings Announcement Drift | | as above
Macro Event Regime Model | macro_event_regime |
News Sentiment Quant Model | | no news feed or text
NLP Sentiment Model | | no news feed or text
News Momentum Model | post_news_mom | news = scheduled releases; same test as Post-News Momentum
News Reversion Model | post_news_mr | same test as Post-News Mean Reversion
Sentiment Extreme Reversion | sentiment_extreme_proxy | VIX as the sentiment gauge
Sentiment-Momentum Divergence | sentiment_mom_div_proxy | VIX as the sentiment gauge
Positioning-Based Entry | | no positioning data reachable (CFTC and broker feeds blocked here)
COT Positioning Model | | CFTC COT not reachable from this environment
Retail Positioning Contrarian Model | | broker client-positioning history not available
Broker Sentiment Model | | as above
Long/Short Ratio Model | | as above
Commitment-of-Traders Extremes | | CFTC COT not reachable
Positioning Z-Score | | no positioning data
Positioning Momentum | | no positioning data
Flow-Based Model | flow_model | tick volume x candle sign as the flow proxy
Volume-Weighted Flow Model | vw_flow |
Cumulative Volume Delta Model | cvd | proxy: signed tick volume per hour (no trade-side data)
Tick-Volume Model | tick_volume |
Tick-Volume Anomaly | tick_vol_anomaly |
Relative-Volume Anomaly | rel_vol_anomaly |
Volume Z-Score | volume_z |
Volume/Price Divergence | vol_price_div |
Price/Volume Regression | price_vol_reg |
Volume-Volatility Relationship | vol_volatility |
Order-Flow Imbalance Model | ofi_proxy | proxy: signed 15-minute tick volume
Market-Microstructure Signal | micro_signal | proxy: close location x relative volume
Bid/Ask Imbalance Model | | no order book
Spread-Change Model | | no bid/ask history in this data
Tick Imbalance Model | tick_imbalance | 15-minute bars stand in for ticks
Trade-Intensity Model | trade_intensity |
Arrival-Rate Model | arrival_rate |
Hawkes Process Model | hawkes |
Jump Detection Model | jump_detection |
Price-Jump Reversion | jump_rev |
Price-Jump Continuation | jump_cont |
Intraday Jump-Diffusion Model | jump_diffusion |
Extreme-Value Theory Entry | evt_entry |
Tail-Event Reversion | evt_tail_rev |
Tail-Event Momentum | evt_tail_mom |
EVT Breakout Model | evt_breakout |
Expected-Move Breakout | expected_move_bo |
Implied-Move Breach | implied_move_breach | S&P only, VIX-implied move
Range-Expansion Probability Model | range_exp_prob |
Bayesian Breakout Probability | bayes_breakout |
Breakout Success-Rate Model | breakout_success |
Conditional Breakout Model | cond_breakout |
Failed-Breakout Probability Model | failed_breakout |
Statistical False-Breakout Model | false_breakout_stat |
Pattern-Conditional Probability | pattern_cond_prob |
Candle-Sequence Model | candle_sequence |
Markov Candle-State Model | markov_candle |
Multi-Candle Return Model | multi_candle |
Return Autocorrelation Model | ret_autocorr_weekly |
Overnight Return Model | overnight_return |
Intraday Return Seasonality | tod_season | same test as Time-of-Day Seasonality
Gap-Return Distribution Model | gap_return_dist |
Open-to-High/Low Model | open_to_hilo |
Close-to-Open Model | close_to_open |
Open-to-Close Momentum | open_to_close_mom |
High-Low Range Model | hl_range |
Range Compression Model | range_compression |
Range Expansion Model | range_expansion |
ATR Compression Model | atr_compression |
Volatility Contraction Pattern | vcp |
Volatility Expansion Continuation | vol_exp_cont |
Volatility Mean Reversion | vol_mr_fade |
Volatility Regime + Trend | vol_regime_trend |
Volatility Regime + Mean Reversion | vol_regime_mr |
Regime-Conditional Momentum | regime_cond_mom |
Regime-Conditional Mean Reversion | regime_cond_mr |
Regime-Conditional Breakout | regime_cond_bo |
Regime-Conditional Trend Following | regime_cond_trend |
Adaptive Moving-Average Model | adaptive_ma |
Kaufman Adaptive Moving Average Model | kama |
Hull Moving Average Quant Model | hma |
Zero-Lag Moving Average Model | zlema |
Adaptive Channel Model | adaptive_channel |
Dynamic Support/Resistance Model | dyn_sr |
Machine-Learned Support/Resistance | ml_sr |
Quantified Swing-Level Model | swing_level |
Price-Level Reaction Probability | level_reaction_prob |
Level-Strength Scoring Model | level_strength |
Bayesian Support/Resistance Model | bayes_sr |
Multi-Timeframe Statistical Alignment | mtf_align |
Multi-Timeframe Probability Model | mtf_prob |
Multi-Timeframe Regime Model | mtf_regime |
HTF Trend Probability + LTF Trigger | htf_trend_ltf |
HTF Mean Reversion + LTF Trigger | htf_mr_ltf |
Ensemble Multi-Timeframe Model | mtf_ensemble |
Walk-Forward Optimized Model | wf_optimized |
Rolling-Parameter Model | rolling_param |
Adaptive Parameter Model | adaptive_param |
Online Learning Model | online_learning |
Expanding-Window Model | expanding_window |
Rolling-Window Model | rolling_window |
Walk-Forward Momentum | wf_momentum |
Walk-Forward Mean Reversion | wf_mr |
Walk-Forward Breakout | wf_breakout |
Portfolio-Level Signal Model | portfolio_signal |
Multi-Asset Signal Ensemble | multi_asset_ensemble |
Risk-Adjusted Signal Selection | risk_adj_selection |
Correlation-Adjusted Signal Selection | corr_adj_selection |
Volatility-Adjusted Signal Selection | voladj_selection |
Dynamic Signal Weighting | dynamic_weighting |
Regime-Dependent Signal Weighting | regime_weighting |
Bayesian Model Averaging | bma |
Ensemble Technical Model | ensemble_tech |
Ensemble Macro Model | ensemble_macro |
Ensemble Cross-Asset Model | ensemble_xasset |
Meta-Labeling | meta_label |
Triple-Barrier Entry Model | triple_barrier |
Event-Based Sampling Model | cusum_event |
Dollar-Bar Model | dollar_bars | price x tick volume
Volatility-Bar Model | vol_bars |
Information-Bar Model | info_bars |
Fractional Differentiation Model | frac_diff |
Feature-Engineered Return Model | ml_feature_return |
Machine-Learned Probability-of-Return Model | ml_prob_return |
Machine-Learned Probability-of-Breakout Model | ml_prob_breakout |
Machine-Learned Probability-of-Reversal Model | ml_prob_reversal |
Machine-Learned Expected-Return Model | ml_expected_return |
Machine-Learned Risk/Reward Model | ml_risk_reward |
Expected-Value Optimization | ev_optimization |
Utility-Maximizing Entry | utility_max |
Probability × Payoff Model | prob_payoff |
Conditional R-Multiple Model | cond_r_multiple |
Kelly-Optimized Entry | kelly_opt_entry |
Risk-of-Ruin Filter | risk_of_ruin |
Drawdown-Aware Signal Model | drawdown_aware_signal |
Capital-Preservation Quant Model | capital_preservation |
Prop-Firm Constraint-Aware Quant Model | prop_firm_aware |
Daily-Loss-Limit-Aware Model | daily_loss_aware |
Maximum-Drawdown-Aware Model | max_dd_aware |
Volatility-Scaled Prop-Firm Model | volscaled_prop |
Dynamic Risk-Sizing Model | dynamic_risk_sizing |
Fixed-Fractional Quant Model | fixed_fractional |
ATR Position-Sizing Model | atr_sizing |
Volatility Target Position Sizing | vol_target_sizing |
Kelly Position Sizing | kelly_sizing |
Fractional-Kelly Position Sizing | frac_kelly_sizing |
Risk-Parity Position Sizing | rp_sizing |
Correlation-Adjusted Position Sizing | corr_adj_sizing |
Expected-Shortfall Position Sizing | es_sizing |
Drawdown-Adjusted Position Sizing | dd_adj_sizing |
"""

# Reference streams the overlays and filters are judged against.
REFERENCES = ["trade_base", "h1_mr_base", "breakout_plain",
              "long_basket_fixed", "vol_targeting_portfolio"]


def entries():
    out = []
    for line in RAW.strip().splitlines():
        name, key, note = (x.strip() for x in line.split("|"))
        out.append((name, key or None, note))
    return out
