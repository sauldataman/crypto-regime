# BTC Regime Data Pipeline — Quality Report

Generated: 2026-04-01 16:16

## Raw Data Summary

| Source | Rows | Date Range | Columns |
|--------|------|------------|---------|
| BTC Price | 3744 | 2016-01-01 → 2026-04-01 | btc_open, btc_high, btc_low, btc_close, btc_volume |
| Macro | 2708 | 2016-01-01 → 2026-04-01 | sp500, nasdaq, gold, crude_oil, cnyusd, eurusd, jpyusd, dxy, treasury_10y, m2, vix, cpi |
| On-chain | 3740 | 2016-01-01 → 2026-03-31 | hash_rate, tx_count, mining_difficulty |

## Processed Dataset

- **Total rows**: 3744
- **Total columns**: 47
- **Date range**: 2016-01-01 → 2026-04-01

### Missing Values (top 10)

| Column | Missing | % |
|--------|---------|---|
| btc_realized_vol_30d_lag7 | 27 | 0.7% |
| btc_realized_vol_30d | 20 | 0.5% |
| gold_lag7 | 10 | 0.3% |
| dxy_lag7 | 10 | 0.3% |
| crude_oil_lag7 | 10 | 0.3% |
| vix_lag7 | 10 | 0.3% |
| treasury_10y_lag7 | 10 | 0.3% |
| sp500_lag7 | 10 | 0.3% |
| nasdaq_lag7 | 10 | 0.3% |
| btc_daily_return_lag7 | 8 | 0.2% |

## Regime Distribution

| Regime | Start | End | Days |
|--------|-------|-----|------|
| early | 2016-01-01 | 2020-10-31 | 1766 |
| late | 2020-11-01 | 2024-01-10 | 1166 |
| post_etf | 2024-01-11 | 2026-04-01 | 812 |

## TimesFM Dataset

- **Training examples**: 462
- **Window size**: 512 days
- **Stride**: 7 days
- **Covariates per example**: 46
- **Covariate names**: btc_close_lag7, btc_daily_return, btc_daily_return_lag7, btc_etf_indicator, btc_high, btc_high_lag7, btc_low, btc_low_lag7, btc_open, btc_open_lag7, btc_realized_vol_30d, btc_realized_vol_30d_lag7, btc_volume, btc_volume_lag7, cnyusd...
