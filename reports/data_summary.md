# BTC Regime Data Pipeline — Quality Report

Generated: 2026-04-07 18:12

## Raw Data Summary

| Source | Rows | Date Range | Columns |
|--------|------|------------|---------|
| BTC Price | 4221 | 2014-09-17 → 2026-04-07 | btc_open, btc_high, btc_low, btc_close, btc_volume |
| Macro | 4298 | 2010-01-01 → 2026-04-07 | sp500, nasdaq, gold, crude_oil, cnyusd, eurusd, jpyusd, dxy, treasury_10y, m2, vix, cpi |
| On-chain | 3746 | 2016-01-01 → 2026-04-06 | hash_rate, tx_count, mining_difficulty |

## Processed Dataset

- **Total rows**: 4221
- **Total columns**: 48
- **Date range**: 2014-09-17 → 2026-04-07

### Missing Values (top 10)

| Column | Missing | % |
|--------|---------|---|
| mining_difficulty_lag7 | 537 | 12.7% |
| tx_count_lag7 | 537 | 12.7% |
| hash_rate_lag7 | 537 | 12.7% |
| mining_difficulty | 530 | 12.6% |
| tx_count | 530 | 12.6% |
| hash_rate | 530 | 12.6% |
| btc_realized_vol_30d_lag7 | 86 | 2.0% |
| btc_realized_vol_30d | 79 | 1.9% |
| btc_daily_return_lag7 | 67 | 1.6% |
| btc_open_lag7 | 66 | 1.6% |

## Regime Distribution

| Regime | Start | End | Days |
|--------|-------|-----|------|
| early | 2014-09-17 | 2020-10-31 | 2237 |
| late | 2020-11-01 | 2024-01-10 | 1166 |
| post_etf | 2024-01-11 | 2026-04-07 | 818 |

## PELT Regime Detection

- **btc**: 9 breakpoints
- **eth**: 5 breakpoints
- **sol**: 4 breakpoints
- **bnb**: 5 breakpoints
