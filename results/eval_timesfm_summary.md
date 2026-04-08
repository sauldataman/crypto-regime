# TimesFM 2.5 Crypto Evaluation Results

Generated: 2026-04-07 17:09

## 1. Zero-shot Capability

| Horizon | Direction Acc | MAE | RMSE | Quantile Cal Dev | VaR 10% Breach |
|---------|-------------|-----|------|-----------------|----------------|
| 1d | 0.464 | 0.01996 | 0.02729 | 0.027 | 0.15693430656934307 |
| 5d | 0.453 | 0.04728 | 0.06045 | 0.090 | 0.025547445255474453 |
| 30d | 0.522 | 0.10546 | 0.13154 | 0.194 | 0.0 |

## 2. vs Traditional Methods (h=1)

| Method | Direction Acc | MAE | Correlation |
|--------|-------------|-----|-------------|
| TIMESFM | 0.464 | 0.01996 | -0.058 |
| AR | 0.400 | 0.02114 | -0.291 |
| GARCH | 0.357 | 0.01910 | -0.314 |

## 3. Fine-tune Effect (BTC daily, h=1)

| Model | Direction Acc | MAE | Quantile Cal Dev |
|-------|-------------|-----|-----------------|
| zero-shot | 0.464 | 0.01996 | 0.027 |

## 4. Cross-Asset (zero-shot, h=1)

| Asset | Direction Acc | MAE | Quantile Cal Dev | VaR 10% Breach |
|-------|-------------|-----|-----------------|----------------|
| BNB | 0.442 | 0.02127 | 0.024 | 0.14233576642335766 |
| BTC | 0.464 | 0.01996 | 0.027 | 0.15693430656934307 |
| DOGE | 0.493 | 0.03827 | 0.031 | 0.12408759124087591 |
| ETH | 0.442 | 0.02738 | 0.039 | 0.17153284671532848 |
| LINK | 0.467 | 0.03958 | 0.032 | 0.14963503649635038 |
| SOL | 0.485 | 0.03523 | 0.021 | 0.11313868613138686 |

## 5. Cross-Frequency (BTC zero-shot, h=1)

| Frequency | Direction Acc | MAE | Quantile Cal Dev |
|-----------|-------------|-----|-----------------|
| daily | 0.464 | 0.01996 | 0.027 |
| hourly | 0.535 | 0.00369 | 0.016 |
| 5min | 0.525 | 0.00110 | 0.043 |

## Bottom Line

*Is TimesFM useful for crypto?*

(Fill in after reviewing results above)
