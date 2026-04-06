# Phase 0 Baseline Results

Data: 3744 rows | 2016-01-01 → 2026-04-01

## Direction Accuracy (1d / 5d / 30d)

| Model | 1d | 5d | 30d | Notes |
|-------|----|----|-----|-------|
| AR(1) | 0.521 | 0.513 | 0.511 | ⚠️ near random — expected for log-returns |
| ARIMA(auto) | 0.303 | 0.279 | 0.290 | ❌ worse than random (overfitting in walk-forward) |
| TimesFM zero-shot | — | — | — | ⏳ requires DGX (JAX + GPU memory) |
| TimesFM + XReg | — | — | — | ⏳ requires DGX (JAX + GPU memory) |

## Key Findings (Local Experiments)

- AR(1) gives ~52% direction accuracy — consistent with paper's finding (53.4% for HMM)
- ARIMA under-performs in walk-forward — overfits to local patterns, degrades on regime shifts
- TimesFM experiments must run on DGX (model too large for CPU, XReg needs JAX)

## Decision Gate

AR(1) baseline established: **52%**
Target for TimesFM+XReg: **>55%** (success) / **>58%** (stretch)
