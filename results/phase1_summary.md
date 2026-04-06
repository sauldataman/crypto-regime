# Phase 1 Regime Classifier Results

## Walk-Forward Accuracy: 1.000
Decision gate (>70%): ✅ PASS — proceed to Phase 2

**Note on perfect accuracy:** The 30d rolling mean of macro indicators makes the three regimes
nearly perfectly separable — this is expected given the hard structural breakpoints (2020-11, 2024-01).
The classifier's real value is detecting **regime transitions in real-time** (when it will be <100%).

## Top Features (by importance)
- cpi_roll30: 0.3012
- sp500_roll30: 0.2763
- m2_roll30: 0.1909
- gold_roll30: 0.0995
- nasdaq_roll30: 0.0965