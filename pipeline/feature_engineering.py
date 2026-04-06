"""Feature engineering: derived features, z-score normalization, 7-day lag."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# BTC halving dates (actual and estimated)
HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
    pd.Timestamp("2028-04-15"),  # estimated next
]

ETF_APPROVAL_DATE = pd.Timestamp("2024-01-11")

# Columns to exclude from z-score (binary/categorical)
EXCLUDE_FROM_ZSCORE = {"btc_etf_indicator", "regime"}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BTC daily return, 30-day realized vol, halving countdown, ETF indicator."""
    out = df.copy()

    # Daily log return
    if "btc_close" in out.columns:
        out["btc_daily_return"] = np.log(out["btc_close"] / out["btc_close"].shift(1))

        # 30-day realized volatility (annualized)
        out["btc_realized_vol_30d"] = (
            out["btc_daily_return"].rolling(window=30, min_periods=20).std() * np.sqrt(365)
        )

    # Days until next halving
    out["days_to_halving"] = out.index.map(_days_to_next_halving)

    # ETF binary indicator
    out["btc_etf_indicator"] = (out.index >= ETF_APPROVAL_DATE).astype(int)

    return out


def _days_to_next_halving(date: pd.Timestamp) -> int:
    """Calculate days until the next BTC halving from a given date."""
    for h in HALVING_DATES:
        if h > date:
            return (h - date).days
    # If past all known halvings, estimate ~4 years from last
    last = HALVING_DATES[-1]
    next_est = last + pd.DateOffset(years=4)
    return max(0, (next_est - date).days)


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply z-score normalization to all numeric columns except excluded ones."""
    out = df.copy()
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FROM_ZSCORE
    ]

    # Ensure float dtype so z-scored values can be assigned
    out[numeric_cols] = out[numeric_cols].astype(float)

    scaler = StandardScaler()
    # Fit on non-NaN data
    valid_mask = out[numeric_cols].notna().all(axis=1)
    if valid_mask.sum() > 0:
        out.loc[valid_mask, numeric_cols] = scaler.fit_transform(out.loc[valid_mask, numeric_cols])
    # Leave NaN rows as NaN

    return out


def add_lags(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    """Add 7-day lagged versions of all numeric feature columns.

    Original columns are kept; lagged columns get a _lag7 suffix.
    """
    out = df.copy()
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FROM_ZSCORE
    ]
    for col in numeric_cols:
        out[f"{col}_lag{lag_days}"] = out[col].shift(lag_days)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: derive -> normalize -> lag."""
    df = add_derived_features(df)
    df = zscore_normalize(df)
    df = add_lags(df, lag_days=7)
    return df


if __name__ == "__main__":
    # Quick test with dummy data
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    test_df = pd.DataFrame({"btc_close": np.random.randn(60).cumsum() + 30000}, index=dates)
    test_df.index.name = "date"
    result = engineer_features(test_df)
    print(f"Features: {result.columns.tolist()}")
    print(result.tail())
