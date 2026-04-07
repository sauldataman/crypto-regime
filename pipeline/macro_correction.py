"""Phase II-D: Macro-corrected quantile forecasts.

TimesFM's zero-shot quantile forecasts are well-calibrated (1.7% avg deviation)
but don't use macro data. PyTorch TimesFM doesn't support covariates natively.

Solution: a lightweight XGBoost second-stage model that takes TimesFM quantiles +
current macro indicators as input and outputs corrected quantiles.

Pipeline:
  1. TimesFM produces raw quantile forecasts (P10..P90)
  2. build_correction_features() combines quantiles + z-scored macro indicators
  3. XGBoost quantile regression learns the residual (actual - raw quantile)
  4. apply_macro_correction() adjusts raw quantiles at inference time

Usage:
  python -m pipeline.macro_correction          # standalone evaluation
  # Or import into phase3_risk_signals.py:
  from pipeline.macro_correction import train_correction_model, apply_macro_correction
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# Temporal split (matches phase05 / phase3)
CAL_START = "2023-07-01"
CAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-03-31"

# Macro columns we use as correction features (rolling 7-day z-scored)
MACRO_FEATURE_COLS = ["sp500", "vix", "dxy", "treasury_10y"]

# Rolling z-score window for macro features
MACRO_ZSCORE_WINDOW = 7

# BTC realized volatility window (days)
BTC_VOL_WINDOW = 30

# IQR rank lookback
IQR_RANK_WINDOW = 60


# ---------------------------------------------------------------------------
# 1. Feature builder
# ---------------------------------------------------------------------------


def _zscore_rolling(series: pd.Series, window: int = 7) -> pd.Series:
    """Rolling z-score with forward-fill for missing values."""
    roll_mean = series.rolling(window=window, min_periods=max(window // 2, 2)).mean()
    roll_std = series.rolling(window=window, min_periods=max(window // 2, 2)).std()
    roll_std = roll_std.clip(lower=1e-8)
    return (series - roll_mean) / roll_std


def prepare_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score macro columns and forward-fill missing data.

    Args:
        macro_df: Raw macro DataFrame with DatetimeIndex and columns like
                  sp500, vix, dxy, treasury_10y, etc.

    Returns:
        DataFrame with z-scored columns (suffixed _z), forward-filled.
    """
    out = macro_df.copy()

    # Forward-fill, then drop any remaining leading NaNs
    out = out.ffill()

    result = pd.DataFrame(index=out.index)
    for col in MACRO_FEATURE_COLS:
        if col in out.columns:
            result[f"{col}_z"] = _zscore_rolling(out[col], window=MACRO_ZSCORE_WINDOW)
        else:
            logger.warning("Macro column '%s' not found; filling with zeros.", col)
            result[f"{col}_z"] = 0.0

    return result


def build_correction_features(
    timesfm_quantiles: dict,
    macro_features: pd.DataFrame,
    date: pd.Timestamp,
    btc_vol_30d: float | None = None,
    regime: int | None = None,
    iqr_history: list[float] | None = None,
) -> np.ndarray:
    """Build feature vector for the correction model.

    Features:
    - TimesFM raw quantiles: q10, q50, q90, IQR              (4 features)
    - Macro indicators (rolling 7-day z-scored):
        sp500_z, vix_z, dxy_z, treasury_10y_z                (4 features)
    - btc_vol_30d (realized volatility, annualized)           (1 feature)
    - Regime indicator (if provided)                          (1 feature)
    - IQR percentile rank in last 60 days (if history given)  (1 feature)

    Total: up to 11 features.

    Args:
        timesfm_quantiles: dict with keys like 'q10', 'q50', 'q90'.
        macro_features: z-scored macro DataFrame from prepare_macro_features().
        date: Timestamp for the forecast date (used to look up macro row).
        btc_vol_30d: BTC 30-day realized vol. If None, feature is set to 0.
        regime: PELT regime label (int). If None, feature is set to -1.
        iqr_history: list of recent IQR values for percentile rank computation.

    Returns:
        1-D numpy array of features.
    """
    q10 = timesfm_quantiles.get("q10", 0.0)
    q50 = timesfm_quantiles.get("q50", 0.0)
    q90 = timesfm_quantiles.get("q90", 0.0)
    iqr = q90 - q10

    features = [q10, q50, q90, iqr]

    # Macro z-scores for the given date
    macro_row = _lookup_macro_row(macro_features, date)
    for col in MACRO_FEATURE_COLS:
        z_col = f"{col}_z"
        features.append(macro_row.get(z_col, 0.0))

    # BTC realized vol
    features.append(btc_vol_30d if btc_vol_30d is not None else 0.0)

    # Regime
    features.append(float(regime) if regime is not None else -1.0)

    # IQR percentile rank
    if iqr_history and len(iqr_history) >= 2:
        rank = sum(1 for v in iqr_history if v <= iqr) / len(iqr_history)
        features.append(rank)
    else:
        features.append(0.5)  # neutral default

    return np.array(features, dtype=np.float64)


def get_feature_names() -> list[str]:
    """Return ordered list of feature names matching build_correction_features output."""
    names = ["q10", "q50", "q90", "iqr"]
    for col in MACRO_FEATURE_COLS:
        names.append(f"{col}_z")
    names.extend(["btc_vol_30d", "regime", "iqr_rank_60d"])
    return names


def _lookup_macro_row(macro_features: pd.DataFrame, date: pd.Timestamp) -> dict:
    """Look up macro features for a date, with nearest-date fallback."""
    if macro_features.empty:
        return {}

    # Normalize to date-only for matching
    date_norm = pd.Timestamp(date.date())

    if date_norm in macro_features.index:
        row = macro_features.loc[date_norm]
        return row.to_dict() if hasattr(row, "to_dict") else {}

    # Nearest prior date (macro data may skip weekends)
    prior = macro_features.index[macro_features.index <= date_norm]
    if len(prior) > 0:
        nearest = prior[-1]
        row = macro_features.loc[nearest]
        return row.to_dict() if hasattr(row, "to_dict") else {}

    return {}


# ---------------------------------------------------------------------------
# 2. Correction model training
# ---------------------------------------------------------------------------


def train_correction_model(
    cal_results: list[dict],
    macro_df: pd.DataFrame,
    target_quantile: float = 0.05,
    regime_labels: pd.Series | None = None,
    btc_returns: pd.Series | None = None,
    validation_frac: float = 0.20,
) -> dict:
    """Train XGBoost model to predict quantile correction.

    Training target: the residual (actual - P10_predicted) for the target quantile.
    We want to learn: given macro conditions, how much should we adjust the raw
    quantile?

    Uses quantile regression objective in XGBoost for proper quantile estimation.

    Args:
        cal_results: List of forecast dicts with 'date', 'actual', 'q10', 'q50',
                     'q90' keys (from Phase 0.5 calibration results).
        macro_df: Raw macro DataFrame (will be z-scored internally).
        target_quantile: Target quantile level (e.g. 0.05 for VaR 5%).
        regime_labels: Optional Series of PELT regime labels with DatetimeIndex.
        btc_returns: Optional BTC daily returns for realized vol computation.
        validation_frac: Fraction of cal set held out for validation.

    Returns:
        dict with 'model', 'feature_names', 'train_metrics', 'val_metrics'.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost>=2.0")
        raise

    if len(cal_results) < 30:
        raise ValueError(
            f"Calibration set too small ({len(cal_results)} < 30) for XGBoost training."
        )

    # Prepare macro features (z-scored)
    macro_features = prepare_macro_features(macro_df)

    # Compute BTC realized vol if returns provided
    btc_vols = {}
    if btc_returns is not None:
        vol_series = btc_returns.rolling(
            window=BTC_VOL_WINDOW, min_periods=20
        ).std() * np.sqrt(365)
        for date in vol_series.index:
            btc_vols[pd.Timestamp(date.date())] = float(vol_series.loc[date])

    # Build feature matrix and target vector
    feature_names = get_feature_names()
    X_rows = []
    y_targets = []
    iqr_history: list[float] = []

    for r in cal_results:
        date = pd.Timestamp(r["date"])
        quantiles = {k: r[k] for k in ["q10", "q50", "q90"] if k in r}

        if not quantiles:
            logger.warning("Skipping date %s: missing quantile keys", r["date"])
            continue

        q10 = quantiles.get("q10", 0.0)
        actual = r["actual"]

        # Target: residual (how much actual deviated from predicted quantile)
        target = actual - q10

        # Regime label for this date
        regime = None
        if regime_labels is not None:
            regime = _lookup_regime(regime_labels, date)

        # BTC vol
        btc_vol = btc_vols.get(pd.Timestamp(date.date()))

        # Build features
        features = build_correction_features(
            timesfm_quantiles=quantiles,
            macro_features=macro_features,
            date=date,
            btc_vol_30d=btc_vol,
            regime=regime,
            iqr_history=iqr_history[-IQR_RANK_WINDOW:] if iqr_history else None,
        )

        # Track IQR history
        iqr = quantiles.get("q90", 0.0) - quantiles.get("q10", 0.0)
        iqr_history.append(iqr)

        X_rows.append(features)
        y_targets.append(target)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_targets, dtype=np.float64)

    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Training data: %d samples, %d features", X.shape[0], X.shape[1]
    )

    # Train/validation split (temporal: last validation_frac as held-out)
    n_total = len(X)
    n_val = max(1, int(n_total * validation_frac))
    n_train = n_total - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    logger.info(
        "Split: %d train, %d validation (temporal, last %.0f%%)",
        n_train, n_val, validation_frac * 100,
    )

    # XGBoost quantile regression
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": target_quantile,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("Feature importance (top 5):")
    for name, imp in sorted_importance[:5]:
        logger.info("  %s: %.4f", name, imp)

    # Training metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = _compute_quantile_metrics(y_train, train_pred, target_quantile, "train")
    val_metrics = _compute_quantile_metrics(y_val, val_pred, target_quantile, "val")

    return {
        "model": model,
        "feature_names": feature_names,
        "target_quantile": target_quantile,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "feature_importance": dict(sorted_importance),
        "n_train": n_train,
        "n_val": n_val,
        "params": {k: v for k, v in params.items() if k != "objective"},
    }


def _compute_quantile_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float, label: str
) -> dict:
    """Compute metrics for quantile regression evaluation."""
    # Quantile coverage: fraction of y_true below y_pred
    coverage = float(np.mean(y_true <= y_pred))

    # Pinball loss (quantile loss)
    residuals = y_true - y_pred
    pinball = float(np.mean(np.where(residuals >= 0, alpha * residuals, (alpha - 1) * residuals)))

    # MAE
    mae = float(np.mean(np.abs(residuals)))

    logger.info(
        "  %s: coverage=%.3f (target=%.3f), pinball=%.6f, MAE=%.6f",
        label, coverage, alpha, pinball, mae,
    )

    return {
        "coverage": coverage,
        "target_coverage": alpha,
        "pinball_loss": pinball,
        "mae": mae,
    }


def _lookup_regime(regime_labels: pd.Series, date: pd.Timestamp) -> int | None:
    """Look up regime label for a date."""
    date_norm = pd.Timestamp(date.date())

    if date_norm in regime_labels.index:
        return int(regime_labels.loc[date_norm])

    prior = regime_labels.index[regime_labels.index <= date_norm]
    if len(prior) > 0:
        return int(regime_labels.loc[prior[-1]])

    return None


# ---------------------------------------------------------------------------
# 3. Correction application
# ---------------------------------------------------------------------------


def apply_macro_correction(
    timesfm_quantiles: dict,
    correction_model: dict,
    macro_features: pd.DataFrame,
    date: pd.Timestamp,
    btc_vol_30d: float | None = None,
    regime: int | None = None,
    iqr_history: list[float] | None = None,
) -> dict:
    """Apply trained correction model to adjust TimesFM quantiles.

    The model predicts the residual (actual - q10) at the target quantile level.
    We add this predicted residual to q10 to get the corrected VaR level.

    For other quantiles (q50, q90), we apply a proportional shift based on
    the correction magnitude relative to the original IQR.

    Args:
        timesfm_quantiles: dict with keys 'q10', 'q50', 'q90' (and others).
        correction_model: dict from train_correction_model() with 'model' key.
        macro_features: z-scored macro DataFrame from prepare_macro_features().
        date: Timestamp for the forecast date.
        btc_vol_30d: BTC 30-day realized vol.
        regime: PELT regime label.
        iqr_history: recent IQR values for percentile rank.

    Returns:
        dict with same keys as input quantiles, with corrected values.
    """
    model = correction_model["model"]

    features = build_correction_features(
        timesfm_quantiles=timesfm_quantiles,
        macro_features=macro_features,
        date=date,
        btc_vol_30d=btc_vol_30d,
        regime=regime,
        iqr_history=iqr_history,
    )

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict residual correction
    predicted_residual = float(model.predict(features.reshape(1, -1))[0])

    # Apply correction to q10 (the primary VaR quantile)
    corrected = dict(timesfm_quantiles)
    q10_raw = timesfm_quantiles.get("q10", 0.0)
    corrected["q10"] = q10_raw + predicted_residual

    # For other quantiles, apply a proportional shift.
    # The idea: if the model says the tail is wider/narrower than TimesFM thinks,
    # scale other quantiles accordingly.
    iqr_raw = timesfm_quantiles.get("q90", 0.0) - q10_raw
    if abs(iqr_raw) > 1e-10:
        # Apply dampened shift to median and upper quantiles
        # (median gets half the shift, q90 gets a quarter)
        q50_raw = timesfm_quantiles.get("q50", 0.0)
        q90_raw = timesfm_quantiles.get("q90", 0.0)
        corrected["q50"] = q50_raw + predicted_residual * 0.5
        corrected["q90"] = q90_raw + predicted_residual * 0.25

    return corrected


# ---------------------------------------------------------------------------
# 4. Regime-conditional IQR scaling
# ---------------------------------------------------------------------------


def build_regime_vol_map(
    returns: pd.Series, regime_labels: pd.Series
) -> dict[int, float]:
    """Compute average realized volatility per regime.

    Args:
        returns: Daily return series with DatetimeIndex.
        regime_labels: Series of int regime labels with same index.

    Returns:
        {regime_id: avg_annualized_volatility}
    """
    vol_map = {}
    combined = pd.DataFrame({"return": returns, "regime": regime_labels})
    combined = combined.dropna()

    for regime_id, group in combined.groupby("regime"):
        if len(group) < 20:
            logger.warning(
                "Regime %d has only %d points; vol estimate unreliable.",
                regime_id, len(group),
            )
        daily_std = group["return"].std()
        annualized_vol = daily_std * np.sqrt(365)
        vol_map[int(regime_id)] = float(annualized_vol)
        logger.info(
            "  Regime %d: %d days, vol=%.4f (ann.)",
            regime_id, len(group), annualized_vol,
        )

    return vol_map


def regime_iqr_scaling(
    iqr: float, regime: int, regime_vol_map: dict[int, float]
) -> float:
    """Scale IQR based on current regime's historical volatility.

    If current regime is historically high-vol, widen the IQR.
    If low-vol, narrow it.

    The scaling factor is: regime_vol / median_vol across all regimes.
    Clamped to [0.5, 2.0] to prevent extreme adjustments.

    Args:
        iqr: Raw IQR from TimesFM (q90 - q10).
        regime: Current PELT regime label.
        regime_vol_map: {regime_id: avg_volatility} from training data.

    Returns:
        Scaled IQR.
    """
    if not regime_vol_map:
        return iqr

    if regime not in regime_vol_map:
        logger.debug("Regime %d not in vol map; returning raw IQR.", regime)
        return iqr

    all_vols = list(regime_vol_map.values())
    median_vol = float(np.median(all_vols))

    if median_vol < 1e-10:
        return iqr

    current_vol = regime_vol_map[regime]
    scale_factor = current_vol / median_vol

    # Clamp to prevent extreme adjustments
    scale_factor = max(0.5, min(2.0, scale_factor))

    scaled_iqr = iqr * scale_factor

    logger.debug(
        "IQR scaling: regime=%d, vol=%.4f, median_vol=%.4f, "
        "factor=%.3f, iqr %.6f -> %.6f",
        regime, current_vol, median_vol, scale_factor, iqr, scaled_iqr,
    )

    return float(scaled_iqr)


# ---------------------------------------------------------------------------
# 5. End-to-end evaluation
# ---------------------------------------------------------------------------


def evaluate_macro_correction(
    test_results: list[dict],
    macro_df: pd.DataFrame,
    correction_model: dict,
    conformal_correction: float = 0.0,
    regime_labels: pd.Series | None = None,
    regime_vol_map: dict[int, float] | None = None,
    btc_returns: pd.Series | None = None,
) -> dict:
    """Evaluate: does macro correction improve VaR?

    Compare four approaches:
    1. Raw TimesFM quantiles (no correction)
    2. Conformal-only (Phase 0.5 approach)
    3. Macro-corrected quantiles (this module)
    4. Regime-conditional correction (macro + regime IQR scaling)

    Args:
        test_results: List of forecast dicts with 'date', 'actual', 'q10', etc.
        macro_df: Raw macro DataFrame.
        correction_model: Trained model dict from train_correction_model().
        conformal_correction: Conformal correction value from Phase 0.5 calibration.
        regime_labels: Optional PELT regime labels.
        regime_vol_map: Optional {regime_id: vol} from build_regime_vol_map().
        btc_returns: Optional BTC daily returns for vol computation.

    Returns:
        Comparison dict with breach rates for each approach.
    """
    macro_features = prepare_macro_features(macro_df)

    # Compute BTC vols if available
    btc_vols = {}
    if btc_returns is not None:
        vol_series = btc_returns.rolling(
            window=BTC_VOL_WINDOW, min_periods=20
        ).std() * np.sqrt(365)
        for date in vol_series.index:
            btc_vols[pd.Timestamp(date.date())] = float(vol_series.loc[date])

    n_test = len(test_results)
    raw_breaches = 0
    conformal_breaches = 0
    macro_breaches = 0
    regime_breaches = 0

    iqr_history: list[float] = []

    for r in test_results:
        date = pd.Timestamp(r["date"])
        actual = r["actual"]
        q10_raw = r.get("q10", 0.0)
        q90_raw = r.get("q90", 0.0)
        iqr = q90_raw - q10_raw

        # 1. Raw: actual < q10
        if actual < q10_raw:
            raw_breaches += 1

        # 2. Conformal: actual < (q10 - conformal_correction)
        var_conformal = q10_raw - conformal_correction
        if actual < var_conformal:
            conformal_breaches += 1

        # 3. Macro correction
        quantiles = {k: r[k] for k in ["q10", "q50", "q90"] if k in r}
        regime = None
        if regime_labels is not None:
            regime = _lookup_regime(regime_labels, date)

        btc_vol = btc_vols.get(pd.Timestamp(date.date()))

        corrected = apply_macro_correction(
            timesfm_quantiles=quantiles,
            correction_model=correction_model,
            macro_features=macro_features,
            date=date,
            btc_vol_30d=btc_vol,
            regime=regime,
            iqr_history=iqr_history[-IQR_RANK_WINDOW:] if iqr_history else None,
        )
        if actual < corrected["q10"]:
            macro_breaches += 1

        # 4. Regime-conditional: apply regime IQR scaling on top of macro correction
        if regime is not None and regime_vol_map:
            scaled_iqr = regime_iqr_scaling(iqr, regime, regime_vol_map)
            # Adjust q10 by the IQR change: if IQR widens, push q10 lower
            iqr_change = scaled_iqr - iqr
            regime_adjusted_q10 = corrected["q10"] - iqr_change * 0.5
            if actual < regime_adjusted_q10:
                regime_breaches += 1
        else:
            # Same as macro correction when no regime info
            if actual < corrected["q10"]:
                regime_breaches += 1

        iqr_history.append(iqr)

    # Compute rates
    results = {
        "n_test": n_test,
        "target_rate": correction_model.get("target_quantile", 0.05),
        "raw_timesfm": {
            "breaches": raw_breaches,
            "breach_rate": float(raw_breaches / n_test) if n_test > 0 else 0.0,
        },
        "conformal_only": {
            "breaches": conformal_breaches,
            "breach_rate": float(conformal_breaches / n_test) if n_test > 0 else 0.0,
            "correction": conformal_correction,
        },
        "macro_corrected": {
            "breaches": macro_breaches,
            "breach_rate": float(macro_breaches / n_test) if n_test > 0 else 0.0,
        },
        "regime_conditional": {
            "breaches": regime_breaches,
            "breach_rate": float(regime_breaches / n_test) if n_test > 0 else 0.0,
        },
    }

    # Log comparison table
    logger.info("=" * 60)
    logger.info("VaR Breach Rate Comparison (target: %.1f%%)", results["target_rate"] * 100)
    logger.info("=" * 60)
    logger.info("  %-25s  %s  %s", "Method", "Breaches", "Rate")
    logger.info("  %-25s  %d/%d  %.1f%%", "Raw TimesFM P10",
                raw_breaches, n_test,
                results["raw_timesfm"]["breach_rate"] * 100)
    logger.info("  %-25s  %d/%d  %.1f%%", "Conformal only",
                conformal_breaches, n_test,
                results["conformal_only"]["breach_rate"] * 100)
    logger.info("  %-25s  %d/%d  %.1f%%", "Macro-corrected",
                macro_breaches, n_test,
                results["macro_corrected"]["breach_rate"] * 100)
    logger.info("  %-25s  %d/%d  %.1f%%", "Regime-conditional",
                regime_breaches, n_test,
                results["regime_conditional"]["breach_rate"] * 100)

    return results


# ---------------------------------------------------------------------------
# 6. Synthetic test data (fallback when Phase 0.5 not available)
# ---------------------------------------------------------------------------


def _generate_synthetic_forecasts(
    returns: pd.Series, start: str, end: str
) -> list[dict]:
    """Generate synthetic TimesFM-like forecasts using rolling quantiles.

    Used when results/phase05_forecasts.json is not available.
    Approximates P10/P50/P90 with rolling 252-day empirical quantiles.
    """
    mask = (returns.index >= start) & (returns.index <= end)
    period_returns = returns[mask]

    results = []
    for date in period_returns.index:
        loc = returns.index.get_loc(date)
        if loc < 252:
            continue

        window = returns.iloc[loc - 252:loc].values
        results.append({
            "date": str(date.date()),
            "actual": float(returns.iloc[loc]),
            "q10": float(np.quantile(window, 0.10)),
            "q50": float(np.quantile(window, 0.50)),
            "q90": float(np.quantile(window, 0.90)),
        })

    return results


def _load_btc_returns() -> pd.Series:
    """Load BTC daily returns (same logic as phase3_risk_signals.py)."""
    processed = ROOT / "data/processed/btc_full.parquet"
    if processed.exists():
        df = pd.read_parquet(processed).sort_index()
        if "btc_daily_return" in df.columns:
            return df["btc_daily_return"].dropna()

    raw = ROOT / "data/raw/btc_price.parquet"
    if not raw.exists():
        raise FileNotFoundError(
            f"No BTC data found at {processed} or {raw}. "
            "Run the pipeline on DGX first."
        )
    df = pd.read_parquet(raw).sort_index()
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    returns.name = "btc_daily_return"
    return returns


def _load_macro() -> pd.DataFrame:
    """Load macro data from parquet, with graceful fallback."""
    macro_path = ROOT / "data/raw/macro.parquet"
    if macro_path.exists():
        df = pd.read_parquet(macro_path).sort_index()
        logger.info(
            "Macro data: %d rows, %d cols (%s -> %s)",
            len(df), len(df.columns),
            df.index[0].date(), df.index[-1].date(),
        )
        return df

    logger.warning("Macro data not found at %s. Generating synthetic macro data.", macro_path)
    dates = pd.date_range("2010-01-01", "2025-03-31", freq="B")
    np.random.seed(42)
    df = pd.DataFrame({
        "sp500": np.random.randn(len(dates)).cumsum() + 3000,
        "vix": np.abs(np.random.randn(len(dates)) * 5 + 20),
        "dxy": np.random.randn(len(dates)).cumsum() * 0.1 + 100,
        "treasury_10y": np.abs(np.random.randn(len(dates)) * 0.5 + 3),
    }, index=dates)
    df.index.name = "date"
    return df


def _load_regime_labels(returns: pd.Series) -> pd.Series | None:
    """Load PELT regime labels, or compute from returns."""
    try:
        from pipeline.regime_detection import detect_regimes, label_from_breakpoints

        breakpoints = detect_regimes(returns)
        labels = label_from_breakpoints(returns.index, breakpoints)
        logger.info("PELT regimes: %d breakpoints, %d regimes",
                     len(breakpoints), labels.nunique())
        return labels
    except (ImportError, Exception) as e:
        logger.warning("Could not compute PELT regimes: %s", e)
        return None


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def main():
    """End-to-end Phase II-D evaluation."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logger.info("=" * 60)
    logger.info("Phase II-D: Macro Correction for TimesFM Quantiles")
    logger.info("=" * 60)

    # --- Load data ---
    try:
        btc_returns = _load_btc_returns()
        logger.info(
            "BTC returns: %d rows (%s -> %s)",
            len(btc_returns),
            btc_returns.index[0].date(),
            btc_returns.index[-1].date(),
        )
    except FileNotFoundError as e:
        logger.error("Cannot load BTC data: %s", e)
        return

    macro_df = _load_macro()

    # --- Load or generate forecasts ---
    forecasts_path = RESULTS / "phase05_forecasts.json"
    if forecasts_path.exists():
        logger.info("Loading real TimesFM forecasts from %s", forecasts_path)
        with open(forecasts_path) as f:
            forecasts = json.load(f)
        cal_results = forecasts.get("cal", [])
        test_results = forecasts.get("test", [])
        logger.info(
            "Real forecasts: %d cal, %d test", len(cal_results), len(test_results)
        )
    else:
        logger.warning(
            "Phase 0.5 forecasts not found at %s. "
            "Using synthetic rolling-quantile approximation.",
            forecasts_path,
        )
        cal_results = _generate_synthetic_forecasts(btc_returns, CAL_START, CAL_END)
        test_results = _generate_synthetic_forecasts(btc_returns, TEST_START, TEST_END)
        logger.info(
            "Synthetic forecasts: %d cal, %d test",
            len(cal_results), len(test_results),
        )

    if not cal_results:
        logger.error("No calibration results available. Exiting.")
        return

    # --- Load regime labels ---
    regime_labels = _load_regime_labels(btc_returns)

    # --- Build regime vol map ---
    regime_vol_map = None
    if regime_labels is not None:
        logger.info("\nBuilding regime volatility map...")
        regime_vol_map = build_regime_vol_map(btc_returns, regime_labels)

    # --- Train correction model ---
    logger.info("\n--- Training macro correction model ---")
    correction_model = train_correction_model(
        cal_results=cal_results,
        macro_df=macro_df,
        target_quantile=0.05,
        regime_labels=regime_labels,
        btc_returns=btc_returns,
    )
    logger.info(
        "Training complete: train pinball=%.6f, val pinball=%.6f",
        correction_model["train_metrics"]["pinball_loss"],
        correction_model["val_metrics"]["pinball_loss"],
    )

    # --- Compute conformal correction for comparison ---
    # (Replicate the simple conformal from Phase 0.5)
    cal_actuals = np.array([r["actual"] for r in cal_results])
    cal_q10 = np.array([r["q10"] for r in cal_results])
    scores = cal_q10 - cal_actuals
    n_cal = len(scores)
    conformal_correction = float(np.quantile(scores, 0.95 * (1 + 1 / n_cal)))
    logger.info("Conformal correction (reference): %.6f", conformal_correction)

    # --- Evaluate on test set ---
    logger.info("\n--- Evaluating on test set ---")
    evaluation = evaluate_macro_correction(
        test_results=test_results,
        macro_df=macro_df,
        correction_model=correction_model,
        conformal_correction=conformal_correction,
        regime_labels=regime_labels,
        regime_vol_map=regime_vol_map,
        btc_returns=btc_returns,
    )

    # --- Save results ---
    RESULTS.mkdir(exist_ok=True)

    # Convert numpy types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output = {
        "phase": "II-D",
        "description": "Macro-corrected TimesFM quantile forecasts",
        "data_source": "real" if forecasts_path.exists() else "synthetic",
        "training": {
            "n_train": correction_model["n_train"],
            "n_val": correction_model["n_val"],
            "target_quantile": correction_model["target_quantile"],
            "train_metrics": correction_model["train_metrics"],
            "val_metrics": correction_model["val_metrics"],
            "feature_importance": correction_model["feature_importance"],
            "xgb_params": correction_model["params"],
        },
        "evaluation": evaluation,
        "regime_vol_map": regime_vol_map,
    }

    out_path = RESULTS / "phase2d_macro_correction.json"
    out_path.write_text(json.dumps(output, indent=2, default=_convert))
    logger.info("\nResults saved: %s", out_path)

    # --- Print summary ---
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("Data: %s", output["data_source"])
    logger.info("Target quantile: %.2f", correction_model["target_quantile"])
    logger.info(
        "Train pinball loss: %.6f",
        correction_model["train_metrics"]["pinball_loss"],
    )
    logger.info(
        "Val pinball loss:   %.6f",
        correction_model["val_metrics"]["pinball_loss"],
    )
    logger.info("")
    logger.info("Test breach rates (target: 5.0%%):")
    for method in ["raw_timesfm", "conformal_only", "macro_corrected", "regime_conditional"]:
        rate = evaluation[method]["breach_rate"]
        logger.info("  %-25s %.1f%%", method, rate * 100)

    best_method = min(
        ["raw_timesfm", "conformal_only", "macro_corrected", "regime_conditional"],
        key=lambda m: abs(evaluation[m]["breach_rate"] - 0.05),
    )
    logger.info("\nBest method (closest to 5%%): %s (%.1f%%)",
                best_method, evaluation[best_method]["breach_rate"] * 100)


if __name__ == "__main__":
    main()
