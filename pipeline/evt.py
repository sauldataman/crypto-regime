"""Extreme Value Theory (EVT) tail extrapolation for VaR estimation.

Phase II-C: Fixes the VaR 1% problem.

Phase I showed that VaR 5% works well via conformal calibration on TimesFM P10
(4.01% breach rate). But VaR 1% fails because P10 is too far from P01 for
conformal correction to bridge the gap (correction = 2.34, way too large).

Solution: Fit a Generalized Pareto Distribution (GPD) to the left tail of
returns beyond P10, then extrapolate to P05 and P01. This is the standard
approach in financial risk management (Peaks-Over-Threshold method).

Key idea for TimesFM integration:
  - Instead of modeling raw returns, we model the *residuals* (actual - P10 prediction)
  - This lets GPD capture how much worse reality can be relative to the model's
    most pessimistic decile
  - The fitted GPD then provides principled corrections for any target alpha

Usage:
  python -m pipeline.evt          # standalone test with BTC data
  # Or import into phase3_risk_signals.py:
  from pipeline.evt import evt_calibrate
"""
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. GPD Tail Estimation
# ---------------------------------------------------------------------------

def fit_gpd_tail(
    returns: np.ndarray,
    threshold_quantile: float = 0.10,
    min_exceedances: int = 20,
) -> dict:
    """Fit Generalized Pareto Distribution to the left tail of returns.

    Uses the Peaks-Over-Threshold (POT) method:
    1. Set threshold at the empirical ``threshold_quantile`` (e.g., 10th pctile)
    2. Extract exceedances: values below the threshold
    3. Fit GPD to the (negated) exceedances using scipy.stats.genpareto

    We negate the exceedances so that GPD models positive values (losses).
    The GPD parametrization in scipy is:
        F(x) = 1 - (1 + xi * x / sigma)^(-1/xi)  for xi != 0
        F(x) = 1 - exp(-x / sigma)                for xi = 0

    Args:
        returns: Array of returns (e.g., log returns from calibration period).
        threshold_quantile: Quantile level for the threshold (0.10 = 10th pctile).
        min_exceedances: Minimum number of tail observations required.

    Returns:
        dict with keys:
            - threshold: the actual return threshold value (negative)
            - shape (xi): GPD shape parameter (>0 = heavy tail)
            - scale (sigma): GPD scale parameter
            - n_exceedances: number of tail observations used
            - n_total: total number of observations
            - converged: whether the MLE fit converged

    Raises:
        ValueError: If fewer than ``min_exceedances`` observations in the tail.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if len(returns) < min_exceedances * 2:
        raise ValueError(
            f"Need at least {min_exceedances * 2} observations, got {len(returns)}"
        )

    # Step 1: empirical threshold
    threshold = float(np.quantile(returns, threshold_quantile))

    # Step 2: exceedances (values below threshold)
    tail_mask = returns < threshold
    tail_values = returns[tail_mask]
    n_exceedances = len(tail_values)

    if n_exceedances < min_exceedances:
        raise ValueError(
            f"Only {n_exceedances} exceedances (need >= {min_exceedances}). "
            f"Threshold={threshold:.6f} at q={threshold_quantile}"
        )

    # Step 3: convert to positive losses (exceedances beyond threshold)
    # losses = threshold - tail_value  (positive, since tail_value < threshold)
    losses = threshold - tail_values
    losses = losses[losses > 0]  # guard against exact-threshold values

    if len(losses) < min_exceedances:
        raise ValueError(
            f"Only {len(losses)} positive exceedances after filtering zeros"
        )

    # Step 4: fit GPD via MLE
    # scipy.stats.genpareto uses parametrization: c = shape (xi)
    # floc=0 fixes location to 0 (standard POT: exceedances start at 0)
    try:
        shape, loc, scale = stats.genpareto.fit(losses, floc=0)
        converged = True
    except (RuntimeError, ValueError) as exc:
        logger.warning("GPD MLE fit failed: %s. Using method of moments.", exc)
        # Fallback: method of moments for exponential tail (shape=0)
        shape = 0.0
        loc = 0.0
        scale = float(np.mean(losses))
        converged = False

    # Sanity: shape parameter should be reasonable for financial returns
    # Typical range: -0.5 to 1.0. Values outside suggest fit issues.
    if shape > 2.0:
        logger.warning(
            "GPD shape=%.3f is unusually large (heavy tail). "
            "Results may be unreliable.",
            shape,
        )
    if shape < -0.5:
        logger.warning(
            "GPD shape=%.3f is very negative (bounded tail). "
            "This is unusual for crypto returns.",
            shape,
        )
    if scale <= 0:
        logger.warning("GPD scale=%.6f <= 0. Falling back to exponential.", scale)
        shape = 0.0
        scale = float(np.mean(losses))
        converged = False

    logger.info(
        "GPD fit: shape=%.4f, scale=%.6f, threshold=%.6f, "
        "n_exceed=%d/%d, converged=%s",
        shape, scale, threshold, len(losses), len(returns), converged,
    )

    return {
        "threshold": threshold,
        "shape": float(shape),
        "scale": float(scale),
        "n_exceedances": len(losses),
        "n_total": len(returns),
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# 2. VaR Extrapolation
# ---------------------------------------------------------------------------

def extrapolate_var(gpd_params: dict, target_quantile: float) -> float:
    """Extrapolate VaR to a target quantile using fitted GPD.

    Uses the standard POT-GPD VaR formula:

        VaR(p) = threshold - (sigma/xi) * ((n_total/(n_exceed) * p)^(-xi) - 1)

    where p = target_quantile (e.g., 0.01 for VaR 1%).

    For xi = 0 (exponential tail):

        VaR(p) = threshold - sigma * log(n_exceed / (n_total * p))

    The result is on the return scale (negative = loss).

    Args:
        gpd_params: Output from :func:`fit_gpd_tail`.
        target_quantile: Target quantile level (e.g., 0.05 for VaR 5%,
            0.01 for VaR 1%). Must be less than the threshold quantile
            used in fitting.

    Returns:
        Extrapolated VaR value (negative for loss).

    Raises:
        ValueError: If target_quantile is above the tail probability.
    """
    threshold = gpd_params["threshold"]
    xi = gpd_params["shape"]
    sigma = gpd_params["scale"]
    n_exceed = gpd_params["n_exceedances"]
    n_total = gpd_params["n_total"]

    # Tail probability at threshold
    tail_prob = n_exceed / n_total

    if target_quantile >= tail_prob:
        raise ValueError(
            f"target_quantile={target_quantile} must be < tail_prob={tail_prob:.4f}. "
            f"GPD extrapolation only works beyond the threshold."
        )

    if target_quantile <= 0:
        raise ValueError("target_quantile must be positive")

    # Ratio: how far into the tail we're going
    # ratio = tail_prob / target_quantile  (always > 1)
    ratio = tail_prob / target_quantile

    EPS = 1e-8
    if abs(xi) < EPS:
        # Exponential tail (xi -> 0): VaR = threshold - sigma * log(ratio)
        excess = sigma * np.log(ratio)
    else:
        # General GPD: VaR = threshold - (sigma/xi) * (ratio^xi - 1)
        excess = (sigma / xi) * (ratio**xi - 1)

    var_value = threshold - excess

    logger.debug(
        "VaR(%.3f): threshold=%.6f, excess=%.6f, var=%.6f",
        target_quantile, threshold, excess, var_value,
    )

    return float(var_value)


# ---------------------------------------------------------------------------
# 3. Combined EVT + TimesFM calibration
# ---------------------------------------------------------------------------

def evt_calibrate(
    cal_results: list[dict],
    target_alpha: float = 0.01,
    threshold_quantile: float = 0.10,
    min_exceedances: int = 20,
) -> dict:
    """Calibrate VaR using EVT on TimesFM quantile residuals.

    Instead of using conformal on P10 directly (which fails for alpha < 0.05),
    this function:

    1. Computes residuals: actual - P10_prediction for each calibration day
    2. Fits GPD to the left tail of residuals (days where actual << P10)
    3. Uses GPD to estimate the correction needed for target_alpha

    This is more principled than conformal for extreme quantiles because
    GPD is specifically designed for tail modeling.

    The correction is defined as:
        VaR_adjusted = P10_prediction - correction

    where correction > 0 shifts the VaR threshold further into the loss tail.

    Args:
        cal_results: List of dicts with ``actual`` and ``q10`` keys
            (from walk-forward calibration forecasts).
        target_alpha: Target VaR level (0.01 = 1%, 0.05 = 5%).
        threshold_quantile: Quantile level for GPD threshold within the
            residual distribution (default 0.10 = worst 10% of residuals).
        min_exceedances: Minimum exceedances for GPD fit.

    Returns:
        dict with:
            - method: ``"evt"``
            - correction: amount to subtract from P10 to get VaR at target_alpha
            - gpd_params: fitted GPD parameters
            - diagnostic: goodness-of-fit info
    """
    if len(cal_results) < 50:
        logger.warning(
            "Calibration set too small (%d < 50). Cannot fit EVT.", len(cal_results)
        )
        return {
            "method": "evt_failed",
            "reason": "cal_set_too_small",
            "correction": 0.0,
            "n_cal": len(cal_results),
        }

    actuals = np.array([r["actual"] for r in cal_results], dtype=np.float64)
    predicted_q10 = np.array([r["q10"] for r in cal_results], dtype=np.float64)

    # Residuals: actual - predicted_P10
    # Negative residuals = actual was worse (more negative) than P10 predicted
    residuals = actuals - predicted_q10

    # Fit GPD to the left tail of residuals
    try:
        gpd_params = fit_gpd_tail(
            residuals,
            threshold_quantile=threshold_quantile,
            min_exceedances=min_exceedances,
        )
    except ValueError as exc:
        logger.warning("EVT fit failed: %s", exc)
        return {
            "method": "evt_failed",
            "reason": str(exc),
            "correction": 0.0,
            "n_cal": len(cal_results),
        }

    # The GPD is fit on residuals. We need to find the residual quantile
    # that corresponds to target_alpha.
    #
    # P10 covers ~10% of the distribution. We want to find how much further
    # beyond P10 we need to go to reach target_alpha.
    #
    # The empirical coverage of P10 on the cal set:
    p10_coverage = float(np.mean(actuals <= predicted_q10))
    logger.info("P10 empirical coverage on cal set: %.3f", p10_coverage)

    # We need the residual quantile at level: target_alpha / p10_coverage
    # But this must be less than the threshold_quantile used for GPD.
    # Actually, we want: P(actual < P10 - correction) = target_alpha
    # Which means: P(residual < -correction) = target_alpha
    # i.e., -correction is the target_alpha quantile of the residual distribution
    #
    # Since we fit GPD to the left tail of residuals with threshold at
    # threshold_quantile, we can extrapolate to target_alpha.
    try:
        residual_var = extrapolate_var(gpd_params, target_quantile=target_alpha)
    except ValueError as exc:
        logger.warning("VaR extrapolation failed: %s", exc)
        return {
            "method": "evt_failed",
            "reason": f"extrapolation_failed: {exc}",
            "correction": 0.0,
            "gpd_params": gpd_params,
            "n_cal": len(cal_results),
        }

    # residual_var is the target_alpha quantile of (actual - P10).
    # It's negative (= actual was much worse than P10).
    # correction = -residual_var (positive, to subtract from P10)
    correction = -residual_var

    # Run diagnostic
    diag = evt_diagnostic(residuals, gpd_params)

    # Validate: correction should be positive (shifting P10 further into loss)
    if correction < 0:
        logger.warning(
            "EVT correction is negative (%.6f). This means P10 already "
            "over-covers at alpha=%.3f. Using correction=0.",
            correction, target_alpha,
        )
        correction = 0.0

    # Validate on calibration set
    cal_var = predicted_q10 + residual_var  # = P10 - correction
    cal_breach_rate = float(np.mean(actuals < cal_var))

    logger.info(
        "EVT calibration: alpha=%.3f, correction=%.6f, "
        "cal_breach_rate=%.3f (target=%.3f)",
        target_alpha, correction, cal_breach_rate, target_alpha,
    )

    return {
        "method": "evt",
        "correction": float(correction),
        "gpd_params": gpd_params,
        "diagnostic": diag,
        "target_alpha": target_alpha,
        "p10_coverage": p10_coverage,
        "cal_breach_rate": cal_breach_rate,
        "n_cal": len(cal_results),
    }


# ---------------------------------------------------------------------------
# 4. Diagnostic / validation
# ---------------------------------------------------------------------------

def evt_diagnostic(
    returns: np.ndarray,
    gpd_params: dict,
    ci_level: float = 0.95,
) -> dict:
    """Run diagnostic checks on the GPD fit.

    Args:
        returns: The same returns array used for fitting (or residuals).
        gpd_params: Output from :func:`fit_gpd_tail`.
        ci_level: Confidence level for shape parameter CI.

    Returns:
        dict with:
            - ks_pvalue: Kolmogorov-Smirnov test p-value for GPD fit
            - ad_pvalue: Anderson-Darling test p-value (if available)
            - mean_excess_linear: whether mean excess plot is approx. linear
            - shape_ci: (lower, upper) confidence interval for shape parameter
            - qq_rmse: root mean squared error of QQ plot residuals
    """
    returns = np.asarray(returns, dtype=np.float64)
    threshold = gpd_params["threshold"]
    xi = gpd_params["shape"]
    sigma = gpd_params["scale"]

    # Extract exceedances (positive losses beyond threshold)
    tail_values = returns[returns < threshold]
    losses = threshold - tail_values
    losses = losses[losses > 0]
    losses = np.sort(losses)

    result: dict = {}

    # --- KS test ---
    # Compare empirical exceedance distribution against fitted GPD
    if len(losses) >= 5:
        try:
            ks_stat, ks_pvalue = stats.kstest(
                losses, "genpareto", args=(xi, 0, sigma)
            )
            result["ks_statistic"] = float(ks_stat)
            result["ks_pvalue"] = float(ks_pvalue)
        except Exception as exc:
            logger.warning("KS test failed: %s", exc)
            result["ks_pvalue"] = None
    else:
        result["ks_pvalue"] = None

    # --- QQ plot residuals ---
    if len(losses) >= 5:
        n = len(losses)
        # Theoretical quantiles from fitted GPD
        empirical_probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.genpareto.ppf(empirical_probs, xi, 0, sigma)
        qq_residuals = losses - theoretical_quantiles
        result["qq_rmse"] = float(np.sqrt(np.mean(qq_residuals**2)))
        result["qq_max_deviation"] = float(np.max(np.abs(qq_residuals)))
    else:
        result["qq_rmse"] = None
        result["qq_max_deviation"] = None

    # --- Mean excess plot linearity check ---
    # For GPD, the mean excess function e(u) = E[X-u | X>u] should be linear in u.
    # We check this by computing e(u) at several thresholds within the tail.
    if len(losses) >= 30:
        thresholds_check = np.quantile(losses, np.linspace(0.0, 0.8, 10))
        mean_excesses = []
        for u in thresholds_check:
            above = losses[losses > u] - u
            if len(above) >= 5:
                mean_excesses.append(float(np.mean(above)))
            else:
                mean_excesses.append(np.nan)

        me_arr = np.array(mean_excesses)
        valid = np.isfinite(me_arr)
        if np.sum(valid) >= 4:
            # Check linearity via correlation with threshold values
            x_valid = thresholds_check[valid]
            y_valid = me_arr[valid]
            if np.std(x_valid) > 0 and np.std(y_valid) > 0:
                corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
                # |corr| > 0.8 suggests reasonably linear
                result["mean_excess_correlation"] = corr
                result["mean_excess_linear"] = bool(abs(corr) > 0.8)
            else:
                result["mean_excess_linear"] = True  # constant = trivially linear
                result["mean_excess_correlation"] = 1.0
        else:
            result["mean_excess_linear"] = None
            result["mean_excess_correlation"] = None
    else:
        result["mean_excess_linear"] = None
        result["mean_excess_correlation"] = None

    # --- Shape parameter confidence interval ---
    # Use profile likelihood or bootstrap approximation.
    # Quick approach: delta method. For MLE of GPD, asymptotic variance of
    # shape estimator is approximately 2*(1+xi)^2 / n_exceedances.
    n_exc = gpd_params["n_exceedances"]
    if n_exc > 0 and gpd_params["converged"]:
        z = stats.norm.ppf(1 - (1 - ci_level) / 2)
        # Asymptotic standard error of xi (Smith, 1987)
        se_xi = (1 + xi) * np.sqrt(2.0 / n_exc)
        ci_lower = xi - z * se_xi
        ci_upper = xi + z * se_xi
        result["shape_ci"] = (float(ci_lower), float(ci_upper))
        result["shape_se"] = float(se_xi)
    else:
        result["shape_ci"] = None
        result["shape_se"] = None

    # --- Overall assessment ---
    ks_ok = result.get("ks_pvalue") is not None and result["ks_pvalue"] > 0.05
    me_ok = result.get("mean_excess_linear") is True
    result["fit_acceptable"] = ks_ok or me_ok  # at least one check passes

    logger.info(
        "GPD diagnostic: KS p=%.3f, QQ RMSE=%.6f, ME linear=%s, "
        "shape CI=%s, fit_ok=%s",
        result.get("ks_pvalue", -1) or -1,
        result.get("qq_rmse", -1) or -1,
        result.get("mean_excess_linear"),
        result.get("shape_ci"),
        result.get("fit_acceptable"),
    )

    return result


# ---------------------------------------------------------------------------
# 5. Main: standalone test
# ---------------------------------------------------------------------------

def _load_btc_returns() -> "pd.Series":
    """Load BTC daily returns (same logic as phase3_risk_signals.py)."""
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).parent.parent
    processed = root / "data/processed/btc_full.parquet"
    if processed.exists():
        df = pd.read_parquet(processed).sort_index()
        if "btc_daily_return" in df.columns:
            return df["btc_daily_return"].dropna()

    raw = root / "data/raw/btc_price.parquet"
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


def _simulate_p10_predictions(
    returns: "pd.Series",
    start: str,
    end: str,
) -> list[dict]:
    """Simulate P10 predictions using historical rolling quantile.

    When TimesFM is not available (running on local Mac), we approximate
    P10 predictions using a rolling 252-day 10th percentile. This is
    a reasonable stand-in for testing the EVT pipeline.
    """
    import pandas as pd

    mask = (returns.index >= start) & (returns.index <= end)
    period_returns = returns[mask]

    results = []
    for date in period_returns.index:
        loc = returns.index.get_loc(date)
        if loc < 252:
            continue
        window = returns.iloc[loc - 252:loc].values
        q10 = float(np.quantile(window, 0.10))
        results.append({
            "date": str(date.date()),
            "actual": float(returns.iloc[loc]),
            "q10": q10,
        })
    return results


def main():
    """Standalone EVT test on BTC daily returns."""
    import json
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    CAL_START = "2023-07-01"
    CAL_END = "2024-06-30"
    TEST_START = "2024-07-01"
    TEST_END = "2025-03-31"

    root = Path(__file__).parent.parent

    logger.info("=" * 60)
    logger.info("EVT Tail Extrapolation Test")
    logger.info("=" * 60)

    # --- Load BTC returns ---
    try:
        returns = _load_btc_returns()
        logger.info(
            "BTC returns: %d rows (%s -> %s)",
            len(returns), returns.index[0].date(), returns.index[-1].date(),
        )
    except FileNotFoundError as exc:
        logger.error("Cannot load BTC data: %s", exc)
        return

    # --- Part A: GPD on raw returns (calibration period) ---
    logger.info("\n--- Part A: GPD on raw returns (cal period) ---")
    cal_mask = (returns.index >= CAL_START) & (returns.index <= CAL_END)
    cal_returns = returns[cal_mask].values

    logger.info("Calibration returns: %d days", len(cal_returns))

    try:
        gpd_raw = fit_gpd_tail(cal_returns, threshold_quantile=0.10)
    except ValueError as exc:
        logger.error("GPD fit on raw returns failed: %s", exc)
        return

    # Extrapolate VaR at 5% and 1%
    var_5_raw = extrapolate_var(gpd_raw, target_quantile=0.05)
    var_1_raw = extrapolate_var(gpd_raw, target_quantile=0.01)

    # Compare with empirical quantiles
    emp_5 = float(np.quantile(cal_returns, 0.05))
    emp_1 = float(np.quantile(cal_returns, 0.01))

    logger.info("  VaR 5%%: GPD=%.6f, empirical=%.6f", var_5_raw, emp_5)
    logger.info("  VaR 1%%: GPD=%.6f, empirical=%.6f", var_1_raw, emp_1)

    # Diagnostic
    diag_raw = evt_diagnostic(cal_returns, gpd_raw)

    # --- Part B: GPD on residuals (simulated P10, or real if available) ---
    logger.info("\n--- Part B: EVT calibration on residuals ---")

    # Try loading real Phase 0.5 forecasts
    forecasts_path = root / "results/phase05_forecasts.json"
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
        logger.info(
            "No real forecasts found. Using rolling-quantile simulation."
        )
        cal_results = _simulate_p10_predictions(returns, CAL_START, CAL_END)
        test_results = _simulate_p10_predictions(returns, TEST_START, TEST_END)
        logger.info(
            "Simulated forecasts: %d cal, %d test",
            len(cal_results), len(test_results),
        )

    # EVT calibration for VaR 1%
    evt_cal_1 = evt_calibrate(cal_results, target_alpha=0.01)
    logger.info("EVT VaR 1%% calibration: %s", json.dumps(
        {k: v for k, v in evt_cal_1.items() if k != "diagnostic"},
        indent=2, default=str,
    ))

    # EVT calibration for VaR 5%
    evt_cal_5 = evt_calibrate(cal_results, target_alpha=0.05)
    logger.info("EVT VaR 5%% calibration: %s", json.dumps(
        {k: v for k, v in evt_cal_5.items() if k != "diagnostic"},
        indent=2, default=str,
    ))

    # --- Part C: Evaluate on test set ---
    logger.info("\n--- Part C: Test set evaluation ---")

    if test_results:
        test_actuals = np.array([r["actual"] for r in test_results])
        test_q10 = np.array([r["q10"] for r in test_results])

        # VaR 5% with EVT
        var5_evt = test_q10 - evt_cal_5.get("correction", 0)
        breach_5_evt = float(np.mean(test_actuals < var5_evt))

        # VaR 1% with EVT
        var1_evt = test_q10 - evt_cal_1.get("correction", 0)
        breach_1_evt = float(np.mean(test_actuals < var1_evt))

        # VaR 5% with raw P10 (no correction)
        breach_5_raw = float(np.mean(test_actuals < test_q10))

        n_test = len(test_results)
        logger.info("  Test set: %d days", n_test)
        logger.info(
            "  VaR 5%% breach rate: EVT=%.1f%% (%d/%d), raw P10=%.1f%% [target: 3-8%%]",
            breach_5_evt * 100, int(breach_5_evt * n_test), n_test,
            breach_5_raw * 100,
        )
        logger.info(
            "  VaR 1%% breach rate: EVT=%.1f%% (%d/%d) [target: 0.5-2%%]",
            breach_1_evt * 100, int(breach_1_evt * n_test), n_test,
        )
    else:
        logger.warning("No test results available for evaluation.")
        breach_5_evt = None
        breach_1_evt = None

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("GPD shape (xi)    : %.4f  %s",
                gpd_raw["shape"],
                "(heavy tail)" if gpd_raw["shape"] > 0 else "(light tail)")
    logger.info("GPD scale (sigma) : %.6f", gpd_raw["scale"])
    logger.info("KS test p-value   : %.3f  %s",
                diag_raw.get("ks_pvalue") or 0,
                "(OK)" if (diag_raw.get("ks_pvalue") or 0) > 0.05 else "(POOR)")
    logger.info("Mean excess linear: %s", diag_raw.get("mean_excess_linear"))
    logger.info("EVT correction 5%%: %.6f", evt_cal_5.get("correction", 0))
    logger.info("EVT correction 1%%: %.6f", evt_cal_1.get("correction", 0))
    if breach_5_evt is not None:
        logger.info("Test VaR 5%% breach: %.1f%%", breach_5_evt * 100)
    if breach_1_evt is not None:
        logger.info("Test VaR 1%% breach: %.1f%%", breach_1_evt * 100)

    # Save results
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    out = {
        "phase": "II-C",
        "gpd_raw_returns": {
            "params": gpd_raw,
            "var_5pct": var_5_raw,
            "var_1pct": var_1_raw,
            "empirical_5pct": emp_5,
            "empirical_1pct": emp_1,
            "diagnostic": diag_raw,
        },
        "evt_calibration": {
            "var_5pct": {
                k: v for k, v in evt_cal_5.items()
                if k != "diagnostic"
            },
            "var_1pct": {
                k: v for k, v in evt_cal_1.items()
                if k != "diagnostic"
            },
        },
        "test_evaluation": {
            "var_5pct_breach_rate": breach_5_evt,
            "var_1pct_breach_rate": breach_1_evt,
            "n_test": len(test_results) if test_results else 0,
        },
    }

    # Convert numpy types for JSON serialization
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

    out_path = results_dir / "phase2c_evt.json"
    out_path.write_text(json.dumps(out, indent=2, default=_convert))
    logger.info("\nResults saved: %s", out_path)


if __name__ == "__main__":
    main()
