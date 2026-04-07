"""
Phase 0.5: End-to-End Smoke Test (runs on DGX)

Zero-shot TimesFM 2.5 + conformal calibration → 1-day VaR 5%
This is the DECISION GATE: if zero-shot already achieves VaR breach rate 3-8%,
fine-tuning (Phase 2) may have low marginal value.

Also verifies:
  - TimesFM loads and runs on DGX
  - TimesFMFinetuner API exists (for Phase 2)
  - Quantile forecasts extract correctly
  - Conformal calibration produces valid intervals

Usage:
  python experiments/phase05_smoke_test.py

Output:
  results/phase05_smoke_test.json
"""
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ── Temporal split ──────────────────────────────────────────
# 30-day gap between splits to prevent future leakage
TRAIN_END = "2022-06-01"
VAL_START = "2022-07-01"
VAL_END = "2023-06-30"
CAL_START = "2023-07-01"
CAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-03-31"

CONTEXT_LEN = 512
HORIZON = 1  # 1-day VaR

# Custom quantile levels for VaR
QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def load_btc_daily() -> pd.Series:
    """Load BTC daily returns from processed data or raw price."""
    processed = ROOT / "data/processed/btc_full.parquet"
    if processed.exists():
        df = pd.read_parquet(processed).sort_index()
        if "btc_daily_return" in df.columns:
            returns = df["btc_daily_return"].dropna()
            logger.info("Loaded BTC daily returns from processed: %d rows (%s -> %s)",
                        len(returns), returns.index[0].date(), returns.index[-1].date())
            return returns

    # Fallback: compute from raw price
    raw = ROOT / "data/raw/btc_price.parquet"
    if not raw.exists():
        raise FileNotFoundError(f"No BTC data found at {processed} or {raw}")
    df = pd.read_parquet(raw).sort_index()
    price_col = [c for c in df.columns if "close" in c.lower()][0]
    returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
    returns.name = "btc_daily_return"
    logger.info("Computed BTC daily returns from raw: %d rows (%s -> %s)",
                len(returns), returns.index[0].date(), returns.index[-1].date())
    return returns


def check_timesfm_api():
    """Verify TimesFM loads and check if TimesFMFinetuner exists."""
    logger.info("=== Checking TimesFM API ===")
    try:
        import timesfm
        import torch

        logger.info("TimesFM version: %s", getattr(timesfm, "__version__", "unknown"))
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            logger.info("GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)

        # Check for Finetuner
        timesfm_attrs = dir(timesfm)
        has_finetuner = any("finetun" in a.lower() for a in timesfm_attrs)
        finetuner_names = [a for a in timesfm_attrs if "finetun" in a.lower()]
        logger.info("TimesFMFinetuner found: %s (%s)", has_finetuner, finetuner_names)

        # List all public classes/functions
        public_attrs = [a for a in timesfm_attrs if not a.startswith("_")]
        logger.info("Public API: %s", public_attrs)

        return {
            "timesfm_available": True,
            "has_finetuner": has_finetuner,
            "finetuner_names": finetuner_names,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError as e:
        logger.error("TimesFM not available: %s", e)
        logger.error("Install: pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]")
        return {"timesfm_available": False, "error": str(e)}


def zero_shot_forecast(returns: pd.Series, start: str, end: str) -> list[dict]:
    """Run walk-forward zero-shot TimesFM forecast with custom quantiles.

    For each day in [start, end], use the previous CONTEXT_LEN days as context,
    forecast 1 day ahead, record actual vs predicted quantiles.
    """
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")

    logger.info("=== Loading TimesFM model ===")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        quantiles=QUANTILE_LEVELS,
    ))
    logger.info("Model loaded and compiled with quantiles: %s", QUANTILE_LEVELS)

    # Get date range
    mask = (returns.index >= start) & (returns.index <= end)
    test_dates = returns.index[mask]
    logger.info("Forecasting %d days (%s -> %s)", len(test_dates), start, end)

    results = []
    t0 = time.time()

    for i, date in enumerate(test_dates):
        # Get context: CONTEXT_LEN days before this date
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue

        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = returns.iloc[loc]

        try:
            point_fc, quantile_fc = model.forecast(
                horizon=HORIZON, inputs=[context]
            )
            # point_fc shape: [1, horizon], quantile_fc shape: [1, horizon, num_quantiles]
            point = float(point_fc[0][0])
            quantiles = {
                f"q{int(q*100):02d}": float(quantile_fc[0][0][j])
                for j, q in enumerate(QUANTILE_LEVELS)
            }

            results.append({
                "date": str(date.date()),
                "actual": float(actual),
                "point_forecast": point,
                **quantiles,
            })

        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Forecast failed at %s: %s", date.date(), e)
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d done (%.1f sec, %.2f sec/step)",
                        i + 1, len(test_dates), elapsed, elapsed / (i + 1))

    elapsed = time.time() - t0
    logger.info("Forecasting complete: %d results in %.1f sec", len(results), elapsed)
    return results


def conformal_calibrate(cal_results: list[dict], alpha: float = 0.05) -> dict:
    """Split conformal prediction on calibration set.

    Computes conformity scores and determines the conformal correction.
    Returns calibration parameters to apply on test set.

    Args:
        cal_results: forecast results on calibration set
        alpha: VaR level (0.05 = 5% VaR)

    Returns:
        dict with calibration parameters
    """
    if len(cal_results) < 50:
        logger.warning("Calibration set too small (%d < 50). Falling back to raw quantiles.", len(cal_results))
        return {"method": "raw", "reason": "cal_set_too_small", "n_cal": len(cal_results)}

    q_key = f"q{int(alpha * 100):02d}"

    actuals = np.array([r["actual"] for r in cal_results])
    predicted_q = np.array([r[q_key] for r in cal_results])

    # Conformity scores: how much the actual exceeded the predicted quantile
    # For lower tail VaR: score = predicted_quantile - actual
    # (positive when actual is below predicted quantile, i.e., quantile was too high)
    scores = predicted_q - actuals

    # Conformal correction: (1-alpha)(1+1/n) quantile of scores
    n = len(scores)
    conformal_quantile = np.quantile(scores, (1 - alpha) * (1 + 1 / n))

    # Check if conformal intervals are reasonable
    raw_coverage = np.mean(actuals <= predicted_q)
    corrected_q = predicted_q - conformal_quantile
    corrected_coverage = np.mean(actuals <= corrected_q)

    # Raw quantile width vs conformal width (using IQR as proxy)
    raw_iqr = np.percentile(predicted_q, 75) - np.percentile(predicted_q, 25)
    correction_magnitude = abs(conformal_quantile)

    logger.info("=== Conformal Calibration (alpha=%.2f) ===", alpha)
    logger.info("  Cal set size: %d", n)
    logger.info("  Raw coverage: %.1f%% (target: %.1f%%)", raw_coverage * 100, alpha * 100)
    logger.info("  Conformal correction: %.6f", conformal_quantile)
    logger.info("  Corrected coverage on cal set: %.1f%%", corrected_coverage * 100)

    # Fallback check: correction too large = conformal not working
    if raw_iqr > 0 and correction_magnitude > 3 * raw_iqr:
        logger.warning("Conformal correction (%.4f) > 3x raw IQR (%.4f). Falling back to isotonic.",
                        correction_magnitude, raw_iqr)
        return {
            "method": "isotonic_fallback",
            "reason": "correction_too_large",
            "raw_coverage": float(raw_coverage),
            "correction": float(conformal_quantile),
            "n_cal": n,
        }

    # Fallback check: coverage way off target
    if corrected_coverage < alpha - 0.10:
        logger.warning("Corrected coverage %.1f%% < target-10%%. Falling back to isotonic.",
                        corrected_coverage * 100)
        return {
            "method": "isotonic_fallback",
            "reason": "coverage_too_low",
            "raw_coverage": float(raw_coverage),
            "corrected_coverage": float(corrected_coverage),
            "n_cal": n,
        }

    return {
        "method": "conformal",
        "alpha": alpha,
        "conformal_correction": float(conformal_quantile),
        "raw_coverage_cal": float(raw_coverage),
        "corrected_coverage_cal": float(corrected_coverage),
        "n_cal": n,
    }


def evaluate_test_set(test_results: list[dict], cal_params: dict, alpha: float = 0.05) -> dict:
    """Evaluate VaR on test set using calibration parameters. Report only, no decisions."""
    q_key = f"q{int(alpha * 100):02d}"

    actuals = np.array([r["actual"] for r in test_results])
    predicted_q = np.array([r[q_key] for r in test_results])

    if cal_params["method"] == "conformal":
        adjusted_q = predicted_q - cal_params["conformal_correction"]
    else:
        # Raw quantiles (isotonic fallback or raw)
        adjusted_q = predicted_q

    # VaR breach: actual return < VaR threshold
    breaches = actuals < adjusted_q
    breach_rate = float(np.mean(breaches))
    n_breaches = int(np.sum(breaches))

    # Quantile calibration: check all quantile levels
    calibration = {}
    for q in QUANTILE_LEVELS:
        qk = f"q{int(q * 100):02d}"
        pred_q = np.array([r[qk] for r in test_results])
        if cal_params["method"] == "conformal":
            # Apply same correction direction for all quantiles
            # (simplified: full conformal would calibrate each quantile separately)
            pred_q_adj = pred_q - cal_params["conformal_correction"]
        else:
            pred_q_adj = pred_q
        actual_coverage = float(np.mean(actuals <= pred_q_adj))
        calibration[qk] = {
            "target": q,
            "actual": actual_coverage,
            "deviation": abs(actual_coverage - q),
        }

    avg_cal_deviation = np.mean([v["deviation"] for v in calibration.values()])

    logger.info("=== Test Set Evaluation ===")
    logger.info("  Test set size: %d", len(test_results))
    logger.info("  Calibration method: %s", cal_params["method"])
    logger.info("  1-day VaR %.0f%% breach rate: %.1f%% (%d/%d)",
                alpha * 100, breach_rate * 100, n_breaches, len(test_results))
    logger.info("  Avg quantile calibration deviation: %.1f%%", avg_cal_deviation * 100)
    logger.info("  Per-quantile calibration:")
    for qk, v in calibration.items():
        status = "OK" if v["deviation"] < 0.05 else "OFF"
        logger.info("    %s: target=%.0f%% actual=%.1f%% dev=%.1f%% [%s]",
                     qk, v["target"] * 100, v["actual"] * 100, v["deviation"] * 100, status)

    # Decision gate
    gate_pass = 0.03 <= breach_rate <= 0.08
    logger.info("\n  ╔══════════════════════════════════════════╗")
    logger.info("  ║  DECISION GATE: VaR 5%% breach rate = %.1f%%  ║", breach_rate * 100)
    if gate_pass:
        logger.info("  ║  RESULT: PASS (3-8%% range)                ║")
        logger.info("  ║  Zero-shot may be sufficient.              ║")
        logger.info("  ║  Fine-tuning has LOW marginal value.       ║")
    else:
        logger.info("  ║  RESULT: FAIL (outside 3-8%% range)        ║")
        logger.info("  ║  Fine-tuning (Phase 2) recommended.        ║")
    logger.info("  ╚══════════════════════════════════════════╝")

    return {
        "n_test": len(test_results),
        "var_alpha": alpha,
        "breach_rate": breach_rate,
        "n_breaches": n_breaches,
        "avg_calibration_deviation": float(avg_cal_deviation),
        "calibration": calibration,
        "gate_pass": gate_pass,
        "gate_recommendation": "zero-shot sufficient" if gate_pass else "proceed to Phase 2",
    }


def main():
    logger.info("=" * 60)
    logger.info("Phase 0.5: End-to-End Smoke Test")
    logger.info("=" * 60)

    # Step 1: Check TimesFM API
    api_check = check_timesfm_api()
    if not api_check.get("timesfm_available"):
        logger.error("Cannot proceed without TimesFM. Exiting.")
        result = {"phase": "0.5", "status": "BLOCKED", "api_check": api_check}
        (RESULTS / "phase05_smoke_test.json").write_text(json.dumps(result, indent=2))
        return

    # Step 2: Load BTC daily returns
    returns = load_btc_daily()

    # Step 3: Zero-shot forecast on calibration set
    logger.info("\n=== Calibration Set Forecast ===")
    cal_results = zero_shot_forecast(returns, CAL_START, CAL_END)
    logger.info("Cal set: %d forecasts", len(cal_results))

    # Step 4: Conformal calibration (decision made HERE, not on test set)
    cal_params_5pct = conformal_calibrate(cal_results, alpha=0.05)
    cal_params_1pct = conformal_calibrate(cal_results, alpha=0.01)

    # Step 5: Zero-shot forecast on test set
    logger.info("\n=== Test Set Forecast ===")
    test_results = zero_shot_forecast(returns, TEST_START, TEST_END)
    logger.info("Test set: %d forecasts", len(test_results))

    # Step 6: Evaluate (report only, no decisions)
    eval_5pct = evaluate_test_set(test_results, cal_params_5pct, alpha=0.05)
    eval_1pct = evaluate_test_set(test_results, cal_params_1pct, alpha=0.01)

    # Save results
    result = {
        "phase": "0.5",
        "status": "DONE",
        "api_check": api_check,
        "temporal_split": {
            "cal": f"{CAL_START} to {CAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
            "context_len": CONTEXT_LEN,
            "horizon": HORIZON,
        },
        "calibration_params": {
            "var_5pct": cal_params_5pct,
            "var_1pct": cal_params_1pct,
        },
        "evaluation": {
            "var_5pct": eval_5pct,
            "var_1pct": eval_1pct,
        },
        "n_cal_forecasts": len(cal_results),
        "n_test_forecasts": len(test_results),
    }

    out_path = RESULTS / "phase05_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("\nResults saved: %s", out_path)

    # Also save raw forecast data for debugging
    raw_path = RESULTS / "phase05_forecasts.json"
    raw_path.write_text(json.dumps({
        "cal": cal_results,
        "test": test_results,
    }, indent=2))
    logger.info("Raw forecasts saved: %s", raw_path)


if __name__ == "__main__":
    main()
