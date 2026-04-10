"""
Phase 3: Risk Signal System (runs on DGX)

Loads fine-tuned model (or zero-shot if Phase 0.5 passed) and produces:
  - 1-day VaR at 5% and 1%
  - Anomaly score (model surprise)
  - Uncertainty ratio
  - Position sizing signal

Uses conformal calibration from Phase 0.5 results.

Usage:
  python experiments/phase3_risk_signals.py                     # use best available model
  python experiments/phase3_risk_signals.py --model zero-shot   # force zero-shot
  python experiments/phase3_risk_signals.py --model progressive # force fine-tuned

Output:
  results/phase3_risk_signals.json
"""
import argparse
import json
import logging
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Same temporal split as Phase 0.5
CAL_START = "2023-07-01"
CAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-03-31"

CONTEXT_LEN = 512
HORIZON = 1  # 1-day

# TimesFM outputs fixed deciles P10-P90. VaR 5%/1% via conformal on P10.
QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load_model(model_choice: str):
    """Load TimesFM model based on choice."""
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")

    base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    if model_choice == "zero-shot":
        logger.info("Using zero-shot model (no fine-tuning)")
    elif model_choice == "progressive":
        ckpt = MODELS / "timesfm_progressive_best.pt"
        if ckpt.exists():
            base_model.model.load_state_dict(torch.load(ckpt, weights_only=True))
            logger.info("Loaded progressive fine-tuned model: %s", ckpt)
        else:
            logger.warning("Progressive checkpoint not found. Falling back to zero-shot.")
            model_choice = "zero-shot"
    elif model_choice == "daily":
        ckpt = MODELS / "timesfm_daily_best.pt"
        if ckpt.exists():
            base_model.model.load_state_dict(torch.load(ckpt, weights_only=True))
            logger.info("Loaded daily fine-tuned model: %s", ckpt)
        else:
            logger.warning("Daily checkpoint not found. Falling back to zero-shot.")
            model_choice = "zero-shot"
    else:  # auto
        # Pick best available
        for name in ["timesfm_progressive_best.pt", "timesfm_daily_best.pt"]:
            ckpt = MODELS / name
            if ckpt.exists():
                base_model.model.load_state_dict(torch.load(ckpt, weights_only=True))
                logger.info("Auto-selected model: %s", ckpt)
                model_choice = name.replace("timesfm_", "").replace("_best.pt", "")
                break
        else:
            logger.info("No fine-tuned model found. Using zero-shot.")
            model_choice = "zero-shot"

    base_model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    ))

    return base_model, model_choice


def load_btc_returns(freq: str = "daily") -> pd.Series:
    """Load BTC returns at specified frequency."""
    if freq == "daily":
        processed = ROOT / "data/processed/btc_full.parquet"
        if processed.exists():
            df = pd.read_parquet(processed).sort_index()
            if "btc_daily_return" in df.columns:
                return df["btc_daily_return"].dropna()
        raw = ROOT / "data/raw/btc_price.parquet"
    elif freq == "hourly":
        raw = ROOT / "data/raw/hourly/btc_1h.parquet"
    elif freq == "5min":
        raw = ROOT / "data/raw/5min/btc_5m.parquet"
    else:
        raise ValueError(f"Unknown freq: {freq}")

    df = pd.read_parquet(raw).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    returns.name = f"btc_{freq}_return"
    return returns


def walk_forward_forecast(model, returns: pd.Series, start: str, end: str) -> list[dict]:
    """Walk-forward 1-day forecast with quantiles."""
    mask = (returns.index >= start) & (returns.index <= end)
    dates = returns.index[mask]
    results = []
    t0 = time.time()

    for i, date in enumerate(dates):
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue

        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = float(returns.iloc[loc])

        try:
            point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
            point = float(point_fc[0][0])
            qf = quantile_fc[0][0]  # [N] values
            n_q = len(qf)
            offset = 1 if n_q >= 10 else 0  # skip mean if present
            quantiles = {}
            for j, q in enumerate(QUANTILE_LEVELS):
                idx = j + offset
                qk = f"q{int(q*100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
            results.append({"date": str(date.date()), "actual": actual, "point": point, **quantiles})
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Forecast failed at %s: %s", date.date(), e)

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d (%.1f sec)", i + 1, len(dates), time.time() - t0)

    logger.info("  %d forecasts in %.1f sec", len(results), time.time() - t0)
    return results


def evt_calibrate_var1(results: list[dict]) -> dict:
    """Use EVT (Generalized Pareto Distribution) for VaR 1% calibration.

    Conformal fails for VaR 1% because P10 is too far from P01.
    EVT models the left tail of residuals and extrapolates to P01.
    """
    try:
        from pipeline.evt import evt_calibrate
        cal_params = evt_calibrate(results, target_alpha=0.01)
        if cal_params.get("method") == "evt":
            logger.info("  EVT VaR 1%%: correction=%.4f", cal_params["correction"])
            return cal_params
        else:
            logger.warning("  EVT failed: %s. Falling back to conformal.", cal_params.get("reason", "unknown"))
    except ImportError:
        logger.warning("  pipeline.evt not available. Falling back to conformal.")
    except Exception as e:
        logger.warning("  EVT error: %s. Falling back to conformal.", e)

    # Fallback to conformal
    return conformal_calibrate(results, 0.01)


def conformal_calibrate(results: list[dict], alpha: float) -> dict:
    """Conformal calibration. Uses P10 as base quantile for all VaR levels."""
    q_key = "q10"  # nearest available quantile to 5% and 1%
    actuals = np.array([r["actual"] for r in results])
    predicted_q = np.array([r[q_key] for r in results])

    if len(results) < 50:
        return {"method": "raw", "correction": 0.0}

    scores = predicted_q - actuals
    n = len(scores)
    correction = float(np.quantile(scores, (1 - alpha) * (1 + 1 / n)))

    # Sanity checks
    raw_iqr = np.percentile(predicted_q, 75) - np.percentile(predicted_q, 25)
    if raw_iqr > 0 and abs(correction) > 3 * raw_iqr:
        return {"method": "isotonic_fallback", "correction": 0.0}

    raw_cov = float(np.mean(actuals <= predicted_q))
    adj_cov = float(np.mean(actuals <= (predicted_q - correction)))
    if adj_cov < alpha - 0.10:
        return {"method": "isotonic_fallback", "correction": 0.0}

    return {"method": "conformal", "correction": correction, "raw_coverage": raw_cov, "adj_coverage": adj_cov}


def compute_risk_signals(results: list[dict], cal_5: dict, cal_1: dict) -> list[dict]:
    """Compute risk signals for each day in results.

    Phase II-A fixes:
      - Anomaly: dynamic threshold via rolling 30-day P90 of anomaly scores
      - Uncertainty: IQR / realized_vol_30d (instead of IQR / |median|)
      - Position weight: rank-based 1 - percentile_rank(IQR) over rolling 60-day window
    """
    signals = []

    # Pre-compute realized volatility (rolling 30-day std of actual returns)
    actuals_arr = np.array([r["actual"] for r in results])
    realized_vols = pd.Series(actuals_arr).rolling(window=30, min_periods=1).std().values

    # Rolling windows for anomaly threshold and IQR percentile rank
    ANOMALY_WINDOW = 30
    POSITION_WINDOW = 60
    ANOMALY_FALLBACK_THRESHOLD = 1.0

    anomaly_score_history: deque[float] = deque(maxlen=ANOMALY_WINDOW)
    iqr_history: deque[float] = deque(maxlen=POSITION_WINDOW)

    for i, r in enumerate(results):
        actual = r["actual"]
        median = r["q50"]
        # VaR 5% and 1%: derived from P10 via conformal correction
        # (conformal shifts P10 downward to approximate P05/P01)
        q10_raw = r["q10"]
        var_5pct = q10_raw - cal_5.get("correction", 0)
        var_1pct = q10_raw - cal_1.get("correction", 0)
        q10 = q10_raw
        q90 = r["q90"]
        iqr = q90 - q10

        # --- Anomaly score: |actual - median| / IQR (unchanged formula) ---
        if abs(iqr) > 1e-10:
            anomaly_score = abs(actual - median) / iqr
        else:
            anomaly_score = 0.0

        # Dynamic anomaly threshold: rolling 30-day P90 of anomaly scores
        if len(anomaly_score_history) >= ANOMALY_WINDOW:
            anomaly_threshold = float(np.percentile(list(anomaly_score_history), 90))
        else:
            anomaly_threshold = ANOMALY_FALLBACK_THRESHOLD

        anomaly_flag = anomaly_score > anomaly_threshold
        anomaly_score_history.append(anomaly_score)

        # --- Uncertainty ratio: IQR / realized_vol_30d ---
        realized_vol = realized_vols[i]
        if realized_vol is not None and realized_vol > 1e-8:
            uncertainty = iqr / realized_vol
        else:
            # Fallback: use IQR directly as uncertainty measure
            uncertainty = iqr

        # --- Position weight: rank-based 1 - percentile_rank(IQR) ---
        iqr_history.append(iqr)
        if len(iqr_history) >= 2:
            # Percentile rank: fraction of values in window that are <= current IQR
            iqr_list = list(iqr_history)
            rank = sum(1 for v in iqr_list if v <= iqr) / len(iqr_list)
            position_weight = 1.0 - rank
        else:
            position_weight = 0.5  # neutral default for first observation

        # Clip to [0.1, 1.0]
        position_weight = max(0.1, min(1.0, position_weight))

        # VaR breaches
        var5_breach = actual < var_5pct
        var1_breach = actual < var_1pct

        signals.append({
            "date": r["date"],
            "actual_return": actual,
            "var_5pct": float(var_5pct),
            "var_1pct": float(var_1pct),
            "var5_breach": bool(var5_breach),
            "var1_breach": bool(var1_breach),
            "anomaly_score": float(anomaly_score),
            "anomaly_flag": bool(anomaly_flag),
            "anomaly_threshold": float(anomaly_threshold),
            "uncertainty_ratio": float(uncertainty),
            "position_weight": float(position_weight),
            "median_forecast": float(median),
            "iqr": float(iqr),
        })

    return signals


def evaluate_signals(signals: list[dict]) -> dict:
    """Evaluate risk signal quality.

    Phase II-A: uses dynamic anomaly_flag field instead of static threshold.
    """
    n = len(signals)
    var5_breaches = sum(1 for s in signals if s["var5_breach"])
    var1_breaches = sum(1 for s in signals if s["var1_breach"])

    var5_rate = var5_breaches / n if n > 0 else 0
    var1_rate = var1_breaches / n if n > 0 else 0

    # Anomaly precision: when anomaly_flag is True, does high vol follow?
    # "High vol" = |actual return| in top 20% of all returns
    all_abs_returns = [abs(s["actual_return"]) for s in signals]
    vol_threshold = np.percentile(all_abs_returns, 80) if all_abs_returns else 0

    # Use dynamic anomaly_flag (backward compatible: fall back to score > 2.0)
    anomaly_flagged = [s for s in signals if s.get("anomaly_flag", s["anomaly_score"] > 2.0)]
    if anomaly_flagged:
        # Check if current day is high vol (simplified without forward-looking)
        anomaly_correct = sum(1 for s in anomaly_flagged if abs(s["actual_return"]) > vol_threshold)
        anomaly_precision = anomaly_correct / len(anomaly_flagged)
    else:
        anomaly_precision = 0.0

    # Dynamic threshold statistics
    thresholds = [s["anomaly_threshold"] for s in signals if "anomaly_threshold" in s]
    avg_threshold = float(np.mean(thresholds)) if thresholds else 0.0
    min_threshold = float(np.min(thresholds)) if thresholds else 0.0
    max_threshold = float(np.max(thresholds)) if thresholds else 0.0

    logger.info("=== Risk Signal Evaluation ===")
    logger.info("  VaR 5%% breach rate: %.1f%% (%d/%d) [target: 3-8%%]", var5_rate * 100, var5_breaches, n)
    logger.info("  VaR 1%% breach rate: %.1f%% (%d/%d) [target: 0.5-2%%]", var1_rate * 100, var1_breaches, n)
    logger.info("  Anomaly flags (dynamic threshold): %d/%d (%.1f%%)", len(anomaly_flagged), n,
                len(anomaly_flagged) / n * 100 if n > 0 else 0)
    logger.info("  Anomaly dynamic threshold: avg=%.3f, min=%.3f, max=%.3f",
                avg_threshold, min_threshold, max_threshold)
    logger.info("  Anomaly precision: %.1f%%", anomaly_precision * 100)
    logger.info("  Avg uncertainty ratio (IQR/realized_vol): %.2f",
                np.mean([s["uncertainty_ratio"] for s in signals]))
    logger.info("  Avg position weight (rank-based): %.2f",
                np.mean([s["position_weight"] for s in signals]))
    logger.info("  Position weight range: [%.2f, %.2f]",
                np.min([s["position_weight"] for s in signals]),
                np.max([s["position_weight"] for s in signals]))

    return {
        "n_days": n,
        "var_5pct_breach_rate": float(var5_rate),
        "var_5pct_breaches": var5_breaches,
        "var_1pct_breach_rate": float(var1_rate),
        "var_1pct_breaches": var1_breaches,
        "anomaly_flags": len(anomaly_flagged),
        "anomaly_precision": float(anomaly_precision),
        "anomaly_threshold_avg": avg_threshold,
        "anomaly_threshold_range": [min_threshold, max_threshold],
        "avg_uncertainty": float(np.mean([s["uncertainty_ratio"] for s in signals])),
        "avg_position_weight": float(np.mean([s["position_weight"] for s in signals])),
        "position_weight_range": [
            float(np.min([s["position_weight"] for s in signals])),
            float(np.max([s["position_weight"] for s in signals])),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["auto", "zero-shot", "progressive", "daily"],
                        default="auto")
    parser.add_argument("--freq", choices=["daily", "hourly", "5min"],
                        default="daily",
                        help="Data frequency for evaluation")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 3: Risk Signal System")
    logger.info("=" * 60)

    # Load model
    try:
        model, model_name = load_model(args.model)
    except ImportError:
        logger.error("TimesFM not available. Install on DGX first.")
        return

    # Load data
    freq = args.freq
    returns = load_btc_returns(freq)
    logger.info("BTC %s returns: %d rows (%s -> %s)", freq, len(returns),
                returns.index[0].date(), returns.index[-1].date())

    # For hourly/5min, use last 20% as test (no fixed date split)
    if freq != "daily":
        n = len(returns)
        cal_start_idx = int(n * 0.6)
        cal_end_idx = int(n * 0.8)
        test_start_idx = int(n * 0.8)
        CAL_START_DYN = str(returns.index[cal_start_idx].date())
        CAL_END_DYN = str(returns.index[cal_end_idx].date())
        TEST_START_DYN = str(returns.index[test_start_idx].date())
        TEST_END_DYN = str(returns.index[-1].date())
        logger.info("  Dynamic split: cal=%s~%s, test=%s~%s",
                     CAL_START_DYN, CAL_END_DYN, TEST_START_DYN, TEST_END_DYN)

    # Use dynamic split for non-daily frequencies
    cal_s = CAL_START_DYN if freq != "daily" else CAL_START
    cal_e = CAL_END_DYN if freq != "daily" else CAL_END
    test_s = TEST_START_DYN if freq != "daily" else TEST_START
    test_e = TEST_END_DYN if freq != "daily" else TEST_END

    # Subsample for speed on high-freq data
    step = 1 if freq == "daily" else (6 if freq == "hourly" else 60)

    # Calibration set forecast
    logger.info("\n=== Calibration Set (%s) ===", freq)
    cal_mask = (returns.index >= cal_s) & (returns.index <= cal_e)
    cal_dates = returns.index[cal_mask][::step]
    cal_results = []
    for date in cal_dates:
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue
        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = float(returns.iloc[loc])
        try:
            point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
            point = float(point_fc[0][0])
            qf = quantile_fc[0][0]
            n_q = len(qf)
            offset = 1 if n_q >= 10 else 0
            quantiles = {}
            for j, q in enumerate(QUANTILE_LEVELS):
                idx = j + offset
                qk = f"q{int(q*100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
            cal_results.append({"date": str(date), "actual": actual, "point": point, **quantiles})
        except (RuntimeError, ValueError, IndexError):
            pass
    logger.info("  Cal forecasts: %d", len(cal_results))

    # Conformal calibration
    cal_5 = conformal_calibrate(cal_results, 0.05)
    cal_1 = evt_calibrate_var1(cal_results)
    logger.info("VaR 5%% calibration: %s (correction=%.6f)", cal_5["method"], cal_5.get("correction", 0))
    logger.info("VaR 1%% calibration: %s (correction=%.6f)", cal_1["method"], cal_1.get("correction", 0))

    # Test set forecast
    logger.info("\n=== Test Set (%s) ===", freq)
    test_mask = (returns.index >= test_s) & (returns.index <= test_e)
    test_dates = returns.index[test_mask][::step]
    test_results = []
    for date in test_dates:
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue
        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = float(returns.iloc[loc])
        try:
            point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
            point = float(point_fc[0][0])
            qf = quantile_fc[0][0]
            n_q = len(qf)
            offset = 1 if n_q >= 10 else 0
            quantiles = {}
            for j, q in enumerate(QUANTILE_LEVELS):
                idx = j + offset
                qk = f"q{int(q*100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
            test_results.append({"date": str(date), "actual": actual, "point": point, **quantiles})
        except (RuntimeError, ValueError, IndexError):
            pass
    logger.info("  Test forecasts: %d", len(test_results))

    # Compute risk signals
    signals = compute_risk_signals(test_results, cal_5, cal_1)

    # Evaluate
    evaluation = evaluate_signals(signals)

    # Compare with Phase 0.5 if available
    comparison = None
    p05_path = RESULTS / "phase05_smoke_test.json"
    if p05_path.exists() and model_name != "zero-shot":
        p05 = json.loads(p05_path.read_text())
        zs_breach = p05.get("evaluation", {}).get("var_5pct", {}).get("breach_rate", None)
        if zs_breach is not None:
            ft_breach = evaluation["var_5pct_breach_rate"]
            improvement = zs_breach - ft_breach  # positive = fine-tuned is better (closer to 5%)
            logger.info("\n=== Comparison: Fine-tuned vs Zero-shot ===")
            logger.info("  Zero-shot VaR 5%% breach rate: %.1f%%", zs_breach * 100)
            logger.info("  Fine-tuned VaR 5%% breach rate: %.1f%%", ft_breach * 100)
            logger.info("  Improvement: %.1f pp", improvement * 100)
            comparison = {
                "zero_shot_breach_rate": zs_breach,
                "fine_tuned_breach_rate": ft_breach,
                "improvement_pp": float(improvement),
            }

    # Save
    output = {
        "phase": "3",
        "model": model_name,
        "calibration": {"var_5pct": cal_5, "var_1pct": cal_1},
        "evaluation": evaluation,
        "comparison_vs_zero_shot": comparison,
        "signals_sample": signals[:10],  # first 10 days as sample
    }

    suffix = f"_{freq}" if freq != "daily" else ""
    out_path = RESULTS / f"phase3_risk_signals{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("\nResults: %s", out_path)

    # Save full signals
    signals_path = RESULTS / f"phase3_signals_full{suffix}.json"
    signals_path.write_text(json.dumps(signals, indent=2))
    logger.info("Full signals: %s", signals_path)


if __name__ == "__main__":
    main()
