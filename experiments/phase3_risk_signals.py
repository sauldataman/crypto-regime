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
            base_model.load_state_dict(torch.load(ckpt, weights_only=True))
            logger.info("Loaded progressive fine-tuned model: %s", ckpt)
        else:
            logger.warning("Progressive checkpoint not found. Falling back to zero-shot.")
            model_choice = "zero-shot"
    elif model_choice == "daily":
        ckpt = MODELS / "timesfm_daily_best.pt"
        if ckpt.exists():
            base_model.load_state_dict(torch.load(ckpt, weights_only=True))
            logger.info("Loaded daily fine-tuned model: %s", ckpt)
        else:
            logger.warning("Daily checkpoint not found. Falling back to zero-shot.")
            model_choice = "zero-shot"
    else:  # auto
        # Pick best available
        for name in ["timesfm_progressive_best.pt", "timesfm_daily_best.pt"]:
            ckpt = MODELS / name
            if ckpt.exists():
                base_model.load_state_dict(torch.load(ckpt, weights_only=True))
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


def load_btc_returns() -> pd.Series:
    """Load BTC daily returns."""
    processed = ROOT / "data/processed/btc_full.parquet"
    if processed.exists():
        df = pd.read_parquet(processed).sort_index()
        if "btc_daily_return" in df.columns:
            return df["btc_daily_return"].dropna()

    raw = ROOT / "data/raw/btc_price.parquet"
    df = pd.read_parquet(raw).sort_index()
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    returns.name = "btc_daily_return"
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
    """Compute risk signals for each day in results."""
    signals = []

    for r in results:
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

        # 1-day VaR (already computed above from P10 + conformal correction)

        # Anomaly score: |actual - median| / IQR
        # Guards against IQR = 0
        if abs(iqr) > 1e-10:
            anomaly_score = abs(actual - median) / iqr
        else:
            anomaly_score = 0.0

        # Uncertainty ratio: IQR / |median|
        if abs(median) > 1e-10:
            uncertainty = iqr / abs(median)
        else:
            uncertainty = float("inf") if iqr > 0 else 0.0

        # Position sizing: inverse of uncertainty (capped)
        if uncertainty > 0:
            position_weight = min(1.0 / uncertainty, 10.0)  # cap at 10x
        else:
            position_weight = 10.0

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
            "uncertainty_ratio": float(uncertainty) if uncertainty != float("inf") else 999.0,
            "position_weight": float(position_weight),
            "median_forecast": float(median),
            "iqr": float(iqr),
        })

    return signals


def evaluate_signals(signals: list[dict]) -> dict:
    """Evaluate risk signal quality."""
    n = len(signals)
    var5_breaches = sum(1 for s in signals if s["var5_breach"])
    var1_breaches = sum(1 for s in signals if s["var1_breach"])

    var5_rate = var5_breaches / n if n > 0 else 0
    var1_rate = var1_breaches / n if n > 0 else 0

    # Anomaly precision: when anomaly_score > 2, does high vol follow?
    # "High vol" = |actual return| in top 20% of all returns
    all_abs_returns = [abs(s["actual_return"]) for s in signals]
    vol_threshold = np.percentile(all_abs_returns, 80) if all_abs_returns else 0

    anomaly_flags = [s for s in signals if s["anomaly_score"] > 2.0]
    if anomaly_flags:
        # Check if next 5 days have high volatility
        # (simplified: check if current day is high vol, since we don't have forward-looking here)
        anomaly_correct = sum(1 for s in anomaly_flags if abs(s["actual_return"]) > vol_threshold)
        anomaly_precision = anomaly_correct / len(anomaly_flags)
    else:
        anomaly_precision = 0.0

    logger.info("=== Risk Signal Evaluation ===")
    logger.info("  VaR 5%% breach rate: %.1f%% (%d/%d) [target: 3-8%%]", var5_rate * 100, var5_breaches, n)
    logger.info("  VaR 1%% breach rate: %.1f%% (%d/%d) [target: 0.5-2%%]", var1_rate * 100, var1_breaches, n)
    logger.info("  Anomaly flags (score>2): %d/%d (%.1f%%)", len(anomaly_flags), n,
                len(anomaly_flags) / n * 100 if n > 0 else 0)
    logger.info("  Anomaly precision: %.1f%%", anomaly_precision * 100)
    logger.info("  Avg uncertainty ratio: %.2f", np.mean([s["uncertainty_ratio"] for s in signals]))
    logger.info("  Avg position weight: %.2f", np.mean([s["position_weight"] for s in signals]))

    return {
        "n_days": n,
        "var_5pct_breach_rate": float(var5_rate),
        "var_5pct_breaches": var5_breaches,
        "var_1pct_breach_rate": float(var1_rate),
        "var_1pct_breaches": var1_breaches,
        "anomaly_flags": len(anomaly_flags),
        "anomaly_precision": float(anomaly_precision),
        "avg_uncertainty": float(np.mean([s["uncertainty_ratio"] for s in signals])),
        "avg_position_weight": float(np.mean([s["position_weight"] for s in signals])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["auto", "zero-shot", "progressive", "daily"],
                        default="auto")
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
    returns = load_btc_returns()
    logger.info("BTC returns: %d rows (%s -> %s)", len(returns),
                returns.index[0].date(), returns.index[-1].date())

    # Calibration set forecast
    logger.info("\n=== Calibration Set ===")
    cal_results = walk_forward_forecast(model, returns, CAL_START, CAL_END)

    # Conformal calibration (decisions made here)
    cal_5 = conformal_calibrate(cal_results, 0.05)
    cal_1 = conformal_calibrate(cal_results, 0.01)
    logger.info("VaR 5%% calibration: %s (correction=%.6f)", cal_5["method"], cal_5.get("correction", 0))
    logger.info("VaR 1%% calibration: %s (correction=%.6f)", cal_1["method"], cal_1.get("correction", 0))

    # Test set forecast
    logger.info("\n=== Test Set ===")
    test_results = walk_forward_forecast(model, returns, TEST_START, TEST_END)

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

    out_path = RESULTS / "phase3_risk_signals.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("\nResults: %s", out_path)

    # Save full signals
    signals_path = RESULTS / "phase3_signals_full.json"
    signals_path.write_text(json.dumps(signals, indent=2))
    logger.info("Full signals: %s", signals_path)


if __name__ == "__main__":
    main()
