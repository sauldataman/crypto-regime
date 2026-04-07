"""
TimesFM 2.5 Crypto Evaluation Framework (runs on DGX)

Core question: Is TimesFM useful for crypto? In what ways?

6 evaluation dimensions:
  1. Zero-shot capability (point forecast, direction, quantile calibration)
  2. vs Traditional methods (AR, ARIMA, GARCH)
  3. Fine-tune effect (zero-shot vs daily-only vs progressive)
  4. Cross-asset (6 assets: which benefits most?)
  5. Cross-frequency (daily vs hourly vs 5min)
  6. Cross-horizon (1d vs 5d vs 30d)

Usage:
  python experiments/eval_timesfm.py                        # full evaluation
  python experiments/eval_timesfm.py --dim zero-shot        # single dimension
  python experiments/eval_timesfm.py --dim cross-asset      # single dimension
  python experiments/eval_timesfm.py --asset btc --freq daily  # specific combo

Output:
  results/eval_timesfm.json           — full results
  results/eval_timesfm_summary.md     — human-readable summary
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
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────
CONTEXT_LEN = 512
HORIZONS = [1, 5, 30]
# TimesFM 2.5 quantile head outputs fixed deciles: mean, P10, P20, ..., P90
# Cannot customize quantile levels. VaR 5% derived via conformal correction on P10.
QUANTILE_INDICES = {"q10": 0, "q20": 1, "q30": 2, "q40": 3, "q50": 4,
                    "q60": 5, "q70": 6, "q80": 7, "q90": 8}
QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

ASSETS = {
    "btc": {"daily": "data/raw/btc_price.parquet", "hourly": "data/raw/hourly/btc_1h.parquet", "5min": "data/raw/5min/btc_5m.parquet"},
    "eth": {"daily": "data/raw/eth_price.parquet", "hourly": "data/raw/hourly/eth_1h.parquet", "5min": "data/raw/5min/eth_5m.parquet"},
    "sol": {"daily": "data/raw/sol_price.parquet", "hourly": "data/raw/hourly/sol_1h.parquet", "5min": "data/raw/5min/sol_5m.parquet"},
    "bnb": {"daily": "data/raw/bnb_price.parquet", "hourly": "data/raw/hourly/bnb_1h.parquet", "5min": "data/raw/5min/bnb_5m.parquet"},
    "doge": {"daily": "data/raw/doge_price.parquet", "hourly": "data/raw/hourly/doge_1h.parquet", "5min": "data/raw/5min/doge_5m.parquet"},
    "link": {"daily": "data/raw/link_price.parquet", "hourly": "data/raw/hourly/link_1h.parquet", "5min": "data/raw/5min/link_5m.parquet"},
}

# Test window (same as Phase 0.5)
TEST_START = "2024-07-01"
TEST_END = "2025-03-31"
# For hourly/5min: use last 20% of data as test
TEST_FRACTION = 0.2


def load_returns(asset: str, freq: str) -> pd.Series:
    """Load return series for a given asset and frequency."""
    path = ROOT / ASSETS[asset][freq]
    if not path.exists():
        logger.warning("File not found: %s", path)
        return pd.Series(dtype=float)

    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        logger.warning("No close column in %s", path)
        return pd.Series(dtype=float)

    close = df[close_col[0]]
    returns = np.log(close / close.shift(1)).dropna()
    returns.name = f"{asset}_{freq}_return"
    return returns


def get_test_indices(returns: pd.Series, freq: str) -> pd.DatetimeIndex:
    """Get test indices depending on frequency."""
    if freq == "daily":
        mask = (returns.index >= TEST_START) & (returns.index <= TEST_END)
        return returns.index[mask]
    else:
        # Use last TEST_FRACTION of data
        n = len(returns)
        start_idx = int(n * (1 - TEST_FRACTION))
        return returns.index[start_idx:]


def load_timesfm_model(model_type: str = "zero-shot"):
    """Load TimesFM model."""
    import timesfm
    import torch
    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    if model_type != "zero-shot":
        ckpt_map = {
            "progressive": "timesfm_progressive_best.pt",
            "daily": "timesfm_daily_best.pt",
            "5min": "timesfm_5min_best.pt",
            "hourly": "timesfm_hourly_best.pt",
        }
        ckpt = MODELS / ckpt_map.get(model_type, f"timesfm_{model_type}_best.pt")
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            logger.info("Loaded checkpoint: %s", ckpt)
        else:
            logger.warning("Checkpoint not found: %s. Using zero-shot.", ckpt)
            model_type = "zero-shot"

    return model, model_type


def compile_model(model, max_horizon: int):
    """Compile model with quantile head enabled."""
    import timesfm
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    ))


# ── Metrics ─────────────────────────────────────────────────

def compute_metrics(actuals: np.ndarray, predictions: np.ndarray,
                     quantile_preds: dict = None) -> dict:
    """Compute evaluation metrics."""
    n = len(actuals)
    if n == 0:
        return {}

    # Point forecast metrics
    mae = float(np.mean(np.abs(actuals - predictions)))
    rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    direction_acc = float(np.mean(np.sign(actuals) == np.sign(predictions)))

    # Correlation
    if np.std(predictions) > 0 and np.std(actuals) > 0:
        corr = float(np.corrcoef(actuals, predictions)[0, 1])
    else:
        corr = 0.0

    metrics = {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_acc,
        "correlation": corr,
    }

    # Quantile calibration
    if quantile_preds:
        calibration = {}
        for q_key, q_values in quantile_preds.items():
            q_level = int(q_key.replace("q", "")) / 100
            if len(q_values) == 0:
                continue
            coverage = float(np.mean(actuals[:len(q_values)] <= q_values))
            calibration[q_key] = {
                "target": q_level,
                "actual": coverage,
                "deviation": abs(coverage - q_level),
            }
        if calibration:
            avg_deviation = np.mean([v["deviation"] for v in calibration.values()])
            metrics["quantile_calibration"] = calibration
            metrics["avg_calibration_deviation"] = float(avg_deviation)

        # VaR metrics: use P10 as proxy for VaR 10%
        # (VaR 5% requires conformal correction on P10, done in phase05/phase3)
        if "q10" in quantile_preds and len(quantile_preds["q10"]) > 0:
            var10_breach = float(np.mean(actuals[:len(quantile_preds["q10"])] < quantile_preds["q10"]))
            metrics["var_10pct_breach_rate"] = var10_breach

    return metrics


# ── Walk-forward forecast engine ────────────────────────────

def walk_forward(model, returns: pd.Series, test_dates: pd.DatetimeIndex,
                  horizon: int, step: int = 1) -> dict:
    """Walk-forward forecast and collect metrics.

    Args:
        model: compiled TimesFM model
        returns: full return series
        test_dates: dates to forecast
        horizon: forecast horizon
        step: subsample every N dates for speed

    Returns:
        dict with actuals, predictions, quantile predictions
    """
    actuals = []
    predictions = []
    quantile_preds = {qk: [] for qk in QUANTILE_INDICES}

    sampled_dates = test_dates[::step]
    t0 = time.time()

    for i, date in enumerate(sampled_dates):
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN or loc + horizon - 1 >= len(returns):
            continue

        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()

        # For multi-step horizon, actual is cumulative return or last step
        if horizon == 1:
            actual = float(returns.iloc[loc])
        else:
            actual = float(returns.iloc[loc:loc + horizon].sum())  # cumulative

        try:
            point_fc, quantile_fc = model.forecast(horizon=horizon, inputs=[context])
            # quantile_fc shape: [1, horizon, 10] where idx 0=mean, 1-9=P10..P90
            # Actually: first element is point forecast, remaining 9 are P10-P90
            # Verify shape and adapt
            qf = quantile_fc[0]  # [horizon, n_quantiles]
            n_q = qf.shape[-1] if len(qf.shape) > 1 else 1

            if horizon == 1:
                pred = float(point_fc[0][0])
                for qk, qi in QUANTILE_INDICES.items():
                    # quantile_fc may include mean as idx 0, deciles start at idx 1
                    idx = qi + 1 if n_q >= 10 else qi  # adapt to actual layout
                    if idx < n_q:
                        quantile_preds[qk].append(float(qf[0][idx]))
                    else:
                        quantile_preds[qk].append(float(qf[0][min(qi, n_q - 1)]))
            else:
                pred = float(np.sum(point_fc[0][:horizon]))
                for qk, qi in QUANTILE_INDICES.items():
                    idx = qi + 1 if n_q >= 10 else qi
                    if idx < n_q:
                        q_sum = float(np.sum(qf[:horizon, idx]))
                    else:
                        q_sum = float(np.sum(qf[:horizon, min(qi, n_q - 1)]))
                    quantile_preds[qk].append(q_sum)

            actuals.append(actual)
            predictions.append(pred)

        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Forecast failed at %s: %s", date, e)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d (%.1f sec)", i + 1, len(sampled_dates), time.time() - t0)

    return {
        "actuals": np.array(actuals),
        "predictions": np.array(predictions),
        "quantile_preds": {k: np.array(v) for k, v in quantile_preds.items()},
    }


# ── Traditional baselines ───────────────────────────────────

def ar_forecast(returns: pd.Series, test_dates: pd.DatetimeIndex,
                 horizon: int, train_window: int = 252) -> dict:
    """AR(1) walk-forward baseline."""
    from statsmodels.tsa.ar_model import AutoReg

    actuals, predictions = [], []
    for date in test_dates[::5]:  # subsample for speed
        loc = returns.index.get_loc(date)
        if loc < train_window or loc + horizon - 1 >= len(returns):
            continue
        train = returns.iloc[loc - train_window:loc].values
        try:
            mdl = AutoReg(train, lags=1, old_names=False).fit(disp=False)
            fc = mdl.forecast(horizon)
            actual = float(returns.iloc[loc:loc + horizon].sum()) if horizon > 1 else float(returns.iloc[loc])
            pred = float(np.sum(fc)) if horizon > 1 else float(fc[0])
            actuals.append(actual)
            predictions.append(pred)
        except (ValueError, np.linalg.LinAlgError):
            pass

    return {"actuals": np.array(actuals), "predictions": np.array(predictions), "quantile_preds": {}}


def garch_forecast(returns: pd.Series, test_dates: pd.DatetimeIndex,
                    horizon: int, train_window: int = 252) -> dict:
    """GARCH(1,1) walk-forward baseline for volatility."""
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not installed. Skipping GARCH. Install: pip install arch")
        return {"actuals": np.array([]), "predictions": np.array([]), "quantile_preds": {}}

    actuals, predictions = [], []
    for date in test_dates[::10]:  # subsample heavily (GARCH is slow)
        loc = returns.index.get_loc(date)
        if loc < train_window or loc + horizon - 1 >= len(returns):
            continue
        train = returns.iloc[loc - train_window:loc].values * 100  # arch likes %

        try:
            mdl = arch_model(train, vol="Garch", p=1, q=1, mean="AR", lags=1, dist="t")
            res = mdl.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=horizon)
            mean_fc = fc.mean.iloc[-1].values / 100  # back to decimal
            actual = float(returns.iloc[loc:loc + horizon].sum()) if horizon > 1 else float(returns.iloc[loc])
            pred = float(np.sum(mean_fc)) if horizon > 1 else float(mean_fc[0])
            actuals.append(actual)
            predictions.append(pred)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            pass

    return {"actuals": np.array(actuals), "predictions": np.array(predictions), "quantile_preds": {}}


# ── Dimension evaluators ────────────────────────────────────

def eval_zero_shot(model, returns: pd.Series, test_dates: pd.DatetimeIndex) -> dict:
    """Dimension 1: Zero-shot capability across horizons."""
    logger.info("=== Dim 1: Zero-shot Capability ===")
    results = {}
    for h in HORIZONS:
        compile_model(model, max_horizon=h)
        step = max(1, len(test_dates) // 200)  # ~200 samples
        data = walk_forward(model, returns, test_dates, horizon=h, step=step)
        metrics = compute_metrics(data["actuals"], data["predictions"], data["quantile_preds"])
        results[f"h{h}"] = metrics
        logger.info("  h=%d: dir=%.3f mae=%.5f cal_dev=%.3f",
                     h, metrics.get("direction_accuracy", 0),
                     metrics.get("mae", 0), metrics.get("avg_calibration_deviation", 0))
    return results


def eval_vs_traditional(returns: pd.Series, test_dates: pd.DatetimeIndex,
                         timesfm_results: dict) -> dict:
    """Dimension 2: vs AR/ARIMA/GARCH."""
    logger.info("=== Dim 2: vs Traditional Methods ===")
    results = {}
    for h in HORIZONS:
        logger.info("  Horizon %d:", h)

        # AR
        ar_data = ar_forecast(returns, test_dates, h)
        ar_metrics = compute_metrics(ar_data["actuals"], ar_data["predictions"])
        results[f"ar_h{h}"] = ar_metrics
        logger.info("    AR: dir=%.3f mae=%.5f", ar_metrics.get("direction_accuracy", 0), ar_metrics.get("mae", 0))

        # GARCH (daily only, too slow for high freq)
        garch_data = garch_forecast(returns, test_dates, h)
        if len(garch_data["actuals"]) > 0:
            garch_metrics = compute_metrics(garch_data["actuals"], garch_data["predictions"])
            results[f"garch_h{h}"] = garch_metrics
            logger.info("    GARCH: dir=%.3f mae=%.5f", garch_metrics.get("direction_accuracy", 0), garch_metrics.get("mae", 0))

        # TimesFM (from dim 1)
        tfm_key = f"h{h}"
        if tfm_key in timesfm_results:
            results[f"timesfm_h{h}"] = timesfm_results[tfm_key]

    return results


def eval_cross_asset(model) -> dict:
    """Dimension 4: Performance across assets."""
    logger.info("=== Dim 4: Cross-Asset ===")
    results = {}
    compile_model(model, max_horizon=1)

    for asset in ASSETS:
        returns = load_returns(asset, "daily")
        if returns.empty:
            continue
        test_dates = get_test_indices(returns, "daily")
        if len(test_dates) < 50:
            logger.warning("  %s: too few test dates (%d)", asset, len(test_dates))
            continue

        step = max(1, len(test_dates) // 150)
        data = walk_forward(model, returns, test_dates, horizon=1, step=step)
        metrics = compute_metrics(data["actuals"], data["predictions"], data["quantile_preds"])
        results[asset] = metrics
        logger.info("  %s: dir=%.3f mae=%.5f cal_dev=%.3f n=%d",
                     asset, metrics.get("direction_accuracy", 0), metrics.get("mae", 0),
                     metrics.get("avg_calibration_deviation", 0), metrics.get("n", 0))

    return results


def eval_cross_frequency(model, asset: str = "btc") -> dict:
    """Dimension 5: Performance across frequencies."""
    logger.info("=== Dim 5: Cross-Frequency (%s) ===", asset)
    results = {}
    compile_model(model, max_horizon=1)

    for freq in ["daily", "hourly", "5min"]:
        returns = load_returns(asset, freq)
        if returns.empty or len(returns) < CONTEXT_LEN + 100:
            logger.warning("  %s %s: insufficient data", asset, freq)
            continue

        test_dates = get_test_indices(returns, freq)
        # Subsample more aggressively for high-freq
        n_samples = 200 if freq == "daily" else 100
        step = max(1, len(test_dates) // n_samples)

        data = walk_forward(model, returns, test_dates, horizon=1, step=step)
        metrics = compute_metrics(data["actuals"], data["predictions"], data["quantile_preds"])
        results[freq] = metrics
        logger.info("  %s %s: dir=%.3f mae=%.5f cal_dev=%.3f n=%d",
                     asset, freq, metrics.get("direction_accuracy", 0), metrics.get("mae", 0),
                     metrics.get("avg_calibration_deviation", 0), metrics.get("n", 0))

    return results


def eval_finetune_effect(returns: pd.Series, test_dates: pd.DatetimeIndex) -> dict:
    """Dimension 3: Zero-shot vs fine-tuned models."""
    logger.info("=== Dim 3: Fine-tune Effect ===")
    results = {}

    model_types = ["zero-shot", "daily", "progressive"]
    for mt in model_types:
        try:
            model, actual_type = load_timesfm_model(mt)
            compile_model(model, max_horizon=1)
            step = max(1, len(test_dates) // 200)
            data = walk_forward(model, returns, test_dates, horizon=1, step=step)
            metrics = compute_metrics(data["actuals"], data["predictions"], data["quantile_preds"])
            results[actual_type] = metrics
            logger.info("  %s: dir=%.3f mae=%.5f cal_dev=%.3f",
                         actual_type, metrics.get("direction_accuracy", 0),
                         metrics.get("mae", 0), metrics.get("avg_calibration_deviation", 0))
        except (ImportError, RuntimeError) as e:
            logger.warning("  %s: failed (%s)", mt, e)

    return results


# ── Summary generator ───────────────────────────────────────

def generate_summary(all_results: dict) -> str:
    """Generate human-readable markdown summary."""
    lines = [
        "# TimesFM 2.5 Crypto Evaluation Results",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Dim 1: Zero-shot
    if "zero_shot" in all_results:
        lines.append("## 1. Zero-shot Capability")
        lines.append("")
        lines.append("| Horizon | Direction Acc | MAE | RMSE | Quantile Cal Dev | VaR 10% Breach |")
        lines.append("|---------|-------------|-----|------|-----------------|----------------|")
        for h in HORIZONS:
            m = all_results["zero_shot"].get(f"h{h}", {})
            lines.append(f"| {h}d | {m.get('direction_accuracy', 0):.3f} | "
                         f"{m.get('mae', 0):.5f} | {m.get('rmse', 0):.5f} | "
                         f"{m.get('avg_calibration_deviation', 0):.3f} | "
                         f"{m.get('var_10pct_breach_rate', 'N/A')} |")
        lines.append("")

    # Dim 2: vs Traditional
    if "vs_traditional" in all_results:
        lines.append("## 2. vs Traditional Methods (h=1)")
        lines.append("")
        lines.append("| Method | Direction Acc | MAE | Correlation |")
        lines.append("|--------|-------------|-----|-------------|")
        for key in ["timesfm_h1", "ar_h1", "garch_h1"]:
            m = all_results["vs_traditional"].get(key, {})
            name = key.replace("_h1", "").upper()
            if m:
                lines.append(f"| {name} | {m.get('direction_accuracy', 0):.3f} | "
                             f"{m.get('mae', 0):.5f} | {m.get('correlation', 0):.3f} |")
        lines.append("")

    # Dim 3: Fine-tune
    if "finetune_effect" in all_results:
        lines.append("## 3. Fine-tune Effect (BTC daily, h=1)")
        lines.append("")
        lines.append("| Model | Direction Acc | MAE | Quantile Cal Dev |")
        lines.append("|-------|-------------|-----|-----------------|")
        for mt, m in all_results["finetune_effect"].items():
            lines.append(f"| {mt} | {m.get('direction_accuracy', 0):.3f} | "
                         f"{m.get('mae', 0):.5f} | {m.get('avg_calibration_deviation', 0):.3f} |")
        lines.append("")

    # Dim 4: Cross-asset
    if "cross_asset" in all_results:
        lines.append("## 4. Cross-Asset (zero-shot, h=1)")
        lines.append("")
        lines.append("| Asset | Direction Acc | MAE | Quantile Cal Dev | VaR 10% Breach |")
        lines.append("|-------|-------------|-----|-----------------|----------------|")
        for asset, m in sorted(all_results["cross_asset"].items()):
            lines.append(f"| {asset.upper()} | {m.get('direction_accuracy', 0):.3f} | "
                         f"{m.get('mae', 0):.5f} | {m.get('avg_calibration_deviation', 0):.3f} | "
                         f"{m.get('var_10pct_breach_rate', 'N/A')} |")
        lines.append("")

    # Dim 5: Cross-frequency
    if "cross_frequency" in all_results:
        lines.append("## 5. Cross-Frequency (BTC zero-shot, h=1)")
        lines.append("")
        lines.append("| Frequency | Direction Acc | MAE | Quantile Cal Dev |")
        lines.append("|-----------|-------------|-----|-----------------|")
        for freq, m in all_results["cross_frequency"].items():
            lines.append(f"| {freq} | {m.get('direction_accuracy', 0):.3f} | "
                         f"{m.get('mae', 0):.5f} | {m.get('avg_calibration_deviation', 0):.3f} |")
        lines.append("")

    # Bottom line
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("*Is TimesFM useful for crypto?*")
    lines.append("")
    lines.append("(Fill in after reviewing results above)")
    lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=["zero-shot", "traditional", "finetune",
                                           "cross-asset", "cross-frequency", "all"],
                        default="all")
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--freq", default="daily")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TimesFM 2.5 Crypto Evaluation Framework")
    logger.info("=" * 60)

    all_results = {}

    # Load base model
    try:
        model, _ = load_timesfm_model("zero-shot")
    except ImportError:
        logger.error("TimesFM not available. Install on DGX first.")
        return

    # Load BTC daily as primary test asset
    btc_returns = load_returns("btc", "daily")
    btc_test_dates = get_test_indices(btc_returns, "daily")
    logger.info("BTC daily: %d rows, %d test dates", len(btc_returns), len(btc_test_dates))

    dims = [args.dim] if args.dim != "all" else [
        "zero-shot", "traditional", "finetune", "cross-asset", "cross-frequency"
    ]

    for dim in dims:
        if dim == "zero-shot":
            all_results["zero_shot"] = eval_zero_shot(model, btc_returns, btc_test_dates)

        elif dim == "traditional":
            # Need zero-shot results first
            if "zero_shot" not in all_results:
                compile_model(model, max_horizon=max(HORIZONS))
                all_results["zero_shot"] = eval_zero_shot(model, btc_returns, btc_test_dates)
            all_results["vs_traditional"] = eval_vs_traditional(
                btc_returns, btc_test_dates, all_results["zero_shot"])

        elif dim == "finetune":
            all_results["finetune_effect"] = eval_finetune_effect(btc_returns, btc_test_dates)

        elif dim == "cross-asset":
            all_results["cross_asset"] = eval_cross_asset(model)

        elif dim == "cross-frequency":
            all_results["cross_frequency"] = eval_cross_frequency(model, asset=args.asset)

    # Save results
    out_path = RESULTS / "eval_timesfm.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Results saved: %s", out_path)

    # Generate summary
    summary = generate_summary(all_results)
    summary_path = RESULTS / "eval_timesfm_summary.md"
    summary_path.write_text(summary)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
