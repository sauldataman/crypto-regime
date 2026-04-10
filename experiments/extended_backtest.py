"""
Extended Walk-Forward Backtest

Runs a much longer backtest than Phase 3:
  - Calibration:  2023-01-01 ~ 2023-12-31 (1 year)
  - Test:         2024-01-01 ~ 2026-03-31 (2+ years)
  - Re-calibrates every 90 days (walk-forward)
  - Both daily AND hourly frequencies
  - Compares zero-shot vs v7 fine-tuned side by side
  - Monthly VaR breach rate breakdown

Usage (on DGX):
  python experiments/extended_backtest.py
  python experiments/extended_backtest.py --freq hourly
  python experiments/extended_backtest.py --freq daily --models zero-shot
  python experiments/extended_backtest.py --freq both

Output:
  results/extended_backtest.json
"""
import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# sys.path fix for pipeline imports (same as phase3)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.evt import evt_calibrate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = _ROOT
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Temporal config
# ---------------------------------------------------------------------------
INITIAL_CAL_START = "2023-01-01"
INITIAL_CAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-03-31"
RECAL_DAYS = 90  # re-calibrate every 90 days

CONTEXT_LEN = 512
HORIZON = 1

QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(model_choices: list[str]) -> dict:
    """Load requested models. Returns {name: model}."""
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")

    loaded = {}
    for choice in model_choices:
        logger.info("Loading model: %s", choice)
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

        if choice == "v7":
            ckpt = MODELS / "timesfm_v7_progressive_best.pt"
            if not ckpt.exists():
                # Fallback to older naming
                ckpt = MODELS / "timesfm_progressive_best.pt"
            if ckpt.exists():
                m.model.load_state_dict(torch.load(ckpt, weights_only=True))
                logger.info("  Loaded checkpoint: %s", ckpt.name)
            else:
                logger.warning("  No v7 checkpoint found. Skipping fine-tuned model.")
                continue

        m.compile(timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        ))
        loaded[choice] = m
        logger.info("  Model '%s' ready.", choice)

    return loaded


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_btc_returns(freq: str) -> pd.Series:
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
    else:
        raise ValueError(f"Unknown freq: {freq}")

    df = pd.read_parquet(raw).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    returns.name = f"btc_{freq}_return"
    return returns


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------
def forecast_window(model, returns: pd.Series, start: str, end: str,
                    step: int = 1) -> list[dict]:
    """Run walk-forward forecasts over [start, end] with subsampling step."""
    mask = (returns.index >= start) & (returns.index <= end)
    dates = returns.index[mask][::step]
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
            qf = quantile_fc[0][0]
            n_q = len(qf)
            offset = 1 if n_q >= 10 else 0
            quantiles = {}
            for j, q in enumerate(QUANTILE_LEVELS):
                idx = j + offset
                qk = f"q{int(q * 100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
            results.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "actual": actual,
                "point": point,
                **quantiles,
            })
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Forecast failed at %s: %s", date, e)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d forecasts (%.1f sec)", i + 1, len(dates), elapsed)

    logger.info("  %d forecasts in %.1f sec", len(results), time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def conformal_calibrate(results: list[dict], alpha: float) -> dict:
    """Conformal calibration using P10 as base quantile."""
    q_key = "q10"
    actuals = np.array([r["actual"] for r in results])
    predicted_q = np.array([r[q_key] for r in results])

    if len(results) < 50:
        return {"method": "raw", "correction": 0.0}

    scores = predicted_q - actuals
    n = len(scores)
    correction = float(np.quantile(scores, (1 - alpha) * (1 + 1 / n)))

    raw_iqr = np.percentile(predicted_q, 75) - np.percentile(predicted_q, 25)
    if raw_iqr > 0 and abs(correction) > 3 * raw_iqr:
        return {"method": "isotonic_fallback", "correction": 0.0}

    raw_cov = float(np.mean(actuals <= predicted_q))
    adj_cov = float(np.mean(actuals <= (predicted_q - correction)))
    if adj_cov < alpha - 0.10:
        return {"method": "isotonic_fallback", "correction": 0.0}

    return {
        "method": "conformal",
        "correction": correction,
        "raw_coverage": raw_cov,
        "adj_coverage": adj_cov,
    }


def calibrate_var(cal_results: list[dict]) -> tuple[dict, dict]:
    """Calibrate VaR 5% (conformal) and VaR 1% (EVT).

    Returns (cal_5, cal_1).
    """
    cal_5 = conformal_calibrate(cal_results, 0.05)
    try:
        cal_1 = evt_calibrate(cal_results, target_alpha=0.01)
        if cal_1.get("method") != "evt":
            logger.warning("  EVT did not converge, falling back to conformal for VaR 1%%")
            cal_1 = conformal_calibrate(cal_results, 0.01)
    except Exception as e:
        logger.warning("  EVT error: %s. Falling back to conformal for VaR 1%%", e)
        cal_1 = conformal_calibrate(cal_results, 0.01)

    return cal_5, cal_1


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def walk_forward_backtest(
    model,
    returns: pd.Series,
    cal_start: str,
    cal_end: str,
    test_start: str,
    test_end: str,
    recal_days: int,
    step: int = 1,
    model_name: str = "",
) -> dict:
    """Full walk-forward backtest with periodic recalibration.

    1. Initial calibration on [cal_start, cal_end].
    2. Walk through test period [test_start, test_end].
    3. Every `recal_days` calendar days, re-calibrate using the most recent
       year of data (expanding from initial cal window).

    Returns dict with all signals and monthly breakdown.
    """
    logger.info("=" * 60)
    logger.info("Walk-forward backtest: %s", model_name)
    logger.info("  Cal: %s ~ %s", cal_start, cal_end)
    logger.info("  Test: %s ~ %s", test_start, test_end)
    logger.info("  Recal every %d days, step=%d", recal_days, step)
    logger.info("=" * 60)

    # Build the full list of test dates
    test_mask = (returns.index >= test_start) & (returns.index <= test_end)
    test_dates = returns.index[test_mask][::step]
    if len(test_dates) == 0:
        logger.warning("No test dates in range!")
        return {"signals": [], "monthly": {}, "overall": {}}

    # Determine recalibration schedule (by calendar date boundaries)
    recal_boundaries = []
    cursor = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(test_end)
    while cursor <= end_ts:
        recal_boundaries.append(cursor)
        cursor += pd.Timedelta(days=recal_days)

    logger.info("  Recalibration points: %d", len(recal_boundaries))

    all_signals = []
    current_cal_5 = None
    current_cal_1 = None
    next_recal_idx = 0
    forecast_count = 0
    t0 = time.time()

    for date in test_dates:
        # Check if we need to recalibrate
        need_recal = (current_cal_5 is None)
        if not need_recal and next_recal_idx < len(recal_boundaries):
            if date >= recal_boundaries[next_recal_idx]:
                need_recal = True

        if need_recal:
            # Calibration window: 1 year ending at the day before current date
            cal_end_dt = date - pd.Timedelta(days=1)
            cal_start_dt = cal_end_dt - pd.Timedelta(days=365)
            cal_s = str(cal_start_dt.date()) if hasattr(cal_start_dt, "date") else str(cal_start_dt)
            cal_e = str(cal_end_dt.date()) if hasattr(cal_end_dt, "date") else str(cal_end_dt)

            logger.info("  [Recal] %s: cal window %s ~ %s", date.date(), cal_s, cal_e)
            cal_results = forecast_window(model, returns, cal_s, cal_e, step=step)
            if len(cal_results) >= 50:
                current_cal_5, current_cal_1 = calibrate_var(cal_results)
                logger.info("    VaR5 correction=%.6f (%s), VaR1 correction=%.6f (%s)",
                            current_cal_5.get("correction", 0), current_cal_5["method"],
                            current_cal_1.get("correction", 0), current_cal_1["method"])
            else:
                logger.warning("    Only %d cal forecasts, keeping previous calibration", len(cal_results))
                if current_cal_5 is None:
                    # First calibration must succeed
                    current_cal_5 = {"method": "raw", "correction": 0.0}
                    current_cal_1 = {"method": "raw", "correction": 0.0}

            # Advance to next recal point
            while next_recal_idx < len(recal_boundaries) and date >= recal_boundaries[next_recal_idx]:
                next_recal_idx += 1

        # Forecast
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
                qk = f"q{int(q * 100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Forecast failed at %s: %s", date, e)
            continue

        q10 = quantiles["q10"]
        var_5pct = q10 - current_cal_5.get("correction", 0)
        var_1pct = q10 - current_cal_1.get("correction", 0)

        all_signals.append({
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "actual": actual,
            "point": point,
            "q10": q10,
            "q50": quantiles["q50"],
            "q90": quantiles["q90"],
            "var_5pct": float(var_5pct),
            "var_1pct": float(var_1pct),
            "var5_breach": actual < var_5pct,
            "var1_breach": actual < var_1pct,
        })

        forecast_count += 1
        if forecast_count % 500 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d test forecasts (%.1f sec)", forecast_count, len(test_dates), elapsed)

    total_time = time.time() - t0
    logger.info("  Completed: %d signals in %.1f sec", len(all_signals), total_time)

    # Compute monthly breakdown
    monthly = compute_monthly_breakdown(all_signals)

    # Overall stats
    n = len(all_signals)
    var5_breaches = sum(1 for s in all_signals if s["var5_breach"])
    var1_breaches = sum(1 for s in all_signals if s["var1_breach"])
    overall = {
        "n_samples": n,
        "var5_breach_rate": var5_breaches / n if n > 0 else 0,
        "var5_breaches": var5_breaches,
        "var1_breach_rate": var1_breaches / n if n > 0 else 0,
        "var1_breaches": var1_breaches,
        "total_time_sec": total_time,
    }

    return {
        "signals": all_signals,
        "monthly": monthly,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------
def compute_monthly_breakdown(signals: list[dict]) -> list[dict]:
    """Group signals by year-month and compute breach rates."""
    by_month: dict[str, list[dict]] = defaultdict(list)
    for s in signals:
        # date is string like "2024-01-15"
        ym = s["date"][:7]  # "2024-01"
        by_month[ym].append(s)

    rows = []
    for ym in sorted(by_month.keys()):
        group = by_month[ym]
        n = len(group)
        v5 = sum(1 for s in group if s["var5_breach"])
        v1 = sum(1 for s in group if s["var1_breach"])
        rows.append({
            "month": ym,
            "n_samples": n,
            "var5_breaches": v5,
            "var5_rate": v5 / n if n > 0 else 0,
            "var1_breaches": v1,
            "var1_rate": v1 / n if n > 0 else 0,
        })

    return rows


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(results: dict[str, dict], freq: str):
    """Print a nicely formatted summary table."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTENDED BACKTEST SUMMARY (%s)", freq.upper())
    logger.info("=" * 80)

    # Overall comparison
    header = f"{'Model':<15} {'N':>7} {'VaR5% rate':>11} {'VaR5 target':>12} {'VaR1% rate':>11} {'VaR1 target':>12}"
    logger.info(header)
    logger.info("-" * 80)
    for model_name, res in results.items():
        ov = res["overall"]
        v5_str = f"{ov['var5_breach_rate']*100:.1f}%"
        v1_str = f"{ov['var1_breach_rate']*100:.1f}%"
        logger.info(
            f"{model_name:<15} {ov['n_samples']:>7} {v5_str:>11} {'3-8%':>12} {v1_str:>11} {'0.5-2%':>12}"
        )

    # Monthly breakdown (side by side)
    model_names = list(results.keys())
    if len(model_names) == 0:
        return

    # Collect all months across all models
    all_months = set()
    for res in results.values():
        for row in res["monthly"]:
            all_months.add(row["month"])

    if not all_months:
        return

    logger.info("")
    logger.info("Monthly VaR 5%% Breach Rates:")
    logger.info("-" * 80)

    # Build header
    parts = [f"{'Month':<10}"]
    for mn in model_names:
        parts.append(f"{mn:>15}")
    logger.info("  ".join(parts))

    # Build month-to-row lookup for each model
    lookups = {}
    for mn in model_names:
        lookup = {}
        for row in results[mn]["monthly"]:
            lookup[row["month"]] = row
        lookups[mn] = lookup

    for month in sorted(all_months):
        parts = [f"{month:<10}"]
        for mn in model_names:
            row = lookups[mn].get(month)
            if row:
                rate = row["var5_rate"] * 100
                n = row["n_samples"]
                parts.append(f"{rate:5.1f}% ({row['var5_breaches']:>3}/{n:<4})")
            else:
                parts.append(f"{'---':>15}")
        logger.info("  ".join(parts))

    # Monthly VaR 1%
    logger.info("")
    logger.info("Monthly VaR 1%% Breach Rates:")
    logger.info("-" * 80)
    parts = [f"{'Month':<10}"]
    for mn in model_names:
        parts.append(f"{mn:>15}")
    logger.info("  ".join(parts))

    for month in sorted(all_months):
        parts = [f"{month:<10}"]
        for mn in model_names:
            row = lookups[mn].get(month)
            if row:
                rate = row["var1_rate"] * 100
                n = row["n_samples"]
                parts.append(f"{rate:5.1f}% ({row['var1_breaches']:>3}/{n:<4})")
            else:
                parts.append(f"{'---':>15}")
        logger.info("  ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extended walk-forward backtest")
    parser.add_argument(
        "--freq", choices=["daily", "hourly", "both"], default="both",
        help="Data frequency (default: both)"
    )
    parser.add_argument(
        "--models", nargs="+", default=["zero-shot", "v7"],
        help="Models to compare (default: zero-shot v7)"
    )
    parser.add_argument(
        "--recal-days", type=int, default=RECAL_DAYS,
        help=f"Recalibration interval in days (default: {RECAL_DAYS})"
    )
    args = parser.parse_args()

    freqs = ["daily", "hourly"] if args.freq == "both" else [args.freq]

    logger.info("=" * 80)
    logger.info("Extended Walk-Forward Backtest")
    logger.info("  Frequencies: %s", freqs)
    logger.info("  Models: %s", args.models)
    logger.info("  Cal: %s ~ %s", INITIAL_CAL_START, INITIAL_CAL_END)
    logger.info("  Test: %s ~ %s", TEST_START, TEST_END)
    logger.info("  Recal every %d days", args.recal_days)
    logger.info("=" * 80)

    # Load models
    try:
        models = load_models(args.models)
    except ImportError:
        logger.error("TimesFM not available. Install on DGX first.")
        return

    if not models:
        logger.error("No models loaded. Check checkpoint paths.")
        return

    all_results = {}

    for freq in freqs:
        logger.info("\n" + "#" * 80)
        logger.info("# FREQUENCY: %s", freq.upper())
        logger.info("#" * 80)

        returns = load_btc_returns(freq)
        logger.info("BTC %s returns: %d rows (%s -> %s)", freq, len(returns),
                     returns.index[0].date(), returns.index[-1].date())

        step = 1 if freq == "daily" else 6

        freq_results = {}
        for model_name, model in models.items():
            res = walk_forward_backtest(
                model=model,
                returns=returns,
                cal_start=INITIAL_CAL_START,
                cal_end=INITIAL_CAL_END,
                test_start=TEST_START,
                test_end=TEST_END,
                recal_days=args.recal_days,
                step=step,
                model_name=f"{model_name} ({freq})",
            )
            freq_results[model_name] = res

        print_summary(freq_results, freq)
        all_results[freq] = freq_results

    # Save results (strip full signal lists for JSON size, keep monthly + overall)
    output = {}
    for freq, freq_results in all_results.items():
        output[freq] = {}
        for model_name, res in freq_results.items():
            output[freq][model_name] = {
                "overall": res["overall"],
                "monthly": res["monthly"],
                "n_signals": len(res["signals"]),
                # Keep first and last 5 signals as sample
                "signals_head": res["signals"][:5],
                "signals_tail": res["signals"][-5:] if len(res["signals"]) > 5 else [],
            }

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "extended_backtest.json"

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_path.write_text(json.dumps(output, indent=2, default=_json_default))
    logger.info("\nResults saved: %s", out_path)

    # Also save full signals for detailed analysis
    for freq, freq_results in all_results.items():
        for model_name, res in freq_results.items():
            slug = f"{freq}_{model_name}".replace("-", "_")
            sig_path = RESULTS / f"extended_backtest_signals_{slug}.json"
            sig_path.write_text(json.dumps(res["signals"], indent=2, default=_json_default))
            logger.info("Full signals: %s (%d entries)", sig_path, len(res["signals"]))


if __name__ == "__main__":
    main()
