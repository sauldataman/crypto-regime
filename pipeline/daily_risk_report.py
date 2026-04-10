"""Daily production risk report for 6 crypto assets.

Loads the latest hourly data, runs TimesFM quantile forecasts, applies
conformal + EVT calibration, and outputs a JSON risk report plus a
human-readable ASCII table.

Usage (on DGX):
  python -m pipeline.daily_risk_report
  python -m pipeline.daily_risk_report --freq daily   # fallback to daily data
  python -m pipeline.daily_risk_report --model v4      # force v4 checkpoint

Output:
  reports/daily_risk_YYYY-MM-DD.json
  reports/daily_risk_latest.json
  stdout: ASCII risk table
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup so ``pipeline.*`` imports work regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["btc", "eth", "sol", "bnb", "doge", "link"]

CONTEXT_LEN = 512
HORIZON = 1

# TimesFM fixed decile quantiles
QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Checkpoint search order
CHECKPOINT_ORDER = [
    ("v7_progressive", "timesfm_v7_progressive_best.pt"),
    ("v4",             "timesfm_v4_best.pt"),
    ("progressive",    "timesfm_progressive_best.pt"),
    ("daily",          "timesfm_daily_best.pt"),
]

# Calibration lookback (hours or days depending on freq)
CAL_LOOKBACK_HOURS = 90 * 24  # ~90 days of hourly data
CAL_LOOKBACK_DAYS = 90


# ===================================================================
# 1. Data loading
# ===================================================================

def load_asset_returns(asset: str, freq: str) -> pd.Series | None:
    """Load hourly or daily returns for a single asset.

    Tries hourly first, falls back to daily if freq='hourly' data is
    missing.  Returns None when no data can be found.
    """
    if freq == "hourly":
        path = ROOT / f"data/raw/hourly/{asset}_1h.parquet"
        if path.exists():
            df = pd.read_parquet(path).sort_index()
            df.index = pd.to_datetime(df.index)
            close_col = _find_close_col(df)
            if close_col is None:
                logger.warning("%s hourly: no close column found", asset)
                return None
            returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
            returns.name = f"{asset}_hourly_return"
            logger.info("  %s hourly: %d rows (%s -> %s)",
                        asset, len(returns), returns.index[0], returns.index[-1])
            return returns

        # Hourly not found — try daily as fallback
        logger.warning("%s: hourly data not found at %s, trying daily", asset, path)

    # Daily path
    daily_path = ROOT / f"data/raw/{asset}_price.parquet"
    if daily_path.exists():
        df = pd.read_parquet(daily_path).sort_index()
        df.index = pd.to_datetime(df.index)
        close_col = _find_close_col(df)
        if close_col is None:
            logger.warning("%s daily: no close column found", asset)
            return None
        returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
        returns.name = f"{asset}_daily_return"
        logger.info("  %s daily: %d rows (%s -> %s)",
                    asset, len(returns), returns.index[0], returns.index[-1])
        return returns

    logger.warning("%s: no price data found (tried hourly and daily)", asset)
    return None


def _find_close_col(df: pd.DataFrame) -> str | None:
    """Find the close-price column regardless of naming convention."""
    for col in df.columns:
        if "close" in col.lower():
            return col
    return None


# ===================================================================
# 2. Model loading
# ===================================================================

def load_model(model_choice: str | None):
    """Load TimesFM with optional fine-tuned weights.

    Returns (model, model_name_str).
    """
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")

    base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model_name = "zero-shot"

    if model_choice and model_choice != "zero-shot":
        # Explicit choice: look for a matching checkpoint
        found = False
        for tag, filename in CHECKPOINT_ORDER:
            if model_choice in tag or model_choice in filename:
                ckpt = MODELS_DIR / filename
                if ckpt.exists():
                    base_model.model.load_state_dict(
                        torch.load(ckpt, weights_only=True)
                    )
                    model_name = tag
                    logger.info("Loaded checkpoint: %s (%s)", tag, ckpt)
                    found = True
                    break
        if not found:
            logger.warning(
                "Requested model '%s' not found. Falling back to auto-detection.",
                model_choice,
            )
            model_choice = None  # trigger auto below

    if model_choice is None:
        # Auto: try each checkpoint in priority order
        for tag, filename in CHECKPOINT_ORDER:
            ckpt = MODELS_DIR / filename
            if ckpt.exists():
                base_model.model.load_state_dict(
                    torch.load(ckpt, weights_only=True)
                )
                model_name = tag
                logger.info("Auto-selected checkpoint: %s (%s)", tag, ckpt)
                break
        else:
            logger.info("No fine-tuned checkpoint found. Using zero-shot model.")

    import timesfm as _tfm
    base_model.compile(_tfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    ))

    return base_model, model_name


# ===================================================================
# 3. Forecasting
# ===================================================================

def forecast_single(model, context: list[float]) -> dict | None:
    """Run a single 1-step-ahead forecast.

    Returns dict with 'point' and 'q10'..'q90' keys, or None on failure.
    """
    try:
        point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
        point = float(point_fc[0][0])
        qf = quantile_fc[0][0]
        n_q = len(qf)
        offset = 1 if n_q >= 10 else 0  # skip mean column if present

        result = {"point": point}
        for j, q in enumerate(QUANTILE_LEVELS):
            idx = j + offset
            qk = f"q{int(q * 100):02d}"
            result[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])
        return result
    except (RuntimeError, ValueError, IndexError) as e:
        logger.warning("Forecast failed: %s", e)
        return None


# ===================================================================
# 4. Calibration
# ===================================================================

def load_or_compute_calibration(
    model,
    returns: pd.Series,
    freq: str,
) -> tuple[dict, dict]:
    """Load pre-computed calibration params or compute from recent data.

    Tries to load from results/phase3_risk_signals_{freq}.json first.
    If that fails, computes fresh calibration over the most recent 90 days
    of data.

    Returns (cal_5pct, cal_1pct) dicts.
    """
    suffix = f"_{freq}" if freq != "daily" else ""
    saved_path = RESULTS_DIR / f"phase3_risk_signals{suffix}.json"

    if saved_path.exists():
        try:
            with open(saved_path) as f:
                saved = json.load(f)
            cal = saved.get("calibration", {})
            cal_5 = cal.get("var_5pct", {})
            cal_1 = cal.get("var_1pct", {})
            if cal_5.get("method") and cal_1.get("method"):
                logger.info(
                    "Loaded calibration from %s (VaR5: %s, VaR1: %s)",
                    saved_path, cal_5["method"], cal_1["method"],
                )
                return cal_5, cal_1
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse saved calibration: %s", e)

    # Compute fresh calibration from recent data
    logger.info("Computing fresh calibration from last 90 days...")
    lookback = CAL_LOOKBACK_HOURS if freq == "hourly" else CAL_LOOKBACK_DAYS
    cal_returns = returns.iloc[-lookback:] if len(returns) > lookback else returns

    # Walk-forward forecasts on calibration window (subsample for speed)
    step = 6 if freq == "hourly" else 1
    cal_dates = cal_returns.index[CONTEXT_LEN::step]
    cal_results = []

    for date in cal_dates:
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue
        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = float(returns.iloc[loc])
        fc = forecast_single(model, context)
        if fc is not None:
            fc["actual"] = actual
            fc["date"] = str(date)
            cal_results.append(fc)

    logger.info("  Fresh calibration: %d forecast samples", len(cal_results))

    if len(cal_results) < 50:
        logger.warning(
            "Too few calibration samples (%d). Using raw quantiles without correction.",
            len(cal_results),
        )
        return {"method": "raw", "correction": 0.0}, {"method": "raw", "correction": 0.0}

    cal_5 = _conformal_calibrate(cal_results, 0.05)

    # VaR 1%: try EVT first
    cal_1 = _evt_calibrate_var1(cal_results)

    logger.info(
        "  VaR 5%% calibration: %s (correction=%.6f)",
        cal_5["method"], cal_5.get("correction", 0),
    )
    logger.info(
        "  VaR 1%% calibration: %s (correction=%.6f)",
        cal_1["method"], cal_1.get("correction", 0),
    )
    return cal_5, cal_1


def _conformal_calibrate(results: list[dict], alpha: float) -> dict:
    """Conformal calibration on P10."""
    actuals = np.array([r["actual"] for r in results])
    predicted_q10 = np.array([r["q10"] for r in results])

    scores = predicted_q10 - actuals
    n = len(scores)
    correction = float(np.quantile(scores, (1 - alpha) * (1 + 1 / n)))

    # Sanity checks
    raw_iqr = np.percentile(predicted_q10, 75) - np.percentile(predicted_q10, 25)
    if raw_iqr > 0 and abs(correction) > 3 * raw_iqr:
        return {"method": "isotonic_fallback", "correction": 0.0}

    raw_cov = float(np.mean(actuals <= predicted_q10))
    adj_cov = float(np.mean(actuals <= (predicted_q10 - correction)))
    if adj_cov < alpha - 0.10:
        return {"method": "isotonic_fallback", "correction": 0.0}

    return {
        "method": "conformal",
        "correction": correction,
        "raw_coverage": raw_cov,
        "adj_coverage": adj_cov,
    }


def _evt_calibrate_var1(results: list[dict]) -> dict:
    """EVT calibration for VaR 1%. Falls back to conformal."""
    try:
        from pipeline.evt import evt_calibrate
        cal_params = evt_calibrate(results, target_alpha=0.01)
        if cal_params.get("method") == "evt":
            logger.info("  EVT VaR 1%%: correction=%.4f", cal_params["correction"])
            return cal_params
        logger.warning("  EVT returned method=%s, falling back to conformal", cal_params.get("method"))
    except ImportError:
        logger.warning("  pipeline.evt not available. Falling back to conformal.")
    except Exception as e:
        logger.warning("  EVT error: %s. Falling back to conformal.", e)

    return _conformal_calibrate(results, 0.01)


# ===================================================================
# 5. Risk signal computation (single snapshot)
# ===================================================================

def compute_asset_risk(
    fc: dict,
    cal_5: dict,
    cal_1: dict,
    recent_returns: np.ndarray,
) -> dict:
    """Compute risk metrics for a single asset from one forecast.

    Args:
        fc: Forecast dict with q10..q90 keys.
        cal_5: VaR 5% calibration params.
        cal_1: VaR 1% calibration params.
        recent_returns: Recent actual returns for realized vol computation.

    Returns:
        Dict with var_5pct, var_1pct, median_forecast, uncertainty_ratio,
        position_weight, anomaly_flag, iqr.
    """
    q10 = fc["q10"]
    q50 = fc["q50"]
    q90 = fc["q90"]
    iqr = q90 - q10

    var_5pct = q10 - cal_5.get("correction", 0.0)
    var_1pct = q10 - cal_1.get("correction", 0.0)

    # Uncertainty ratio: IQR / realized_vol_30d
    if len(recent_returns) >= 30:
        realized_vol = float(np.std(recent_returns[-720:]))  # ~30 days hourly
    else:
        realized_vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 1e-6

    if realized_vol > 1e-8:
        uncertainty_ratio = iqr / realized_vol
    else:
        uncertainty_ratio = iqr  # fallback

    # Position weight: inverse of uncertainty (higher uncertainty -> lower weight)
    # Clipped to [0.1, 1.0]
    if uncertainty_ratio > 0:
        position_weight = max(0.1, min(1.0, 1.0 / uncertainty_ratio))
    else:
        position_weight = 0.5

    # Anomaly flag: simple threshold on recent return being extreme
    # Use the latest actual return vs the forecast spread
    anomaly_flag = False
    if len(recent_returns) > 0:
        latest_return = recent_returns[-1]
        if abs(iqr) > 1e-10:
            anomaly_score = abs(latest_return - q50) / iqr
            anomaly_flag = anomaly_score > 1.0
        else:
            anomaly_score = 0.0
    else:
        anomaly_score = 0.0

    return {
        "var_5pct": round(float(var_5pct), 6),
        "var_1pct": round(float(var_1pct), 6),
        "median_forecast": round(float(q50), 6),
        "uncertainty_ratio": round(float(uncertainty_ratio), 2),
        "position_weight": round(float(position_weight), 2),
        "anomaly_flag": bool(anomaly_flag),
        "iqr": round(float(iqr), 6),
    }


# ===================================================================
# 6. Regime detection (optional)
# ===================================================================

def get_regime_status(returns: pd.Series) -> dict | None:
    """Try to detect current regime using PELT. Returns None on failure."""
    try:
        from pipeline.regime_detection import detect_regimes, label_from_breakpoints

        # Use last 2 years of data for regime detection
        lookback = min(len(returns), 365 * 24 * 2)  # 2 years hourly
        recent = returns.iloc[-lookback:]

        breakpoints = detect_regimes(recent)
        if not breakpoints:
            return {"regime_id": 0, "n_breakpoints": 0, "last_transition": None}

        labels = label_from_breakpoints(recent.index, breakpoints)
        current_regime = int(labels.iloc[-1])
        last_bp = str(breakpoints[-1].date()) if breakpoints else None

        return {
            "regime_id": current_regime,
            "n_breakpoints": len(breakpoints),
            "last_transition": last_bp,
        }
    except ImportError:
        logger.warning("ruptures not available; skipping regime detection")
        return None
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return None


# ===================================================================
# 7. Event detection (optional)
# ===================================================================

def get_recent_events(returns: pd.Series, lookback_hours: int = 168) -> dict:
    """Check for recent events in the last ``lookback_hours`` hours."""
    try:
        from pipeline.event_detection import detect_events

        recent = returns.iloc[-lookback_hours:]
        if len(recent) < 10:
            return {"any_event": False, "n_events": 0}

        events = detect_events(recent)
        n_events = int(events["event_flag"].sum())
        any_event = n_events > 0

        # Most recent event details
        latest_event = None
        if any_event:
            flagged = events[events["event_flag"] == 1]
            last = flagged.iloc[-1]
            latest_event = {
                "time": str(flagged.index[-1]),
                "type": last.get("event_type"),
                "zscore": round(float(last.get("zscore_value", 0)), 2),
            }

        return {
            "any_event": any_event,
            "n_events": n_events,
            "lookback_hours": lookback_hours,
            "latest_event": latest_event,
        }
    except ImportError:
        logger.warning("Event detection modules not available; skipping")
        return {"any_event": False, "n_events": 0}
    except Exception as e:
        logger.warning("Event detection failed: %s", e)
        return {"any_event": False, "n_events": 0}


# ===================================================================
# 8. Report generation
# ===================================================================

def generate_report(
    asset_risks: dict[str, dict],
    model_name: str,
    freq: str,
    regime_status: dict | None,
    event_status: dict[str, dict],
) -> dict:
    """Build the final JSON report structure."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Portfolio summary
    valid_assets = {k: v for k, v in asset_risks.items() if v is not None}

    if valid_assets:
        highest_risk = max(valid_assets, key=lambda a: abs(valid_assets[a]["var_5pct"]))
        lowest_risk = min(valid_assets, key=lambda a: abs(valid_assets[a]["var_5pct"]))
        avg_uncertainty = round(
            float(np.mean([v["uncertainty_ratio"] for v in valid_assets.values()])), 2
        )
        any_anomaly = any(v["anomaly_flag"] for v in valid_assets.values())
    else:
        highest_risk = None
        lowest_risk = None
        avg_uncertainty = 0.0
        any_anomaly = False

    any_event = any(
        ev.get("any_event", False) for ev in event_status.values()
    )

    report = {
        "date": today,
        "generated_at": now.isoformat(),
        "model": model_name,
        "frequency": freq,
        "assets": {k: v for k, v in asset_risks.items() if v is not None},
        "portfolio_summary": {
            "highest_risk": highest_risk.upper() if highest_risk else None,
            "lowest_risk": lowest_risk.upper() if lowest_risk else None,
            "avg_uncertainty": avg_uncertainty,
            "any_anomaly": any_anomaly,
            "any_event": any_event,
        },
    }

    if regime_status is not None:
        report["regime_status"] = regime_status

    if event_status:
        report["event_status"] = event_status

    return report


# ===================================================================
# 9. ASCII table output
# ===================================================================

def print_report_table(report: dict) -> None:
    """Print a human-readable ASCII risk table to stdout."""
    date = report["date"]
    model = report["model"]
    freq = report["frequency"]
    assets = report.get("assets", {})
    summary = report.get("portfolio_summary", {})

    header_text = f"  Daily Risk Report -- {date}  (model: {model}, freq: {freq})"
    width = max(60, len(header_text) + 4)

    # Top border
    print()
    print("+" + "=" * (width - 2) + "+")
    print("|" + header_text.ljust(width - 2) + "|")
    print("+" + "=" * (width - 2) + "+")

    # Column headers
    fmt = "| {asset:<6} | {var5:>8} | {var1:>8} | {uncert:>6} | {pos:>6} | {anom:>7} |"
    sep = "+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 9 + "+"

    print(sep)
    print(fmt.format(
        asset="Asset", var5="VaR 5%", var1="VaR 1%",
        uncert="Uncert", pos="Pos Wt", anom="Anomaly",
    ))
    print(sep)

    for asset_name, metrics in assets.items():
        var5_str = f"{metrics['var_5pct'] * 100:.1f}%"
        var1_str = f"{metrics['var_1pct'] * 100:.1f}%"
        uncert_str = f"{metrics['uncertainty_ratio']:.1f}"
        pos_str = f"{metrics['position_weight']:.2f}"
        anom_str = "!YES!" if metrics["anomaly_flag"] else "no"

        print(fmt.format(
            asset=asset_name.upper(),
            var5=var5_str, var1=var1_str,
            uncert=uncert_str, pos=pos_str, anom=anom_str,
        ))

    print(sep)

    # Portfolio summary
    print("|" + " Portfolio Summary".ljust(width - 2) + "|")
    hr = summary.get("highest_risk", "N/A")
    lr = summary.get("lowest_risk", "N/A")
    au = summary.get("avg_uncertainty", 0)
    aa = summary.get("any_anomaly", False)
    ae = summary.get("any_event", False)
    print(f"|   Highest risk: {hr:<8}  Lowest risk: {lr:<8}".ljust(width - 1) + "|")
    print(f"|   Avg uncertainty: {au:<6.1f}   Anomaly: {'YES' if aa else 'no':<5}  Event: {'YES' if ae else 'no':<5}".ljust(width - 1) + "|")

    # Regime status
    regime = report.get("regime_status")
    if regime:
        rid = regime.get("regime_id", "?")
        lt = regime.get("last_transition", "N/A")
        print(f"|   Regime: {rid}  Last transition: {lt}".ljust(width - 1) + "|")

    print("+" + "=" * (width - 2) + "+")
    print()


# ===================================================================
# 10. Save report
# ===================================================================

def save_report(report: dict) -> tuple[Path, Path]:
    """Save report as dated JSON and as latest symlink."""
    date_str = report["date"]
    dated_path = REPORTS_DIR / f"daily_risk_{date_str}.json"
    latest_path = REPORTS_DIR / "daily_risk_latest.json"

    payload = json.dumps(report, indent=2, default=str)

    dated_path.write_text(payload)
    latest_path.write_text(payload)

    logger.info("Report saved: %s", dated_path)
    logger.info("Latest report: %s", latest_path)
    return dated_path, latest_path


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily crypto risk report")
    parser.add_argument(
        "--model", default=None,
        help="Model checkpoint to use (v7, v4, progressive, daily, zero-shot). "
             "Default: auto-detect best available.",
    )
    parser.add_argument(
        "--freq", choices=["hourly", "daily"], default="hourly",
        help="Data frequency (default: hourly)",
    )
    parser.add_argument(
        "--skip-regime", action="store_true",
        help="Skip PELT regime detection (faster)",
    )
    parser.add_argument(
        "--skip-events", action="store_true",
        help="Skip event detection (faster)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Daily Risk Report")
    logger.info("=" * 60)

    t0 = time.time()

    # --- Load model ---
    try:
        model, model_name = load_model(args.model)
    except ImportError:
        logger.error(
            "TimesFM not available. Install with: pip install timesfm"
        )
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("Model: %s", model_name)
    logger.info("Frequency: %s", args.freq)

    # --- Process each asset ---
    asset_risks: dict[str, dict | None] = {}
    event_status: dict[str, dict] = {}
    regime_status: dict | None = None
    btc_returns: pd.Series | None = None

    for asset in ASSETS:
        logger.info("--- %s ---", asset.upper())
        returns = load_asset_returns(asset, args.freq)

        if returns is None or len(returns) < CONTEXT_LEN + 1:
            logger.warning(
                "%s: insufficient data (%s rows). Skipping.",
                asset, len(returns) if returns is not None else 0,
            )
            asset_risks[asset] = None
            continue

        # Remember BTC returns for regime detection
        if asset == "btc":
            btc_returns = returns

        # Load or compute calibration (once per run, using BTC or first available)
        # In production all assets share the same calibration since the model
        # is trained on normalised returns and the quantile structure is similar.
        # Per-asset calibration could be added later as a refinement.
        if asset == "btc" or not any(v is not None for v in asset_risks.values()):
            cal_5, cal_1 = load_or_compute_calibration(model, returns, args.freq)

        # Get latest context window
        context = returns.iloc[-CONTEXT_LEN:].values.tolist()
        recent_returns = returns.iloc[-720:].values  # ~30 days hourly

        # Forecast
        fc = forecast_single(model, context)
        if fc is None:
            logger.warning("%s: forecast failed. Skipping.", asset)
            asset_risks[asset] = None
            continue

        # Compute risk
        risk = compute_asset_risk(fc, cal_5, cal_1, recent_returns)
        asset_risks[asset] = risk
        logger.info(
            "  %s: VaR5=%.4f  VaR1=%.4f  Uncert=%.1f  PosWt=%.2f  Anomaly=%s",
            asset.upper(), risk["var_5pct"], risk["var_1pct"],
            risk["uncertainty_ratio"], risk["position_weight"],
            risk["anomaly_flag"],
        )

        # Event detection per asset
        if not args.skip_events:
            ev = get_recent_events(returns)
            event_status[asset] = ev
            if ev.get("any_event"):
                logger.info("  %s: %d events in last week", asset.upper(), ev["n_events"])

    # --- Regime detection (on BTC) ---
    if not args.skip_regime and btc_returns is not None:
        logger.info("--- Regime Detection (BTC) ---")
        regime_status = get_regime_status(btc_returns)
        if regime_status:
            logger.info(
                "  Regime: %d  Breakpoints: %d  Last transition: %s",
                regime_status["regime_id"],
                regime_status["n_breakpoints"],
                regime_status.get("last_transition", "N/A"),
            )

    # --- Build and save report ---
    report = generate_report(
        asset_risks, model_name, args.freq, regime_status, event_status,
    )
    save_report(report)

    # --- Print ASCII table ---
    print_report_table(report)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
