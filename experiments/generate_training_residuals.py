"""
Generate TimesFM residuals over the FULL training period for macro correction.

Current problem: macro correction XGBoost only has 293 training samples (cal period).
Solution: Run TimesFM walk-forward on 2014-2022, generate ~3000 residual samples.

Each sample: (TimesFM_quantiles, actual_return, macro_features, regime) on a given day.
This gives the XGBoost much more data to learn the relationship between
macro conditions and TimesFM forecast errors.

Usage:
  python experiments/generate_training_residuals.py

Output:
  results/training_residuals.json (~3000 samples)
  Then re-run pipeline/macro_correction.py with this extended dataset.
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

CONTEXT_LEN = 512
HORIZON = 1

# Full training period (much larger than just cal period)
TRAIN_START = "2016-01-01"  # need 512 days before this for context
TRAIN_END = "2022-06-01"
# Cal period (for validation of the correction model)
CAL_START = "2023-07-01"
CAL_END = "2024-06-30"

QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load_btc_returns() -> pd.Series:
    """Load BTC daily returns."""
    for path in [ROOT / "data/processed/btc_full.parquet", ROOT / "data/raw/btc_price.parquet"]:
        if path.exists():
            df = pd.read_parquet(path).sort_index()
            df.index = pd.to_datetime(df.index)
            if "btc_daily_return" in df.columns:
                return df["btc_daily_return"].dropna()
            close_col = [c for c in df.columns if "close" in c.lower()]
            if close_col:
                returns = np.log(df[close_col[0]] / df[close_col[0]].shift(1)).dropna()
                returns.name = "btc_daily_return"
                return returns
    raise FileNotFoundError("No BTC data found")


def walk_forward_residuals(returns: pd.Series, start: str, end: str,
                            step: int = 1) -> list[dict]:
    """Run TimesFM walk-forward and collect residuals + quantile predictions.

    For each day in [start, end], record:
      - actual return
      - TimesFM point forecast
      - TimesFM quantile forecasts (P10-P90)
      - residual (actual - point)
      - date
    """
    import timesfm

    logger.info("Loading TimesFM...")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    ))

    mask = (returns.index >= start) & (returns.index <= end)
    dates = returns.index[mask]
    logger.info("Walk-forward: %d days (%s to %s), step=%d",
                len(dates), start, end, step)

    results = []
    t0 = time.time()

    for i, date in enumerate(dates[::step]):
        loc = returns.index.get_loc(date)
        if loc < CONTEXT_LEN:
            continue

        context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
        actual = float(returns.iloc[loc])

        try:
            point_fc, quantile_fc = m.forecast(horizon=HORIZON, inputs=[context])
            point = float(point_fc[0][0])
            qf = quantile_fc[0][0]
            n_q = len(qf)
            offset = 1 if n_q >= 10 else 0

            quantiles = {}
            for j, q in enumerate(QUANTILE_LEVELS):
                idx = j + offset
                qk = f"q{int(q*100):02d}"
                quantiles[qk] = float(qf[idx]) if idx < n_q else float(qf[-1])

            results.append({
                "date": str(date.date()),
                "actual": actual,
                "point_forecast": point,
                "residual": actual - point,
                **quantiles,
            })

        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning("Failed at %s: %s", date.date(), e)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d (%.1f sec, %.2f sec/step)",
                        i + 1, len(dates[::step]), elapsed, elapsed / (i + 1))

    elapsed = time.time() - t0
    logger.info("Done: %d residuals in %.1f sec", len(results), elapsed)
    return results


def main():
    logger.info("=" * 60)
    logger.info("Generating training residuals for macro correction")
    logger.info("=" * 60)

    returns = load_btc_returns()
    logger.info("BTC returns: %d rows (%s to %s)",
                len(returns), returns.index[0].date(), returns.index[-1].date())

    # Generate residuals for FULL training period
    logger.info("\n=== Training period residuals ===")
    train_residuals = walk_forward_residuals(
        returns, TRAIN_START, TRAIN_END, step=1
    )

    # Also generate for cal period (for validation)
    logger.info("\n=== Calibration period residuals ===")
    cal_residuals = walk_forward_residuals(
        returns, CAL_START, CAL_END, step=1
    )

    # Save
    output = {
        "train": train_residuals,
        "cal": cal_residuals,
        "metadata": {
            "train_period": f"{TRAIN_START} to {TRAIN_END}",
            "cal_period": f"{CAL_START} to {CAL_END}",
            "n_train": len(train_residuals),
            "n_cal": len(cal_residuals),
            "context_len": CONTEXT_LEN,
            "horizon": HORIZON,
        },
    }

    out_path = RESULTS / "training_residuals.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("\nSaved: %s", out_path)
    logger.info("Train samples: %d, Cal samples: %d",
                len(train_residuals), len(cal_residuals))

    # Quick stats
    if train_residuals:
        residuals = [r["residual"] for r in train_residuals]
        logger.info("Residual stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                     np.mean(residuals), np.std(residuals),
                     np.min(residuals), np.max(residuals))


if __name__ == "__main__":
    main()
