"""
Portfolio backtest: use TimesFM risk signals for asset allocation.

Strategy:
  - 6 crypto assets: BTC, ETH, SOL, BNB, DOGE, LINK
  - Every rebalance period (default 24h):
    1. Get TimesFM uncertainty (IQR) for each asset
    2. Low uncertainty → high weight, high uncertainty → low weight
    3. Weight = 1/IQR, normalized to sum to 1
    4. If anomaly flag → reduce that asset to 50% of target
    5. If VaR 5% breach yesterday → reduce to 25% of target
  - Compare vs equal-weight buy-and-hold

Metrics: cumulative return, Sharpe ratio, max drawdown, Calmar ratio

Usage:
  python experiments/portfolio_backtest.py
  python experiments/portfolio_backtest.py --freq daily
  python experiments/portfolio_backtest.py --start 2024-01-01 --end 2026-03-31
"""
import argparse
import json
import logging
import sys
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
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CONTEXT_LEN = 512
HORIZON = 1
QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

ASSETS = {
    "btc": {"hourly": "data/raw/hourly/btc_1h.parquet", "daily": "data/raw/btc_price.parquet"},
    "eth": {"hourly": "data/raw/hourly/eth_1h.parquet", "daily": "data/raw/eth_price.parquet"},
    "sol": {"hourly": "data/raw/hourly/sol_1h.parquet", "daily": "data/raw/sol_price.parquet"},
    "bnb": {"hourly": "data/raw/hourly/bnb_1h.parquet", "daily": "data/raw/bnb_price.parquet"},
    "doge": {"hourly": "data/raw/hourly/doge_1h.parquet", "daily": "data/raw/doge_price.parquet"},
    "link": {"hourly": "data/raw/hourly/link_1h.parquet", "daily": "data/raw/link_price.parquet"},
}


def load_returns(asset: str, freq: str) -> pd.Series:
    """Load log returns for an asset."""
    path = ROOT / ASSETS[asset][freq]
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    returns = np.log(close / close.shift(1)).dropna()
    return returns


def load_model(model_type="auto"):
    """Load TimesFM model."""
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    if model_type != "zero-shot":
        import torch
        for ckpt_name in ["timesfm_v7_progressive_best.pt", "timesfm_v4_progressive_best.pt",
                          "timesfm_progressive_best.pt", "timesfm_daily_best.pt"]:
            ckpt = ROOT / "models" / ckpt_name
            if ckpt.exists():
                m.model.load_state_dict(torch.load(ckpt, weights_only=True))
                logger.info("Loaded: %s", ckpt.name)
                break
        else:
            logger.info("No fine-tuned checkpoint found, using zero-shot")

    m.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    ))
    return m


def get_signal(model, returns: pd.Series, loc: int) -> dict:
    """Get TimesFM risk signal for a single time point."""
    if loc < CONTEXT_LEN:
        return None

    context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
    try:
        point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
        qf = quantile_fc[0][0]
        n_q = len(qf)
        offset = 1 if n_q >= 10 else 0

        q10 = float(qf[0 + offset])
        q50 = float(qf[4 + offset]) if 4 + offset < n_q else float(qf[n_q // 2])
        q90 = float(qf[8 + offset]) if 8 + offset < n_q else float(qf[-1])
        iqr = q90 - q10

        return {
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "iqr": max(iqr, 1e-10),
            "point": float(point_fc[0][0]),
        }
    except (RuntimeError, ValueError, IndexError):
        return None


def run_backtest(model, all_returns: dict, start: str, end: str,
                  rebalance_every: int = 24, freq: str = "hourly") -> dict:
    """Run portfolio backtest with TimesFM-guided allocation.

    Args:
        model: TimesFM model
        all_returns: {asset: pd.Series} returns for all assets
        start, end: backtest period
        rebalance_every: rebalance every N steps (24 = daily for hourly freq)
        freq: "hourly" or "daily"
    """
    # Find common date range
    common_start = pd.Timestamp(start)
    common_end = pd.Timestamp(end)

    assets = list(all_returns.keys())
    n_assets = len(assets)

    # Align all returns to common dates
    aligned = {}
    for asset in assets:
        r = all_returns[asset]
        mask = (r.index >= common_start) & (r.index <= common_end)
        aligned[asset] = r[mask]

    # Find common dates (where all assets have data)
    common_dates = aligned[assets[0]].index
    for asset in assets[1:]:
        common_dates = common_dates.intersection(aligned[asset].index)
    common_dates = common_dates.sort_values()

    logger.info("Backtest: %s to %s, %d steps, %d assets, rebal every %d",
                common_dates[0].date(), common_dates[-1].date(),
                len(common_dates), n_assets, rebalance_every)

    # Initialize
    # Strategy: uncertainty-weighted allocation
    strat_nav = [1.0]
    # Benchmark: equal-weight buy-and-hold
    bench_nav = [1.0]

    equal_weight = 1.0 / n_assets
    current_weights = {a: equal_weight for a in assets}

    anomaly_history = {a: deque(maxlen=30) for a in assets}
    rebal_count = 0
    signal_log = []

    step = 6 if freq == "hourly" else 1  # subsample for speed
    eval_dates = common_dates[::step]

    t0 = time.time()

    for i, date in enumerate(eval_dates):
        # Get actual returns for this step
        step_returns = {}
        for asset in assets:
            loc_in_aligned = aligned[asset].index.get_loc(date)
            step_returns[asset] = float(aligned[asset].iloc[loc_in_aligned])

        # Portfolio return (weighted)
        strat_return = sum(current_weights[a] * step_returns[a] for a in assets)
        strat_nav.append(strat_nav[-1] * np.exp(strat_return))

        # Benchmark return (equal weight)
        bench_return = sum(equal_weight * step_returns[a] for a in assets)
        bench_nav.append(bench_nav[-1] * np.exp(bench_return))

        # Rebalance?
        if (i + 1) % rebalance_every == 0:
            rebal_count += 1

            # Get signals for all assets
            signals = {}
            for asset in assets:
                full_returns = all_returns[asset]
                full_loc = full_returns.index.get_loc(date) if date in full_returns.index else None
                if full_loc is not None:
                    sig = get_signal(model, full_returns, full_loc)
                    if sig:
                        signals[asset] = sig

            if len(signals) >= 2:
                # Weight by inverse IQR (low uncertainty = high weight)
                inv_iqrs = {a: 1.0 / signals[a]["iqr"] for a in signals}
                total_inv = sum(inv_iqrs.values())

                new_weights = {}
                for a in assets:
                    if a in inv_iqrs:
                        base_weight = inv_iqrs[a] / total_inv
                    else:
                        base_weight = equal_weight

                    # Anomaly check
                    actual = step_returns.get(a, 0)
                    if a in signals:
                        s = signals[a]
                        anom_score = abs(actual - s["q50"]) / s["iqr"] if s["iqr"] > 1e-10 else 0
                        anomaly_history[a].append(anom_score)

                        # Dynamic anomaly threshold (rolling P90)
                        if len(anomaly_history[a]) >= 10:
                            threshold = np.percentile(list(anomaly_history[a]), 90)
                        else:
                            threshold = 1.0

                        if anom_score > threshold:
                            base_weight *= 0.5  # reduce anomalous asset

                        # VaR breach check
                        if actual < s["q10"]:
                            base_weight *= 0.25  # big reduce after VaR breach

                    new_weights[a] = base_weight

                # Normalize
                total_w = sum(new_weights.values())
                if total_w > 0:
                    current_weights = {a: w / total_w for a, w in new_weights.items()}

                if rebal_count % 100 == 0:
                    weights_str = ", ".join(f"{a}:{w:.2f}" for a, w in sorted(current_weights.items()))
                    logger.info("  Rebal %d: %s", rebal_count, weights_str)

        if (i + 1) % 5000 == 0:
            logger.info("  Step %d/%d (%.1f sec)", i + 1, len(eval_dates), time.time() - t0)

    # Compute metrics
    strat_nav = np.array(strat_nav)
    bench_nav = np.array(bench_nav)

    strat_returns_arr = np.diff(np.log(strat_nav))
    bench_returns_arr = np.diff(np.log(bench_nav))

    def compute_metrics(nav, returns_arr, name):
        total_return = (nav[-1] / nav[0] - 1) * 100
        ann_factor = 252 if freq == "daily" else 252 * 24 / step
        ann_return = total_return / (len(returns_arr) / ann_factor) if len(returns_arr) > 0 else 0
        vol = np.std(returns_arr) * np.sqrt(ann_factor) * 100 if len(returns_arr) > 0 else 0
        sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(ann_factor) if np.std(returns_arr) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        max_dd = float(np.min(dd)) * 100

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return {
            "name": name,
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "annualized_vol_pct": round(vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "n_steps": len(returns_arr),
            "n_rebalances": rebal_count,
        }

    strat_metrics = compute_metrics(strat_nav, strat_returns_arr, "TimesFM Risk-Weighted")
    bench_metrics = compute_metrics(bench_nav, bench_returns_arr, "Equal-Weight Buy&Hold")

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("Period: %s to %s (%s)", start, end, freq)
    logger.info("Assets: %s", ", ".join(a.upper() for a in assets))
    logger.info("")

    header = f"{'Metric':<25} {'Strategy':>15} {'Benchmark':>15}"
    logger.info(header)
    logger.info("-" * 55)

    for key in ["total_return_pct", "annualized_return_pct", "annualized_vol_pct",
                "sharpe_ratio", "max_drawdown_pct", "calmar_ratio"]:
        s_val = strat_metrics[key]
        b_val = bench_metrics[key]
        unit = "%" if "pct" in key else ""
        label = key.replace("_pct", "").replace("_", " ").title()
        logger.info(f"{label:<25} {s_val:>14.2f}{unit} {b_val:>14.2f}{unit}")

    # Who won?
    logger.info("")
    if strat_metrics["sharpe_ratio"] > bench_metrics["sharpe_ratio"]:
        improvement = strat_metrics["sharpe_ratio"] - bench_metrics["sharpe_ratio"]
        logger.info("Strategy WINS: Sharpe %.3f vs %.3f (+%.3f)",
                     strat_metrics["sharpe_ratio"], bench_metrics["sharpe_ratio"], improvement)
    else:
        logger.info("Benchmark wins: Sharpe %.3f vs %.3f",
                     bench_metrics["sharpe_ratio"], strat_metrics["sharpe_ratio"])

    if abs(strat_metrics["max_drawdown_pct"]) < abs(bench_metrics["max_drawdown_pct"]):
        logger.info("Strategy has LOWER drawdown: %.1f%% vs %.1f%%",
                     strat_metrics["max_drawdown_pct"], bench_metrics["max_drawdown_pct"])

    # Save
    output = {
        "backtest": {
            "start": start,
            "end": end,
            "freq": freq,
            "rebalance_every": rebalance_every,
            "assets": assets,
            "n_steps": len(eval_dates),
        },
        "strategy": strat_metrics,
        "benchmark": bench_metrics,
        "nav": {
            "strategy": strat_nav[::max(1, len(strat_nav)//500)].tolist(),
            "benchmark": bench_nav[::max(1, len(bench_nav)//500)].tolist(),
        },
    }

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--model", choices=["auto", "zero-shot"], default="auto")
    parser.add_argument("--rebal", type=int, default=None,
                        help="Rebalance every N steps (default: 24 for hourly, 1 for daily)")
    args = parser.parse_args()

    rebal = args.rebal or (24 if args.freq == "hourly" else 1)

    logger.info("=" * 60)
    logger.info("Portfolio Backtest: TimesFM Risk-Weighted Allocation")
    logger.info("=" * 60)

    # Load model
    try:
        model = load_model(args.model)
    except ImportError:
        logger.error("TimesFM not available.")
        return

    # Load all asset returns
    logger.info("\nLoading returns...")
    all_returns = {}
    for asset in ASSETS:
        r = load_returns(asset, args.freq)
        if len(r) > CONTEXT_LEN:
            all_returns[asset] = r
            logger.info("  %s: %d rows (%s ~ %s)",
                        asset, len(r), r.index[0].date(), r.index[-1].date())
        else:
            logger.warning("  %s: insufficient data (%d rows)", asset, len(r))

    if len(all_returns) < 2:
        logger.error("Need at least 2 assets")
        return

    # Run backtest
    result = run_backtest(model, all_returns, args.start, args.end, rebal, args.freq)

    # Save
    out_path = RESULTS / "portfolio_backtest.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("\nResults: %s", out_path)


if __name__ == "__main__":
    main()
