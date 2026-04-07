"""Regime detection using ruptures PELT algorithm.

Key design choice: auto-penalty via elbow method instead of BIC.
BIC-based penalty fails on crypto returns because variance is tiny (~0.0004),
making the penalty so high that PELT finds 0 breakpoints.
"""
import json
import logging
import numpy as np
import pandas as pd
import ruptures

logger = logging.getLogger(__name__)


def _find_elbow_penalty(signal: np.ndarray, model: str = "l2",
                         min_size: int = 60, n_candidates: int = 20) -> float:
    """Find optimal PELT penalty via elbow method.

    Scan penalties from high (1 breakpoint) to low (many breakpoints).
    Find the "elbow" where adding more breakpoints gives diminishing cost reduction.

    This is more robust than BIC for financial time series where variance is tiny.
    """
    algo = ruptures.Pelt(model=model, min_size=min_size).fit(signal)

    # Generate candidate penalties on log scale
    # Start from a very high penalty (likely 0-1 breakpoints) down to low (many)
    signal_var = np.var(signal)
    n = len(signal)

    # Heuristic range: from 10x BIC down to 0.01x BIC
    bic_pen = np.log(n) * signal_var
    pen_high = max(bic_pen * 10, 0.1)   # ensure minimum
    pen_low = max(bic_pen * 0.01, 1e-6)

    penalties = np.logspace(np.log10(pen_low), np.log10(pen_high), n_candidates)[::-1]

    results = []
    for pen in penalties:
        bps = algo.predict(pen=pen)
        n_bps = len([b for b in bps if b < n])  # exclude terminal breakpoint
        cost = algo.cost.sum_of_costs(bps)
        results.append({"pen": pen, "n_bps": n_bps, "cost": cost})

    if not results:
        return bic_pen

    # Find elbow: biggest drop in cost per additional breakpoint
    best_pen = penalties[0]
    best_score = 0

    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]

        if curr["n_bps"] <= prev["n_bps"]:
            continue

        delta_bps = curr["n_bps"] - prev["n_bps"]
        delta_cost = prev["cost"] - curr["cost"]

        if delta_bps > 0 and delta_cost > 0:
            # Score: cost reduction per breakpoint, penalized by total breakpoints
            # (prefer fewer breakpoints with big cost reduction)
            score = (delta_cost / delta_bps) / max(curr["n_bps"], 1)
            if score > best_score:
                best_score = score
                best_pen = curr["pen"]

    logger.info("  Elbow penalty: %.6f (BIC would be %.6f)", best_pen, bic_pen)
    return best_pen


def detect_regimes(returns: pd.Series, pen: float = None, model: str = "l2",
                    min_size: int = 60, target_n_regimes: tuple = (2, 6)) -> list[pd.Timestamp]:
    """Detect regime breakpoints using PELT on return series.

    Args:
        returns: Daily return series with DatetimeIndex
        pen: Penalty parameter. If None, uses elbow method.
        model: Cost model for PELT ("rbf", "l2", "normal")
        min_size: Minimum segment length in days
        target_n_regimes: (min, max) expected number of regimes. Used for validation.

    Returns:
        List of breakpoint timestamps
    """
    signal = returns.dropna().values.reshape(-1, 1)
    dates = returns.dropna().index

    if len(signal) < 100:
        logger.warning("Too few data points (%d) for PELT.", len(signal))
        return []

    algo = ruptures.Pelt(model=model, min_size=min_size).fit(signal)

    if pen is None:
        pen = _find_elbow_penalty(signal, model=model, min_size=min_size)

    breakpoints = algo.predict(pen=pen)
    # Remove the terminal breakpoint (always == len(signal))
    breakpoints = [bp for bp in breakpoints if bp < len(signal)]
    n_regimes = len(breakpoints) + 1

    # Validate against expected range
    min_r, max_r = target_n_regimes
    if n_regimes < min_r:
        logger.warning("PELT found %d regimes (< %d). Trying lower penalty...", n_regimes, min_r)
        pen_low, pen_high = pen * 0.001, pen
        best_in_range = None
        for _ in range(15):
            pen_mid = (pen_low + pen_high) / 2
            bps_mid = algo.predict(pen=pen_mid)
            n_mid = len([b for b in bps_mid if b < len(signal)])
            if min_r <= n_mid + 1 <= max_r:
                breakpoints = [b for b in bps_mid if b < len(signal)]
                pen = pen_mid
                best_in_range = (breakpoints, pen)
                break
            elif n_mid + 1 < min_r:
                pen_high = pen_mid
            else:  # too many
                pen_low = pen_mid
                if n_mid + 1 <= max_r + 1:
                    best_in_range = ([b for b in bps_mid if b < len(signal)], pen_mid)
        else:
            if best_in_range:
                breakpoints, pen = best_in_range
            else:
                bps_final = algo.predict(pen=pen_low)
                breakpoints = [b for b in bps_final if b < len(signal)]
                pen = pen_low

    elif n_regimes > max_r:
        logger.warning("PELT found %d regimes (> %d). Trying higher penalty...", n_regimes, max_r)
        pen_low, pen_high = pen, pen * 100
        best_in_range = None
        for _ in range(15):
            pen_mid = (pen_low + pen_high) / 2
            bps_mid = algo.predict(pen=pen_mid)
            n_mid = len([b for b in bps_mid if b < len(signal)])
            if min_r <= n_mid + 1 <= max_r:
                breakpoints = [b for b in bps_mid if b < len(signal)]
                pen = pen_mid
                best_in_range = (breakpoints, pen)
                break
            elif n_mid + 1 > max_r:
                pen_low = pen_mid
            else:  # too few
                pen_high = pen_mid
                # Save as "closest" if within 1 of range
                if n_mid + 1 >= min_r - 1:
                    best_in_range = ([b for b in bps_mid if b < len(signal)], pen_mid)
        else:
            if best_in_range:
                breakpoints, pen = best_in_range
            else:
                # Use max_r breakpoints as upper bound
                bps_final = algo.predict(pen=pen_low)
                breakpoints = [b for b in bps_final if b < len(signal)]
                pen = pen_low

    n_final = len(breakpoints)
    logger.info("  PELT: %d breakpoints, %d regimes (pen=%.6f)", n_final, n_final + 1, pen)

    bp_dates = [dates[bp] for bp in breakpoints if bp < len(dates)]
    return bp_dates


def label_from_breakpoints(index: pd.DatetimeIndex, breakpoints: list[pd.Timestamp]) -> pd.Series:
    """Assign regime labels (0, 1, 2, ...) based on breakpoint dates."""
    labels = pd.Series(0, index=index, dtype=int)
    for i, bp in enumerate(sorted(breakpoints)):
        labels[index >= bp] = i + 1
    return labels


def cross_asset_sync(breakpoints_dict: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    """Compare PELT breakpoints across assets.

    Returns:
        DataFrame with pairwise breakpoint lag analysis
    """
    records = []
    assets = list(breakpoints_dict.keys())
    for i, a1 in enumerate(assets):
        for a2 in assets[i+1:]:
            bps1 = breakpoints_dict[a1]
            bps2 = breakpoints_dict[a2]
            if bps1 and bps2:
                lags = []
                for bp1 in bps1:
                    nearest = min(bps2, key=lambda x: abs((x - bp1).days))
                    lags.append(abs((nearest - bp1).days))
                avg_lag = np.mean(lags)
            else:
                avg_lag = float('nan')
            records.append({
                "asset_1": a1,
                "asset_2": a2,
                "n_breakpoints_1": len(bps1),
                "n_breakpoints_2": len(bps2),
                "avg_lag_days": round(avg_lag, 1) if not np.isnan(avg_lag) else None,
            })
    return pd.DataFrame(records)


def run_all_assets(asset_returns: dict[str, pd.Series], pen: float = None) -> dict:
    """Run PELT on multiple assets and produce sync report."""
    breakpoints = {}
    for asset, returns in asset_returns.items():
        logger.info("Running PELT on %s (%d points)...", asset, len(returns))
        bps = detect_regimes(returns, pen=pen)
        breakpoints[asset] = bps
        bp_str = [str(d.date()) for d in bps]
        logger.info("  %s: %d breakpoints at %s", asset, len(bps), bp_str)

    sync_df = cross_asset_sync(breakpoints)

    result = {
        "breakpoints": {k: [str(d.date()) for d in v] for k, v in breakpoints.items()},
        "sync_report": sync_df.to_dict(orient="records"),
    }
    return result


if __name__ == "__main__":
    # Test on real BTC data
    from pathlib import Path
    ROOT = Path(__file__).parent.parent

    btc_path = ROOT / "data/processed/btc_full.parquet"
    if btc_path.exists():
        df = pd.read_parquet(btc_path).sort_index()
        returns = df["btc_daily_return"].dropna()
        print(f"BTC daily returns: {len(returns)} rows ({returns.index[0].date()} -> {returns.index[-1].date()})")
        print(f"Return variance: {np.var(returns.values):.6f}")

        bps = detect_regimes(returns)
        print(f"\nBreakpoints ({len(bps)}):")
        for bp in bps:
            print(f"  {bp.date()}")
    else:
        print("No BTC data found. Run run_pipeline.py first.")
        # Synthetic test
        np.random.seed(42)
        dates = pd.date_range("2016-01-01", periods=3000, freq="D")
        returns = pd.Series(np.random.randn(3000) * 0.02, index=dates)
        returns.iloc[1000:2000] += 0.005  # regime shift
        returns.iloc[2000:] -= 0.003      # another shift
        bps = detect_regimes(returns)
        print(f"Synthetic breakpoints: {[str(d.date()) for d in bps]}")
