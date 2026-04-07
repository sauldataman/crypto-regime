"""Regime detection using ruptures PELT algorithm."""
import json
import logging
import numpy as np
import pandas as pd
import ruptures

logger = logging.getLogger(__name__)

def detect_regimes(returns: pd.Series, pen: float = None, model: str = "rbf") -> list[pd.Timestamp]:
    """Detect regime breakpoints using PELT on return series.

    Args:
        returns: Daily return series with DatetimeIndex
        pen: Penalty parameter. If None, uses BIC-based auto-selection.
        model: Cost model for PELT ("rbf", "l2", "normal")

    Returns:
        List of breakpoint timestamps
    """
    signal = returns.dropna().values.reshape(-1, 1)
    dates = returns.dropna().index

    if len(signal) < 100:
        logger.warning(f"Too few data points ({len(signal)}) for PELT. Returning no breakpoints.")
        return []

    algo = ruptures.Pelt(model=model, min_size=60).fit(signal)

    if pen is None:
        # BIC-based penalty: log(n) * dim * variance
        n = len(signal)
        pen = np.log(n) * np.var(signal) * 2

    breakpoints = algo.predict(pen=pen)
    # Remove the last breakpoint (always == len(signal))
    breakpoints = [bp for bp in breakpoints if bp < len(signal)]

    if len(breakpoints) == 0:
        logger.warning("PELT found 0 breakpoints. Consider lowering penalty.")
    elif len(breakpoints) > 20:
        logger.warning(f"PELT found {len(breakpoints)} breakpoints. Consider raising penalty.")

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

    Args:
        breakpoints_dict: {asset_name: [breakpoint_dates]}

    Returns:
        DataFrame with pairwise breakpoint lag analysis
    """
    records = []
    assets = list(breakpoints_dict.keys())
    for i, a1 in enumerate(assets):
        for a2 in assets[i+1:]:
            bps1 = breakpoints_dict[a1]
            bps2 = breakpoints_dict[a2]
            # For each bp in a1, find nearest bp in a2
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
        logger.info(f"Running PELT on {asset} ({len(returns)} points)...")
        bps = detect_regimes(returns, pen=pen)
        breakpoints[asset] = bps
        logger.info(f"  {asset}: {len(bps)} breakpoints at {[str(d.date()) for d in bps]}")

    sync_df = cross_asset_sync(breakpoints)

    # Serialize for JSON
    result = {
        "breakpoints": {k: [str(d.date()) for d in v] for k, v in breakpoints.items()},
        "sync_report": sync_df.to_dict(orient="records"),
    }
    return result

if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2016-01-01", periods=3000, freq="D")
    returns = pd.Series(np.random.randn(3000) * 0.02, index=dates)
    # Insert a regime shift
    returns.iloc[1500:] += 0.01
    bps = detect_regimes(returns)
    print(f"Breakpoints: {bps}")
