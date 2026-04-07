"""Fast-layer event detection for short-term anomalies.

Complements the slow-layer PELT regime detection (daily, macro shifts) with
fast event detection on hourly/5min data: flash crashes, sudden pumps,
liquidity dry-ups.

Two methods:
  1. Rolling z-score: flags single-point price spikes (|z| > threshold)
  2. CUSUM: flags sustained trend shifts (cumulative bias exceeds threshold)

Integration:
  TimesFM quantile (base)
    -> PELT regime correction (slow layer)
    -> Event detection override (fast layer: force high uncertainty during events)
    -> Final risk signals
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Rolling Z-score
# ---------------------------------------------------------------------------


def detect_zscore_events(
    returns: pd.Series,
    window: int = 168,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Detect single-point price anomalies using rolling z-score.

    Args:
        returns: Return series (hourly or 5min) with DatetimeIndex.
        window: Rolling window size (168 = 1 week of hourly data).
        threshold: Z-score threshold for flagging events.

    Returns:
        DataFrame with columns:
            zscore_flag  - 0/1
            zscore_value - the z-score at each point
            zscore_dir   - +1 (pump) / -1 (crash) / 0 (no event)
    """
    if returns.empty:
        logger.warning("detect_zscore_events: empty input series")
        return pd.DataFrame(
            {"zscore_flag": pd.Series(dtype=int),
             "zscore_value": pd.Series(dtype=float),
             "zscore_dir": pd.Series(dtype=int)},
        )

    if len(returns) < window:
        logger.warning(
            "detect_zscore_events: series length %d < window %d; "
            "using expanding window instead",
            len(returns), window,
        )
        roll_mean = returns.expanding(min_periods=2).mean()
        roll_std = returns.expanding(min_periods=2).std()
    else:
        roll_mean = returns.rolling(window=window, min_periods=max(window // 2, 2)).mean()
        roll_std = returns.rolling(window=window, min_periods=max(window // 2, 2)).std()

    # Guard against zero / near-zero variance
    roll_std = roll_std.replace(0, np.nan)

    zscore = (returns - roll_mean) / roll_std

    flag = (zscore.abs() > threshold).astype(int)
    direction = np.sign(zscore).where(flag == 1, 0).astype(int)

    result = pd.DataFrame({
        "zscore_flag": flag,
        "zscore_value": zscore,
        "zscore_dir": direction,
    }, index=returns.index)

    n_events = flag.sum()
    logger.info(
        "Z-score events: %d flagged out of %d points (threshold=%.1f, window=%d)",
        n_events, len(returns), threshold, window,
    )
    return result


# ---------------------------------------------------------------------------
# 2. CUSUM
# ---------------------------------------------------------------------------


def detect_cusum_events(
    returns: pd.Series,
    drift: float = 0.5,
    threshold: float = 5.0,
    rolling_std_window: int = 168,
) -> pd.DataFrame:
    """Detect sustained trend shifts using CUSUM (Cumulative Sum).

    Standardises returns with a rolling std, then tracks the cumulative sum
    of (standardised return - drift) upward and -(standardised return + drift)
    downward.  An alarm fires when either CUSUM exceeds *threshold*, after
    which the statistic resets to zero.

    Args:
        returns: Return series with DatetimeIndex.
        drift: Target drift to detect (in std units).
        threshold: CUSUM alarm threshold (in std units).
        rolling_std_window: Window for the rolling std used to standardise.

    Returns:
        DataFrame with columns:
            cusum_flag   - 0/1
            cusum_value  - max(|S_pos|, |S_neg|) at each point
            cusum_dir    - +1 (upward alarm) / -1 (downward alarm) / 0
    """
    if returns.empty:
        logger.warning("detect_cusum_events: empty input series")
        return pd.DataFrame(
            {"cusum_flag": pd.Series(dtype=int),
             "cusum_value": pd.Series(dtype=float),
             "cusum_dir": pd.Series(dtype=int)},
        )

    # Rolling std for standardisation
    if len(returns) < rolling_std_window:
        roll_std = returns.expanding(min_periods=2).std()
    else:
        roll_std = returns.rolling(
            window=rolling_std_window,
            min_periods=max(rolling_std_window // 2, 2),
        ).std()

    roll_std = roll_std.replace(0, np.nan)
    standardised = (returns / roll_std).fillna(0.0)

    # Run CUSUM (imperative loop -- necessary for the reset logic)
    n = len(standardised)
    vals = standardised.values

    s_pos = np.zeros(n, dtype=np.float64)
    s_neg = np.zeros(n, dtype=np.float64)
    flags = np.zeros(n, dtype=np.int64)
    dirs = np.zeros(n, dtype=np.int64)
    cusum_out = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        s_pos[i] = max(0.0, s_pos[i - 1] + vals[i] - drift)
        s_neg[i] = max(0.0, s_neg[i - 1] - vals[i] - drift)

        cusum_out[i] = max(s_pos[i], s_neg[i])

        if s_pos[i] > threshold:
            flags[i] = 1
            dirs[i] = 1
            s_pos[i] = 0.0  # reset after alarm
        elif s_neg[i] > threshold:
            flags[i] = 1
            dirs[i] = -1
            s_neg[i] = 0.0  # reset after alarm

    result = pd.DataFrame({
        "cusum_flag": flags,
        "cusum_value": cusum_out,
        "cusum_dir": dirs,
    }, index=returns.index)

    n_events = flags.sum()
    logger.info(
        "CUSUM events: %d alarms out of %d points (drift=%.2f, threshold=%.1f)",
        n_events, n, drift, threshold,
    )
    return result


# ---------------------------------------------------------------------------
# 3. Combined detector
# ---------------------------------------------------------------------------


def detect_events(
    returns: pd.Series,
    window: int = 168,
    zscore_threshold: float = 3.0,
    cusum_drift: float = 0.5,
    cusum_threshold: float = 5.0,
) -> pd.DataFrame:
    """Combined event detection: z-score OR CUSUM.

    Args:
        returns: Return series with DatetimeIndex.
        window: Rolling window for z-score (and CUSUM std).
        zscore_threshold: Z-score flag threshold.
        cusum_drift: CUSUM drift parameter (std units).
        cusum_threshold: CUSUM alarm threshold (std units).

    Returns:
        DataFrame with columns:
            zscore_flag   - 0/1
            cusum_flag    - 0/1
            event_flag    - 0/1 (OR of both)
            zscore_value  - the z-score
            cusum_value   - the CUSUM statistic
            event_type    - 'zscore' | 'cusum' | 'both' | None
    """
    # Validate input
    if returns.empty:
        logger.warning("detect_events: empty input series")
        return pd.DataFrame(
            columns=["zscore_flag", "cusum_flag", "event_flag",
                      "zscore_value", "cusum_value", "event_type"],
        )

    clean = returns.dropna()
    if len(clean) < 3:
        logger.warning("detect_events: too few non-NaN points (%d)", len(clean))
        return pd.DataFrame(
            columns=["zscore_flag", "cusum_flag", "event_flag",
                      "zscore_value", "cusum_value", "event_type"],
            index=returns.index,
        )

    # Check for zero variance (constant series)
    if clean.std() == 0:
        logger.warning("detect_events: zero variance in returns, no events possible")
        out = pd.DataFrame(index=returns.index)
        out["zscore_flag"] = 0
        out["cusum_flag"] = 0
        out["event_flag"] = 0
        out["zscore_value"] = 0.0
        out["cusum_value"] = 0.0
        out["event_type"] = None
        return out

    zs = detect_zscore_events(clean, window=window, threshold=zscore_threshold)
    cu = detect_cusum_events(
        clean, drift=cusum_drift, threshold=cusum_threshold,
        rolling_std_window=window,
    )

    combined = pd.DataFrame(index=clean.index)
    combined["zscore_flag"] = zs["zscore_flag"]
    combined["cusum_flag"] = cu["cusum_flag"]
    combined["event_flag"] = ((combined["zscore_flag"] | combined["cusum_flag"])).astype(int)
    combined["zscore_value"] = zs["zscore_value"]
    combined["cusum_value"] = cu["cusum_value"]

    # Classify event type
    def _event_type(row):
        zf = row["zscore_flag"]
        cf = row["cusum_flag"]
        if zf and cf:
            return "both"
        if zf:
            return "zscore"
        if cf:
            return "cusum"
        return None

    combined["event_type"] = combined.apply(_event_type, axis=1)

    total = combined["event_flag"].sum()
    zs_only = (combined["event_type"] == "zscore").sum()
    cu_only = (combined["event_type"] == "cusum").sum()
    both = (combined["event_type"] == "both").sum()
    logger.info(
        "Combined events: %d total (%d zscore-only, %d cusum-only, %d both) "
        "out of %d points",
        total, zs_only, cu_only, both, len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# 4. Integration helper
# ---------------------------------------------------------------------------


def apply_event_override(
    signals: list[dict],
    events: pd.DataFrame,
    min_position_weight: float = 0.05,
) -> list[dict]:
    """Override risk signals during detected events.

    When an event is active the position_weight is forced to *min_position_weight*
    (effectively maximum uncertainty) and the event metadata is attached to the
    signal dict.

    The matching is done on the 'date' key in each signal dict against the
    events DataFrame index.  If the signal's date does not appear in *events*
    the signal is passed through unchanged.

    Args:
        signals: List of risk-signal dicts (from phase3_risk_signals).
                 Each must have at least a 'date' key and a 'position_weight' key.
        events: DataFrame produced by :func:`detect_events`.
        min_position_weight: Position weight to force during events.

    Returns:
        New list of signal dicts with event overrides applied.
    """
    if events.empty or not signals:
        return signals

    # Build a lookup from the events index for fast matching
    event_lookup: dict[pd.Timestamp, pd.Series] = {}
    for ts in events.index[events["event_flag"] == 1]:
        event_lookup[ts] = events.loc[ts]

    overridden = []
    n_overrides = 0

    for sig in signals:
        sig = dict(sig)  # shallow copy
        raw_date = sig.get("date")

        # Try to match: the signal date may be a string or Timestamp
        if isinstance(raw_date, str):
            try:
                ts = pd.Timestamp(raw_date)
            except (ValueError, TypeError):
                overridden.append(sig)
                continue
        elif isinstance(raw_date, pd.Timestamp):
            ts = raw_date
        else:
            overridden.append(sig)
            continue

        if ts in event_lookup:
            row = event_lookup[ts]
            sig["position_weight"] = min_position_weight
            sig["event_flag"] = 1
            sig["event_type"] = row.get("event_type")
            sig["zscore_value"] = float(row.get("zscore_value", 0.0))
            sig["cusum_value"] = float(row.get("cusum_value", 0.0))
            n_overrides += 1
        else:
            sig["event_flag"] = 0
            sig["event_type"] = None

        overridden.append(sig)

    logger.info(
        "Event override: %d of %d signals overridden (position_weight -> %.3f)",
        n_overrides, len(signals), min_position_weight,
    )
    return overridden


# ---------------------------------------------------------------------------
# 5. CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    ROOT = Path(__file__).parent.parent
    hourly_path = ROOT / "data/raw/hourly/btc_1h.parquet"

    if hourly_path.exists():
        df = pd.read_parquet(hourly_path).sort_index()

        # Determine the return column
        if "return" in df.columns:
            ret_col = "return"
        elif "close" in df.columns:
            # Compute hourly log-returns from close prices
            df["_return"] = np.log(df["close"] / df["close"].shift(1))
            ret_col = "_return"
        else:
            # Fall back to first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns in {hourly_path}")
            ret_col = numeric_cols[0]
            logger.info("Using column '%s' as return proxy", ret_col)

        returns = df[ret_col].dropna()
        print(f"BTC hourly returns: {len(returns)} rows "
              f"({returns.index[0]} -> {returns.index[-1]})")
        print(f"  mean={returns.mean():.6f}  std={returns.std():.6f}")

        events = detect_events(returns)

        # --- Summary ---
        total = events["event_flag"].sum()
        zs_events = (events["event_type"] == "zscore").sum()
        cu_events = (events["event_type"] == "cusum").sum()
        both_events = (events["event_type"] == "both").sum()

        print(f"\n{'='*60}")
        print(f"Event detection summary")
        print(f"{'='*60}")
        print(f"  Total events:       {total}")
        print(f"  Z-score only:       {zs_events}")
        print(f"  CUSUM only:         {cu_events}")
        print(f"  Both:               {both_events}")
        print(f"  Event rate:         {total / len(events) * 100:.2f}%")

        # --- Top 10 most extreme events ---
        flagged = events[events["event_flag"] == 1].copy()
        if len(flagged) > 0:
            flagged["severity"] = flagged["zscore_value"].abs() + flagged["cusum_value"]
            top10 = flagged.nlargest(10, "severity")

            print(f"\nTop 10 most extreme events:")
            print(f"{'Date':<22} {'Type':<8} {'Z-score':>9} {'CUSUM':>9} {'Return':>10}")
            print("-" * 60)
            for ts, row in top10.iterrows():
                ret_val = returns.get(ts, float("nan"))
                print(f"{str(ts):<22} {str(row['event_type']):<8} "
                      f"{row['zscore_value']:>9.2f} {row['cusum_value']:>9.2f} "
                      f"{ret_val:>10.4%}")
        else:
            print("\nNo events detected.")

    else:
        print(f"Hourly data not found at {hourly_path}")
        print("Generating synthetic hourly data for testing...\n")

        # Synthetic test: 4 weeks of hourly data with injected events
        np.random.seed(42)
        n = 168 * 4  # 4 weeks
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        returns = pd.Series(np.random.randn(n) * 0.005, index=dates)

        # Inject a flash crash at hour 300
        returns.iloc[300] = -0.08
        # Inject a pump at hour 500
        returns.iloc[500] = 0.06
        # Inject sustained downtrend around hour 400-420
        returns.iloc[400:420] -= 0.01

        print(f"Synthetic hourly returns: {n} rows")
        events = detect_events(returns)

        total = events["event_flag"].sum()
        zs_events = (events["event_type"] == "zscore").sum()
        cu_events = (events["event_type"] == "cusum").sum()
        both_events = (events["event_type"] == "both").sum()

        print(f"\n{'='*60}")
        print(f"Event detection summary (synthetic)")
        print(f"{'='*60}")
        print(f"  Total events:       {total}")
        print(f"  Z-score only:       {zs_events}")
        print(f"  CUSUM only:         {cu_events}")
        print(f"  Both:               {both_events}")

        flagged = events[events["event_flag"] == 1].copy()
        if len(flagged) > 0:
            flagged["severity"] = flagged["zscore_value"].abs() + flagged["cusum_value"]
            top10 = flagged.nlargest(10, "severity")

            print(f"\nTop 10 most extreme events:")
            print(f"{'Date':<22} {'Type':<8} {'Z-score':>9} {'CUSUM':>9} {'Return':>10}")
            print("-" * 60)
            for ts, row in top10.iterrows():
                ret_val = returns.get(ts, float("nan"))
                print(f"{str(ts):<22} {str(row['event_type']):<8} "
                      f"{row['zscore_value']:>9.2f} {row['cusum_value']:>9.2f} "
                      f"{ret_val:>10.4%}")
