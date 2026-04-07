"""
合并 Kraken 早期数据 + Binance 数据
Kraken CSV: timestamp,open,high,low,close,volume,trades (no header)
输出: 保存原始 1min 到 data/raw/1min/, 合并 5min 到 data/raw/5min/
"""
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
KRAKEN_DIR = ROOT / "data/raw/kraken"
RAW_1M = ROOT / "data/raw/1min"
RAW_5M = ROOT / "data/raw/5min"
RAW_1M.mkdir(parents=True, exist_ok=True)

# Kraken symbol -> our name, Binance start date (use Kraken data before this)
ASSETS = {
    "XBTUSD": {"name": "btc", "binance_5m_start": "2017-08-17"},
    "ETHUSD": {"name": "eth", "binance_5m_start": "2017-08-17"},
    "XDGUSD": {"name": "doge", "binance_5m_start": "2019-07-05"},
}


def load_kraken_csv(path: Path) -> pd.DataFrame:
    """Load Kraken CSV (no header): timestamp,open,high,low,close,volume,trades"""
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    # Remove rows with zero volume (no actual trading)
    df = df[df["volume"] > 0]
    logger.info("  Loaded %s: %s rows (%s -> %s)", path.name, f"{len(df):,}", df.index[0], df.index[-1])
    return df


def aggregate_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min OHLCV to 5-min."""
    df_5m = df_1m.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    return df_5m


def merge_with_binance(early_df: pd.DataFrame, binance_path: Path, cutoff: str, name: str) -> pd.DataFrame:
    """Merge early (Kraken) data with existing Binance data, preferring Binance where overlap."""
    if not binance_path.exists():
        logger.warning("  No Binance file at %s, using Kraken only", binance_path)
        return early_df

    binance_df = pd.read_parquet(binance_path)
    binance_df.index = pd.to_datetime(binance_df.index)

    # Use Kraken data before Binance start, Binance data after
    cutoff_dt = pd.Timestamp(cutoff)
    kraken_early = early_df[early_df.index < cutoff_dt]

    if kraken_early.empty:
        logger.info("  %s: no Kraken data before Binance start (%s)", name, cutoff)
        return binance_df

    merged = pd.concat([kraken_early, binance_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]  # prefer Binance

    logger.info(
        "  %s merged: %s rows (%s -> %s). Kraken added %s early rows.",
        name, f"{len(merged):,}", merged.index[0], merged.index[-1], f"{len(kraken_early):,}",
    )
    return merged


def main():
    for kraken_sym, config in ASSETS.items():
        name = config["name"]
        bn_start = config["binance_5m_start"]

        logger.info("\n=== Processing %s (%s) ===", name.upper(), kraken_sym)

        # --- 1min: save Kraken raw ---
        path_1m_kraken = KRAKEN_DIR / f"{kraken_sym}_1.csv"
        path_1m_out = RAW_1M / f"{name}_1m.parquet"

        if path_1m_kraken.exists():
            df_1m = load_kraken_csv(path_1m_kraken)
            # Save full Kraken 1min (useful for future event detection)
            df_1m.to_parquet(path_1m_out)
            logger.info("  Saved 1min: %s (%s rows)", path_1m_out, f"{len(df_1m):,}")
        else:
            logger.warning("  No 1min file: %s", path_1m_kraken)
            df_1m = pd.DataFrame()

        # --- 5min: load Kraken 5min CSV, merge with Binance ---
        path_5m_kraken = KRAKEN_DIR / f"{kraken_sym}_5.csv"
        path_5m_binance = RAW_5M / f"{name}_5m.parquet"

        if path_5m_kraken.exists():
            df_5m_kraken = load_kraken_csv(path_5m_kraken)
            merged_5m = merge_with_binance(df_5m_kraken, path_5m_binance, bn_start, f"{name}_5m")
            merged_5m.to_parquet(path_5m_binance)
            logger.info("  Updated 5min: %s (%s rows)", path_5m_binance, f"{len(merged_5m):,}")
        elif not df_1m.empty:
            # Fallback: aggregate 1min to 5min
            logger.info("  No Kraken 5min CSV, aggregating from 1min...")
            df_5m_from_1m = aggregate_to_5min(df_1m)
            merged_5m = merge_with_binance(df_5m_from_1m, path_5m_binance, bn_start, f"{name}_5m")
            merged_5m.to_parquet(path_5m_binance)
            logger.info("  Updated 5min: %s (%s rows)", path_5m_binance, f"{len(merged_5m):,}")

    # Summary
    logger.info("\n=== SUMMARY ===")
    for freq_dir, freq_name in [(RAW_1M, "1min"), (RAW_5M, "5min")]:
        for f in sorted(freq_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            logger.info("  %s %s: %s rows (%s -> %s)", freq_name, f.stem, f"{len(df):,}", df.index[0], df.index[-1])


if __name__ == "__main__":
    main()
