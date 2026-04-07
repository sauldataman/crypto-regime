"""
拉取 1 分钟 K 线 (CryptoCompare histominute API)
用于 BTC/ETH/DOGE 的 Binance 之前的早期数据
保存原始 1min + 聚合 5min 拼接到现有数据
"""
import argparse
import logging
import os
import time

import pandas as pd
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_1M = ROOT / "data/raw/1min"
RAW_5M = ROOT / "data/raw/5min"
RAW_1M.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]


def fetch_cc_minute(fsym: str, tsym: str, since_str: str, until_str: str) -> pd.DataFrame:
    """Fetch minute-level data from CryptoCompare histominute API.

    CryptoCompare returns max 2000 rows per call, paginated backwards.
    """
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY", "")
    if not api_key:
        logger.error("CRYPTOCOMPARE_API_KEY not set")
        return pd.DataFrame()

    since_ts = int(pd.Timestamp(since_str).timestamp())
    until_ts = int(pd.Timestamp(until_str).timestamp())

    logger.info("Fetching %s 1min from CryptoCompare (%s -> %s)...", fsym, since_str, until_str)

    all_rows = []
    to_ts = until_ts
    batch_count = 0

    while to_ts > since_ts:
        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": 2000,
            "toTs": to_ts,
            "api_key": api_key,
        }

        resp = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning("Attempt %d failed: %s. Retry in %ds", attempt + 1, e, delay)
                time.sleep(delay)
        else:
            logger.error("Failed after %d retries for %s at toTs=%d", MAX_RETRIES, fsym, to_ts)
            break

        if data.get("Response") == "Error":
            logger.error("CryptoCompare error: %s", data.get("Message", "unknown"))
            break

        rows = data.get("Data", {}).get("Data", [])
        if not rows:
            break

        # Filter zero-volume rows
        valid = [r for r in rows if r.get("volumefrom", 0) > 0 or r.get("volumeto", 0) > 0]
        all_rows.extend(valid)

        earliest_ts = rows[0]["time"]
        if earliest_ts <= since_ts:
            break
        to_ts = earliest_ts - 1

        batch_count += 1
        if batch_count % 50 == 0:
            logger.info("  %s: %d batches, %d rows so far...", fsym, batch_count, len(all_rows))

        time.sleep(0.15)  # rate limit: ~6 req/sec

    if not all_rows:
        logger.warning("%s: no 1min data returned", fsym)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df["volume"] = df.get("volumeto", 0)
    df = df[["datetime", "open", "high", "low", "close", "volume"]].set_index("datetime")
    df = df[df.index >= since_str].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info("  %s 1min: %s rows (%s -> %s)", fsym, f"{len(df):,}", df.index[0], df.index[-1])
    return df


def aggregate_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV to 5-minute."""
    if df_1m.empty:
        return pd.DataFrame()

    df_5m = df_1m.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    logger.info("  Aggregated 1min -> 5min: %s rows", f"{len(df_5m):,}")
    return df_5m


def merge_5min(early_5m: pd.DataFrame, existing_path: Path, name: str) -> pd.DataFrame:
    """Merge early 5min (from CryptoCompare) with existing Binance 5min."""
    if not existing_path.exists():
        logger.warning("  No existing 5min file at %s", existing_path)
        return early_5m

    existing = pd.read_parquet(existing_path)
    existing.index = pd.to_datetime(existing.index)

    bn_start = existing.index[0]
    early_only = early_5m[early_5m.index < bn_start]

    if early_only.empty:
        logger.info("  %s: no early data before Binance start (%s)", name, bn_start)
        return existing

    merged = pd.concat([early_only, existing]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]  # prefer Binance
    logger.info("  %s 5min merged: %s rows (%s -> %s)", name, f"{len(merged):,}", merged.index[0], merged.index[-1])
    return merged


# Assets to fetch early 1min data for
ASSETS = [
    {
        "name": "btc",
        "fsym": "BTC",
        "tsym": "USD",
        "since": "2014-01-01",
        "until": "2017-08-01",  # Binance starts ~2017-08 for BTC/USDT
    },
    {
        "name": "eth",
        "fsym": "ETH",
        "tsym": "USD",
        "since": "2016-01-01",
        "until": "2017-09-01",  # Binance starts ~2017-08 for ETH/USDT
    },
    {
        "name": "doge",
        "fsym": "DOGE",
        "tsym": "USD",
        "since": "2017-01-01",
        "until": "2019-07-01",  # Binance starts ~2019-07 for DOGE/USDT
    },
]


def main():
    parser = argparse.ArgumentParser(description="Fetch early 1min data from CryptoCompare")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if 1min file exists")
    parser.add_argument("--no-merge", action="store_true", help="Only fetch 1min, don't merge to 5min")
    args = parser.parse_args()

    for asset in ASSETS:
        name = asset["name"]
        path_1m = RAW_1M / f"{name}_1m.parquet"
        path_5m = RAW_5M / f"{name}_5m.parquet"

        # Fetch 1min
        if path_1m.exists() and not args.force:
            logger.info("%s 1min: already exists, loading from disk", name)
            df_1m = pd.read_parquet(path_1m)
            df_1m.index = pd.to_datetime(df_1m.index)
        else:
            df_1m = fetch_cc_minute(asset["fsym"], asset["tsym"], asset["since"], asset["until"])
            if not df_1m.empty:
                df_1m.to_parquet(path_1m)
                logger.info("  Saved 1min: %s (%s rows)", path_1m, f"{len(df_1m):,}")

        if df_1m.empty:
            continue

        # Aggregate to 5min and merge
        if not args.no_merge:
            df_5m_early = aggregate_to_5min(df_1m)
            if not df_5m_early.empty:
                merged = merge_5min(df_5m_early, path_5m, name)
                merged.to_parquet(path_5m)
                logger.info("  Updated 5min: %s (%s rows)", path_5m, f"{len(merged):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
