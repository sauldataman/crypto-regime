"""
拉取小时级数据：BTC/ETH/SOL/BNB/DOGE/AVAX/LINK 从最早可用日期开始
每条序列大幅增加训练数据
"""
import argparse
import logging
import time

import ccxt
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path(__file__).parent.parent / "data/raw/hourly"
RAW.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # exponential backoff in seconds


def fetch_with_retry(exchange, symbol, timeframe, since, limit=1000):
    """Fetch OHLCV with exponential backoff retry on API errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except (ccxt.RateLimitExceeded, ccxt.RequestTimeout, ccxt.NetworkError) as e:
            delay = RETRY_DELAYS[attempt]
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                attempt + 1, MAX_RETRIES, symbol, type(e).__name__, delay,
            )
            time.sleep(delay)
        except ccxt.BaseError as e:
            logger.error("Non-retryable CCXT error for %s: %s", symbol, e)
            raise
    # Final attempt — let it raise if it fails
    return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


def fetch_hourly(symbol, since_str, name):
    logger.info("Fetching %s hourly from %s...", name, since_str)
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_str}T00:00:00Z")
    ohlcv = []
    while True:
        batch = fetch_with_retry(exchange, symbol, "1h", since)
        if not batch:
            break
        ohlcv.extend(batch)
        since = batch[-1][0] + 3600_000
        if since > exchange.milliseconds():
            break
        time.sleep(0.3)
    if not ohlcv:
        logger.warning("%s: no data returned from API", name)
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").drop(columns=["ts"])
    df.columns = [f"{name}_{c}" for c in df.columns]
    logger.info("  %s: %s total rows (%s -> %s)", name, f"{len(df):,}", df.index[0], df.index[-1])
    return df


# Earliest available dates on Binance for each asset
assets = [
    ("BTC/USDT", "2017-01-01", "btc"),
    ("ETH/USDT", "2017-01-01", "eth"),
    ("SOL/USDT", "2020-04-01", "sol"),
    ("BNB/USDT", "2017-11-01", "bnb"),
    ("DOGE/USDT", "2019-07-01", "doge"),
    ("AVAX/USDT", "2020-09-01", "avax"),
    ("LINK/USDT", "2017-09-01", "link"),
]


def main():
    parser = argparse.ArgumentParser(description="Fetch hourly OHLCV data from Binance")
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing files and re-fetch all data",
    )
    args = parser.parse_args()

    for symbol, since, name in assets:
        path = RAW / f"{name}_1h.parquet"

        if path.exists():
            if args.force:
                logger.info("%s: --force passed, deleting existing file", name)
                path.unlink()
            else:
                logger.info("%s: already exists, skip (use --force to re-fetch)", name)
                continue

        df = fetch_hourly(symbol, since, name)
        if not df.empty:
            df.to_parquet(path)
            logger.info("  Saved: %s", path)
        time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
