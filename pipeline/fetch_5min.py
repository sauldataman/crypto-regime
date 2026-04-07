"""5分钟K线，11个资产，从2017年开始（Binance 2017-07上线）"""
import argparse
import logging
import time

import ccxt
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path(__file__).parent.parent / "data/raw/5min"
RAW.mkdir(parents=True, exist_ok=True)

assets = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
    "MATIC/USDT", "XRP/USDT", "ADA/USDT", "DOT/USDT"
]

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


def main():
    parser = argparse.ArgumentParser(description="Fetch 5min OHLCV data from Binance")
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing files and re-fetch all data",
    )
    args = parser.parse_args()

    exchange = ccxt.binance({"enableRateLimit": True})
    SINCE = exchange.parse8601("2017-01-01T00:00:00Z")

    for symbol in assets:
        name = symbol.replace("/USDT", "").lower()
        path = RAW / f"{name}_5m.parquet"

        if path.exists():
            if args.force:
                logger.info("%s: --force passed, deleting existing file", name)
                path.unlink()
            else:
                logger.info("%s: already exists, skip (use --force to re-fetch)", name)
                continue

        logger.info("Fetching %s 5m...", name)
        ohlcv = []
        since = SINCE
        while True:
            batch = fetch_with_retry(exchange, symbol, "5m", since)
            if not batch:
                break
            ohlcv.extend(batch)
            since = batch[-1][0] + 300_000  # +5min
            if since > exchange.milliseconds():
                break
            time.sleep(0.05)

        if not ohlcv:
            logger.warning("%s: no data returned from API", name)
            continue

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("datetime").drop(columns=["ts"])
        df.to_parquet(path)
        logger.info("%s: %s total rows fetched -> %s", name, f"{len(df):,}", path.name)
        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    main()
