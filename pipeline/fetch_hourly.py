"""
拉取小时级数据：6个核心资产
- BTC/ETH/DOGE: CryptoCompare (早期) + Binance (2017+) 拼接
- SOL/BNB/LINK: Binance only (无更早数据)
"""
import argparse
import logging
import os
import time

import ccxt
import pandas as pd
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path(__file__).parent.parent / "data/raw/hourly"
RAW.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]


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
    return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


def fetch_binance_hourly(symbol: str, since_str: str, name: str) -> pd.DataFrame:
    """Fetch hourly data from Binance."""
    logger.info("Fetching %s hourly from Binance (%s)...", name, since_str)
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
        logger.warning("%s: no Binance data returned", name)
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").drop(columns=["ts"])
    logger.info("  %s Binance: %s rows (%s -> %s)", name, f"{len(df):,}", df.index[0], df.index[-1])
    return df


def fetch_cryptocompare_hourly(fsym: str, tsym: str, since_str: str, until_str: str) -> pd.DataFrame:
    """Fetch hourly data from CryptoCompare histohour API.

    CryptoCompare returns max 2000 rows per call, paginated backwards from toTs.
    """
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY", "")
    if not api_key:
        logger.warning("CRYPTOCOMPARE_API_KEY not set, skipping early data for %s", fsym)
        return pd.DataFrame()

    logger.info("Fetching %s hourly from CryptoCompare (%s -> %s)...", fsym, since_str, until_str)

    since_ts = int(pd.Timestamp(since_str).timestamp())
    until_ts = int(pd.Timestamp(until_str).timestamp())

    all_rows = []
    to_ts = until_ts

    while to_ts > since_ts:
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": 2000,
            "toTs": to_ts,
            "api_key": api_key,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 4
                logger.warning("CryptoCompare attempt %d failed: %s. Retry in %ds", attempt + 1, e, delay)
                time.sleep(delay)
        else:
            logger.error("CryptoCompare failed after %d retries for %s", MAX_RETRIES, fsym)
            break

        if data.get("Response") == "Error":
            logger.error("CryptoCompare error: %s", data.get("Message", "unknown"))
            break

        rows = data.get("Data", {}).get("Data", [])
        if not rows:
            break

        # Filter out rows with 0 volume (no trading)
        valid_rows = [r for r in rows if r.get("volumefrom", 0) > 0 or r.get("volumeto", 0) > 0]
        all_rows.extend(valid_rows)

        earliest_ts = rows[0]["time"]
        if earliest_ts <= since_ts:
            break
        to_ts = earliest_ts - 1
        time.sleep(0.3)

    if not all_rows:
        logger.warning("%s: no CryptoCompare data returned", fsym)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
    df["volume"] = df.get("volumeto", 0)  # volume in quote currency
    df = df[["datetime", "open", "high", "low", "close", "volume"]].set_index("datetime")
    df = df[df.index >= since_str].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info("  %s CryptoCompare: %s rows (%s -> %s)", fsym, f"{len(df):,}", df.index[0], df.index[-1])
    return df


def merge_hourly(cc_df: pd.DataFrame, bn_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge CryptoCompare (early) + Binance (later), preferring Binance where overlap."""
    if cc_df.empty:
        return bn_df
    if bn_df.empty:
        return cc_df

    # Use Binance data where available (higher quality), CryptoCompare for earlier period
    bn_start = bn_df.index[0]
    cc_early = cc_df[cc_df.index < bn_start]

    if cc_early.empty:
        logger.info("  %s: no early CryptoCompare data before Binance start (%s)", name, bn_start)
        return bn_df

    merged = pd.concat([cc_early, bn_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]  # prefer Binance
    logger.info("  %s merged: %s rows (%s -> %s)", name, f"{len(merged):,}", merged.index[0], merged.index[-1])
    return merged


# Asset config: (binance_symbol, binance_since, name, cc_fsym, cc_tsym, cc_since, cc_until)
# cc_until = when Binance data starts, so CryptoCompare fills the gap before that
ASSETS = [
    {
        "name": "btc",
        "binance_symbol": "BTC/USDT",
        "binance_since": "2017-07-01",
        "cc_fsym": "BTC",
        "cc_tsym": "USD",
        "cc_since": "2014-01-01",
        "cc_until": "2017-07-01",
    },
    {
        "name": "eth",
        "binance_symbol": "ETH/USDT",
        "binance_since": "2017-08-01",
        "cc_fsym": "ETH",
        "cc_tsym": "USD",
        "cc_since": "2016-01-01",
        "cc_until": "2017-08-01",
    },
    {
        "name": "sol",
        "binance_symbol": "SOL/USDT",
        "binance_since": "2020-04-01",
        "cc_fsym": None,  # no early data needed
        "cc_tsym": None,
        "cc_since": None,
        "cc_until": None,
    },
    {
        "name": "bnb",
        "binance_symbol": "BNB/USDT",
        "binance_since": "2017-11-01",
        "cc_fsym": None,
        "cc_tsym": None,
        "cc_since": None,
        "cc_until": None,
    },
    {
        "name": "doge",
        "binance_symbol": "DOGE/USDT",
        "binance_since": "2019-07-01",
        "cc_fsym": "DOGE",
        "cc_tsym": "USD",
        "cc_since": "2017-01-01",
        "cc_until": "2019-07-01",
    },
    {
        "name": "link",
        "binance_symbol": "LINK/USDT",
        "binance_since": "2017-09-01",
        "cc_fsym": None,
        "cc_tsym": None,
        "cc_since": None,
        "cc_until": None,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Fetch hourly OHLCV data (CryptoCompare + Binance)")
    parser.add_argument("--force", action="store_true", help="Delete existing files and re-fetch")
    args = parser.parse_args()

    for asset in ASSETS:
        name = asset["name"]
        path = RAW / f"{name}_1h.parquet"

        if path.exists():
            if args.force:
                logger.info("%s: --force passed, deleting existing file", name)
                path.unlink()
            else:
                logger.info("%s: already exists, skip (use --force to re-fetch)", name)
                continue

        # Fetch Binance data
        bn_df = fetch_binance_hourly(asset["binance_symbol"], asset["binance_since"], name)

        # Fetch CryptoCompare early data if configured
        cc_df = pd.DataFrame()
        if asset["cc_fsym"]:
            cc_df = fetch_cryptocompare_hourly(
                asset["cc_fsym"], asset["cc_tsym"],
                asset["cc_since"], asset["cc_until"],
            )

        # Merge
        final_df = merge_hourly(cc_df, bn_df, name)

        if not final_df.empty:
            final_df.to_parquet(path)
            logger.info("  Saved: %s (%s rows)", path, f"{len(final_df):,}")
        time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
