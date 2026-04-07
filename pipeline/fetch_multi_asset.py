"""
Multi-asset data fetcher: ETH, SOL, BNB, DOGE, AVAX, LINK
"""
import logging
import time
from pathlib import Path

import ccxt
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path(__file__).parent.parent / "data/raw"
RAW.mkdir(exist_ok=True)


def fetch_ohlcv_ccxt(symbol: str, since_str: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance via CCXT"""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_str}T00:00:00Z")
    ohlcv = []
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
        except (ccxt.RateLimitExceeded, ccxt.RequestTimeout, ccxt.NetworkError) as e:
            logger.warning("CCXT transient error for %s: %s. Retrying in 2s...", symbol, e)
            time.sleep(2)
            continue
        except ccxt.BaseError as e:
            logger.error("CCXT error fetching %s: %s", symbol, e)
            raise
        if not batch:
            break
        ohlcv.extend(batch)
        since = batch[-1][0] + 86400_000
        if since > exchange.milliseconds():
            break
        time.sleep(0.3)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
    df = df.set_index("date").drop(columns=["ts"])
    return df


def fetch_asset(ticker_yf: str, symbol_ccxt: str, name: str, since: str) -> pd.DataFrame:
    logger.info("Fetching %s via CCXT (%s) from %s...", name, symbol_ccxt, since)
    try:
        df = fetch_ohlcv_ccxt(symbol_ccxt, since)
        if len(df) > 100:
            df.columns = [f"{name}_{c}" for c in df.columns]
            logger.info("  %s: %d rows via CCXT", name, len(df))
            return df
    except ccxt.BaseError as e:
        logger.warning("CCXT failed for %s: %s, trying Yahoo Finance...", name, e)

    logger.info("Fetching %s via Yahoo (%s)...", name, ticker_yf)
    try:
        raw = yf.download(ticker_yf, start=since, progress=False, auto_adjust=True)
    except Exception as e:
        logger.error("Yahoo Finance failed for %s: %s", name, e)
        return pd.DataFrame()

    if raw.empty:
        logger.warning("No data for %s from either source", name)
        return pd.DataFrame()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw.columns = [f"{name}_{c}" for c in raw.columns]
    logger.info("  %s: %d rows via Yahoo", name, len(raw))
    return raw


if __name__ == "__main__":
    assets = [
        # (yahoo_ticker, ccxt_symbol, name, earliest_date)
        ("ETH-USD",  "ETH/USDT",  "eth",  "2015-08-01"),  # ETH launched Aug 2015
        ("SOL-USD",  "SOL/USDT",  "sol",  "2020-04-01"),  # earliest available
        ("BNB-USD",  "BNB/USDT",  "bnb",  "2017-11-01"),  # earliest on Binance
        ("DOGE-USD", "DOGE/USDT", "doge", "2013-12-01"),  # DOGE launched Dec 2013
        ("AVAX-USD", "AVAX/USDT", "avax", "2020-09-01"),  # AVAX mainnet Sep 2020
        ("LINK-USD", "LINK/USDT", "link", "2017-09-01"),  # LINK launched Sep 2017
    ]

    for ticker, ccxt_sym, name, since in assets:
        df = fetch_asset(ticker, ccxt_sym, name, since)
        if not df.empty:
            path = RAW / f"{name}_price.parquet"
            df.to_parquet(path)
            logger.info("  Saved: %s (%d rows)", path, len(df))
        time.sleep(0.5)

    print("\nDone.")
