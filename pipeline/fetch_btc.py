"""Fetch BTC-USD daily price data from Yahoo Finance."""

import logging
import warnings

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2010-01-01"


def fetch_btc(end_date: str | None = None) -> pd.DataFrame:
    """Download BTC-USD OHLCV data from Yahoo Finance.

    Returns DataFrame with columns: btc_open, btc_high, btc_low, btc_close, btc_volume
    indexed by date.
    """
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=START_DATE, end=end_date, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.rename(columns={
        "open": "btc_open",
        "high": "btc_high",
        "low": "btc_low",
        "close": "btc_close",
        "volume": "btc_volume",
    })
    # Drop duplicate index entries
    df = df[~df.index.duplicated(keep="first")]

    # --- Data quality checks ---
    _check_data_quality(df)

    return df


def _check_data_quality(df: pd.DataFrame) -> None:
    """Validate fetched BTC data: positive prices, missing day count."""
    # Assert all prices are positive
    price_cols = ["btc_open", "btc_high", "btc_low", "btc_close"]
    for col in price_cols:
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            raise ValueError(
                f"Data quality error: {non_positive} non-positive values in {col}"
            )

    # Check for missing trading days
    if len(df) < 2:
        logger.warning("Too few rows (%d) to assess missing days", len(df))
        return

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    # BTC trades 365 days/year, so every calendar day should have data
    missing_days = len(full_range) - len(df)
    total_days = len(full_range)
    missing_pct = missing_days / total_days * 100

    logger.info(
        "BTC data quality: %d rows, %d missing days out of %d (%.1f%%)",
        len(df), missing_days, total_days, missing_pct,
    )

    if missing_pct > 5.0:
        warnings.warn(
            f"BTC data has {missing_pct:.1f}% missing days ({missing_days}/{total_days}). "
            f"This exceeds the 5% threshold and may affect downstream analysis.",
            stacklevel=2,
        )


if __name__ == "__main__":
    df = fetch_btc()
    print(f"BTC data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(df.tail())
