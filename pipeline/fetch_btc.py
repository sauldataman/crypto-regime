"""Fetch BTC-USD daily price data from Yahoo Finance."""

import pandas as pd
import yfinance as yf

START_DATE = "2016-01-01"


def fetch_btc(end_date: str | None = None) -> pd.DataFrame:
    """Download BTC-USD OHLCV data from Yahoo Finance.

    Returns DataFrame with columns: open, high, low, close, volume
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
    return df


if __name__ == "__main__":
    df = fetch_btc()
    print(f"BTC data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(df.tail())
