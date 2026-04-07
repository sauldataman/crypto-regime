"""Fetch macro/financial data from Yahoo Finance and FRED."""

import logging
import os
import warnings

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2010-01-01"

# Yahoo Finance tickers
# Note: All these tickers have data available from 2010 or earlier on Yahoo Finance.
YF_TICKERS = {
    "^GSPC": "sp500",      # S&P 500 — available from 1950+
    "^IXIC": "nasdaq",     # NASDAQ Composite — available from 1971+
    "GC=F": "gold",        # Gold futures — available from ~2000+
    "CL=F": "crude_oil",   # Crude oil futures — available from ~2000+
    "CNY=X": "cnyusd",     # CNY/USD — available from ~2003+
    "EURUSD=X": "eurusd",  # EUR/USD — available from ~2003+
    "JPY=X": "jpyusd",     # JPY/USD — available from ~1996+
}

# FRED series
# Note on data availability from FRED:
# - DTWEXBGS (DXY broad): available from 2006-01-02 (covers 2010+)
# - DGS10 (10Y Treasury): available from 1962-01-02 (covers 2010+)
# - M2SL (M2 money supply): available from 1959-01-01, monthly (covers 2010+)
# - VIXCLS (VIX): available from 1990-01-02 (covers 2010+)
# - CPIAUCSL (CPI): available from 1947-01-01, monthly (covers 2010+)
# All FRED series cover 2010+ without gaps.
FRED_SERIES = {
    "DTWEXBGS": "dxy",
    "DGS10": "treasury_10y",
    "M2SL": "m2",
    "VIXCLS": "vix",
    "CPIAUCSL": "cpi",
}


def fetch_yf_macro(end_date: str | None = None) -> pd.DataFrame:
    """Fetch macro tickers from Yahoo Finance (close prices)."""
    frames = []
    for ticker_symbol, col_name in YF_TICKERS.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=START_DATE, end=end_date, auto_adjust=True)
            if hist.empty:
                logger.warning("No data returned for %s (%s)", ticker_symbol, col_name)
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist.index.name = "date"
            series = hist["Close"].rename(col_name)
            series = series[~series.index.duplicated(keep="first")]
            actual_start = series.index.min()
            logger.info(
                "  %s (%s): %d rows, starts %s",
                col_name, ticker_symbol, len(series), actual_start.date(),
            )
            if actual_start > pd.Timestamp("2010-12-31"):
                logger.warning(
                    "  NOTE: %s starts at %s, later than 2010-01-01",
                    col_name, actual_start.date(),
                )
            frames.append(series)
        except Exception as e:
            logger.error("Failed to fetch %s (%s): %s", ticker_symbol, col_name, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def fetch_fred_macro(end_date: str | None = None) -> pd.DataFrame:
    """Fetch macro series from FRED. Requires FRED_API_KEY env var."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        warnings.warn("FRED_API_KEY not set -- skipping FRED data. VIX will come from Yahoo Finance.")
        return _fetch_vix_fallback(end_date)

    from fredapi import Fred
    fred = Fred(api_key=api_key)

    frames = []
    for series_id, col_name in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=START_DATE, observation_end=end_date)
            s.index = pd.to_datetime(s.index).normalize()
            s.index.name = "date"
            s.name = col_name
            s = s[~s.index.duplicated(keep="first")]
            actual_start = s.index.min()
            logger.info(
                "  %s (%s): %d rows, starts %s",
                col_name, series_id, len(s), actual_start.date(),
            )
            frames.append(s)
        except Exception as e:
            logger.error("Failed to fetch FRED %s (%s): %s", series_id, col_name, e)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    # M2 and CPI are monthly -- forward-fill to daily
    if "m2" in df.columns:
        df["m2"] = df["m2"].ffill()
    if "cpi" in df.columns:
        df["cpi"] = df["cpi"].ffill()
    return df


def _fetch_vix_fallback(end_date: str | None = None) -> pd.DataFrame:
    """Fetch VIX from Yahoo Finance as fallback when FRED unavailable."""
    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(start=START_DATE, end=end_date, auto_adjust=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        hist.index.name = "date"
        df = hist[["Close"]].rename(columns={"Close": "vix"})
        df = df[~df.index.duplicated(keep="first")]
        logger.info("  VIX (Yahoo fallback): %d rows", len(df))
        return df
    except Exception as e:
        logger.error("Failed to fetch VIX from Yahoo: %s", e)
        return pd.DataFrame()


def fetch_macro(end_date: str | None = None) -> pd.DataFrame:
    """Fetch all macro data, merging YF and FRED sources."""
    yf_df = fetch_yf_macro(end_date)
    fred_df = fetch_fred_macro(end_date)

    if yf_df.empty and fred_df.empty:
        return pd.DataFrame()
    if yf_df.empty:
        return fred_df
    if fred_df.empty:
        return yf_df

    merged = yf_df.join(fred_df, how="outer")
    return merged.sort_index()


if __name__ == "__main__":
    df = fetch_macro()
    print(f"Macro data: {len(df)} rows, {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(df.tail())
