"""Fetch macro/financial data from Yahoo Finance and FRED."""

import os
import warnings

import pandas as pd
import yfinance as yf

START_DATE = "2016-01-01"

# Yahoo Finance tickers
YF_TICKERS = {
    "^GSPC": "sp500",
    "^IXIC": "nasdaq",
    "GC=F": "gold",
    "CL=F": "crude_oil",
    "CNY=X": "cnyusd",
    "EURUSD=X": "eurusd",
    "JPY=X": "jpyusd",
}

# FRED series
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
                warnings.warn(f"No data for {ticker_symbol}")
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist.index.name = "date"
            series = hist["Close"].rename(col_name)
            series = series[~series.index.duplicated(keep="first")]
            frames.append(series)
        except Exception as e:
            warnings.warn(f"Failed to fetch {ticker_symbol}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def fetch_fred_macro(end_date: str | None = None) -> pd.DataFrame:
    """Fetch macro series from FRED. Requires FRED_API_KEY env var."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        warnings.warn("FRED_API_KEY not set — skipping FRED data. VIX will come from Yahoo Finance.")
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
            frames.append(s)
        except Exception as e:
            warnings.warn(f"Failed to fetch FRED {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    # M2 and CPI are monthly — forward-fill to daily
    df["m2"] = df["m2"].ffill() if "m2" in df.columns else None
    df["cpi"] = df["cpi"].ffill() if "cpi" in df.columns else None
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
        return df
    except Exception as e:
        warnings.warn(f"Failed to fetch VIX from Yahoo: {e}")
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
    print(df.tail())
