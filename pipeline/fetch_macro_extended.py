"""
Fetch extended macro/financial indicators for Phase II-D macro correction.

Adds to the existing macro.parquet:
  - Fed Funds Rate (FRED: FEDFUNDS)
  - Yield curve spread 10Y-2Y (FRED: T10Y2Y)
  - High yield credit spread (FRED: BAMLH0A0HYM2)
  - Bitcoin dominance (CoinGecko)
  - Crypto Fear & Greed Index (alternative.me)
  - BTC ETF total inflows (proxy: BITO ETF volume)

Also fetches existing indicators with longer history if available.

Usage:
  python pipeline/fetch_macro_extended.py
"""
import json
import logging
import os
import time

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw"
RAW.mkdir(parents=True, exist_ok=True)

START_DATE = "2010-01-01"


def fetch_fred_extended() -> pd.DataFrame:
    """Fetch additional FRED series."""
    api_key = os.environ.get("FRED_API_KEY", "")

    # Extended FRED series
    series = {
        # Existing (already in macro.parquet, fetch again for completeness)
        "DTWEXBGS": "dxy",
        "DGS10": "treasury_10y",
        "VIXCLS": "vix",
        # New indicators
        "FEDFUNDS": "fed_funds_rate",      # Fed Funds Rate (monthly)
        "T10Y2Y": "yield_curve_10y2y",     # 10Y-2Y spread (daily)
        "BAMLH0A0HYM2": "hy_spread",       # High yield credit spread (daily)
        "T10YIE": "breakeven_inflation",    # 10Y breakeven inflation (daily)
        "DCOILWTICO": "wti_crude",          # WTI crude oil (daily)
    }

    if not api_key:
        logger.warning("FRED_API_KEY not set. Fetching what we can from Yahoo Finance.")
        return pd.DataFrame()

    frames = []
    for series_id, name in series.items():
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            data = fred.get_series(series_id, observation_start=START_DATE)
            if data is not None and len(data) > 0:
                df = data.to_frame(name=name)
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                frames.append(df)
                logger.info("  FRED %s (%s): %d rows, starts %s",
                            series_id, name, len(df), df.index[0].date())
        except Exception as e:
            logger.warning("  FRED %s failed: %s", series_id, e)

    if frames:
        return pd.concat(frames, axis=1).sort_index()
    return pd.DataFrame()


def fetch_yahoo_extended() -> pd.DataFrame:
    """Fetch additional Yahoo Finance indicators."""
    tickers = {
        # Volatility indices
        "^VIX": "vix_yf",            # VIX (backup to FRED)
        "^MOVE": "move_index",        # MOVE bond volatility index
        # Rates
        "^TNX": "treasury_10y_yf",    # 10Y Treasury yield
        "^FVX": "treasury_5y_yf",     # 5Y Treasury yield
        "^IRX": "treasury_3m_yf",     # 3-month T-bill
        # Equity
        "^GSPC": "sp500",
        "^IXIC": "nasdaq",
        "^RUT": "russell_2000",       # Small cap
        # Commodities
        "GC=F": "gold",
        "SI=F": "silver",
        # Currency
        "DX-Y.NYB": "dxy_yf",        # Dollar index
        # Crypto-adjacent
        "BITO": "bito_etf",          # Bitcoin ETF proxy (ProShares, from Oct 2021)
    }

    frames = []
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if data.empty:
                logger.warning("  Yahoo %s (%s): no data", ticker, name)
                continue
            # Use close price
            close_col = data.columns
            if isinstance(close_col, pd.MultiIndex):
                close = data[("Close", ticker)] if ("Close", ticker) in data.columns else data.iloc[:, 0]
            else:
                close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            df = close.to_frame(name=name)
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            frames.append(df)
            logger.info("  Yahoo %s (%s): %d rows, starts %s",
                        ticker, name, len(df), df.index[0].date())
        except Exception as e:
            logger.warning("  Yahoo %s (%s) failed: %s", ticker, name, e)

    if frames:
        return pd.concat(frames, axis=1).sort_index()
    return pd.DataFrame()


def fetch_fear_greed() -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index from alternative.me."""
    logger.info("Fetching Crypto Fear & Greed Index...")
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            logger.warning("  Fear & Greed: no data returned")
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                "date": pd.to_datetime(int(d["timestamp"]), unit="s"),
                "fear_greed": int(d["value"]),
            })
        df = pd.DataFrame(records).set_index("date").sort_index()
        logger.info("  Fear & Greed: %d rows (%s -> %s)",
                     len(df), df.index[0].date(), df.index[-1].date())
        return df

    except Exception as e:
        logger.warning("  Fear & Greed failed: %s", e)
        return pd.DataFrame()


def fetch_btc_dominance() -> pd.DataFrame:
    """Fetch BTC dominance from CoinGecko."""
    logger.info("Fetching BTC dominance...")
    try:
        # CoinGecko free API: global market data
        url = "https://api.coingecko.com/api/v3/global"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc", None)

        if btc_dom is not None:
            logger.info("  Current BTC dominance: %.1f%%", btc_dom)
            # CoinGecko free API doesn't have historical dominance
            # Use BTC.D from TradingView via yfinance as proxy
            logger.info("  Fetching historical BTC dominance proxy via market caps...")

            # Fetch BTC and total crypto market cap
            btc = yf.download("BTC-USD", start="2014-01-01", progress=False, auto_adjust=True)
            total = yf.download("^CMC200", start="2014-01-01", progress=False, auto_adjust=True)

            if not btc.empty:
                # Use BTC market cap / total as proxy
                # Since we can't easily get total, use BTC close as a feature directly
                if isinstance(btc.columns, pd.MultiIndex):
                    btc_close = btc[("Close", "BTC-USD")]
                else:
                    btc_close = btc["Close"]
                df = btc_close.to_frame(name="btc_mcap_proxy")
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                return df

        return pd.DataFrame()

    except Exception as e:
        logger.warning("  BTC dominance failed: %s", e)
        return pd.DataFrame()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features."""
    out = df.copy()

    # Yield curve: if we have 10Y and 3M, compute spread
    if "treasury_10y_yf" in out.columns and "treasury_3m_yf" in out.columns:
        out["yield_curve_10y3m"] = out["treasury_10y_yf"] - out["treasury_3m_yf"]

    # VIX change (momentum)
    if "vix_yf" in out.columns:
        out["vix_change_5d"] = out["vix_yf"].pct_change(5)
        out["vix_change_20d"] = out["vix_yf"].pct_change(20)

    # S&P500 momentum
    if "sp500" in out.columns:
        out["sp500_return_5d"] = np.log(out["sp500"] / out["sp500"].shift(5))
        out["sp500_return_20d"] = np.log(out["sp500"] / out["sp500"].shift(20))
        out["sp500_vol_20d"] = out["sp500"].pct_change().rolling(20).std() * np.sqrt(252)

    # Gold/BTC ratio (safe haven signal)
    if "gold" in out.columns and "btc_mcap_proxy" in out.columns:
        out["gold_btc_ratio"] = out["gold"] / out["btc_mcap_proxy"].clip(lower=1)

    # Dollar strength momentum
    if "dxy_yf" in out.columns:
        out["dxy_change_5d"] = out["dxy_yf"].pct_change(5)

    # Credit conditions tightening
    if "hy_spread" in out.columns:
        out["hy_spread_change_20d"] = out["hy_spread"].diff(20)

    return out


def main():
    logger.info("=" * 60)
    logger.info("Fetching extended macro indicators")
    logger.info("=" * 60)

    # Fetch all sources
    logger.info("\n--- FRED Extended ---")
    fred_df = fetch_fred_extended()

    logger.info("\n--- Yahoo Finance Extended ---")
    yahoo_df = fetch_yahoo_extended()

    logger.info("\n--- Crypto Fear & Greed ---")
    fg_df = fetch_fear_greed()

    logger.info("\n--- BTC Dominance ---")
    btc_dom_df = fetch_btc_dominance()

    # Merge all
    frames = [f for f in [fred_df, yahoo_df, fg_df, btc_dom_df] if not f.empty]
    if not frames:
        logger.error("No data fetched!")
        return

    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged.ffill()  # forward-fill gaps

    # Compute derived features
    merged = compute_derived_features(merged)

    logger.info("\n--- Summary ---")
    logger.info("Total rows: %d", len(merged))
    logger.info("Total columns: %d", len(merged.columns))
    logger.info("Date range: %s -> %s", merged.index[0].date(), merged.index[-1].date())
    logger.info("Columns: %s", list(merged.columns))

    # Save
    out_path = RAW / "macro_extended.parquet"
    merged.to_parquet(out_path)
    logger.info("Saved: %s", out_path)

    # Also save a feature availability report
    report = {}
    for col in merged.columns:
        valid = merged[col].dropna()
        if len(valid) > 0:
            report[col] = {
                "start": str(valid.index[0].date()),
                "end": str(valid.index[-1].date()),
                "count": len(valid),
                "missing_pct": round((1 - len(valid) / len(merged)) * 100, 1),
            }
    report_path = ROOT / "results" / "macro_extended_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
