"""
Multi-asset data fetcher: ETH, SOL + BTC extended to 2010
"""
import yfinance as yf, pandas as pd, numpy as np
from pathlib import Path
import ccxt, time

RAW = Path(__file__).parent.parent / "data/raw"
RAW.mkdir(exist_ok=True)

def fetch_ohlcv_ccxt(symbol: str, since_str: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance via CCXT"""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_str}T00:00:00Z")
    ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
        if not batch:
            break
        ohlcv.extend(batch)
        since = batch[-1][0] + 86400_000
        if since > exchange.milliseconds():
            break
        time.sleep(0.3)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
    df = df.set_index("date").drop(columns=["ts"])
    return df

def fetch_asset(ticker_yf: str, symbol_ccxt: str, name: str, since: str) -> pd.DataFrame:
    print(f"  Fetching {name} via CCXT ({symbol_ccxt}) from {since}...")
    try:
        df = fetch_ohlcv_ccxt(symbol_ccxt, since)
        if len(df) > 100:
            df.columns = [f"{name}_{c}" for c in df.columns]
            print(f"  → {len(df)} rows via CCXT")
            return df
    except Exception as e:
        print(f"  CCXT failed: {e}, trying Yahoo Finance...")
    
    print(f"  Fetching {name} via Yahoo ({ticker_yf})...")
    raw = yf.download(ticker_yf, start=since, progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  ❌ No data for {name}")
        return pd.DataFrame()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw.columns = [f"{name}_{c}" for c in raw.columns]
    print(f"  → {len(raw)} rows via Yahoo")
    return raw

if __name__ == "__main__":
    assets = [
        ("ETH-USD",  "ETH/USDT", "eth", "2016-01-01"),
        ("SOL-USD",  "SOL/USDT", "sol", "2020-04-01"),
        ("BNB-USD",  "BNB/USDT", "bnb", "2017-11-01"),
    ]
    
    for ticker, ccxt_sym, name, since in assets:
        df = fetch_asset(ticker, ccxt_sym, name, since)
        if not df.empty:
            path = RAW / f"{name}_price.parquet"
            df.to_parquet(path)
            print(f"  Saved: {path} ({len(df)} rows)")
    
    print("\nDone.")
