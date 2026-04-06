"""
拉取小时级数据：BTC/ETH/SOL/BNB 过去4年的小时K线
每条序列约 35000 个时间点，大幅增加训练数据
"""
import ccxt, pandas as pd, numpy as np, time
from pathlib import Path

RAW = Path(__file__).parent.parent / "data/raw/hourly"
RAW.mkdir(parents=True, exist_ok=True)

def fetch_hourly(symbol, since_str, name):
    print(f"Fetching {name} hourly from {since_str}...")
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_str}T00:00:00Z")
    ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
        if not batch: break
        ohlcv.extend(batch)
        since = batch[-1][0] + 3600_000
        if since > exchange.milliseconds(): break
        time.sleep(0.3)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").drop(columns=["ts"])
    df.columns = [f"{name}_{c}" for c in df.columns]
    print(f"  → {len(df)} rows ({df.index[0]} → {df.index[-1]})")
    return df

assets = [
    ("BTC/USDT", "2021-01-01", "btc"),
    ("ETH/USDT", "2021-01-01", "eth"),
    ("SOL/USDT", "2021-01-01", "sol"),
    ("BNB/USDT", "2021-01-01", "bnb"),
    ("DOGE/USDT","2021-01-01", "doge"),
    ("AVAX/USDT","2021-01-01", "avax"),
    ("LINK/USDT","2021-01-01", "link"),
]

for symbol, since, name in assets:
    df = fetch_hourly(symbol, since, name)
    if not df.empty:
        path = RAW / f"{name}_1h.parquet"
        df.to_parquet(path)
        print(f"  Saved: {path}")
    time.sleep(1)

print("\nDone.")
