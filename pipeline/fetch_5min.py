"""5分钟K线，11个资产，2021-2026"""
import ccxt, pandas as pd, time
from pathlib import Path

RAW = Path(__file__).parent.parent / "data/raw/5min"
RAW.mkdir(parents=True, exist_ok=True)

assets = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT",
    "DOGE/USDT","AVAX/USDT","LINK/USDT",
    "MATIC/USDT","XRP/USDT","ADA/USDT","DOT/USDT"
]

exchange = ccxt.binance({"enableRateLimit": True})
SINCE = exchange.parse8601("2021-01-01T00:00:00Z")

for symbol in assets:
    name = symbol.replace("/USDT","").lower()
    path = RAW / f"{name}_5m.parquet"
    if path.exists():
        print(f"  {name}: already exists, skip")
        continue
    
    print(f"Fetching {name} 5m...", flush=True)
    ohlcv = []
    since = SINCE
    while True:
        batch = exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
        if not batch: break
        ohlcv.extend(batch)
        since = batch[-1][0] + 300_000  # +5min
        if since > exchange.milliseconds(): break
        time.sleep(0.05)
    
    if not ohlcv:
        print(f"  {name}: no data"); continue
    
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").drop(columns=["ts"])
    df.to_parquet(path)
    print(f"  {name}: {len(df):,} rows → {path.name}", flush=True)
    time.sleep(0.5)

print("Done.")
