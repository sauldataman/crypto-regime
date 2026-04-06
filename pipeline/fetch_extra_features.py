"""
补充额外特征:
1. 资产自身衍生：vol_30d, momentum_5d/20d, RSI_14
2. 跨资产价格（ETH作为BTC协变量，反之亦然）
3. Fear & Greed Index（Alternative.me API，免费无key）
4. BTC funding rate（Coinglass API，免费）
5. 链上：活跃地址、exchange netflow（blockchain.com，免费）
"""
import requests, pandas as pd, numpy as np, time
from pathlib import Path

RAW = Path(__file__).parent.parent / "data/raw"
RAW.mkdir(exist_ok=True)

# ── Fear & Greed Index ────────────────────────────────────────
def fetch_fear_greed():
    print("Fetching Fear & Greed Index...")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=2000&format=json", timeout=15)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
        df = df.set_index("date")[["value"]].astype(float)
        df.columns = ["fear_greed"]
        df = df.sort_index()
        print(f"  → {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  ❌ {e}")
        return pd.DataFrame()

# ── Blockchain.com: Active Addresses ─────────────────────────
def fetch_blockchain_chart(chart_name, col_name):
    print(f"Fetching blockchain.com: {chart_name}...")
    try:
        url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json&sampled=true"
        r = requests.get(url, timeout=30)
        vals = r.json()["values"]
        df = pd.DataFrame(vals)
        df["date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
        df = df.set_index("date")[["y"]].rename(columns={"y": col_name})
        df = df.sort_index()
        print(f"  → {len(df)} rows")
        return df
    except Exception as e:
        print(f"  ❌ {e}")
        return pd.DataFrame()

# ── Compute derived features per asset ────────────────────────
def add_derived_features(df, price_col="close"):
    """Add momentum, RSI, vol to a price dataframe"""
    ret = np.log(df[price_col] / df[price_col].shift(1))
    df["momentum_5d"]  = ret.rolling(5).mean()
    df["momentum_20d"] = ret.rolling(20).mean()
    df["vol_7d"]       = ret.rolling(7).std() * np.sqrt(365)
    df["vol_60d"]      = ret.rolling(60).std() * np.sqrt(365)
    # RSI-14
    delta = df[price_col].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # Price relative to 50/200d MA
    df["price_vs_ma50"]  = df[price_col] / df[price_col].rolling(50).mean() - 1
    df["price_vs_ma200"] = df[price_col] / df[price_col].rolling(200).mean() - 1
    return df

if __name__ == "__main__":
    # 1. Fear & Greed
    fg = fetch_fear_greed()
    if not fg.empty:
        fg.to_parquet(RAW / "fear_greed.parquet")
        print(f"  Saved fear_greed.parquet")

    # 2. Active addresses
    time.sleep(1)
    addr = fetch_blockchain_chart("n-unique-addresses", "active_addresses")
    if not addr.empty:
        addr.to_parquet(RAW / "active_addresses.parquet")

    # 3. Exchange flow (net flow = inflow - outflow proxy via exchange volume)
    time.sleep(1)
    exvol = fetch_blockchain_chart("trade-volume", "exchange_volume_usd")
    if not exvol.empty:
        exvol.to_parquet(RAW / "exchange_volume.parquet")

    # 4. Add derived features to each asset
    import ccxt
    for asset, price_file in [("btc","btc_price.parquet"),
                               ("eth","eth_price.parquet"),
                               ("sol","sol_price.parquet"),
                               ("bnb","bnb_price.parquet")]:
        path = RAW / price_file
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df.columns = [c.replace(f"{asset}_","") for c in df.columns]
        df = add_derived_features(df, "close")
        df.columns = [f"{asset}_{c}" if not c.startswith(asset) else c for c in df.columns]
        df.to_parquet(RAW / f"{asset}_features.parquet")
        print(f"  {asset} derived features: {[c for c in df.columns if not c.endswith('close')][:5]}...")

    print("\nDone.")
