"""
Build multi-asset TimesFM training dataset
Assets: BTC + ETH + SOL + BNB, each with regime-specific covariates
"""
import pandas as pd, numpy as np, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data/raw"
PROC = ROOT / "data/processed"
PROC.mkdir(exist_ok=True)

# Load shared macro + on-chain data (already built)
macro_df = pd.read_parquet(RAW / "macro.parquet")
onchain_df = pd.read_parquet(RAW / "onchain.parquet")

# Regime labels from BTC breakpoints
REGIME_BREAKS = [pd.Timestamp("2020-11-11"), pd.Timestamp("2024-01-11")]
def label_regime(date):
    if date < REGIME_BREAKS[0]: return "early"
    elif date < REGIME_BREAKS[1]: return "late"
    else: return "post_etf"

ASSETS = {
    "btc": ("data/raw/btc_price.parquet", "2016-01-01"),
    "eth": ("data/raw/eth_price.parquet", "2016-01-01"),
    "sol": ("data/raw/sol_price.parquet", "2020-04-01"),
    "bnb": ("data/raw/bnb_price.parquet", "2017-11-01"),
}

COVARIATE_COLS = [
    "sp500","nasdaq","gold","crude_oil","cnyusd","eurusd","jpyusd",
    "dxy","treasury_10y","m2","vix","cpi",
    "hash_rate","tx_count","mining_difficulty"
]

all_samples = []

for asset, (price_path, start) in ASSETS.items():
    print(f"\n[{asset.upper()}] Loading from {start}...")
    price_df = pd.read_parquet(ROOT / price_path)
    # normalize column names
    price_df.columns = [c.replace(f"{asset}_","") for c in price_df.columns]
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df[price_df.index >= start].sort_index()

    # compute returns and vol
    price_df["daily_return"] = np.log(price_df["close"] / price_df["close"].shift(1))
    price_df["vol_30d"] = price_df["daily_return"].rolling(30).std() * np.sqrt(365)
    price_df["etf_flag"] = (price_df.index >= "2024-01-11").astype(float)

    # merge macro + onchain
    df = price_df.join(macro_df, how="left").join(onchain_df, how="left")
    df["regime"] = df.index.map(label_regime)

    # z-score + 7-day lag on covariates
    cov_cols = [c for c in COVARIATE_COLS if c in df.columns]
    for col in cov_cols:
        roll_mean = df[col].rolling(252).mean()
        roll_std  = df[col].rolling(252).std().replace(0, 1)
        df[f"{col}_z"] = (df[col] - roll_mean) / roll_std
        df[f"{col}_z_lag7"] = df[f"{col}_z"].shift(7)

    df = df.dropna(subset=["daily_return","vol_30d"])
    z_lag_cols = [c for c in df.columns if c.endswith("_z_lag7")]
    
    # sliding windows
    WINDOW, STRIDE = 512, 7
    prices = df["daily_return"].values
    covs   = df[z_lag_cols].values
    dates  = df.index
    regimes= df["regime"].values

    n_samples = 0
    for i in range(WINDOW, len(prices) - 30, STRIDE):
        context = prices[i-WINDOW:i].tolist()
        future  = prices[i:i+30].tolist()
        cov_window = covs[i-WINDOW:i+30].tolist()
        dominant_regime = pd.Series(regimes[i-WINDOW:i]).value_counts().idxmax()
        
        if any(np.isnan(context)) or any(np.isnan(future)):
            continue
        
        sample = {
            "asset": asset,
            "end_date": str(dates[i].date()),
            "regime": dominant_regime,
            "context": context,
            "future": future,
            "covariates": cov_window,
            "covariate_names": z_lag_cols,
        }
        all_samples.append(sample)
        n_samples += 1

    print(f"  {n_samples} samples, {len(df)} rows, regime dist: {df['regime'].value_counts().to_dict()}")

# Save
out_path = PROC / "multi_asset_train.jsonl"
with open(out_path, "w") as f:
    for s in all_samples:
        f.write(json.dumps(s) + "\n")

print(f"\n✅ Total: {len(all_samples)} samples → {out_path}")

# Stats by asset and regime
import collections
by_asset  = collections.Counter(s["asset"] for s in all_samples)
by_regime = collections.Counter(s["regime"] for s in all_samples)
print("\nBy asset:", dict(by_asset))
print("By regime:", dict(by_regime))
