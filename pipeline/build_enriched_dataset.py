"""
重建训练集：加入所有新特征
新增协变量（共 ~35个）：
- 原有15个宏观+链上
- Fear & Greed (2020+)
- BTC: momentum_5d/20d, vol_7d/60d, rsi_14, price_vs_ma50/200, active_addresses, exchange_volume
- 跨资产：eth_close_z, sol_close_z, bnb_close_z（对BTC样本）
"""
import pandas as pd, numpy as np, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data/raw"
PROC = ROOT / "data/processed"

# Load all data
macro   = pd.read_parquet(RAW / "macro.parquet")
onchain = pd.read_parquet(RAW / "onchain.parquet")
fg      = pd.read_parquet(RAW / "fear_greed.parquet") if (RAW/"fear_greed.parquet").exists() else pd.DataFrame()
addr    = pd.read_parquet(RAW / "active_addresses.parquet") if (RAW/"active_addresses.parquet").exists() else pd.DataFrame()
exvol   = pd.read_parquet(RAW / "exchange_volume.parquet") if (RAW/"exchange_volume.parquet").exists() else pd.DataFrame()

# Cross-asset closes for correlation features
btc_feat = pd.read_parquet(RAW / "btc_features.parquet") if (RAW/"btc_features.parquet").exists() else pd.DataFrame()
eth_feat = pd.read_parquet(RAW / "eth_features.parquet") if (RAW/"eth_features.parquet").exists() else pd.DataFrame()
sol_feat = pd.read_parquet(RAW / "sol_features.parquet") if (RAW/"sol_features.parquet").exists() else pd.DataFrame()
bnb_feat = pd.read_parquet(RAW / "bnb_features.parquet") if (RAW/"bnb_features.parquet").exists() else pd.DataFrame()

REGIME_BREAKS = [pd.Timestamp("2020-11-11"), pd.Timestamp("2024-01-11")]
def label_regime(date):
    if date < REGIME_BREAKS[0]: return "early"
    elif date < REGIME_BREAKS[1]: return "late"
    else: return "post_etf"

BASE_COV = ["sp500","nasdaq","gold","crude_oil","cnyusd","eurusd","jpyusd",
            "dxy","treasury_10y","m2","vix","cpi",
            "hash_rate","tx_count","mining_difficulty"]

DERIVED_COLS = ["momentum_5d","momentum_20d","vol_7d","vol_60d","rsi_14",
                "price_vs_ma50","price_vs_ma200"]

ASSETS = {
    "btc": ("btc_price.parquet", "2016-01-01"),
    "eth": ("eth_price.parquet", "2016-01-01"),
    "sol": ("sol_price.parquet", "2020-04-01"),
    "bnb": ("bnb_price.parquet", "2017-11-01"),
}

all_samples = []

for asset, (price_file, start) in ASSETS.items():
    print(f"\n[{asset.upper()}]")
    price_df = pd.read_parquet(RAW / price_file)
    price_df.columns = [c.replace(f"{asset}_","") for c in price_df.columns]
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df[price_df.index >= start].sort_index()

    # returns
    price_df["daily_return"] = np.log(price_df["close"] / price_df["close"].shift(1))
    price_df["vol_30d"]      = price_df["daily_return"].rolling(30).std() * np.sqrt(365)
    price_df["etf_flag"]     = (price_df.index >= "2024-01-11").astype(float)

    # merge all extras
    df = price_df.join(macro, how="left").join(onchain, how="left")
    if not fg.empty:    df = df.join(fg, how="left")
    if not addr.empty:  df = df.join(addr, how="left")
    if not exvol.empty: df = df.join(exvol, how="left")

    # add derived features for this asset
    feat_path = RAW / f"{asset}_features.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        feat_cols = [c for c in feat.columns if any(d in c for d in DERIVED_COLS)]
        feat_cols = [c for c in feat_cols if c in feat.columns]
        feat_renamed = feat[feat_cols].rename(columns={c: c.replace(f"{asset}_","") for c in feat_cols})
        df = df.join(feat_renamed, how="left")

    # cross-asset closes
    for other_asset, other_feat in [("eth",eth_feat),("sol",sol_feat),("bnb",bnb_feat),("btc",btc_feat)]:
        if other_asset == asset: continue
        col = f"{other_asset}_close"
        if col in other_feat.columns:
            df = df.join(other_feat[[col]].rename(columns={col: f"x_{other_asset}_close"}), how="left")

    df["regime"] = df.index.map(label_regime)

    # identify all covariate columns
    exclude = {"daily_return","vol_30d","etf_flag","regime","open","high","low","close","volume"}
    all_cov_raw = [c for c in df.columns if c not in exclude]

    # z-score + lag everything
    z_lag_cols = []
    for col in all_cov_raw:
        if df[col].dtype == object: continue
        roll_mean = df[col].rolling(252, min_periods=60).mean()
        roll_std  = df[col].rolling(252, min_periods=60).std().replace(0, np.nan).fillna(1)
        z = (df[col] - roll_mean) / roll_std
        z = z.clip(-5, 5)
        colname = f"{col}_z_lag7"
        df[colname] = z.shift(7)
        z_lag_cols.append(colname)

    df = df.dropna(subset=["daily_return","vol_30d"])
    # drop cols with too many NaN
    valid_covs = [c for c in z_lag_cols if df[c].notna().mean() > 0.5]
    df_valid = df.dropna(subset=valid_covs[:10])  # need at least first 10 cols

    print(f"  Rows: {len(df_valid)}, Covariates: {len(valid_covs)}")

    WINDOW, STRIDE = 512, 7
    prices  = df_valid["daily_return"].values
    covs    = df_valid[valid_covs].fillna(0).values
    dates   = df_valid.index
    regimes = df_valid["regime"].values

    n = 0
    for i in range(WINDOW, len(prices) - 30, STRIDE):
        ctx = prices[i-WINDOW:i].tolist()
        fut = prices[i:i+30].tolist()
        cov = covs[i-WINDOW:i+30].tolist()
        dominant = pd.Series(regimes[i-WINDOW:i]).value_counts().idxmax()
        if any(np.isnan(ctx)) or any(np.isnan(fut)): continue
        all_samples.append({
            "asset": asset,
            "end_date": str(dates[i].date()),
            "regime": dominant,
            "context": ctx,
            "future": fut,
            "covariates": cov,
            "covariate_names": valid_covs,
        })
        n += 1
    print(f"  Samples: {n}")

out = PROC / "multi_asset_enriched.jsonl"
with open(out,"w") as f:
    for s in all_samples:
        f.write(json.dumps(s) + "\n")

print(f"\n✅ Total: {len(all_samples)} samples → {out}")
from collections import Counter
print("By asset:", dict(Counter(s["asset"] for s in all_samples)))
print("By regime:", dict(Counter(s["regime"] for s in all_samples)))
print(f"Covariate dims: {len(all_samples[0]['covariate_names'])} per sample")
print("Sample covariates:", all_samples[0]['covariate_names'][:8], "...")
