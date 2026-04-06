"""
构建小时级 TimesFM 训练集
窗口: 512小时上下文 → 预测48小时
Stride: 24小时
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
HOURLY = ROOT / "data/raw/hourly"
PROC   = ROOT / "data/processed"

REGIME_BREAKS = [pd.Timestamp("2020-11-11"), pd.Timestamp("2024-01-11")]
def label_regime(dt):
    d = dt.normalize() if hasattr(dt, 'normalize') else pd.Timestamp(dt).normalize()
    if d < REGIME_BREAKS[0]: return "early"
    elif d < REGIME_BREAKS[1]: return "late"
    else: return "post_etf"

assets = ["btc","eth","sol","bnb","doge","avax","link"]
all_samples = []

for asset in assets:
    path = HOURLY / f"{asset}_1h.parquet"
    if not path.exists(): continue
    df = pd.read_parquet(path)
    df.columns = [c.replace(f"{asset}_","") for c in df.columns]
    df.index = pd.to_datetime(df.index)

    ret = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
    vol  = pd.Series(ret).rolling(24).std().values * np.sqrt(8760)  # annualized
    dates = df.index

    WINDOW, STRIDE, HORIZON = 512, 24, 48
    n = 0
    for i in range(WINDOW, len(ret) - HORIZON, STRIDE):
        ctx = ret[i-WINDOW:i].tolist()
        fut = ret[i:i+HORIZON].tolist()
        if np.isnan(ctx).any() or np.isnan(fut).any(): continue
        # simple vol covariate (available hourly)
        cov_vol = vol[i-WINDOW:i+HORIZON]
        cov_vol = np.nan_to_num(cov_vol, nan=0.0).tolist()
        regime = label_regime(dates[i])
        all_samples.append({
            "asset": asset,
            "granularity": "1h",
            "end_datetime": str(dates[i]),
            "regime": regime,
            "context": ctx,
            "future": fut,
            "covariates": [[v] for v in cov_vol],
            "covariate_names": ["realized_vol_24h_z"],
        })
        n += 1
    print(f"  {asset}: {n} samples")

out = PROC / "hourly_train.jsonl"
with open(out,"w") as f:
    for s in all_samples:
        f.write(json.dumps(s) + "\n")

print(f"\n✅ Total hourly samples: {len(all_samples)} → {out}")
print("By asset:", dict(Counter(s["asset"] for s in all_samples)))
print("By regime:", dict(Counter(s["regime"] for s in all_samples)))
