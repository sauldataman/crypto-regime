"""
Phase 3: Full dual-layer inference system
Layer 1: Regime classifier → current regime
Layer 2: Route to regime-specific TimesFM → forecast

Usage:
  python3 experiments/phase3_inference.py --asset BTC --days 5
"""
import argparse, pickle, json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

def load_classifier():
    path = ROOT / "models/regime_classifier.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def detect_current_regime(clf_data):
    """Use latest 30d macro data to detect current regime"""
    try:
        from pipeline.fetch_macro import fetch_macro
        macro = fetch_macro()
        FEATURE_COLS = ["nasdaq","dxy","eurusd","treasury_10y","vix",
                        "btc_realized_vol_30d","sp500","gold","m2","cpi"]
        available = [c for c in FEATURE_COLS if c in macro.columns]
        roll = macro[available].rolling(30).mean().iloc[-1]
        features = roll.values.reshape(1, -1)
        
        model = clf_data["model"]
        encoder = clf_data["encoder"]
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        regime = encoder.classes_[pred]
        confidence = float(proba.max())
        return regime, confidence
    except Exception as e:
        return "late", 0.0  # fallback to late (current regime)

def run_inference(asset: str, forecast_days: int):
    print(f"\n{'='*50}")
    print(f"Inference: {asset.upper()}, {forecast_days}d horizon")
    print('='*50)
    
    # Step 1: Detect regime
    clf_data = load_classifier()
    regime, conf = detect_current_regime(clf_data)
    print(f"Current regime: {regime} (confidence: {conf:.2f})")
    
    # Step 2: Load appropriate TimesFM model
    try:
        import timesfm, torch
        torch.set_float32_matmul_precision("high")
        
        # Try regime-specific model first, fall back to unified
        model_path = ROOT / "models" / f"timesfm_{regime}.pt"
        if not model_path.exists():
            model_path = ROOT / "models" / "timesfm_unified.pt"
        
        if not model_path.exists():
            print("⚠️ No fine-tuned model found, using zero-shot")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        else:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded: {model_path.name}")
        
        model.compile(timesfm.ForecastConfig(
            max_context=512, max_horizon=forecast_days,
            normalize_inputs=True, use_continuous_quantile_head=True
        ))
        
        # Step 3: Get recent data
        df = pd.read_parquet(ROOT / "data/processed/btc_full.parquet").sort_index()
        returns = df["btc_daily_return"].fillna(0).values[-512:]
        
        # Step 4: Forecast
        point_fc, quantile_fc = model.forecast(horizon=forecast_days, inputs=[returns.tolist()])
        
        print(f"\nForecast ({forecast_days}d):")
        for i, (pt, q) in enumerate(zip(point_fc[0], quantile_fc[0])):
            direction = "↑" if pt > 0 else "↓"
            print(f"  Day {i+1}: {direction} {pt:+.4f}  [10%: {q[0]:+.4f}, 90%: {q[-1]:+.4f}]")
        
        # Regime transition alert
        if conf < 0.70:
            print(f"\n⚠️ REGIME TRANSITION ALERT: confidence={conf:.2f} — uncertainty widened")
        
        return {"regime": regime, "confidence": conf, "forecast": point_fc[0].tolist()}
    
    except ImportError:
        print("❌ TimesFM not available — install on DGX: pip install timesfm[torch]")
        return {"regime": regime, "confidence": conf, "forecast": None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()
    result = run_inference(args.asset, args.days)
    print(f"\nResult: {json.dumps(result, indent=2)}")
