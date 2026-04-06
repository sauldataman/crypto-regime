"""
Phase 0: Baseline Experiments
AR(1), ARIMA, TimesFM zero-shot, TimesFM+XReg
Walk-forward: 252d train, 1/5/30d horizon, 1d step
"""
import pandas as pd, numpy as np, json, time, warnings
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/btc_full.parquet"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_parquet(DATA).sort_index().dropna(subset=["btc_daily_return"])
returns = df["btc_daily_return"].values
dates   = df.index

TRAIN_W = 252
HORIZONS = [1, 5, 30]
COVARIATES = [c for c in df.columns if c.endswith("_lag7") and c != "btc_close_lag7"]

print(f"Data: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
print(f"Covariates: {len(COVARIATES)}")

def direction_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def evaluate(name, preds_by_h):
    """preds_by_h: {horizon: [(y_true, y_pred), ...]}"""
    out = {}
    for h, pairs in preds_by_h.items():
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        out[h] = {
            "dir_acc": round(direction_accuracy(yt, yp), 4),
            "mae":     round(float(np.mean(np.abs(yt - yp))), 6),
            "rmse":    round(float(np.sqrt(np.mean((yt - yp)**2))), 6),
            "n":       len(pairs),
        }
        print(f"  {name} h={h:2d}: dir={out[h]['dir_acc']:.3f}  mae={out[h]['mae']:.5f}")
    return out

results = {}

# ── 0.1 AR(1) ────────────────────────────────────────────────
print("\n[0.1] AR(1)")
preds = {h: [] for h in HORIZONS}
for i in range(TRAIN_W, len(returns) - max(HORIZONS)):
    train = returns[i-TRAIN_W:i]
    try:
        mdl = AutoReg(train, lags=1, old_names=False).fit(disp=False)
        for h in HORIZONS:
            forecast = mdl.forecast(h)[-1]
            preds[h].append((returns[i+h-1], forecast))
    except: pass
results["ar1"] = evaluate("AR(1)", preds)

# ── 0.2 ARIMA ────────────────────────────────────────────────
print("\n[0.2] ARIMA (auto) — subsampled every 5d for speed")
try:
    from pmdarima import auto_arima
    preds = {h: [] for h in HORIZONS}
    for i in range(TRAIN_W, len(returns) - max(HORIZONS), 5):  # step=5 for speed
        train = returns[i-TRAIN_W:i]
        try:
            mdl = auto_arima(train, max_p=3, max_q=3, seasonal=False, suppress_warnings=True, error_action="ignore")
            fc = mdl.predict(max(HORIZONS))
            for h in HORIZONS:
                preds[h].append((returns[i+h-1], fc[h-1]))
        except: pass
    results["arima"] = evaluate("ARIMA", preds)
except ImportError:
    print("  pmdarima not installed, skipping")
    results["arima"] = "skipped"

# ── 0.3 TimesFM zero-shot ─────────────────────────────────────
print("\n[0.3] TimesFM zero-shot")
try:
    import timesfm, torch
    torch.set_float32_matmul_precision("high")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=30, normalize_inputs=True))

    preds = {h: [] for h in HORIZONS}
    step = 10  # subsample for speed
    for i in range(TRAIN_W, len(returns) - max(HORIZONS), step):
        ctx = returns[i-TRAIN_W:i].tolist()
        try:
            pt, _ = tfm.forecast(horizon=30, inputs=[ctx])
            fc = pt[0]
            for h in HORIZONS:
                preds[h].append((returns[i+h-1], float(fc[h-1])))
        except: pass
    results["timesfm_zeroshot"] = evaluate("TimesFM zero-shot", preds)
except Exception as e:
    print(f"  TimesFM not available: {e}")
    results["timesfm_zeroshot"] = f"error: {e}"

# ── 0.4 TimesFM + XReg ───────────────────────────────────────
print("\n[0.4] TimesFM + XReg")
try:
    import timesfm
    # XReg requires JAX — check availability
    import jax
    preds = {h: [] for h in HORIZONS}
    step = 10
    cov_data = df[COVARIATES].values
    for i in range(TRAIN_W, len(returns) - max(HORIZONS), step):
        ctx = returns[i-TRAIN_W:i].tolist()
        cov_future = cov_data[i:i+30].tolist()
        cov_hist   = cov_data[i-TRAIN_W:i].tolist()
        try:
            pt, _ = tfm.forecast_with_covariates(
                horizon=30, inputs=[ctx],
                dynamic_numerical_covariates={c: [[v[j] for v in cov_hist] + [cov_future[k][j] for k in range(30)]]
                                               for j, c in enumerate(COVARIATES)},
                covariate_mode="timesfm+xreg"
            )
            fc = pt[0]
            for h in HORIZONS:
                preds[h].append((returns[i+h-1], float(fc[h-1])))
        except: pass
    results["timesfm_xreg"] = evaluate("TimesFM+XReg", preds)
except Exception as e:
    print(f"  XReg not available (needs JAX): {e}")
    results["timesfm_xreg"] = f"error: {e}"

# ── Save results ──────────────────────────────────────────────
out_path = RESULTS / "phase0_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# ── Summary ───────────────────────────────────────────────────
summary = ["# Phase 0 Baseline Results\n"]
summary.append(f"Data: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}\n\n")
summary.append("## Direction Accuracy (1d / 5d / 30d)\n\n")
summary.append("| Model | 1d | 5d | 30d | Decision |\n")
summary.append("|-------|----|----|-----|----------|\n")
for name, res in results.items():
    if isinstance(res, dict):
        d1 = res.get(1,{}).get("dir_acc","?")
        d5 = res.get(5,{}).get("dir_acc","?")
        d30= res.get(30,{}).get("dir_acc","?")
        gate = "✅ beat random" if isinstance(d1,float) and d1 > 0.52 else "⚠️ near random"
        summary.append(f"| {name} | {d1} | {d5} | {d30} | {gate} |\n")
    else:
        summary.append(f"| {name} | — | — | — | {res} |\n")

summary_path = RESULTS / "phase0_summary.md"
summary_path.write_text("".join(summary))
print(f"Summary: {summary_path}")
