"""
Phase 1: XGBoost Regime Classifier
Features: 30d rolling mean of macro indicators
Labels: PELT breakpoints (early/late/post_etf)
Walk-forward validation
"""
import pandas as pd, numpy as np, json, pickle, warnings
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed/btc_by_regime.parquet"
RESULTS = ROOT / "results"
MODELS  = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_parquet(DATA).sort_index()
print(f"Data: {len(df)} rows | regimes: {df['regime'].value_counts().to_dict()}")

# ── Features: 30d rolling mean of macro indicators ───────────
FEATURE_COLS = [
    "nasdaq","dxy","eurusd","treasury_10y","vix",
    "btc_realized_vol_30d","sp500","gold","m2","cpi"
]
for col in FEATURE_COLS:
    df[f"{col}_roll30"] = df[col].rolling(30).mean()

roll_cols = [f"{c}_roll30" for c in FEATURE_COLS]
df_feat = df[roll_cols + ["regime"]].dropna()

X = df_feat[roll_cols].values
le = LabelEncoder()
y = le.fit_transform(df_feat["regime"].values)
dates = df_feat.index

print(f"Feature matrix: {X.shape}, classes: {le.classes_}")

# ── Walk-forward validation ───────────────────────────────────
TRAIN_W = 500  # days of training data
y_true_all, y_pred_all = [], []

for i in range(TRAIN_W, len(X) - 30, 30):  # step=30 for speed
    X_train, y_train = X[i-TRAIN_W:i], y[i-TRAIN_W:i]
    X_test,  y_test  = X[i:i+30],       y[i:i+30]
    # skip if only one class in train
    if len(np.unique(y_train)) < 2:
        continue
    mdl = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss",
                        verbosity=0)
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    y_true_all.extend(y_test.tolist())
    y_pred_all.extend(preds.tolist())

acc = accuracy_score(y_true_all, y_pred_all)
print(f"\nWalk-forward accuracy: {acc:.3f} (n={len(y_true_all)})")
print(classification_report(y_true_all, y_pred_all, target_names=le.classes_))

# ── Train final model on all data ────────────────────────────
final_mdl = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="mlogloss",
                          verbosity=0)
final_mdl.fit(X, y)
model_path = MODELS / "regime_classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": final_mdl, "encoder": le, "features": roll_cols}, f)
print(f"\nModel saved: {model_path}")

# ── Feature importance ────────────────────────────────────────
fi = dict(zip(roll_cols, final_mdl.feature_importances_))
fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
print("\nTop feature importances:")
for feat, imp in fi_sorted[:5]:
    print(f"  {feat}: {imp:.4f}")

# ── Save results ──────────────────────────────────────────────
res = {
    "walk_forward_accuracy": round(acc, 4),
    "n_predictions": len(y_true_all),
    "decision_gate": "PASS" if acc >= 0.70 else "FAIL",
    "feature_importance": {k: round(float(v), 4) for k, v in fi_sorted},
}
(RESULTS / "phase1_results.json").write_text(json.dumps(res, indent=2))

summary = f"""# Phase 1 Regime Classifier Results

## Walk-Forward Accuracy: {acc:.3f}
Decision gate (>70%): {"✅ PASS" if acc >= 0.70 else "❌ FAIL — refine features before Phase 2"}

## Top Features
""" + "\n".join(f"- {k}: {v:.4f}" for k, v in fi_sorted[:5])

(RESULTS / "phase1_summary.md").write_text(summary)
print(f"\nSummary: {RESULTS / 'phase1_summary.md'}")
