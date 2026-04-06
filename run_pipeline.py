"""Main pipeline entry point: fetch, merge, engineer, label, export."""

import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_TFM = BASE_DIR / "data"
REPORTS = BASE_DIR / "reports"

for d in [DATA_RAW, DATA_PROC, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    # ── 1. Fetch BTC price ──
    step("1/7  Fetching BTC price data")
    from pipeline.fetch_btc import fetch_btc
    btc_df = fetch_btc()
    btc_df.to_parquet(DATA_RAW / "btc_price.parquet")
    print(f"  → {len(btc_df)} rows")

    # ── 2. Fetch macro data ──
    step("2/7  Fetching macro data")
    from pipeline.fetch_macro import fetch_macro
    macro_df = fetch_macro()
    macro_df.to_parquet(DATA_RAW / "macro.parquet")
    print(f"  → {len(macro_df)} rows, cols: {macro_df.columns.tolist()}")

    # ── 3. Fetch on-chain data ──
    step("3/7  Fetching on-chain data")
    from pipeline.fetch_onchain import fetch_onchain
    onchain_df = fetch_onchain()
    onchain_df.to_parquet(DATA_RAW / "onchain.parquet")
    print(f"  → {len(onchain_df)} rows, cols: {onchain_df.columns.tolist()}")

    # ── 4. Merge all data ──
    step("4/7  Merging datasets")
    # Join on date index, BTC is the base
    merged = btc_df.join(macro_df, how="left").join(onchain_df, how="left")
    # Forward-fill macro/onchain data (they may have gaps on weekends)
    merged = merged.ffill()
    print(f"  → {len(merged)} rows, {len(merged.columns)} columns")

    # ── 5. Feature engineering ──
    step("5/7  Feature engineering")
    from pipeline.feature_engineering import engineer_features
    full_df = engineer_features(merged)
    full_df.to_parquet(DATA_PROC / "btc_full.parquet")
    print(f"  → {len(full_df)} rows, {len(full_df.columns)} columns")

    # ── 6. Regime labeling ──
    step("6/7  Regime labeling")
    from pipeline.regime_labeling import label_regimes, regime_summary
    regime_df = label_regimes(full_df)
    regime_df.to_parquet(DATA_PROC / "btc_by_regime.parquet")
    summary = regime_summary(regime_df)
    print(summary)

    # ── 7. Build TimesFM dataset ──
    step("7/7  Building TimesFM dataset")
    from pipeline.build_timesfm_dataset import build_timesfm_dataset, save_jsonl
    examples = build_timesfm_dataset(regime_df, target_col="btc_close", stride=7)
    jsonl_path = DATA_TFM / "timesfm_train.jsonl"
    save_jsonl(examples, str(jsonl_path))
    print(f"  → {len(examples)} training examples written to {jsonl_path}")

    # ── Generate data quality report ──
    step("Generating data quality report")
    report = generate_report(btc_df, macro_df, onchain_df, full_df, regime_df, examples)
    report_path = REPORTS / "data_summary.md"
    report_path.write_text(report)
    print(f"  → Report written to {report_path}")

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")


def generate_report(btc_df, macro_df, onchain_df, full_df, regime_df, examples) -> str:
    """Generate markdown data quality report."""
    lines = [
        "# BTC Regime Data Pipeline — Quality Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Raw Data Summary",
        "",
        "| Source | Rows | Date Range | Columns |",
        "|--------|------|------------|---------|",
        f"| BTC Price | {len(btc_df)} | {btc_df.index.min().date()} → {btc_df.index.max().date()} | {', '.join(btc_df.columns)} |",
        f"| Macro | {len(macro_df)} | {macro_df.index.min().date()} → {macro_df.index.max().date()} | {', '.join(macro_df.columns)} |",
        f"| On-chain | {len(onchain_df)} | {onchain_df.index.min().date()} → {onchain_df.index.max().date()} | {', '.join(onchain_df.columns)} |",
        "",
        "## Processed Dataset",
        "",
        f"- **Total rows**: {len(full_df)}",
        f"- **Total columns**: {len(full_df.columns)}",
        f"- **Date range**: {full_df.index.min().date()} → {full_df.index.max().date()}",
        "",
        "### Missing Values (top 10)",
        "",
        "| Column | Missing | % |",
        "|--------|---------|---|",
    ]

    missing = full_df.isna().sum().sort_values(ascending=False).head(10)
    for col, count in missing.items():
        pct = count / len(full_df) * 100
        lines.append(f"| {col} | {count} | {pct:.1f}% |")

    lines += [
        "",
        "## Regime Distribution",
        "",
        "| Regime | Start | End | Days |",
        "|--------|-------|-----|------|",
    ]

    from pipeline.regime_labeling import regime_summary
    summary = regime_summary(regime_df)
    for regime_name, row in summary.iterrows():
        lines.append(
            f"| {regime_name} | {row['start_date'].date()} | {row['end_date'].date()} | {row['n_days']} |"
        )

    lines += [
        "",
        "## TimesFM Dataset",
        "",
        f"- **Training examples**: {len(examples)}",
        f"- **Window size**: 512 days",
        f"- **Stride**: 7 days",
    ]

    if examples:
        lines += [
            f"- **Covariates per example**: {len(examples[0]['xreg'])}",
            f"- **Covariate names**: {', '.join(sorted(examples[0]['xreg'].keys())[:15])}{'...' if len(examples[0]['xreg']) > 15 else ''}",
        ]

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
