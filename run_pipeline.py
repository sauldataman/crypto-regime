"""Main pipeline entry point: fetch, merge, engineer, PELT regime detection, label, export."""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
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
    step("1/8  Fetching BTC price data")
    from pipeline.fetch_btc import fetch_btc
    btc_df = fetch_btc()
    btc_df.to_parquet(DATA_RAW / "btc_price.parquet")
    print(f"  → {len(btc_df)} rows")

    # ── 2. Fetch macro data ──
    step("2/8  Fetching macro data")
    from pipeline.fetch_macro import fetch_macro
    macro_df = fetch_macro()
    macro_df.to_parquet(DATA_RAW / "macro.parquet")
    print(f"  → {len(macro_df)} rows, cols: {macro_df.columns.tolist()}")

    # ── 3. Fetch on-chain data ──
    step("3/8  Fetching on-chain data")
    from pipeline.fetch_onchain import fetch_onchain
    onchain_df = fetch_onchain()
    onchain_df.to_parquet(DATA_RAW / "onchain.parquet")
    print(f"  → {len(onchain_df)} rows, cols: {onchain_df.columns.tolist()}")

    # ── 4. Merge all data ──
    step("4/8  Merging datasets")
    # Join on date index, BTC is the base
    merged = btc_df.join(macro_df, how="left").join(onchain_df, how="left")
    # Forward-fill macro/onchain data (they may have gaps on weekends)
    merged = merged.ffill()
    print(f"  → {len(merged)} rows, {len(merged.columns)} columns")

    # ── 5. Feature engineering ──
    step("5/8  Feature engineering")
    from pipeline.feature_engineering import engineer_features
    full_df = engineer_features(merged)
    full_df.to_parquet(DATA_PROC / "btc_full.parquet")
    print(f"  → {len(full_df)} rows, {len(full_df.columns)} columns")

    # ── 6. PELT regime detection ──
    step("6/8  PELT regime detection")
    from pipeline.regime_detection import detect_regimes, run_all_assets, label_from_breakpoints
    # Run PELT on BTC daily returns
    btc_returns = full_df["btc_daily_return"].dropna()
    asset_returns = {"btc": btc_returns}
    # If multi-asset price files exist, add them
    for asset_name in ["eth", "sol", "bnb"]:
        asset_path = DATA_RAW / f"{asset_name}_price.parquet"
        if asset_path.exists():
            try:
                adf = pd.read_parquet(asset_path)
                close_col = [c for c in adf.columns if "close" in c.lower()]
                if close_col:
                    adf_returns = np.log(adf[close_col[0]] / adf[close_col[0]].shift(1)).dropna()
                    adf_returns.index = pd.to_datetime(adf_returns.index)
                    asset_returns[asset_name] = adf_returns
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to load {asset_name} for PELT: {e}")
    pelt_result = run_all_assets(asset_returns)
    bp_path = DATA_PROC / "regime_breakpoints.json"
    with open(bp_path, "w") as f:
        json.dump(pelt_result, f, indent=2)
    print(f"  → PELT breakpoints saved to {bp_path}")
    for asset, bps in pelt_result["breakpoints"].items():
        print(f"    {asset}: {len(bps)} breakpoints")

    # Apply PELT labels to BTC data
    btc_bp_dates = [pd.Timestamp(d) for d in pelt_result["breakpoints"].get("btc", [])]
    full_df["regime"] = label_from_breakpoints(full_df.index, btc_bp_dates)

    # ── 7. Regime labeling (legacy — for backward compat) ──
    step("7/8  Regime labeling")
    from pipeline.regime_labeling import label_regimes, regime_summary
    regime_df = label_regimes(full_df)
    regime_df.to_parquet(DATA_PROC / "btc_by_regime.parquet")
    summary = regime_summary(regime_df)
    print(summary)

    # ── 8. Generate data quality report ──
    step("8/8  Generating data quality report")
    report = generate_report(btc_df, macro_df, onchain_df, full_df, regime_df)
    report_path = REPORTS / "data_summary.md"
    report_path.write_text(report)
    print(f"  → Report written to {report_path}")

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")


def generate_report(btc_df, macro_df, onchain_df, full_df, regime_df) -> str:
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
        "## PELT Regime Detection",
        "",
    ]
    bp_path = BASE_DIR / "data" / "processed" / "regime_breakpoints.json"
    if bp_path.exists():
        try:
            with open(bp_path) as f:
                bp_data = json.load(f)
            for asset, bps in bp_data.get("breakpoints", {}).items():
                lines.append(f"- **{asset}**: {len(bps)} breakpoints")
        except (json.JSONDecodeError, KeyError) as e:
            lines.append(f"- Failed to read breakpoints: {e}")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
