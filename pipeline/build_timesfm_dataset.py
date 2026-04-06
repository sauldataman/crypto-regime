"""Build TimesFM fine-tuning dataset with sliding window approach."""

import json

import numpy as np
import pandas as pd

WINDOW_SIZE = 512  # days per training example
STRIDE = 1  # sliding window stride


def build_timesfm_dataset(
    df: pd.DataFrame,
    target_col: str = "btc_close",
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[dict]:
    """Create TimesFM fine-tune examples using sliding windows.

    Each example contains:
    - target: list of target values (btc_close z-scored)
    - xreg: dict of covariate arrays (all other numeric columns)
    - dates: list of date strings
    - regime: regime label at window end
    - metadata: window position info
    """
    # Use only rows without NaN in target
    valid_df = df.dropna(subset=[target_col])
    numeric_cols = [
        c for c in valid_df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    examples = []
    n_rows = len(valid_df)

    for start_idx in range(0, n_rows - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = valid_df.iloc[start_idx:end_idx]

        # Skip windows with too many NaNs in covariates (>20%)
        nan_ratio = window[numeric_cols].isna().mean().mean() if numeric_cols else 0
        if nan_ratio > 0.2:
            continue

        target_values = window[target_col].tolist()
        dates = window.index.strftime("%Y-%m-%d").tolist()

        # Build covariate dict — fill remaining NaNs with 0 for JSON
        xreg = {}
        for col in numeric_cols:
            xreg[col] = window[col].fillna(0).tolist()

        regime = window["regime"].iloc[-1] if "regime" in window.columns else "unknown"

        example = {
            "target": target_values,
            "dates": dates,
            "xreg": xreg,
            "regime": str(regime),
            "metadata": {
                "start_date": dates[0],
                "end_date": dates[-1],
                "window_size": window_size,
            },
        }
        examples.append(example)

    return examples


def save_jsonl(examples: list[dict], path: str) -> None:
    """Write examples to JSONL file."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    # Quick test
    dates = pd.date_range("2016-01-01", periods=600, freq="D")
    test_df = pd.DataFrame({
        "btc_close": np.random.randn(600).cumsum() + 30000,
        "sp500": np.random.randn(600),
        "regime": "early",
    }, index=dates)
    test_df.index.name = "date"
    examples = build_timesfm_dataset(test_df, window_size=512, stride=30)
    print(f"Generated {len(examples)} examples")
    if examples:
        print(f"First example keys: {examples[0].keys()}")
        print(f"Target length: {len(examples[0]['target'])}")
        print(f"XReg keys: {list(examples[0]['xreg'].keys())}")
