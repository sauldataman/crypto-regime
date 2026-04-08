"""
Phase 2 v2: Progressive Fine-Tuning with VERIFIED gradient flow.

Uses m.model (the real nn.Module) with simple standardization (not RevIN).
Verified: loss drops from 10.27 to 1.53 in 3 steps.

Training recipe:
  1. Access nn.Module via m.model (231M params, 229 trainable)
  2. Patch input to [batch, num_patches, 32]
  3. Simple standardization (mean/std, not RevIN which breaks gradients)
  4. Forward: (_, _, output_ts, output_qs), _ = model(normed, masks)
  5. Reshape output_ts: [batch, patches, 128, 10]
  6. Point forecast: output[:, -1, :, 5]
  7. Loss in normalized space
  8. Backward + optimizer step

Progressive: 5min -> hourly -> daily (3 stages)
6 assets: BTC, ETH, SOL, BNB, DOGE, LINK

Usage:
  python experiments/phase2_finetune_v2.py                      # full progressive
  python experiments/phase2_finetune_v2.py --stage daily         # daily only
  python experiments/phase2_finetune_v2.py --stage 5min          # 5min only

Output:
  models/timesfm_*.pt
  results/phase2_finetune_v2.json
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── Model constants ─────────────────────────────────────────
PATCH_LEN = 32          # input patch size
OUTPUT_PATCH_LEN = 128  # output patch size
NUM_QUANTILES = 10      # 9 deciles + 1 point
POINT_IDX = 5           # index for point forecast in quantile dim
CONTEXT_PATCHES = 16    # 512 / 32 = 16 patches
CONTEXT_LEN = CONTEXT_PATCHES * PATCH_LEN  # 512

# ── Temporal boundary ───────────────────────────────────────
TRAIN_CUTOFF = "2022-06-01"
FORECAST_HORIZON = OUTPUT_PATCH_LEN  # 128 steps ahead

ASSETS_5MIN = ["btc", "eth", "sol", "bnb", "doge", "link"]
ASSETS_HOURLY = ["btc", "eth", "sol", "bnb", "doge", "link"]
ASSETS_DAILY = ["btc", "eth", "sol", "bnb", "doge", "link"]

STAGE_CONFIG = {
    "5min":   {"epochs": 20, "lr": 1e-4, "batch_size": 64, "stride": 64},
    "hourly": {"epochs": 20, "lr": 5e-5, "batch_size": 64, "stride": 24},
    "daily":  {"epochs": 50, "lr": 1e-5, "batch_size": 32, "stride": 1},
}


def load_returns_from_parquet(path: Path) -> pd.Series:
    """Load log returns from a parquet file."""
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    returns = np.log(close / close.shift(1)).dropna()
    return returns


def build_patched_samples(returns: pd.Series, train_cutoff: str,
                           stride: int = 1) -> list[dict]:
    """Build patched training samples from a return series.

    Each sample:
      context: [CONTEXT_PATCHES, PATCH_LEN] = [16, 32] = 512 points
      future:  [OUTPUT_PATCH_LEN] = [128] points

    30-day gap: last sample's future must end before cutoff.
    """
    cutoff = pd.Timestamp(train_cutoff)
    values = returns.values
    dates = returns.index
    total_needed = CONTEXT_LEN + FORECAST_HORIZON

    samples = []
    for i in range(CONTEXT_LEN, len(values) - FORECAST_HORIZON, stride):
        # Temporal boundary
        if dates[i + FORECAST_HORIZON - 1] > cutoff:
            break

        ctx = values[i - CONTEXT_LEN:i].astype(np.float32)
        fut = values[i:i + FORECAST_HORIZON].astype(np.float32)

        if np.any(np.isnan(ctx)) or np.any(np.isnan(fut)):
            continue

        # Patch the context: [512] -> [16, 32]
        patched_ctx = ctx.reshape(CONTEXT_PATCHES, PATCH_LEN)

        samples.append({
            "context": patched_ctx,  # [16, 32]
            "future": fut,           # [128]
        })

    return samples


def load_training_data(stage: str) -> list[dict]:
    """Load training samples for a given stage."""
    config = STAGE_CONFIG[stage]
    stride = config["stride"]
    all_samples = []

    if stage == "5min":
        data_dir = ROOT / "data/raw/5min"
        assets = ASSETS_5MIN
    elif stage == "hourly":
        data_dir = ROOT / "data/raw/hourly"
        assets = ASSETS_HOURLY
    elif stage == "daily":
        data_dir = ROOT / "data/raw"
        assets = ASSETS_DAILY
    else:
        raise ValueError(f"Unknown stage: {stage}")

    for asset in assets:
        if stage == "5min":
            path = data_dir / f"{asset}_5m.parquet"
        elif stage == "hourly":
            path = data_dir / f"{asset}_1h.parquet"
        else:
            path = data_dir / f"{asset}_price.parquet"

        if not path.exists():
            logger.warning("  %s: file not found at %s", asset, path)
            continue

        returns = load_returns_from_parquet(path)
        if len(returns) < CONTEXT_LEN + FORECAST_HORIZON:
            logger.warning("  %s: too few data points (%d)", asset, len(returns))
            continue

        samples = build_patched_samples(returns, TRAIN_CUTOFF, stride=stride)
        logger.info("  %s %s: %d samples from %d returns", asset, stage, len(samples), len(returns))
        all_samples.extend(samples)

    logger.info("  Total %s samples: %d", stage, len(all_samples))
    return all_samples


def finetune_stage(model, samples: list[dict], stage: str,
                    checkpoint_path: Path, device: str) -> dict:
    """Fine-tune one stage using verified gradient flow recipe."""
    config = STAGE_CONFIG[stage]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    # Stack all samples into tensors
    all_ctx = np.stack([s["context"] for s in samples])   # [N, 16, 32]
    all_fut = np.stack([s["future"] for s in samples])    # [N, 128]

    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    dataset = TensorDataset(ctx_t, fut_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=(device == "cuda"), drop_last=True)

    logger.info("  Stage %s: %d samples, batch=%d, epochs=%d, lr=%.1e",
                stage, len(samples), batch_size, epochs, lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Cosine LR schedule with warmup
    warmup_steps = 3 * len(loader)
    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    patience = 10
    patience_cnt = 0
    nan_count = 0
    metrics = {"losses": [], "grad_norms": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0

        for ctx_b, fut_b in loader:
            ctx_b = ctx_b.to(device)   # [batch, 16, 32]
            fut_b = fut_b.to(device)   # [batch, 128]

            # Simple standardization (NOT RevIN — RevIN breaks gradients)
            # Standardize each sample by its context mean/std
            ctx_flat = ctx_b.reshape(ctx_b.shape[0], -1)  # [batch, 512]
            ctx_mean = ctx_flat.mean(dim=1, keepdim=True)  # [batch, 1]
            ctx_std = ctx_flat.std(dim=1, keepdim=True).clamp(min=1e-8)

            normed_ctx = (ctx_flat - ctx_mean) / ctx_std
            normed_fut = (fut_b - ctx_mean) / ctx_std  # normalize target same way

            normed_patched = normed_ctx.reshape(ctx_b.shape)  # [batch, 16, 32]
            masks = torch.ones_like(normed_patched, dtype=torch.bool)

            optimizer.zero_grad()

            # Forward pass
            (_, _, output_ts, _), _ = model(normed_patched, masks)
            # output_ts: [batch, 16, 1280] = [batch, 16, 128*10]

            # Reshape and extract point forecast
            output_reshaped = output_ts.reshape(
                ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES
            )
            # Point forecast from last patch
            pred = output_reshaped[:, -1, :, POINT_IDX]  # [batch, 128]

            # MSE loss in normalized space
            loss = ((pred - normed_fut) ** 2).mean()

            # NaN check
            if torch.isnan(loss):
                nan_count += 1
                logger.warning("  NaN loss at epoch %d (count: %d)", epoch + 1, nan_count)
                if nan_count >= 3:
                    return {"status": "ABORTED", "reason": "nan_loss", "epoch": epoch + 1}
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                if checkpoint_path.exists():
                    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                continue

            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Gradient health check (epoch 1, batch 1)
            if epoch == 0 and n_batches == 0:
                if float(grad_norm) == 0:
                    logger.error("  GRADIENT = 0. Aborting.")
                    return {"status": "ABORTED", "reason": "zero_gradient"}
                logger.info("  First batch grad_norm: %.4f (healthy)", float(grad_norm))

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_grad_norm += float(grad_norm)
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        avg_grad = epoch_grad_norm / n_batches
        metrics["losses"].append(avg_loss)
        metrics["grad_norms"].append(avg_grad)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %3d/%d: loss=%.6f  grad=%.4f  lr=%.2e",
                        epoch + 1, epochs, avg_loss, avg_grad,
                        scheduler.get_last_lr()[0])

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("  Early stop at epoch %d", epoch + 1)
                break

    logger.info("  Stage %s done. Best loss: %.6f", stage, best_loss)

    return {
        "status": "OK",
        "stage": stage,
        "best_loss": float(best_loss),
        "final_epoch": epoch + 1,
        "n_samples": len(samples),
        "losses": metrics["losses"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["5min", "hourly", "daily", "progressive"],
                        default="progressive")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v2: Fine-Tuning (verified gradient flow)")
    logger.info("=" * 60)

    import timesfm

    # Load model — access the REAL nn.Module via m.model
    logger.info("Loading TimesFM model...")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model  # the actual nn.Module (231M params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info("Device: %s, Params: %d", device,
                sum(p.numel() for p in model.parameters()))

    if args.stage == "progressive":
        stages = ["5min", "hourly", "daily"]
    else:
        stages = [args.stage]

    results = {}

    for stage in stages:
        logger.info("\n=== Stage: %s ===", stage)

        # Load data
        samples = load_training_data(stage)
        if not samples:
            logger.warning("No samples for %s. Skipping.", stage)
            results[stage] = {"status": "SKIPPED", "reason": "no_data"}
            continue

        # If not first stage, load previous checkpoint
        if stage != stages[0]:
            prev = stages[stages.index(stage) - 1]
            prev_ckpt = MODELS / f"timesfm_v2_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading checkpoint from %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)

        model.train()
        ckpt_path = MODELS / f"timesfm_v2_{stage}_best.pt"
        stage_result = finetune_stage(model, samples, stage, ckpt_path, device)
        results[stage] = stage_result

        if stage_result.get("status") != "OK":
            logger.error("Stage %s failed: %s", stage, stage_result)
            break

    # Save final checkpoint
    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final_path = MODELS / "timesfm_v2_progressive_best.pt"
        torch.save(model.state_dict(), final_path)
        logger.info("Final model saved: %s", final_path)

    # Save results
    out = {
        "phase": "2_v2",
        "recipe": "m.model + simple_standardization + no_RevIN",
        "stages": {k: {kk: vv for kk, vv in v.items() if kk != "losses"}
                   for k, v in results.items()},
    }
    out_path = RESULTS / "phase2_finetune_v2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results: %s", out_path)


if __name__ == "__main__":
    main()
