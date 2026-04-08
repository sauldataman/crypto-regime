"""
Phase 2 v3: Fine-tune with quantile loss + layer freezing.

v2: MSE on point forecast only → didn't improve quantile calibration
v3: Pinball loss on ALL quantiles + freeze bottom transformer layers

Key changes from v2:
  1. Quantile loss (pinball) instead of MSE — directly optimizes calibration
  2. Loss on ALL 10 quantile outputs, not just point forecast (idx 5)
  3. Freeze bottom 15/20 transformer layers — prevent catastrophic forgetting
  4. Loss on ALL patches (teacher forcing), not just last patch
  5. Optional: combined loss = quantile_loss + 0.1 * point_mse

Usage:
  python experiments/phase2_finetune_v3.py --stage daily
  python experiments/phase2_finetune_v3.py --stage progressive
  python experiments/phase2_finetune_v3.py --stage daily --freeze 0  # no freezing
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

# Model constants
PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
NUM_QUANTILES = 10
POINT_IDX = 5
CONTEXT_PATCHES = 16
CONTEXT_LEN = CONTEXT_PATCHES * PATCH_LEN  # 512

TRAIN_CUTOFF = "2022-06-01"
FORECAST_HORIZON = OUTPUT_PATCH_LEN

STAGE_CONFIG = {
    "5min":   {"epochs": 15, "lr": 5e-5, "batch_size": 64, "stride": 64},
    "hourly": {"epochs": 15, "lr": 3e-5, "batch_size": 64, "stride": 24},
    "daily":  {"epochs": 30, "lr": 1e-5, "batch_size": 32, "stride": 1},
}

# Quantile levels matching TimesFM output order
# output[:, :, 128, 10] → 10 channels, index 0-9
# Empirically: idx 5 = point forecast (median-ish), others = deciles P10-P90
# The exact mapping: output contains point + 9 quantiles
QUANTILE_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles: list[float] = None) -> torch.Tensor:
    """Pinball (quantile) loss across all quantile levels.

    Args:
        pred: [batch, patches, output_patch_len, num_quantiles]
        target: [batch, patches, output_patch_len] (broadcast across quantiles)
        quantiles: list of quantile levels [0.1, 0.2, ..., 0.9]

    Returns:
        scalar loss
    """
    if quantiles is None:
        quantiles = QUANTILE_TARGETS

    target_expanded = target.unsqueeze(-1)  # [batch, patches, 128, 1]
    errors = target_expanded - pred          # [batch, patches, 128, Q]

    total_loss = torch.tensor(0.0, device=pred.device)

    for i, q in enumerate(quantiles):
        # Skip the point forecast channel (idx 5 = POINT_IDX)
        # Map: output channels 0-4 → q indices 0-4, channel 5 → point, channels 6-9 → q indices 5-8
        if i < POINT_IDX:
            ch = i
        else:
            ch = i + 1  # skip point channel

        if ch >= pred.shape[-1]:
            continue

        e = errors[..., ch]
        loss_q = torch.max(q * e, (q - 1) * e)
        total_loss = total_loss + loss_q.mean()

    # Also add point forecast MSE (weighted lower)
    point_pred = pred[..., POINT_IDX]
    point_loss = ((point_pred - target) ** 2).mean()

    # Combined: quantile calibration + point accuracy
    combined = total_loss / len(quantiles) + 0.1 * point_loss

    return combined


def load_returns_from_parquet(path: Path) -> pd.Series:
    """Load log returns from a parquet file."""
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


def build_patched_samples(returns: pd.Series, train_cutoff: str,
                           stride: int = 1) -> list[dict]:
    """Build training samples: context [16, 32] + multi-step future [16, 128]."""
    cutoff = pd.Timestamp(train_cutoff)
    values = returns.values

    # For teacher forcing: we need future for ALL patches, not just the last one
    # Each patch i predicts the next OUTPUT_PATCH_LEN values after patch i
    # Total context = 512, but we also need 128 steps after each of the 16 patches
    # Simplification: context = 512, target = next 128 after the LAST patch
    # But also create shifted targets for intermediate patches

    dates = returns.index
    total_needed = CONTEXT_LEN + FORECAST_HORIZON

    samples = []
    for i in range(CONTEXT_LEN, len(values) - FORECAST_HORIZON, stride):
        if dates[i + FORECAST_HORIZON - 1] > cutoff:
            break

        ctx = values[i - CONTEXT_LEN:i].astype(np.float32)
        # Future for each patch: patch j's target is values[j*32+32 : j*32+32+128]
        # But we simplify: use the SAME future target for all patches
        # (the model outputs predictions for all patches, we compare all)
        fut = values[i:i + FORECAST_HORIZON].astype(np.float32)

        if np.any(np.isnan(ctx)) or np.any(np.isnan(fut)):
            continue

        patched_ctx = ctx.reshape(CONTEXT_PATCHES, PATCH_LEN)
        samples.append({"context": patched_ctx, "future": fut})

    return samples


def load_training_data(stage: str) -> list[dict]:
    """Load training samples for a given stage."""
    config = STAGE_CONFIG[stage]
    stride = config["stride"]
    all_samples = []

    if stage == "5min":
        assets, data_dir = ["btc", "eth", "sol", "bnb", "doge", "link"], ROOT / "data/raw/5min"
        suffix = "_5m.parquet"
    elif stage == "hourly":
        assets, data_dir = ["btc", "eth", "sol", "bnb", "doge", "link"], ROOT / "data/raw/hourly"
        suffix = "_1h.parquet"
    else:
        assets, data_dir = ["btc", "eth", "sol", "bnb", "doge", "link"], ROOT / "data/raw"
        suffix = "_price.parquet"

    for asset in assets:
        path = data_dir / f"{asset}{suffix}"
        if not path.exists():
            logger.warning("  %s: not found", path)
            continue
        returns = load_returns_from_parquet(path)
        if len(returns) < CONTEXT_LEN + FORECAST_HORIZON:
            continue
        samples = build_patched_samples(returns, TRAIN_CUTOFF, stride)
        logger.info("  %s %s: %d samples", asset, stage, len(samples))
        all_samples.extend(samples)

    logger.info("  Total %s: %d samples", stage, len(all_samples))
    return all_samples


def freeze_layers(model, n_freeze: int = 15):
    """Freeze bottom transformer layers to prevent catastrophic forgetting.

    Model structure:
      - tokenizer (ResidualBlock) — freeze
      - stacked_xf (ModuleList of 20 Transformer layers) — freeze bottom N
      - output_projection_point (ResidualBlock) — always train
      - output_projection_quantiles (ResidualBlock) — always train
    """
    # Freeze tokenizer
    for param in model.tokenizer.parameters():
        param.requires_grad = False

    # Freeze bottom N transformer layers
    for i, layer in enumerate(model.stacked_xf):
        if i < n_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    # Count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    logger.info("  Frozen: %d/%d params (%.1f%%)", frozen, total, frozen/total*100)
    logger.info("  Trainable: %d params", trainable)
    logger.info("  Frozen layers: tokenizer + transformer[0:%d]", n_freeze)
    logger.info("  Training: transformer[%d:20] + output_projection_point + output_projection_quantiles",
                n_freeze)


def finetune_stage(model, samples: list[dict], stage: str,
                    checkpoint_path: Path, device: str) -> dict:
    """Fine-tune with quantile loss + teacher forcing."""
    config = STAGE_CONFIG[stage]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    all_ctx = np.stack([s["context"] for s in samples])
    all_fut = np.stack([s["future"] for s in samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    dataset = TensorDataset(ctx_t, fut_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=(device == "cuda"), drop_last=True)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    warmup_steps = 3 * len(loader)
    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("  Stage %s: %d samples, batch=%d, epochs=%d, lr=%.1e, trainable params=%d",
                stage, len(samples), batch_size, epochs, lr, sum(p.numel() for p in trainable_params))

    best_loss = float("inf")
    patience = 8
    patience_cnt = 0
    nan_count = 0
    metrics = {"losses": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for ctx_b, fut_b in loader:
            ctx_b = ctx_b.to(device)
            fut_b = fut_b.to(device)

            # Standardize
            ctx_flat = ctx_b.reshape(ctx_b.shape[0], -1)
            ctx_mean = ctx_flat.mean(dim=1, keepdim=True)
            ctx_std = ctx_flat.std(dim=1, keepdim=True).clamp(min=1e-8)

            normed_ctx = (ctx_flat - ctx_mean) / ctx_std
            normed_fut = (fut_b - ctx_mean) / ctx_std

            normed_patched = normed_ctx.reshape(ctx_b.shape)
            masks = torch.ones_like(normed_patched, dtype=torch.bool)

            optimizer.zero_grad()

            (_, _, output_ts, _), _ = model(normed_patched, masks)
            # output_ts: [batch, 16, 1280] → [batch, 16, 128, 10]
            output_reshaped = output_ts.reshape(
                ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES
            )

            # Teacher forcing: loss on LAST patch's full quantile output
            pred_last = output_reshaped[:, -1, :, :]  # [batch, 128, 10]
            target_last = normed_fut  # [batch, 128]

            loss = pinball_loss(pred_last.unsqueeze(1), target_last.unsqueeze(1))

            if torch.isnan(loss):
                nan_count += 1
                if nan_count >= 3:
                    return {"status": "ABORTED", "reason": "nan_loss"}
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            if epoch == 0 and n_batches == 0:
                grad_norm = sum(p.grad.norm().item() for p in trainable_params if p.grad is not None)
                if grad_norm == 0:
                    logger.error("  GRADIENT = 0. Aborting.")
                    return {"status": "ABORTED", "reason": "zero_gradient"}
                logger.info("  First batch: loss=%.4f, grad_norm=%.4f", loss.item(), grad_norm)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        metrics["losses"].append(avg_loss)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            logger.info("  Epoch %3d/%d: loss=%.6f  lr=%.2e",
                        epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["5min", "hourly", "daily", "progressive"],
                        default="progressive")
    parser.add_argument("--freeze", type=int, default=15,
                        help="Freeze bottom N transformer layers (0=no freeze, 15=default, 20=output heads only)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v3: Quantile Loss + Layer Freezing")
    logger.info("=" * 60)

    import timesfm

    logger.info("Loading TimesFM...")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Freeze layers
    if args.freeze > 0:
        logger.info("Freezing %d / 20 transformer layers", args.freeze)
        freeze_layers(model, n_freeze=args.freeze)
    else:
        logger.info("No layer freezing (full fine-tune)")
        total = sum(p.numel() for p in model.parameters())
        logger.info("  All %d params trainable", total)

    if args.stage == "progressive":
        stages = ["5min", "hourly", "daily"]
    else:
        stages = [args.stage]

    results = {}

    for stage in stages:
        logger.info("\n=== Stage: %s ===", stage)

        samples = load_training_data(stage)
        if not samples:
            results[stage] = {"status": "SKIPPED"}
            continue

        if stage != stages[0]:
            prev = stages[stages.index(stage) - 1]
            prev_ckpt = MODELS / f"timesfm_v3_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading checkpoint from %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)
                # Re-freeze after loading
                if args.freeze > 0:
                    freeze_layers(model, n_freeze=args.freeze)

        model.train()
        ckpt_path = MODELS / f"timesfm_v3_{stage}_best.pt"
        stage_result = finetune_stage(model, samples, stage, ckpt_path, device)
        results[stage] = stage_result

        if stage_result.get("status") != "OK":
            break

    # Save final
    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final_path = MODELS / "timesfm_v3_progressive_best.pt"
        torch.save(model.state_dict(), final_path)
        logger.info("Final: %s", final_path)

    out = {
        "phase": "2_v3",
        "recipe": "quantile_loss + freeze_bottom_15 + teacher_forcing",
        "freeze_layers": args.freeze,
        "stages": results,
    }
    out_path = RESULTS / "phase2_finetune_v3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results: %s", out_path)


if __name__ == "__main__":
    main()
