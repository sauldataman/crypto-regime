"""
Phase 2 v5: Maximum fine-tune — less freezing, all-patch loss, data augmentation.

v4 proved fine-tune works (VaR 1% hourly: 11.1% → 0.6%).
v5 pushes harder:
  1. Freeze only 10/20 layers (50% trainable, vs v4's 25%)
  2. Loss on ALL 16 patches (teacher forcing), not just last patch
  3. Data augmentation: Gaussian noise + scale jitter
  4. 70% crypto / 30% tradfi sampling weight
  5. More epochs: 25/25/40 (vs v4's 15/15/30)
  6. Higher learning rates with warmup

Usage:
  python experiments/phase2_finetune_v5.py --stage daily
  python experiments/phase2_finetune_v5.py --stage progressive
"""
import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
NUM_QUANTILES = 10
POINT_IDX = 5
CONTEXT_PATCHES = 16
CONTEXT_LEN = CONTEXT_PATCHES * PATCH_LEN

TRAIN_CUTOFF = "2022-06-01"
FORECAST_HORIZON = OUTPUT_PATCH_LEN

QUANTILE_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

STAGE_CONFIG = {
    "5min":   {"epochs": 25, "lr": 1e-4, "batch_size": 48, "stride": 64},
    "hourly": {"epochs": 25, "lr": 5e-5, "batch_size": 48, "stride": 24},
    "daily":  {"epochs": 40, "lr": 2e-5, "batch_size": 32, "stride": 1},
}

CRYPTO_ASSETS = {
    "5min": {
        "btc": "data/raw/5min/btc_5m.parquet",
        "eth": "data/raw/5min/eth_5m.parquet",
        "sol": "data/raw/5min/sol_5m.parquet",
        "bnb": "data/raw/5min/bnb_5m.parquet",
        "doge": "data/raw/5min/doge_5m.parquet",
        "link": "data/raw/5min/link_5m.parquet",
    },
    "hourly": {
        "btc": "data/raw/hourly/btc_1h.parquet",
        "eth": "data/raw/hourly/eth_1h.parquet",
        "sol": "data/raw/hourly/sol_1h.parquet",
        "bnb": "data/raw/hourly/bnb_1h.parquet",
        "doge": "data/raw/hourly/doge_1h.parquet",
        "link": "data/raw/hourly/link_1h.parquet",
    },
    "daily": {
        "btc": "data/raw/btc_price.parquet",
        "eth": "data/raw/eth_price.parquet",
        "sol": "data/raw/sol_price.parquet",
        "bnb": "data/raw/bnb_price.parquet",
        "doge": "data/raw/doge_price.parquet",
        "link": "data/raw/link_price.parquet",
    },
}


def pinball_loss_all_patches(pred, target_patches, quantiles=None):
    """Pinball loss on ALL patches and ALL quantile channels.

    Args:
        pred: [batch, 16, 128, 10] — all patch outputs
        target_patches: [batch, 16, 128] — target for each patch
    """
    if quantiles is None:
        quantiles = QUANTILE_TARGETS

    target_expanded = target_patches.unsqueeze(-1)  # [batch, 16, 128, 1]
    errors = target_expanded - pred  # [batch, 16, 128, 10]

    total_loss = torch.tensor(0.0, device=pred.device)
    n_q = 0

    for i, q in enumerate(quantiles):
        ch = i if i < POINT_IDX else i + 1
        if ch >= pred.shape[-1]:
            continue
        e = errors[..., ch]
        loss_q = torch.max(q * e, (q - 1) * e)
        total_loss = total_loss + loss_q.mean()
        n_q += 1

    # Point forecast MSE
    point_pred = pred[..., POINT_IDX]
    point_loss = ((point_pred - target_patches) ** 2).mean()

    return total_loss / max(n_q, 1) + 0.1 * point_loss


def augment_batch(ctx, fut, noise_std=0.02, scale_range=(0.8, 1.2)):
    """Data augmentation: Gaussian noise + random scale jitter.

    Applied in normalized space, so noise_std and scale are relative.
    """
    batch = ctx.shape[0]

    # Random scale jitter per sample
    scales = torch.empty(batch, 1, device=ctx.device).uniform_(*scale_range)
    ctx_aug = ctx * scales.unsqueeze(-1)  # [batch, 16, 32] * [batch, 1, 1]
    fut_aug = fut * scales  # [batch, 128] * [batch, 1]

    # Gaussian noise
    ctx_noise = torch.randn_like(ctx_aug) * noise_std
    ctx_aug = ctx_aug + ctx_noise

    return ctx_aug, fut_aug


def load_returns_from_parquet(path):
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


def build_samples_with_all_patch_targets(returns, train_cutoff, stride=1):
    """Build samples where each patch has its own future target.

    For patch i (covering context[i*32 : (i+1)*32]):
      target = returns[(i+1)*32 : (i+1)*32 + 128]

    This enables loss on ALL patches, not just the last one.
    """
    cutoff = pd.Timestamp(train_cutoff)
    values = returns.values
    dates = returns.index

    # Need: 512 context + 128 for each of the 16 patches' targets
    # Actually the last patch's target extends 128 beyond context end
    total_needed = CONTEXT_LEN + FORECAST_HORIZON

    samples = []
    for i in range(CONTEXT_LEN, len(values) - FORECAST_HORIZON, stride):
        if dates[i + FORECAST_HORIZON - 1] > cutoff:
            break

        ctx = values[i - CONTEXT_LEN:i].astype(np.float32)

        # Build target for each patch
        # Patch j covers ctx[j*32 : (j+1)*32]
        # Its target is the next 128 values after the patch
        patch_targets = []
        for j in range(CONTEXT_PATCHES):
            patch_end = (i - CONTEXT_LEN) + (j + 1) * PATCH_LEN
            target_start = patch_end
            target_end = target_start + FORECAST_HORIZON
            if target_end > len(values):
                break
            t = values[target_start:target_end].astype(np.float32)
            if len(t) < FORECAST_HORIZON or np.any(np.isnan(t)):
                break
            patch_targets.append(t)

        if len(patch_targets) < CONTEXT_PATCHES:
            continue
        if np.any(np.isnan(ctx)):
            continue

        patched_ctx = ctx.reshape(CONTEXT_PATCHES, PATCH_LEN)
        stacked_targets = np.stack(patch_targets)  # [16, 128]

        samples.append({
            "context": patched_ctx,
            "targets": stacked_targets,  # [16, 128] — one target per patch
        })

    return samples


def load_all_training_data(stage, crypto_weight=0.7):
    config = STAGE_CONFIG[stage]
    stride = config["stride"]
    crypto_samples = []
    tradfi_samples = []

    # Crypto
    crypto_paths = CRYPTO_ASSETS.get(stage, CRYPTO_ASSETS["daily"])
    for name, rel_path in crypto_paths.items():
        path = ROOT / rel_path
        if not path.exists():
            continue
        returns = load_returns_from_parquet(path)
        if len(returns) < CONTEXT_LEN + FORECAST_HORIZON + CONTEXT_LEN:
            continue
        samples = build_samples_with_all_patch_targets(returns, TRAIN_CUTOFF, stride)
        logger.info("    %s: %d samples", name, len(samples))
        crypto_samples.extend(samples)

    # TradFi (daily from macro_extended)
    macro_path = ROOT / "data/raw/macro_extended.parquet"
    if macro_path.exists():
        macro_df = pd.read_parquet(macro_path)
        macro_df.index = pd.to_datetime(macro_df.index)
        for name in ["sp500", "nasdaq", "russell_2000", "gold", "silver",
                     "vix_yf", "dxy_yf", "treasury_10y_yf", "treasury_5y_yf"]:
            if name not in macro_df.columns:
                continue
            series = macro_df[name].dropna()
            if len(series) < CONTEXT_LEN + FORECAST_HORIZON + CONTEXT_LEN:
                continue
            returns = np.log(series / series.shift(1)).dropna()
            samples = build_samples_with_all_patch_targets(returns, TRAIN_CUTOFF, stride=1)
            logger.info("    %s (tradfi): %d samples", name, len(samples))
            tradfi_samples.extend(samples)

    all_samples = crypto_samples + tradfi_samples
    n_c, n_t = len(crypto_samples), len(tradfi_samples)

    weights = []
    if n_c > 0 and n_t > 0:
        w_c = crypto_weight / n_c
        w_t = (1 - crypto_weight) / n_t
        weights = [w_c] * n_c + [w_t] * n_t
    else:
        weights = [1.0 / max(len(all_samples), 1)] * len(all_samples)

    logger.info("  Total: %d (%d crypto + %d tradfi, weight %.0f%%/%.0f%%)",
                len(all_samples), n_c, n_t, crypto_weight * 100, (1 - crypto_weight) * 100)

    return all_samples, weights, {"n_crypto": n_c, "n_tradfi": n_t, "n_total": len(all_samples)}


def freeze_layers(model, n_freeze=10):
    for param in model.tokenizer.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.stacked_xf):
        if i < n_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Frozen: %d/%d (%.1f%%), Trainable: %d",
                total - trainable, total, (total - trainable) / total * 100, trainable)


def finetune_stage(model, samples, weights, stage, checkpoint_path, device):
    config = STAGE_CONFIG[stage]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    # Stack: context [N, 16, 32], targets [N, 16, 128]
    all_ctx = np.stack([s["context"] for s in samples])
    all_tgt = np.stack([s["targets"] for s in samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    tgt_t = torch.tensor(all_tgt, dtype=torch.float32)

    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    dataset = TensorDataset(ctx_t, tgt_t)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=(device == "cuda"), drop_last=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    warmup_steps = 5 * len(loader)
    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("  Stage %s: %d samples, batch=%d, epochs=%d, lr=%.1e, trainable=%d",
                stage, len(samples), batch_size, epochs, lr,
                sum(p.numel() for p in trainable_params))

    best_loss = float("inf")
    patience_cnt = 0
    nan_count = 0
    losses_log = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for ctx_b, tgt_b in loader:
            ctx_b = ctx_b.to(device)   # [batch, 16, 32]
            tgt_b = tgt_b.to(device)   # [batch, 16, 128]

            # Standardize
            ctx_flat = ctx_b.reshape(ctx_b.shape[0], -1)
            ctx_mean = ctx_flat.mean(dim=1, keepdim=True)
            ctx_std = ctx_flat.std(dim=1, keepdim=True).clamp(min=1e-8)

            normed_ctx = (ctx_flat - ctx_mean) / ctx_std
            # Normalize all 16 patch targets with same stats
            normed_tgt = (tgt_b - ctx_mean.unsqueeze(-1)) / ctx_std.unsqueeze(-1)

            normed_patched = normed_ctx.reshape(ctx_b.shape)

            # Data augmentation (50% chance per batch)
            if torch.rand(1).item() < 0.5:
                normed_patched, normed_tgt = augment_batch(normed_patched, normed_tgt.reshape(ctx_b.shape[0], -1))
                normed_tgt = normed_tgt.reshape(ctx_b.shape[0], CONTEXT_PATCHES, FORECAST_HORIZON)

            masks = torch.ones_like(normed_patched, dtype=torch.bool)

            optimizer.zero_grad()

            (_, _, output_ts, _), _ = model(normed_patched, masks)
            # [batch, 16, 1280] → [batch, 16, 128, 10]
            output_reshaped = output_ts.reshape(
                ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES
            )

            # ALL-PATCH LOSS: loss on every patch's prediction
            loss = pinball_loss_all_patches(output_reshaped, normed_tgt)

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
                    return {"status": "ABORTED", "reason": "zero_gradient"}
                logger.info("  First batch: loss=%.4f, grad=%.4f", loss.item(), grad_norm)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        losses_log.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %3d/%d: loss=%.6f  lr=%.2e",
                        epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                logger.info("  Early stop at epoch %d", epoch + 1)
                break

    logger.info("  Best loss: %.6f", best_loss)
    return {"status": "OK", "stage": stage, "best_loss": float(best_loss),
            "final_epoch": epoch + 1, "n_samples": len(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["5min", "hourly", "daily", "progressive"],
                        default="progressive")
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--crypto-weight", type=float, default=0.7)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v5: Maximum Fine-tune")
    logger.info("  freeze=%d, crypto_weight=%.0f%%, all-patch loss, augmentation",
                args.freeze, args.crypto_weight * 100)
    logger.info("=" * 60)

    import timesfm

    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    freeze_layers(model, n_freeze=args.freeze)

    stages = ["5min", "hourly", "daily"] if args.stage == "progressive" else [args.stage]
    results = {}

    for stage in stages:
        logger.info("\n=== Stage: %s ===", stage)

        samples, weights, stats = load_all_training_data(stage, args.crypto_weight)
        if not samples:
            results[stage] = {"status": "SKIPPED"}
            continue

        if stage != stages[0]:
            prev = stages[stages.index(stage) - 1]
            prev_ckpt = MODELS / f"timesfm_v5_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading checkpoint: %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)
                freeze_layers(model, n_freeze=args.freeze)

        model.train()
        ckpt = MODELS / f"timesfm_v5_{stage}_best.pt"
        result = finetune_stage(model, samples, weights, stage, ckpt, device)
        result["data_stats"] = stats
        results[stage] = result

        if result.get("status") != "OK":
            break

    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final = MODELS / "timesfm_v5_progressive_best.pt"
        torch.save(model.state_dict(), final)
        logger.info("Final: %s", final)

    out = {
        "phase": "2_v5",
        "recipe": "quantile_loss + all_patch_loss + augment + freeze_10 + crypto_70",
        "crypto_weight": args.crypto_weight,
        "freeze_layers": args.freeze,
        "stages": results,
    }
    (RESULTS / "phase2_finetune_v5.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results saved")


if __name__ == "__main__":
    main()
