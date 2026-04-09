"""
Phase 2 v6: Refined fine-tune based on v4 findings.

Changes from v4:
  A) Tail-weighted quantile loss — P10 gets 3x weight, P20 gets 2x
     (directly optimizes what VaR cares about)
  B) More epochs — 30/40 for hourly/daily (v4 was 15/30)
  C) Skip 5min — start from hourly (5min is too noisy for risk management)
     Progressive is now: hourly → daily (2 stages)

Everything else stays v4:
  - Freeze 15/20 layers (proven sweet spot)
  - Last-patch-only loss (avoid attention cheating)
  - 50% crypto / 50% tradfi
  - No data augmentation (preserves calibration)

Usage:
  python experiments/phase2_finetune_v6.py                       # hourly → daily
  python experiments/phase2_finetune_v6.py --stage daily          # daily only
  python experiments/phase2_finetune_v6.py --stage hourly         # hourly only
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

# Tail-weighted quantile targets
# P10 and P20 get higher weight because VaR 5%/1% depends on left tail accuracy
QUANTILE_CONFIG = [
    # (quantile_level, weight)
    (0.10, 3.0),   # P10 — VaR foundation, 3x weight
    (0.20, 2.0),   # P20 — 2x weight
    (0.30, 1.0),
    (0.40, 1.0),
    (0.50, 1.0),   # median
    (0.60, 1.0),
    (0.70, 1.0),
    (0.80, 2.0),   # P80 — upper tail, 2x for symmetry
    (0.90, 3.0),   # P90 — upper tail, 3x for symmetry
]

STAGE_CONFIG = {
    # No 5min — too noisy for risk management
    "hourly": {"epochs": 30, "lr": 5e-5, "batch_size": 64, "stride": 24},
    "daily":  {"epochs": 40, "lr": 1e-5, "batch_size": 32, "stride": 1},
}

CRYPTO_ASSETS = {
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


def tail_weighted_pinball_loss(pred, target):
    """Pinball loss with higher weight on tail quantiles (P10, P20, P80, P90).

    Args:
        pred: [batch, 128, 10] — last patch output
        target: [batch, 128] — future returns
    """
    target_expanded = target.unsqueeze(-1)  # [batch, 128, 1]
    errors = target_expanded - pred          # [batch, 128, 10]

    total_loss = torch.tensor(0.0, device=pred.device)
    total_weight = 0.0

    for i, (q, w) in enumerate(QUANTILE_CONFIG):
        # Map quantile index to output channel (skip POINT_IDX=5)
        ch = i if i < POINT_IDX else i + 1
        if ch >= pred.shape[-1]:
            continue

        e = errors[..., ch]
        loss_q = torch.max(q * e, (q - 1) * e)
        total_loss = total_loss + w * loss_q.mean()
        total_weight += w

    # Point forecast MSE (small weight)
    point_pred = pred[..., POINT_IDX]
    point_loss = ((point_pred - target) ** 2).mean()

    return total_loss / total_weight + 0.1 * point_loss


def load_returns_from_parquet(path):
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


def build_patched_samples(returns, train_cutoff, stride=1):
    cutoff = pd.Timestamp(train_cutoff)
    values = returns.values
    dates = returns.index
    samples = []

    for i in range(CONTEXT_LEN, len(values) - FORECAST_HORIZON, stride):
        if dates[i + FORECAST_HORIZON - 1] > cutoff:
            break
        ctx = values[i - CONTEXT_LEN:i].astype(np.float32)
        fut = values[i:i + FORECAST_HORIZON].astype(np.float32)
        if np.any(np.isnan(ctx)) or np.any(np.isnan(fut)):
            continue
        patched_ctx = ctx.reshape(CONTEXT_PATCHES, PATCH_LEN)
        samples.append({"context": patched_ctx, "future": fut})

    return samples


def load_all_training_data(stage, crypto_weight=0.5):
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
        if len(returns) < CONTEXT_LEN + FORECAST_HORIZON:
            continue
        samples = build_patched_samples(returns, TRAIN_CUTOFF, stride)
        logger.info("    %s: %d samples", name, len(samples))
        crypto_samples.extend(samples)

    # TradFi
    macro_path = ROOT / "data/raw/macro_extended.parquet"
    if macro_path.exists():
        macro_df = pd.read_parquet(macro_path)
        macro_df.index = pd.to_datetime(macro_df.index)
        for name in ["sp500", "nasdaq", "russell_2000", "gold", "silver",
                     "vix_yf", "dxy_yf", "treasury_10y_yf", "treasury_5y_yf"]:
            if name not in macro_df.columns:
                continue
            series = macro_df[name].dropna()
            if len(series) < CONTEXT_LEN + FORECAST_HORIZON:
                continue
            returns = np.log(series / series.shift(1)).dropna()
            samples = build_patched_samples(returns, TRAIN_CUTOFF, stride=1)
            logger.info("    %s (tradfi): %d samples", name, len(samples))
            tradfi_samples.extend(samples)

    all_samples = crypto_samples + tradfi_samples
    n_c, n_t = len(crypto_samples), len(tradfi_samples)

    if n_c > 0 and n_t > 0:
        w_c = crypto_weight / n_c
        w_t = (1 - crypto_weight) / n_t
        weights = [w_c] * n_c + [w_t] * n_t
    else:
        weights = [1.0 / max(len(all_samples), 1)] * len(all_samples)

    logger.info("  Total: %d (%d crypto + %d tradfi)", len(all_samples), n_c, n_t)
    return all_samples, weights, {"n_crypto": n_c, "n_tradfi": n_t, "n_total": len(all_samples)}


def freeze_layers(model, n_freeze=15):
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

    all_ctx = np.stack([s["context"] for s in samples])
    all_fut = np.stack([s["future"] for s in samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    dataset = TensorDataset(ctx_t, fut_t)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=(device == "cuda"), drop_last=True)

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

    logger.info("  Stage %s: %d samples, batch=%d, epochs=%d, lr=%.1e",
                stage, len(samples), batch_size, epochs, lr)

    best_loss = float("inf")
    patience_cnt = 0
    nan_count = 0

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
            output_reshaped = output_ts.reshape(
                ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES
            )

            # Last-patch loss with tail weighting
            pred_last = output_reshaped[:, -1, :, :]  # [batch, 128, 10]
            loss = tail_weighted_pinball_loss(pred_last, normed_fut)

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
    parser.add_argument("--stage", choices=["hourly", "daily", "progressive"],
                        default="progressive")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v6: Refined fine-tune")
    logger.info("  Tail-weighted loss (P10 3x, P20 2x)")
    logger.info("  Hourly → Daily (no 5min)")
    logger.info("  Freeze 15, crypto 50%%, no augmentation")
    logger.info("=" * 60)

    import timesfm

    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    freeze_layers(model, n_freeze=15)

    # Progressive: hourly → daily (no 5min)
    stages = ["hourly", "daily"] if args.stage == "progressive" else [args.stage]
    results = {}

    for stage in stages:
        logger.info("\n=== Stage: %s ===", stage)

        samples, weights, stats = load_all_training_data(stage, crypto_weight=0.5)
        if not samples:
            results[stage] = {"status": "SKIPPED"}
            continue

        if stage != stages[0]:
            prev = stages[stages.index(stage) - 1]
            prev_ckpt = MODELS / f"timesfm_v6_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading checkpoint: %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)
                freeze_layers(model, n_freeze=15)

        model.train()
        ckpt = MODELS / f"timesfm_v6_{stage}_best.pt"
        result = finetune_stage(model, samples, weights, stage, ckpt, device)
        result["data_stats"] = stats
        results[stage] = result

        if result.get("status") != "OK":
            break

    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final = MODELS / "timesfm_v6_progressive_best.pt"
        torch.save(model.state_dict(), final)
        logger.info("Final: %s", final)

    out = {
        "phase": "2_v6",
        "recipe": "tail_weighted_pinball + freeze_15 + hourly_daily_only + no_augment",
        "changes_from_v4": [
            "A: P10/P90 weight 3x, P20/P80 weight 2x in pinball loss",
            "B: More epochs (30/40 vs 15/30)",
            "C: Skip 5min stage (too noisy for risk management)",
        ],
        "stages": results,
    }
    (RESULTS / "phase2_finetune_v6.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results saved")


if __name__ == "__main__":
    main()
