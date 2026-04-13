"""
Phase 2 v8: AutoSearch best config — progressive fine-tune.

Config found by overnight auto_search.py (50 trials, best P10=0.110):
  freeze_layers: 16
  lr: 1e-4 (5min), 5e-5 (hourly), 1e-5 (daily)
  crypto_weight: 0.4
  tail_weight_p10: 5.0
  batch_size: 64
  epochs: 25/25/40

Usage:
  python experiments/phase2_finetune_v8.py --stage progressive
"""
import argparse
import json
import logging
import sys
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
sys.path.insert(0, str(ROOT))
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

# === AutoSearch best config ===
FREEZE_LAYERS = 16
CRYPTO_WEIGHT = 0.4
TAIL_WEIGHT_P10 = 5.0

STAGE_CONFIG = {
    "5min":   {"epochs": 25, "lr": 1e-4, "batch_size": 64, "stride": 64},
    "hourly": {"epochs": 25, "lr": 5e-5, "batch_size": 64, "stride": 24},
    "daily":  {"epochs": 40, "lr": 1e-5, "batch_size": 64, "stride": 1},
}

QUANTILE_CONFIG = [
    (0.10, TAIL_WEIGHT_P10),
    (0.20, max(TAIL_WEIGHT_P10 * 0.66, 1.0)),
    (0.30, 1.0), (0.40, 1.0), (0.50, 1.0),
    (0.60, 1.0), (0.70, 1.0),
    (0.80, max(TAIL_WEIGHT_P10 * 0.66, 1.0)),
    (0.90, TAIL_WEIGHT_P10),
]

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


def tail_weighted_pinball_loss(pred, target):
    target_exp = target.unsqueeze(-1)
    errors = target_exp - pred
    total_loss = torch.tensor(0.0, device=pred.device)
    total_w = 0.0
    for i, (q, w) in enumerate(QUANTILE_CONFIG):
        ch = i if i < POINT_IDX else i + 1
        if ch >= pred.shape[-1]:
            continue
        e = errors[..., ch]
        total_loss = total_loss + w * torch.max(q * e, (q - 1) * e).mean()
        total_w += w
    point_loss = ((pred[..., POINT_IDX] - target) ** 2).mean()
    return total_loss / total_w + 0.1 * point_loss


def load_returns(path):
    df = pd.read_parquet(ROOT / path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


def build_samples(returns, stride=1):
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
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
        samples.append({"context": ctx.reshape(CONTEXT_PATCHES, PATCH_LEN), "future": fut})
    return samples


def load_all_data(stage):
    config = STAGE_CONFIG[stage]
    stride = config["stride"]
    crypto_samples, tradfi_samples = [], []

    crypto_paths = CRYPTO_ASSETS.get(stage, CRYPTO_ASSETS["daily"])
    for name, path in crypto_paths.items():
        r = load_returns(path)
        if len(r) >= CONTEXT_LEN + FORECAST_HORIZON:
            s = build_samples(r, stride)
            logger.info("    %s: %d samples", name, len(s))
            crypto_samples.extend(s)

    macro_path = ROOT / "data/raw/macro_extended.parquet"
    if macro_path.exists():
        macro_df = pd.read_parquet(macro_path)
        macro_df.index = pd.to_datetime(macro_df.index)
        for col in ["sp500", "nasdaq", "russell_2000", "gold", "silver",
                     "vix_yf", "dxy_yf", "treasury_10y_yf", "treasury_5y_yf"]:
            if col not in macro_df.columns:
                continue
            series = macro_df[col].dropna()
            if len(series) < CONTEXT_LEN + FORECAST_HORIZON:
                continue
            r = np.log(series / series.shift(1)).dropna()
            s = build_samples(r, stride=1)
            logger.info("    %s (tradfi): %d samples", col, len(s))
            tradfi_samples.extend(s)

    all_samples = crypto_samples + tradfi_samples
    n_c, n_t = len(crypto_samples), len(tradfi_samples)
    if n_c > 0 and n_t > 0:
        w_c = CRYPTO_WEIGHT / n_c
        w_t = (1 - CRYPTO_WEIGHT) / n_t
        weights = [w_c] * n_c + [w_t] * n_t
    else:
        weights = [1.0 / max(len(all_samples), 1)] * len(all_samples)

    logger.info("  Total: %d (%d crypto + %d tradfi, %.0f%%/%.0f%%)",
                len(all_samples), n_c, n_t, CRYPTO_WEIGHT * 100, (1 - CRYPTO_WEIGHT) * 100)
    return all_samples, weights, {"n_crypto": n_c, "n_tradfi": n_t, "n_total": len(all_samples)}


def freeze_layers(model):
    for param in model.tokenizer.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.stacked_xf):
        if i < FREEZE_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Freeze %d/20: %d/%d trainable (%.1f%%)",
                FREEZE_LAYERS, trainable, total, trainable / total * 100)


def finetune_stage(model, samples, weights, stage, ckpt_path, device):
    config = STAGE_CONFIG[stage]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    all_ctx = np.stack([s["context"] for s in samples])
    all_fut = np.stack([s["future"] for s in samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    loader = DataLoader(TensorDataset(ctx_t, fut_t), batch_size=batch_size,
                        sampler=sampler, pin_memory=(device == "cuda"), drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    warmup = 3 * len(loader)
    total = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup, 1) if s < warmup else
        0.5 * (1 + np.cos(np.pi * (s - warmup) / max(total - warmup, 1))))

    logger.info("  %s: %d samples, batch=%d, epochs=%d, lr=%.1e",
                stage, len(samples), batch_size, epochs, lr)

    best_loss = float("inf")
    patience_cnt = 0

    for epoch in range(epochs):
        epoch_loss, n_b = 0.0, 0
        for ctx_b, fut_b in loader:
            ctx_b, fut_b = ctx_b.to(device), fut_b.to(device)
            ctx_flat = ctx_b.reshape(ctx_b.shape[0], -1)
            mu = ctx_flat.mean(dim=1, keepdim=True)
            std = ctx_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
            normed_ctx = ((ctx_flat - mu) / std).reshape(ctx_b.shape)
            normed_fut = (fut_b - mu) / std
            masks = torch.ones_like(normed_ctx, dtype=torch.bool)

            optimizer.zero_grad()
            (_, _, out, _), _ = model(normed_ctx, masks)
            pred = out.reshape(ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES)[:, -1, :, :]
            loss = tail_weighted_pinball_loss(pred, normed_fut)

            if torch.isnan(loss):
                return {"status": "ABORTED", "reason": "nan_loss"}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)

            if epoch == 0 and n_b == 0:
                gn = sum(p.grad.norm().item() for p in trainable if p.grad is not None)
                if gn == 0:
                    return {"status": "ABORTED", "reason": "zero_gradient"}
                logger.info("  First batch: loss=%.4f, grad=%.4f", loss.item(), gn)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_b += 1

        avg = epoch_loss / max(n_b, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %3d/%d: loss=%.6f  lr=%.2e", epoch + 1, epochs, avg, scheduler.get_last_lr()[0])

        if avg < best_loss:
            best_loss = avg
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
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
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v8: AutoSearch Best Config")
    logger.info("  freeze=%d, crypto=%.0f%%, tail_p10=%.0fx, batch=64",
                FREEZE_LAYERS, CRYPTO_WEIGHT * 100, TAIL_WEIGHT_P10)
    logger.info("=" * 60)

    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    freeze_layers(model)

    stages = ["5min", "hourly", "daily"] if args.stage == "progressive" else [args.stage]
    results = {}

    for stage in stages:
        logger.info("\n=== Stage: %s ===", stage)
        samples, weights, stats = load_all_data(stage)
        if not samples:
            results[stage] = {"status": "SKIPPED"}
            continue

        if stage != stages[0]:
            prev = stages[stages.index(stage) - 1]
            prev_ckpt = MODELS / f"timesfm_v8_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading: %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)
                freeze_layers(model)

        model.train()
        ckpt = MODELS / f"timesfm_v8_{stage}_best.pt"
        result = finetune_stage(model, samples, weights, stage, ckpt, device)
        result["data_stats"] = stats
        results[stage] = result
        if result.get("status") != "OK":
            break

    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final = MODELS / "timesfm_v8_progressive_best.pt"
        torch.save(model.state_dict(), final)
        logger.info("Final: %s", final)

    out = {
        "phase": "2_v8",
        "recipe": "autosearch_best: freeze16 + tail5x + crypto40% + progressive",
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "crypto_weight": CRYPTO_WEIGHT,
            "tail_weight_p10": TAIL_WEIGHT_P10,
        },
        "stages": results,
    }
    (RESULTS / "phase2_finetune_v8.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results saved")


if __name__ == "__main__":
    main()
