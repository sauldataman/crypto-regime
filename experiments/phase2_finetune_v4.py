"""
Phase 2 v4: Multi-asset fine-tune with traditional financial data.

Key idea: train TimesFM on crypto + traditional finance together.
PFN paper showed cross-asset transfer works. TimesFM was pre-trained on diverse
time series, so fine-tuning on diverse financial data matches its pre-training distribution.

Training data:
  Crypto:  BTC, ETH, SOL, BNB, DOGE, LINK (5min + hourly + daily)
  Equity:  S&P500, NASDAQ, Russell 2000 (daily)
  Rates:   Treasury 10Y, 5Y, 3M yields (daily)
  Commodities: Gold, Silver, Crude Oil (daily)
  Currency: DXY (daily)
  Volatility: VIX (daily)

Uses v3 recipe: quantile loss + layer freezing.
New: 50:50 sampling ratio between crypto and traditional finance.

Usage:
  python experiments/phase2_finetune_v4.py --stage daily        # daily only
  python experiments/phase2_finetune_v4.py --stage progressive  # 5min->hourly->daily
  python experiments/phase2_finetune_v4.py --stage daily --crypto-weight 0.7  # 70% crypto
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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

# Quantile targets for pinball loss
QUANTILE_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

STAGE_CONFIG = {
    "5min":   {"epochs": 15, "lr": 5e-5, "batch_size": 64, "stride": 64},
    "hourly": {"epochs": 15, "lr": 3e-5, "batch_size": 64, "stride": 24},
    "daily":  {"epochs": 30, "lr": 1e-5, "batch_size": 32, "stride": 1},
}

# ── Asset definitions ───────────────────────────────────────

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

# Traditional finance: daily only (no high-freq data available)
# These get mixed in during the daily stage of progressive fine-tune
TRADFI_ASSETS = {
    # Equity indices
    "sp500": {"ticker": "^GSPC", "start": "1990-01-01"},
    "nasdaq": {"ticker": "^IXIC", "start": "1990-01-01"},
    "russell2000": {"ticker": "^RUT", "start": "2000-01-01"},
    # Commodities
    "gold": {"ticker": "GC=F", "start": "2000-01-01"},
    "silver": {"ticker": "SI=F", "start": "2000-01-01"},
    "crude_oil": {"ticker": "CL=F", "start": "2000-01-01"},
    # Rates (use as price series, returns = yield changes)
    "treasury_10y": {"ticker": "^TNX", "start": "2000-01-01"},
    "treasury_5y": {"ticker": "^FVX", "start": "2000-01-01"},
    # Volatility
    "vix": {"ticker": "^VIX", "start": "2000-01-01"},
    # Currency
    "dxy": {"ticker": "DX-Y.NYB", "start": "2005-01-01"},
}


def pinball_loss(pred, target, quantiles=None):
    """Pinball (quantile) loss on all quantile channels."""
    if quantiles is None:
        quantiles = QUANTILE_TARGETS

    target_expanded = target.unsqueeze(-1)
    errors = target_expanded - pred

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

    # Point forecast MSE (weighted lower)
    point_pred = pred[..., POINT_IDX]
    point_loss = ((point_pred - target) ** 2).mean()

    return total_loss / max(n_q, 1) + 0.1 * point_loss


def load_returns_from_parquet(path: Path) -> pd.Series:
    """Load log returns from a parquet file."""
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


def fetch_tradfi_returns(ticker: str, start: str) -> pd.Series:
    """Fetch traditional finance returns from Yahoo Finance."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            close = data[("Close", ticker)]
        else:
            close = data["Close"]
        returns = np.log(close / close.shift(1)).dropna()
        returns.index = pd.to_datetime(returns.index)
        return returns
    except Exception as e:
        logger.warning("  Failed to fetch %s: %s", ticker, e)
        return pd.Series(dtype=float)


def build_patched_samples(returns: pd.Series, train_cutoff: str,
                           stride: int = 1) -> list[dict]:
    """Build patched training samples."""
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


def load_all_training_data(stage: str, crypto_weight: float = 0.5) -> tuple:
    """Load crypto + tradfi training data with sampling weights.

    Returns:
        (all_samples, sample_weights, stats)
    """
    config = STAGE_CONFIG[stage]
    stride = config["stride"]

    crypto_samples = []
    tradfi_samples = []

    # Load crypto assets
    logger.info("  Loading crypto assets...")
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

    logger.info("  Crypto total: %d samples", len(crypto_samples))

    # Load tradfi assets (daily only, mixed into all stages)
    if stage == "daily" or True:  # Always include tradfi for diversity
        logger.info("  Loading tradfi assets...")

        # First try from macro_extended.parquet (already downloaded)
        macro_ext_path = ROOT / "data/raw/macro_extended.parquet"
        if macro_ext_path.exists():
            macro_df = pd.read_parquet(macro_ext_path)
            macro_df.index = pd.to_datetime(macro_df.index)
            for name in ["sp500", "nasdaq", "russell_2000", "gold", "silver",
                         "vix_yf", "dxy_yf", "treasury_10y_yf", "treasury_5y_yf"]:
                if name not in macro_df.columns:
                    continue
                series = macro_df[name].dropna()
                if len(series) < CONTEXT_LEN + FORECAST_HORIZON:
                    continue
                returns = np.log(series / series.shift(1)).dropna()
                # For tradfi daily, always use stride=1
                samples = build_patched_samples(returns, TRAIN_CUTOFF, stride=1)
                logger.info("    %s (from macro_extended): %d samples", name, len(samples))
                tradfi_samples.extend(samples)
        else:
            # Fetch from Yahoo Finance
            for name, info in TRADFI_ASSETS.items():
                returns = fetch_tradfi_returns(info["ticker"], info["start"])
                if len(returns) < CONTEXT_LEN + FORECAST_HORIZON:
                    continue
                samples = build_patched_samples(returns, TRAIN_CUTOFF, stride=1)
                logger.info("    %s: %d samples", name, len(samples))
                tradfi_samples.extend(samples)

    logger.info("  TradFi total: %d samples", len(tradfi_samples))

    # Combine with weighted sampling
    all_samples = crypto_samples + tradfi_samples

    # Create sampling weights: crypto_weight for crypto, (1-crypto_weight) for tradfi
    if tradfi_samples:
        n_crypto = len(crypto_samples)
        n_tradfi = len(tradfi_samples)
        # Weight per sample so total expected draws match the target ratio
        w_crypto = crypto_weight / n_crypto if n_crypto > 0 else 0
        w_tradfi = (1 - crypto_weight) / n_tradfi if n_tradfi > 0 else 0
        weights = [w_crypto] * n_crypto + [w_tradfi] * n_tradfi
    else:
        weights = [1.0 / len(all_samples)] * len(all_samples)

    stats = {
        "n_crypto": len(crypto_samples),
        "n_tradfi": len(tradfi_samples),
        "n_total": len(all_samples),
        "crypto_weight": crypto_weight,
    }

    logger.info("  Combined: %d total (%d crypto + %d tradfi, weight=%.0f%%/%.0f%%)",
                len(all_samples), len(crypto_samples), len(tradfi_samples),
                crypto_weight * 100, (1 - crypto_weight) * 100)

    return all_samples, weights, stats


def freeze_layers(model, n_freeze: int = 15):
    """Freeze bottom transformer layers."""
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


def finetune_stage(model, samples: list[dict], weights: list[float],
                    stage: str, checkpoint_path: Path, device: str) -> dict:
    """Fine-tune with quantile loss + weighted sampling."""
    config = STAGE_CONFIG[stage]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    all_ctx = np.stack([s["context"] for s in samples])
    all_fut = np.stack([s["future"] for s in samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    # Weighted random sampler for crypto/tradfi balance
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(samples),
        replacement=True,
    )

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

    logger.info("  Training: %d samples, batch=%d, epochs=%d, lr=%.1e",
                len(samples), batch_size, epochs, lr)

    best_loss = float("inf")
    patience_cnt = 0
    nan_count = 0
    metrics = {"losses": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for ctx_b, fut_b in loader:
            ctx_b = ctx_b.to(device)
            fut_b = fut_b.to(device)

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

            pred_last = output_reshaped[:, -1, :, :]
            loss = pinball_loss(pred_last.unsqueeze(1), normed_fut.unsqueeze(1))

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
            if patience_cnt >= 8:
                logger.info("  Early stop at epoch %d", epoch + 1)
                break

    logger.info("  Best loss: %.6f", best_loss)
    return {"status": "OK", "stage": stage, "best_loss": float(best_loss),
            "final_epoch": epoch + 1, "n_samples": len(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["5min", "hourly", "daily", "progressive"],
                        default="progressive")
    parser.add_argument("--freeze", type=int, default=15)
    parser.add_argument("--crypto-weight", type=float, default=0.5,
                        help="Sampling weight for crypto vs tradfi (0.5 = equal)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 v4: Multi-asset + TradFi Fine-tune")
    logger.info("  Quantile loss + freeze %d layers + %.0f%% crypto / %.0f%% tradfi",
                args.freeze, args.crypto_weight * 100, (1 - args.crypto_weight) * 100)
    logger.info("=" * 60)

    import timesfm

    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model = m.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if args.freeze > 0:
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
            prev_ckpt = MODELS / f"timesfm_v4_{prev}_best.pt"
            if prev_ckpt.exists():
                logger.info("Loading checkpoint: %s", prev_ckpt.name)
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))
                model = model.to(device)
                if args.freeze > 0:
                    freeze_layers(model, n_freeze=args.freeze)

        model.train()
        ckpt = MODELS / f"timesfm_v4_{stage}_best.pt"
        result = finetune_stage(model, samples, weights, stage, ckpt, device)
        result["data_stats"] = stats
        results[stage] = result

        if result.get("status") != "OK":
            break

    if all(results.get(s, {}).get("status") == "OK" for s in stages):
        final = MODELS / "timesfm_v4_progressive_best.pt"
        torch.save(model.state_dict(), final)
        logger.info("Final: %s", final)

    out = {
        "phase": "2_v4",
        "recipe": "quantile_loss + freeze_15 + crypto_tradfi_mixed",
        "crypto_weight": args.crypto_weight,
        "freeze_layers": args.freeze,
        "stages": results,
    }
    (RESULTS / "phase2_finetune_v4.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results saved")


if __name__ == "__main__":
    main()
