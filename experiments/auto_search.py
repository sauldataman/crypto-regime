"""
Automated hyperparameter search (inspired by Karpathy's autoresearch).

Runs overnight: randomly samples hyperparameter configs, fine-tunes daily stage
(~15 min each), evaluates raw P10 coverage, keeps best config.

Target: raw P10 coverage as close to 0.10 as possible (perfect calibration).
Current best (v7): 0.116. Can we find better?

Search space:
  - freeze_layers: [12, 13, 14, 15, 16, 17]
  - lr: [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
  - crypto_weight: [0.3, 0.4, 0.5, 0.6, 0.7]
  - epochs: [15, 20, 25, 30]
  - tail_weight_p10: [1.0, 2.0, 3.0, 5.0]
  - batch_size: [16, 32, 48, 64]

Each trial: fine-tune daily stage → quick eval on cal set → score

Usage:
  python experiments/auto_search.py                     # run all night
  python experiments/auto_search.py --max-trials 10     # quick test
  python experiments/auto_search.py --resume             # continue from last run
"""
import argparse
import json
import logging
import random
import sys
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

QUANTILE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Search space
SEARCH_SPACE = {
    "freeze_layers": [12, 13, 14, 15, 16, 17],
    "lr": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
    "crypto_weight": [0.3, 0.4, 0.5, 0.6, 0.7],
    "epochs": [15, 20, 25, 30],
    "tail_weight_p10": [1.0, 2.0, 3.0, 5.0],
    "batch_size": [16, 32, 48, 64],
}

CRYPTO_DAILY = {
    "btc": "data/raw/btc_price.parquet",
    "eth": "data/raw/eth_price.parquet",
    "sol": "data/raw/sol_price.parquet",
    "bnb": "data/raw/bnb_price.parquet",
    "doge": "data/raw/doge_price.parquet",
    "link": "data/raw/link_price.parquet",
}


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


def load_all_data(crypto_weight):
    crypto_samples = []
    tradfi_samples = []

    for name, path in CRYPTO_DAILY.items():
        r = load_returns(path)
        if len(r) >= CONTEXT_LEN + FORECAST_HORIZON:
            s = build_samples(r, stride=1)
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
            tradfi_samples.extend(s)

    all_samples = crypto_samples + tradfi_samples
    n_c, n_t = len(crypto_samples), len(tradfi_samples)

    if n_c > 0 and n_t > 0:
        w_c = crypto_weight / n_c
        w_t = (1 - crypto_weight) / n_t
        weights = [w_c] * n_c + [w_t] * n_t
    else:
        weights = [1.0 / max(len(all_samples), 1)] * len(all_samples)

    return all_samples, weights


def make_loss_fn(tail_weight_p10):
    """Create pinball loss with configurable P10 weight."""
    quantile_config = [
        (0.10, tail_weight_p10),
        (0.20, max(tail_weight_p10 * 0.66, 1.0)),
        (0.30, 1.0), (0.40, 1.0), (0.50, 1.0),
        (0.60, 1.0), (0.70, 1.0),
        (0.80, max(tail_weight_p10 * 0.66, 1.0)),
        (0.90, tail_weight_p10),
    ]

    def loss_fn(pred, target):
        target_exp = target.unsqueeze(-1)
        errors = target_exp - pred
        total_loss = torch.tensor(0.0, device=pred.device)
        total_w = 0.0
        for i, (q, w) in enumerate(quantile_config):
            ch = i if i < POINT_IDX else i + 1
            if ch >= pred.shape[-1]:
                continue
            e = errors[..., ch]
            total_loss = total_loss + w * torch.max(q * e, (q - 1) * e).mean()
            total_w += w
        point_loss = ((pred[..., POINT_IDX] - target) ** 2).mean()
        return total_loss / total_w + 0.1 * point_loss

    return loss_fn


def quick_eval(model, eval_returns, n_samples=200):
    """Quick evaluation: compute raw P10 coverage on BTC hourly or daily."""
    import timesfm

    # Use the wrapper for inference
    wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    # Copy fine-tuned weights into wrapper's model
    wrapper.model.load_state_dict(model.state_dict())
    wrapper.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN, max_horizon=1,
        normalize_inputs=True, use_continuous_quantile_head=True,
    ))

    # Evaluate on last portion of data
    n = len(eval_returns)
    start_idx = max(CONTEXT_LEN, n - n_samples * 2)
    step = max(1, (n - start_idx) // n_samples)

    breaches = 0
    total = 0

    for idx in range(start_idx, n, step):
        if idx < CONTEXT_LEN:
            continue
        context = eval_returns.iloc[idx - CONTEXT_LEN:idx].values.tolist()
        actual = float(eval_returns.iloc[idx])
        try:
            _, qf = wrapper.forecast(horizon=1, inputs=[context])
            q = qf[0][0]
            offset = 1 if len(q) >= 10 else 0
            p10 = float(q[0 + offset])
            if actual < p10:
                breaches += 1
            total += 1
        except (RuntimeError, ValueError, IndexError):
            pass

    if total == 0:
        return {"p10_coverage": 0.5, "n_eval": 0}

    coverage = breaches / total
    return {"p10_coverage": coverage, "n_eval": total}


def run_trial(trial_num, config, all_samples, weights, eval_returns, base_model_state):
    """Run one fine-tune trial and evaluate."""
    import timesfm

    logger.info("\n" + "=" * 50)
    logger.info("Trial %d: %s", trial_num, json.dumps(config, default=str))
    logger.info("=" * 50)

    t0 = time.time()

    # Fresh model from base
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model = m.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Freeze layers
    n_freeze = config["freeze_layers"]
    for param in model.tokenizer.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.stacked_xf):
        if i < n_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    # Data
    batch_size = config["batch_size"]
    all_ctx = np.stack([s["context"] for s in all_samples])
    all_fut = np.stack([s["future"] for s in all_samples])
    ctx_t = torch.tensor(all_ctx, dtype=torch.float32)
    fut_t = torch.tensor(all_fut, dtype=torch.float32)

    sampler = WeightedRandomSampler(weights, num_samples=len(all_samples), replacement=True)
    dataset = TensorDataset(ctx_t, fut_t)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=(device == "cuda"), drop_last=True)

    # Optimizer
    lr = config["lr"]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    epochs = config["epochs"]

    warmup_steps = 3 * len(loader)
    total_steps = epochs * len(loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    loss_fn = make_loss_fn(config["tail_weight_p10"])

    # Train
    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_b = 0
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
            out_r = out.reshape(ctx_b.shape[0], CONTEXT_PATCHES, OUTPUT_PATCH_LEN, NUM_QUANTILES)
            pred = out_r[:, -1, :, :]

            loss = loss_fn(pred, normed_fut)
            if torch.isnan(loss):
                return {"status": "FAILED", "reason": "nan_loss", "config": config}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_b += 1

        avg_loss = epoch_loss / max(n_b, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d/%d: loss=%.6f", epoch + 1, epochs, avg_loss)

    train_time = time.time() - t0

    # Quick eval
    logger.info("  Evaluating...")
    eval_result = quick_eval(model, eval_returns, n_samples=300)
    p10_cov = eval_result["p10_coverage"]

    # Score: distance from perfect 0.10
    score = abs(p10_cov - 0.10)

    logger.info("  P10 coverage: %.4f (target: 0.10, score: %.4f)", p10_cov, score)
    logger.info("  Best loss: %.6f, Time: %.0fs, Trainable: %d", best_loss, train_time, n_trainable)

    # Save model if promising
    if score < 0.02:  # within 2% of perfect
        ckpt_path = MODELS / f"auto_trial_{trial_num}_p10_{p10_cov:.4f}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  Saved: %s", ckpt_path.name)

    return {
        "status": "OK",
        "trial": trial_num,
        "config": config,
        "best_loss": float(best_loss),
        "p10_coverage": float(p10_cov),
        "score": float(score),
        "train_time_sec": round(train_time),
        "n_trainable": n_trainable,
        "n_eval": eval_result["n_eval"],
    }


def sample_config():
    """Random sample from search space."""
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials", type=int, default=50,
                        help="Max trials (default 50, ~12 hours)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("AutoSearch: Overnight Hyperparameter Optimization")
    logger.info("  Max trials: %d", args.max_trials)
    logger.info("  Search space: %s", {k: f"{len(v)} options" for k, v in SEARCH_SPACE.items()})
    logger.info("  Total combinations: %d", np.prod([len(v) for v in SEARCH_SPACE.values()]))
    logger.info("=" * 60)

    # Load data once
    logger.info("\nLoading data...")
    all_samples, weights = load_all_data(crypto_weight=0.5)
    logger.info("  Samples: %d", len(all_samples))

    # Load eval returns (BTC daily for quick eval)
    btc_returns = load_returns("data/raw/btc_price.parquet")
    logger.info("  BTC eval: %d rows", len(btc_returns))

    # Load or initialize results
    results_path = RESULTS / "auto_search.json"
    if args.resume and results_path.exists():
        all_results = json.loads(results_path.read_text())
        logger.info("  Resuming from %d previous trials", len(all_results["trials"]))
    else:
        all_results = {"trials": [], "best": None}

    best_score = float("inf")
    best_config = None
    if all_results["best"]:
        best_score = all_results["best"]["score"]
        best_config = all_results["best"]["config"]
        logger.info("  Previous best: score=%.4f, P10=%.4f", best_score,
                     all_results["best"]["p10_coverage"])

    start_trial = len(all_results["trials"])
    t_start = time.time()

    for trial_num in range(start_trial, start_trial + args.max_trials):
        config = sample_config()

        # Skip if we've tried very similar configs
        result = run_trial(trial_num, config, all_samples, weights, btc_returns, None)

        all_results["trials"].append(result)

        if result["status"] == "OK" and result["score"] < best_score:
            best_score = result["score"]
            best_config = result["config"]
            all_results["best"] = result
            logger.info("  *** NEW BEST! Score=%.4f, P10=%.4f ***", best_score, result["p10_coverage"])

        # Save after every trial
        results_path.write_text(json.dumps(all_results, indent=2, default=str))

        elapsed = time.time() - t_start
        avg_time = elapsed / (trial_num - start_trial + 1)
        remaining = avg_time * (args.max_trials - (trial_num - start_trial + 1))
        logger.info("  Progress: %d/%d, Elapsed: %.0fm, ETA: %.0fm",
                     trial_num - start_trial + 1, args.max_trials,
                     elapsed / 60, remaining / 60)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("AUTOSEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info("  Trials: %d", len(all_results["trials"]))
    logger.info("  Best score: %.4f", best_score)
    if best_config:
        logger.info("  Best config: %s", json.dumps(best_config, default=str))
        logger.info("  Best P10 coverage: %.4f", all_results["best"]["p10_coverage"])

    # Top 5 results
    ok_trials = [t for t in all_results["trials"] if t["status"] == "OK"]
    top5 = sorted(ok_trials, key=lambda x: x["score"])[:5]
    logger.info("\n  Top 5:")
    for i, t in enumerate(top5):
        logger.info("    %d. score=%.4f P10=%.4f | freeze=%d lr=%.0e epochs=%d tail=%.0f crypto=%.0f%%",
                     i + 1, t["score"], t["p10_coverage"],
                     t["config"]["freeze_layers"], t["config"]["lr"],
                     t["config"]["epochs"], t["config"]["tail_weight_p10"],
                     t["config"]["crypto_weight"] * 100)


if __name__ == "__main__":
    main()
