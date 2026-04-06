"""
Phase 2: TimesFM Fine-Tuning (runs on DGX)

Datasets:
  daily:  data/processed/multi_asset_enriched.jsonl  (820 samples, 26 covs)
  hourly: data/processed/hourly_train.jsonl          (13258 samples, 1 cov)
  early:  data/processed/early_samples.jsonl         (1264 samples, 15 covs)

Hyperparameters (aligned with PFN arXiv:2412.09880):
  Optimizer: SGD, momentum=0.9
  LR: 1e-4 peak, linear warmup 5 epochs, cosine decay
  batch_size: 256 (reduced from PFN's 1024 for single GPU)
  epochs: 100 with early stopping (patience=10)
  gradient_clip: 1.0
  Loss: log-MSE (MSE on log-returns)

Experiments:
  2.1 Unified daily (all assets, all regimes, 26 covs)
  2.2 Per-regime daily (3 models)
  2.3 Unified + regime indicator
  2.4 Hourly pre-train → daily fine-tune (PFN approach adapted)

Usage:
  python3 experiments/phase2_finetune.py --exp 2.1
  python3 experiments/phase2_finetune.py --exp 2.4  # recommended
"""
import argparse, json, numpy as np, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

def load_jsonl(path, regime_filter=None, add_regime_cov=False):
    samples = [json.loads(l) for l in open(path)]
    if regime_filter:
        samples = [s for s in samples if s["regime"] == regime_filter]
    ctxs, futs, covs = [], [], []
    for s in samples:
        ctx = np.nan_to_num(np.array(s["context"], dtype=np.float32), 0)
        fut = np.nan_to_num(np.array(s["future"],  dtype=np.float32), 0)
        cov = np.nan_to_num(np.array(s["covariates"], dtype=np.float32), 0)
        if add_regime_cov:
            enc = {"early":0.0,"late":0.5,"post_etf":1.0}
            indicator = np.full((len(cov),1), enc.get(s.get("regime","late"),0.5))
            cov = np.concatenate([cov, indicator], axis=1)
        ctxs.append(ctx); futs.append(fut); covs.append(cov)
    return ctxs, futs, covs, len(samples)

def log_mse_loss(pred, target):
    """Log-transformed MSE loss (PFN recommendation)"""
    import torch
    eps = 1e-8
    # shift returns to be positive before log
    pred_s   = pred   + 1.0
    target_s = target + 1.0
    pred_s   = pred_s.clamp(min=eps)
    target_s = target_s.clamp(min=eps)
    return ((torch.log(pred_s) - torch.log(target_s)) ** 2).mean()

def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup + cosine decay"""
    import torch
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def finetune(model, ctxs, futs, tag, epochs=100, batch_size=256, lr=1e-4):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}, samples: {len(ctxs)}, batch: {batch_size}")
    model = model.to(device)
    model.train()

    # Pad contexts to same length
    max_ctx = max(len(c) for c in ctxs)
    max_fut = max(len(f) for f in futs)
    ctx_t = torch.tensor(np.stack([np.pad(c,(max_ctx-len(c),0)) for c in ctxs]), dtype=torch.float32)
    fut_t = torch.tensor(np.stack([np.pad(f,(0,max_fut-len(f))) for f in futs]), dtype=torch.float32)

    dataset = TensorDataset(ctx_t, fut_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs,
                               steps_per_epoch=len(loader))

    best_loss, patience, patience_cnt = float("inf"), 10, 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for ctx_b, fut_b in loader:
            ctx_b, fut_b = ctx_b.to(device), fut_b.to(device)
            optimizer.zero_grad()
            # TimesFM forward: inputs are list of arrays
            inputs = [ctx_b[i].cpu().numpy() for i in range(len(ctx_b))]
            try:
                pred, _ = model.forecast(horizon=max_fut, inputs=inputs)
                pred_t = torch.tensor(pred, dtype=torch.float32).to(device)
                loss = log_mse_loss(pred_t, fut_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            except Exception as e:
                continue

        avg_loss = epoch_loss / len(loader)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.5f}  lr={scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), MODELS / f"timesfm_{tag}_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    print(f"  Best loss: {best_loss:.5f} → models/timesfm_{tag}_best.pt")
    return best_loss

def run(exp_id):
    print(f"\n{'='*60}\nExperiment {exp_id}\n{'='*60}")
    try:
        import timesfm, torch
        torch.set_float32_matmul_precision("high")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ {e}\nInstall: pip install timesfm[torch]")
        return

    DAILY   = ROOT / "data/processed/multi_asset_enriched.jsonl"
    HOURLY  = ROOT / "data/processed/hourly_train.jsonl"
    EARLY   = ROOT / "data/processed/early_samples.jsonl"

    if exp_id == "2.1":
        ctxs, futs, covs, n = load_jsonl(DAILY)
        print(f"Unified daily: {n} samples, 26 covs")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        finetune(model, ctxs, futs, "unified")

    elif exp_id == "2.2":
        for regime in ["early","late","post_etf"]:
            src = EARLY if regime == "early" else DAILY
            ctxs, futs, covs, n = load_jsonl(src, regime_filter=regime)
            if n < 30:
                print(f"  {regime}: too few ({n}), skip"); continue
            print(f"\nRegime {regime}: {n} samples")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            finetune(model, ctxs, futs, regime, epochs=100)

    elif exp_id == "2.3":
        ctxs, futs, covs, n = load_jsonl(DAILY, add_regime_cov=True)
        print(f"Unified + regime indicator: {n} samples")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        finetune(model, ctxs, futs, "unified_regime_indicator")

    elif exp_id == "2.4":
        # Step 1: 5min pre-train (20496 samples, 5.9M time points)
        FIVEMIN = ROOT / "data/processed/5min_train.jsonl"
        print("Step 1: 5-min pre-training (20,496 samples, 5.9M pts)...")
        ctxs_5m, futs_5m, _, n_5m = load_jsonl(FIVEMIN)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        finetune(model, ctxs_5m, futs_5m, "5min_pretrain", epochs=30, batch_size=512, lr=1e-4)

        # Step 2: hourly fine-tune
        print("\nStep 2: Hourly fine-tune (13,258 samples)...")
        ctxs_h, futs_h, _, _ = load_jsonl(HOURLY)
        model.load_state_dict(torch.load(MODELS/"timesfm_5min_pretrain_best.pt"))
        finetune(model, ctxs_h, futs_h, "5min_then_hourly", epochs=30, batch_size=256, lr=5e-5)

        # Step 3: daily fine-tune (26 covs, regime-aware)
        print("\nStep 3: Daily fine-tune with covariates (820 samples)...")
        ctxs_d, futs_d, _, _ = load_jsonl(DAILY)
        model.load_state_dict(torch.load(MODELS/"timesfm_5min_then_hourly_best.pt"))
        finetune(model, ctxs_d, futs_d, "final_daily", epochs=50, batch_size=64, lr=1e-5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["2.1","2.2","2.3","2.4","all"], default="2.4")
    args = parser.parse_args()
    if args.exp == "all":
        for e in ["2.1","2.2","2.3","2.4"]: run(e)
    else:
        run(args.exp)
