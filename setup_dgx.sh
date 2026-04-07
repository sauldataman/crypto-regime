#!/bin/bash
# DGX setup: two options
#   Option A (recommended): Docker — CUDA compatibility guaranteed
#   Option B: Native install — faster iteration but CUDA may need manual config
#
# Run this on DGX after cloning the repo and scp-ing data/

set -e
echo "=== crypto-regime DGX setup ==="

# ── Option A: Docker (recommended) ──────────────────────────
setup_docker() {
    echo ""
    echo "=== Option A: Docker Setup ==="
    echo "Building Docker image from NGC PyTorch container..."
    echo "This ensures CUDA/cuDNN/PyTorch compatibility."
    echo ""

    docker build -t crypto-regime .

    echo ""
    echo "=== Docker build complete ==="
    echo ""
    echo "Run experiments:"
    echo "  bash docker-run.sh eval         # Full TimesFM evaluation"
    echo "  bash docker-run.sh phase05      # Phase 0.5 smoke test"
    echo "  bash docker-run.sh phase2       # Progressive fine-tune"
    echo "  bash docker-run.sh phase3       # Risk signal output"
    echo "  bash docker-run.sh              # Interactive shell"
    echo ""
    echo "Or run everything:"
    echo "  bash docker-run.sh all"
}

# ── Option B: Native install ────────────────────────────────
setup_native() {
    echo ""
    echo "=== Option B: Native Install ==="
    echo "Installing directly (no Docker). Make sure CUDA is configured."
    echo ""

    # TimesFM from GitHub (latest, with Finetuner API)
    pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"

    # Project dependencies
    pip install ruptures>=1.1.8 mapie>=0.8 arch>=7.0
    pip install yfinance fredapi ccxt>=4.0 xgboost>=2.0
    pip install scikit-learn statsmodels>=0.14 pmdarima>=2.0
    pip install pyarrow matplotlib

    echo ""
    echo "=== Verifying installation ==="
    python3 -c "
import timesfm
import torch
import ruptures
import mapie

print(f'TimesFM: OK')
print(f'  API: {[a for a in dir(timesfm) if \"Finetun\" in a or \"finetun\" in a]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'ruptures: OK')
print(f'mapie: OK')
"

    echo ""
    echo "=== Setup complete ==="
    echo ""
    echo "Run experiments:"
    echo "  python3 experiments/eval_timesfm.py          # Full TimesFM evaluation"
    echo "  python3 experiments/phase05_smoke_test.py     # Phase 0.5 smoke test"
    echo "  python3 experiments/phase2_finetune.py        # Progressive fine-tune"
    echo "  python3 experiments/phase3_risk_signals.py    # Risk signal output"
}

# ── Choose setup method ─────────────────────────────────────
echo ""
echo "Choose setup method:"
echo "  A) Docker (recommended — guaranteed CUDA compatibility)"
echo "  B) Native install (faster iteration, may need CUDA config)"
echo ""
read -p "Enter A or B: " choice

case "$choice" in
    [Aa]) setup_docker ;;
    [Bb]) setup_native ;;
    *)    echo "Invalid choice. Run again with A or B." ;;
esac
