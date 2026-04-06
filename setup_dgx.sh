#!/bin/bash
# Run this on DGX after scp-ing the project
echo "=== Setting up TimesFM BTC environment on DGX ==="

# 1. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -q yfinance fredapi ccxt xgboost scikit-learn pyarrow pmdarima
pip install -q timesfm[torch]
pip install -q "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Set FRED key
export FRED_API_KEY="9fd4f7d25a037dc92c642dde3d79131a"

# 4. Verify
python3 -c "import timesfm; print('TimesFM OK')"
python3 -c "import jax; print('JAX OK:', jax.devices())"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. python3 experiments/phase0_baselines.py   (TimesFM zero-shot + XReg)"
echo "  2. python3 experiments/phase2_finetune.py --experiment 2.1"
echo "  3. python3 experiments/phase2_finetune.py --experiment 2.2"
echo "  4. python3 experiments/phase3_inference.py --days 5"
