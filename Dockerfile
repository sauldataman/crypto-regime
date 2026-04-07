# Use 25.11 (CUDA 13.0.2) — compatible with driver 580.x
# 26.02 requires driver 590.48+ which is too new for DGX Spark (580.142)
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace/crypto-regime

# Install TimesFM from GitHub (not PyPI) to get latest Finetuner API
RUN pip install --no-cache-dir \
    "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"

# Install project dependencies
RUN pip install --no-cache-dir \
    ruptures>=1.1.8 \
    mapie>=0.8 \
    arch>=7.0 \
    yfinance \
    fredapi \
    ccxt>=4.0 \
    xgboost>=2.0 \
    scikit-learn \
    statsmodels>=0.14 \
    pmdarima>=2.0 \
    pyarrow \
    matplotlib

# Verify installation
RUN python3 -c "import timesfm; print('TimesFM OK'); print(dir(timesfm))" && \
    python3 -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')" && \
    python3 -c "import ruptures; print('ruptures OK')" && \
    python3 -c "import mapie; print('mapie OK')"

CMD ["bash"]
