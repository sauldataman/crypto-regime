FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace/btc-regime

# Install dependencies
RUN pip install --no-cache-dir \
    timesfm[torch] \
    yfinance \
    fredapi \
    ccxt \
    xgboost \
    scikit-learn \
    pyarrow \
    pmdarima \
    matplotlib

# Verify
RUN python3 -c "import timesfm; print('TimesFM OK')" && \
    python3 -c "import torch; print('PyTorch OK, CUDA:', torch.cuda.is_available())"

ENV FRED_API_KEY=9fd4f7d25a037dc92c642dde3d79131a

CMD ["bash"]
