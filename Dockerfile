FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace/crypto-regime

# Install TimesFM from GitHub (not PyPI) to get latest Finetuner API
RUN pip install --no-cache-dir \
    git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]

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
