# TimesFM 2.5 Crypto Fine-tuning Research Report

## 1. 研究目标

评估 Google TimesFM 2.5（200M 参数时间序列基础模型）对加密货币市场的适用性，并通过 fine-tuning 改善其在 crypto 上的 quantile 预测校准。

核心问题：
1. TimesFM zero-shot 在 crypto 上能做什么、不能做什么？
2. Fine-tune 能否改善？怎么 fine-tune（没有官方 API）？
3. 如何将 quantile 预测转化为可用的风险信号（VaR）？

## 2. 相关工作

- **TimesFM 2.5**（Google Research, 2025）：通用时间序列基础模型，200M 参数，支持 zero-shot 预测和 quantile head（P10-P90 deciles）
- **PFN timesfm_fin**（arXiv:2412.09880）：在 1 亿个金融时间点上 fine-tune TimesFM 1.0，Crypto Sharpe 0.26。仅做点预测，无 quantile 输出
- **TimesFM for VaR**（arXiv:2410.11773）：首次用 TimesFM 做 VaR（S&P100），beat GARCH/GAS。发现 quantile head "have not been calibrated after pretraining"
- **Fish Lab**（2026）："When Did Bitcoin Stop Being Bitcoin?" — PELT breakpoints, BOCPD 在 BTC 上失败, 2020-11 是真正的 regime shift

## 3. 数据

### 3.1 Crypto 资产

| 资产 | Daily | Hourly | 5min | 1min |
|------|-------|--------|------|------|
| BTC | 2014-09 (Yahoo) | 2014-01 (CryptoCompare+Binance) | 2013-10 (Kraken+Binance) | 2013-10 (Kraken) |
| ETH | 2015-08 | 2016-01 | 2015-08 (Kraken+Binance) | 2015-08 (Kraken) |
| SOL | 2020-04 | 2020-08 | 2020-08 | — |
| BNB | 2017-07 | 2017-11 | 2017-11 | — |
| DOGE | 2013-12 | 2017-01 | 2019-07 | 2019-12 (Kraken) |
| LINK | 2017-09 | 2019-01 | 2017-09 | — |

### 3.2 传统金融

| 来源 | 数据 | 起始 |
|------|------|------|
| Yahoo Finance | S&P500, NASDAQ, Russell 2000, Gold, Silver, VIX, DXY, Treasury yields | 1990-2010 |
| FRED | Fed Funds Rate, Yield curve 10Y-2Y, HY credit spread | 2000+ |
| alternative.me | Crypto Fear & Greed Index | 2018+ |

### 3.3 Temporal Split

```
Train:       2011-01 ~ 2022-06-01
Val:         2022-07-01 ~ 2023-06-30
Calibration: 2023-07-01 ~ 2024-06-30 (~250 days daily, ~3500 hourly)
Test:        2024-07-01 ~ 2025-03-31
```

30 天 gap 防止滑动窗口 future 目标泄漏。

## 4. 方法

### 4.1 TimesFM 模型结构

通过逆向工程发现 TimesFM PyTorch 版本的内部结构：

```
TimesFM_2p5_200M_torch (wrapper, NOT nn.Module)
  └── .model (nn.Module, 231M params) ← fine-tune 入口
        ├── tokenizer (ResidualBlock)
        │     input: [batch, patches, 64] (32 values + 32 mask)
        │     output: [batch, patches, 1280]
        ├── stacked_xf (ModuleList, 20 Transformer layers)
        │     每层: multi-head attention (16 heads, dim=1280)
        ├── output_projection_point (ResidualBlock)
        │     output: [batch, patches, 1280] → reshape [batch, patches, 128, 10]
        └── output_projection_quantiles (ResidualBlock)
              output: [batch, patches, 10240] → reshape [batch, patches, 1024, 10]
```

关键发现：
- **wrapper 不是 nn.Module**，不能直接 `.to(device)` 或 `.parameters()`
- 真正的 nn.Module 是 `m.model`，有 231M 可训练参数
- `forward(inputs, masks)` 需要 patched 输入 `[batch, num_patches, 32]`
- output shape `[batch, patches, 1280]` = `[batch, patches, 128*10]`（128 步 × 10 quantile）
- index 5 = 点预测，其余 9 个 = P10-P90 deciles

### 4.2 RevIN 梯度问题

TimesFM 的 `decode()` 方法使用 RevIN（Reversible Instance Normalization）做输入标准化。在 `torch.no_grad()` 下运行。

直接使用 RevIN 的 denormalization 会**切断梯度流**（grad_norm = 0）。

解决方案：使用简单的 per-sample standardization（mean/std），在 normalized 空间计算 loss。

```python
# 不能用 RevIN (梯度断裂)
# 使用简单 standardization:
ctx_mean = ctx_flat.mean(dim=1, keepdim=True)
ctx_std = ctx_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
normed_ctx = (ctx_flat - ctx_mean) / ctx_std
normed_fut = (fut - ctx_mean) / ctx_std
# Loss 在 normalized 空间计算
loss = pinball_loss(pred, normed_fut)
```

### 4.3 Fine-tuning 方案演进

| 版本 | 损失函数 | 冻结层 | 训练数据 | 结果 |
|------|---------|--------|---------|------|
| v2 | MSE (点预测) | 全部训练 | 6 crypto | VaR 1% = 15.0% ✗ |
| v3 | Quantile (pinball) | freeze 15/20 | 6 crypto | VaR 1% = 14.6% ✗ |
| v4 | Quantile (pinball) | freeze 15/20 | 6 crypto + 9 tradfi | **VaR 1% = 0.6% ✓** |
| v5 | Quantile + all-patch | freeze 10/20 | 6 crypto + 9 tradfi | VaR 1% = 11.5% ✗ (过拟合) |
| v6 | Tail-weighted pinball | freeze 15/20 | 6 crypto + 9 tradfi (skip 5min) | VaR 1% = 10.6% ✗ |
| v7 | Tail-weighted pinball | freeze 15/20 | 6 crypto + 9 tradfi (含 5min) | **VaR 1% = 0.6% ✓ (with EVT)** |

#### 关键发现

1. **MSE 不能改善 quantile 校准**（v2）：优化目标和评估指标不对齐
2. **冻结层数是关键**（v4 vs v5）：freeze 15/20（训练 25% 参数）是 sweet spot。freeze 10（训练 50%）导致过拟合
3. **All-patch loss 在 transformer 中有根本性缺陷**（v5）：全局 attention 让中间 patch 通过 attention 机制直接"看到"其 target（因为 target 是后续 patch 的 context），等于作弊
4. **5min pre-training 对 VaR 1% 至关重要**（v4 vs v6）：5min 数据包含大量极端事件（flash crash, pump），是模型学习尾部风险的关键信号。跳过 5min 后 VaR 1% 从 0.6% 退到 10.6%
5. **传统金融数据有正则化效果**（v3 vs v4）：加入 S&P500, Gold 等 30K 样本防止 crypto-only 过拟合

#### 最终 fine-tune 配置 (v7)

```
Progressive: 5min → hourly → daily
冻结: tokenizer + 底部 15/20 transformer 层 (训练 ~58M / 231M params)
损失: Tail-weighted pinball loss
  P10/P90: 3x 权重
  P20/P80: 2x 权重
  其他 quantiles: 1x
  + 0.1 * point MSE
数据: 50% crypto + 50% tradfi (weighted sampling)
优化器: AdamW, weight_decay=1e-4
学习率: 5e-5 (5min) → 3e-5 (hourly) → 1e-5 (daily), cosine decay + warmup
Epochs: 25 / 25 / 40
```

### 4.4 VaR 校准方法

| VaR Level | 方法 | 原理 |
|-----------|------|------|
| VaR 5% | Conformal Prediction | 在 calibration set 上计算 P10 预测和实际值的残差分位数，得到修正量 |
| VaR 1% | EVT (Extreme Value Theory) | 用 Generalized Pareto Distribution 建模残差的左尾，外推到 P01 |

Conformal 对 VaR 5% 有效（correction ~0.003），但对 VaR 1% 不可靠（correction 过大时触发 fallback）。EVT 专为尾部建模设计，在 VaR 1% 上更稳定。

### 4.5 PELT Regime Detection

使用 ruptures PELT（Pruned Exact Linear Time）算法检测 regime 断点。

关键实现细节：
- 使用 **raw returns**（不是 z-scored），因为 z-score 会抹平 PELT 需要检测的 level shifts
- Cost model: `normal`（同时检测均值和方差变化），不是 `rbf`（太慢）或 `l2`（只检测均值）
- min_size=365（BTC）确保检测的是宏观 regime，不是短期波动

BTC 检测到 9 个断点，包括 2022-11 FTX crash 和 2024-02 ETF 新 regime。

## 5. 实验结果

### 5.1 TimesFM Zero-shot 能力评估

**6 个维度全面评估**（eval_timesfm.py）：

#### 方向预测：没用

| 频率 | Direction Accuracy |
|------|-------------------|
| Daily | 46.4% (低于随机) |
| Hourly | 53.5% (唯一超过 50%) |
| 5min | 52.5% |

对比传统方法（BTC daily）：AR(1) = 40.0%, GARCH = 35.7%。**所有模型在 crypto daily 上的 correlation 都是负的。**

#### Quantile 校准：非常好

Raw quantile calibration（无 conformal correction, daily test set）：

| Quantile | Target | Actual | Deviation |
|----------|--------|--------|-----------|
| P20 | 20% | 20.8% | 0.8% |
| P30 | 30% | 29.6% | 0.4% |
| P50 | 50% | 50.7% | 0.7% |
| P70 | 70% | 69.7% | 0.3% |
| **平均** | | | **1.7%** |

**TimesFM 的 raw quantile 在 crypto 上的平均校准偏差只有 1.7%。**

#### 跨资产：全部可用

| 资产 | Cal Deviation | VaR 10% Breach |
|------|--------------|----------------|
| SOL | 2.1% | 11.3% |
| BNB | 2.4% | 14.2% |
| BTC | 2.7% | 15.7% |
| DOGE | 3.1% | 12.4% |
| LINK | 3.2% | 15.0% |
| ETH | 3.9% | 17.2% |

6 个资产的 quantile calibration 全部在 4% 以内。

#### 跨频率：Hourly 最佳

| 频率 | Direction | Cal Deviation |
|------|-----------|--------------|
| Daily | 46.4% | 2.7% |
| **Hourly** | **53.5%** | **1.6%** |
| 5min | 52.5% | 4.3% |

#### 跨 Horizon：1-step 是唯一可用的

| Horizon | Cal Deviation |
|---------|--------------|
| 1-step | 2.7% |
| 5-step | 9.0% |
| 30-step | 19.4% |

### 5.2 Fine-tune 效果 (Hourly 评估, 3541 test samples)

| 模型 | VaR 5% | VaR 1% | Raw P10 |
|------|--------|--------|---------|
| Zero-shot | 4.0% | 11.1% | 0.126 |
| v2 (MSE, full params) | 4.0% | — | — |
| v3 (quantile, freeze 15) | 4.0% | — | — |
| v4 (quantile, tradfi, freeze 15) | 4.0% | 0.6%* | 0.110 |
| v5 (all-patch, freeze 10) | 3.9% | 11.5% | 0.130 |
| v6 (tail-weight, skip 5min) | 4.0% | 10.6% | 0.116 |
| **v7 (tail-weight, 含 5min) + EVT** | **3.8%** | **0.6%** | **0.116** |

*v4 的 VaR 1% = 0.6% 依赖 conformal，在不同 raw P10 coverage 下不稳定。v7 + EVT 更稳健。

Target: VaR 5% = 3-8%, VaR 1% = 0.5-2%。**两个都达标。**

### 5.3 未生效的尝试

#### 宏观数据 XGBoost 修正（Phase II-D）

| 方法 | Breach Rate |
|------|------------|
| Conformal | 4.0% ✅ |
| XGBoost macro correction (293 samples) | 13.5% ✗ |
| XGBoost macro correction (2243 samples) | 19.3% ✗ |

宏观指标和 TimesFM quantile residuals 之间的关系太弱或不稳定。Conformal 的统计修正远优于 ML 修正。

**结论：对于 TimesFM 的 quantile 校准，不需要宏观数据。**

#### All-patch loss (v5)

Loss 从 0.350 降到 0.063（5.6x 更低），但 VaR 1% 从 0.6% 退到 11.5%。

原因：Transformer 全局 attention 允许中间 patch 通过 attention 直接"看到" target 值（因为 target 是后续 patch 的 context），构成信息泄漏。Loss 下降是模型学会了作弊，不是学会了预测。

### 5.4 事件检测 (Phase II-B)

BTC hourly, 106K 数据点：
- Z-score (|z|>3σ) 检测到 1999 个单点异常
- CUSUM 检测到 899 个趋势偏移
- 合并后 2273 events (2.14% event rate)

Top events 全部对应真实事件：2020-03-12 COVID crash (-20.1%), 2019-09-24 flash crash (-9.4%), 2017-08-17 (+54.9%)

### 5.5 EVT VaR 1% (Phase II-C)

| 方法 | VaR 1% Breach Rate |
|------|-------------------|
| Conformal | 14.6% (isotonic fallback) |
| EVT (GPD) on daily | 0.36% |
| EVT (GPD) on hourly | **0.6%** |

GPD fit diagnostic: KS p-value = 0.25 (acceptable), converged = true.

## 6. 技术贡献

### 6.1 发现 TimesFM PyTorch 的 fine-tune 路径

TimesFM 2.5 的 PyTorch 版本没有官方 fine-tune API（TimesFMFinetuner 不存在于当前版本）。通过逆向工程发现：

1. `m.model` 是真正的 nn.Module（wrapper 不是）
2. Forward signature: `(inputs: [batch, patches, 32], masks: [batch, patches, 32])`
3. RevIN denormalization 会切断梯度 → 使用简单 standardization
4. Output `[batch, patches, 1280]` = `[batch, patches, 128, 10]`
5. 梯度正常流通：229/232 参数有梯度

### 6.2 Optimal fine-tune 配置

- **Freeze 15/20 transformer 层是 sweet spot**：保护预训练的通用时序表示
- **5min pre-training 对尾部风险校准至关重要**：极端事件集中在高频数据
- **传统金融数据有正则化效果**：50/50 crypto/tradfi 比 crypto-only 好
- **Last-patch-only loss 是唯一安全的选择**：all-patch loss 在 transformer 中导致 attention 作弊

### 6.3 VaR 校准：Conformal + EVT 组合

- VaR 5%: Conformal prediction（简单、稳定、修正量小）
- VaR 1%: EVT GPD tail model（conformal 对极端 quantile 不可靠）

## 7. 结论

### TimesFM 对 crypto 的适用性

| 能力 | 评估 |
|------|------|
| 方向预测 | **没用。** Daily 46.4%，不如猜硬币 |
| Quantile 校准 | **非常好。** 平均偏差 1.7%（P20-P70 全在 2% 以内） |
| VaR 5% | **可用。** Zero-shot + conformal = 4.0% breach rate |
| VaR 1% | **Fine-tune + EVT 后可用。** 0.6% breach rate |
| 跨资产 | **通用。** 6 个 crypto 资产全部可用 |
| 最佳频率 | **Hourly。** 方向 53.5% + 校准 1.6% |
| Fine-tune 效果 | **对 VaR 1% 有显著改善（11.1% → 0.6%）。** VaR 5% 边际改善 |

### 核心发现

1. **TimesFM 的价值在 calibrated uncertainty，不在 direction prediction**
2. **Fine-tune 有效，但需要正确的方法**（quantile loss + freeze 15 + 5min pre-training + tradfi regularization）
3. **VaR 1% 需要 EVT，conformal 不够**
4. **宏观数据对 quantile 校准没有帮助**（conformal 足够）
5. **5min 数据对尾部风险建模至关重要**（不是噪音，是极端事件的训练信号）

## 8. 局限性

1. **评估主要在 BTC 上**：其他资产的 fine-tune 效果未充分验证
2. **Test period 较短**（2024-07 ~ 2025-03, 9 个月）：可能不覆盖所有市场状态
3. **TimesFM quantile levels 固定为 deciles**：无法直接输出 P05/P01
4. **GPU 限制**（DGX Spark GB10）：batch size 和训练时间受限
5. **RevIN 替代方案未充分探索**：简单 standardization 可能不是最优选择
6. **No official fine-tune API**：依赖逆向工程，未来版本可能 break

## 9. 未来工作

1. **跨资产评估**：用 v7 模型评估 ETH/SOL/DOGE 的 VaR 表现
2. **更长回测期**：扩展 test period 到 2024-01 ~ 2025-12
3. **Autoregressive fine-tuning**：当前只优化单步预测，多步预测需要 autoregressive training
4. **JAX fine-tune 探索**：JAX 版本支持 covariates，可能有更好的 API
5. **Daily 生产 pipeline**：自动化每日风险信号生成
6. **Conformal prediction 改进**：adaptive conformal inference 处理非平稳性
7. **更多金融数据源**：外汇、期货 funding rate、链上数据

## 附录

### A. 硬件环境

- 训练/推理: NVIDIA DGX Spark (Grace Blackwell GB10, 128GB, Driver 580.142)
- Docker: NGC pytorch:25.11-py3 (CUDA 13.0.2)
- 代码开发: Mac Mini (M-series, Cloudflare Tunnel `ssh-macmini1`)

### B. 数据量统计

| Stage | Crypto Samples | TradFi Samples | Total |
|-------|---------------|----------------|-------|
| 5min | 40,329 | 30,231 | 70,560 |
| Hourly | 10,706 | 30,231 | 40,937 |
| Daily | 5,350 | 30,231 | 35,581 |

### C. 代码结构

```
crypto-regime/
├── pipeline/
│   ├── regime_detection.py      # PELT regime detection
│   ├── event_detection.py       # Z-score + CUSUM event detection
│   ├── evt.py                   # EVT (GPD) tail model
│   ├── macro_correction.py      # XGBoost macro correction (未生效)
│   ├── feature_engineering.py   # Rolling 252d z-score
│   └── fetch_*.py               # 数据获取
├── experiments/
│   ├── eval_timesfm.py          # 6 维度全面评估
│   ├── phase05_smoke_test.py    # Zero-shot + conformal baseline
│   ├── phase2_finetune_v7.py    # 最终 fine-tune 脚本
│   ├── phase3_risk_signals.py   # 风险信号输出
│   └── probe_*.py               # 模型逆向工程
└── results/                     # 所有实验结果 JSON
```
