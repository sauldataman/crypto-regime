# Crypto Regime v2 — 改进方案

## 从 btc-regime 继承的问题

### 致命 Bug（必须修）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| B1 | fine-tune 梯度断裂 | `phase2_finetune.py:103-112` | model.forecast() 返回 numpy，包装成新 tensor 后 backward 到不了模型参数。模型零学习 |
| B2 | 静默吞异常 | 所有 `except: continue/pass` | 训练失败被隐藏，空模型被存为"best" |
| B3 | 数据格式不匹配 | builder 输出 target/xreg，consumer 读 context/future/covariates | 数据读不进去 |

### 设计缺陷（必须改）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| D1 | z-score 前视偏差 | `feature_engineering.py:56-73` | StandardScaler 在全量数据上 fit，未来信息泄露到过去。baseline 数字被污染 |
| D2 | regime 标签硬编码 | `regime_labeling.py` | XGBoost 分类器100%准确是因为在判断"哪一年"，对实时 regime 检测无意义 |
| D3 | 代码 vs 战略错位 | 全局 | 计划说"风险/anomaly/regime"，代码做"方向预测" |

---

## v2 架构设计

### 核心定位转变

```
v1: TimesFM → 预测价格方向 → direction accuracy
v2: TimesFM → 概率预测 → 量化不确定性 → 风险信号
```

v2 不追求"预测涨跌"，追求"告诉我现在的风险有多大"。

### 三层架构

```
Layer 1: Regime Detection
    输入：宏观 + 链上 + 价格
    方法：BOCPD (Bayesian Online Changepoint Detection) 或 rolling PELT
    输出：当前 regime 概率分布 + transition alert

Layer 2: TimesFM Probabilistic Forecast
    输入：近512天价格 + covariates（regime-conditional）
    方法：TimesFM 2.5 quantile head（原生支持 10/25/50/75/90 分位数）
    输出：未来 N 天的分位数预测

Layer 3: Risk Signal Generator
    输入：quantile forecasts + actual returns
    输出：
      - VaR (5%, 1%): 尾部风险估计
      - Anomaly score: |actual - median| / IQR，超过阈值报警
      - Uncertainty ratio: IQR / median，高 = 市场不确定性大
      - Position sizing signal: 1 / uncertainty（不确定性越大仓位越小）
```

### 文件结构

```
crypto-regime/
├── PLAN.md                          # 本文件
├── Dockerfile                       # NGC PyTorch + TimesFM
├── docker-run.sh                    # 容器启动脚本
├── requirements.txt
│
├── pipeline/                        # 数据管道（大部分复用 v1）
│   ├── fetch_btc.py                 # 复用
│   ├── fetch_macro.py               # 复用
│   ├── fetch_onchain.py             # 复用
│   ├── feature_engineering.py       # 修：rolling z-score 替代全局
│   ├── regime_detection.py          # 新：BOCPD 替代硬编码
│   └── build_dataset.py             # 修：统一 schema，context/future/covariates
│
├── experiments/
│   ├── phase0_baselines.py          # 修：修 z-score 后重跑 baseline
│   ├── phase1_regime.py             # 修：BOCPD + transition detection 评估
│   ├── phase2_finetune.py           # 重写：用 TimesFM 原生 fine-tune API
│   └── phase3_risk_signals.py       # 新：VaR / anomaly / uncertainty 输出
│
├── models/                          # 模型存储
├── results/                         # 实验结果
└── data/
    ├── raw/                         # 原始数据（复用）
    └── processed/                   # 处理后数据
```

---

## 修复计划（按优先级）

### Phase 0: 修基础设施（1天）

**0.1 修 feature_engineering.py — rolling z-score**
```python
# 旧：全局 StandardScaler（前视偏差）
scaler = StandardScaler()
out[cols] = scaler.fit_transform(out[cols])

# 新：rolling 252天窗口
for col in numeric_cols:
    rolling_mean = df[col].rolling(252, min_periods=60).mean()
    rolling_std  = df[col].rolling(252, min_periods=60).std()
    df[f"{col}"] = (df[col] - rolling_mean) / rolling_std.clip(lower=1e-8)
```

**0.2 统一数据 schema**
```python
# 所有 builder 统一输出格式：
{
    "context": [...],         # 历史价格序列（512点）
    "future": [...],          # 未来价格序列（预测目标）
    "covariates": [[...]],    # 协变量矩阵
    "regime": "late",         # regime 标签
    "dates": [...]            # 日期
}
```

**0.3 修 exception handling**
所有 `except: continue` 改成：
```python
except Exception as e:
    logger.warning(f"Batch {batch_idx} failed: {e}")
    failed_count += 1
    if failed_count > max_failures:
        raise RuntimeError(f"Too many failures ({failed_count})")
```

### Phase 1: 修 Regime Detection（1天）

**1.1 用 BOCPD 替代硬编码**
```python
# Bayesian Online Changepoint Detection
# 不需要预设断点，实时检测 regime 变化
from bayesian_changepoint_detection.online_changepoint_detection import OnlineCPD

# 或者用 ruptures 库的 PELT：
import ruptures
algo = ruptures.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=10)
```

**1.2 Regime classifier 改为 transition detector**
评估指标从"分类准确率"改为：
- Transition detection lag：检测到 regime 变化的延迟（天数）
- False alarm rate：误报率
- Soft probability calibration：概率输出是否校准

### Phase 2: 修 Fine-tune（核心，2天）

**2.1 用 TimesFM 原生 fine-tune API**

TimesFM 2.5 提供了 `TimesFMFinetuner`，不需要自己写训练循环：

```python
import timesfm

# 加载模型
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# 准备数据（pandas DataFrame 格式）
# TimesFM fine-tune 接受 DataFrame，不需要自己切 context/future
train_df = pd.DataFrame({
    "unique_id": ids,        # 时间序列标识
    "ds": dates,             # 时间戳
    "y": values,             # 目标值
    # covariates 作为额外列
})

# Fine-tune（内部处理梯度、scheduler、early stopping）
finetuner = timesfm.TimesFMFinetuner(
    model=model,
    train_dataset=train_df,
    learning_rate=1e-4,
    num_epochs=50,
    batch_size=256,
    early_stopping_patience=10,
)
finetuner.train()
model.save_pretrained("models/finetuned")
```

如果 TimesFM 的 fine-tune API 不支持 PyTorch（只支持 JAX），
则需要用 JAX 版本：
```python
model = timesfm.TimesFM_2p5_200M_jax.from_pretrained(
    "google/timesfm-2.5-200m-jax"
)
```

**2.2 备选方案：手写训练循环但修梯度流**

如果原生 API 不可用，手写循环需要：
```python
# 不用 model.forecast()，直接调用模型的 forward pass
model.train()
for ctx_b, fut_b in loader:
    ctx_b, fut_b = ctx_b.to(device), fut_b.to(device)
    optimizer.zero_grad()

    # 直接调用模型的 forward 方法（不是 forecast）
    # TimesFM 的内部结构是 decoder-only transformer
    # 需要查看源码确认 forward 的签名
    output = model.backbone(ctx_b)  # 保持在计算图内
    pred = model.head(output, horizon=fut_b.shape[1])

    loss = log_mse_loss(pred, fut_b)
    loss.backward()  # 梯度正确流回 model 参数
    optimizer.step()
```

**2.3 实验设计**

| Exp | 描述 | 数据 | 预期 |
|-----|------|------|------|
| 2.1 | Zero-shot baseline | 无 fine-tune | Sharpe ~0.26（参考论文）|
| 2.2 | Daily fine-tune | 820 daily samples | Sharpe 0.3-0.5 |
| 2.3 | Progressive: 5min→hourly→daily | 20K→13K→820 | Sharpe 0.4-0.6 |
| 2.4 | Regime-conditional | 分 regime fine-tune | 比 2.3 稍好 |

### Phase 3: 风险信号系统（核心价值，1天）

**3.1 Quantile forecast 提取**
```python
# TimesFM 2.5 原生支持 quantile output
model.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=30,
    use_continuous_quantile_head=True,  # 开启分位数输出
))

point_forecast, quantile_forecast = model.forecast(
    horizon=30, inputs=[context]
)
# quantile_forecast shape: [batch, horizon, num_quantiles]
# 默认 quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**3.2 风险信号计算**
```python
def compute_risk_signals(actual, quantiles):
    median = quantiles[:, 4]      # 50th percentile
    q10 = quantiles[:, 0]         # 10th percentile
    q90 = quantiles[:, 8]         # 90th percentile
    iqr = q90 - q10               # interquartile range

    return {
        "var_5pct": q10,                              # VaR (5%)
        "anomaly_score": abs(actual - median) / iqr,  # 超过2 = 异常
        "uncertainty": iqr / abs(median).clip(1e-8),   # 不确定性比率
        "position_weight": 1.0 / uncertainty,          # 仓位建议
        "regime_stress": regime_transition_prob,        # 来自 Layer 1
    }
```

**3.3 评估指标**

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| Quantile calibration | 实际落在各分位数内的比例 vs 理论比例 | 偏差 < 5% |
| VaR breach rate | 实际 < VaR 的频率 | 接近 5%（不是0%也不是20%） |
| Anomaly precision | anomaly_score > 2 时，后续5天是否真的有大波动 | > 60% |
| Uncertainty-weighted Sharpe | 用 position_weight 做仓位管理后的 Sharpe | > 0.5 |
| Regime transition lag | 从 BOCPD 检测到真实 regime 变化的延迟 | < 14天 |

---

## 对 CoinSummer 的实际价值

这套系统不是交易策略，是风险仪表盘：

1. **每日 VaR report**：告诉你"今天持仓的最大预期亏损是多少"
2. **Anomaly alert**：市场行为偏离模型预期时报警（可能是黑天鹅前兆）
3. **Regime transition signal**：宏观环境在变，提前调仓
4. **Position sizing**：不确定性高的时候自动减仓
5. **跨资产比较**：同一个模型跑 BTC/ETH/SOL，比较谁的 uncertainty 更大

---

## 时间线

| 阶段 | 工作 | 耗时 | 在哪跑 |
|------|------|------|--------|
| Phase 0 | 修基础设施（z-score、schema、exception） | 1天 | Mac |
| Phase 1 | BOCPD regime detection | 1天 | Mac |
| Phase 2 | TimesFM fine-tune（修梯度流） | 2天 | DGX |
| Phase 3 | 风险信号系统 | 1天 | DGX |
| 验证 | 跑全量实验 + 写报告 | 1天 | DGX |
| **总计** | | **~6天** | |

---

## 待确认

1. TimesFM PyTorch 版本是否有原生 fine-tune API？需要在 DGX 上 `pip install timesfm[torch]` 后检查 `dir(timesfm)` 确认。如果没有，需要用 JAX 版本或手写正确的训练循环
2. BOCPD 库是否支持 ARM？备选是 `ruptures`（纯 Python，肯定能跑）
3. 是否需要扩展到 ETH/SOL？当前只有 BTC 数据管道
