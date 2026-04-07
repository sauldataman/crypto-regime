# Phase II Roadmap

## 当前状态 (Phase I 完成)

### 已验证
- TimesFM quantile 校准好 (1.7% avg deviation on P20-P90)
- VaR 5% 可用 (4.01% breach rate, zero-shot + conformal)
- PELT regime detection 工作 (BTC 9 breakpoints, 检测到 FTX/ETF 等)
- 6 资产全部可用, hourly 是最佳频率

### 已发现限制
- TimesFMFinetuner API 不存在 (当前版本)
- PyTorch TimesFM 不支持 covariates
- Quantile levels 固定 P10-P90
- VaR 1% 不可用 (P10 离 P01 太远, conformal correction 过大)
- Anomaly score 阈值 2.0 太高 (0 flags), uncertainty ratio 除以 median 不稳定

---

## Phase II-A: 信号定义修复 (优先级最高, ~2h CC)

### Anomaly Score
- 当前: `|actual - median| / IQR`, 阈值 2.0 → 0 flags
- 修复: 动态阈值, rolling 30天 anomaly_score 的 P90

### Uncertainty Ratio
- 当前: `IQR / |median|` → median ≈ 0 时爆炸 (ratio > 10000)
- 修复: `IQR / realized_vol_30d`

### Position Weight
- 当前: `1 / uncertainty` → 极端值
- 修复: rank-based `1 - percentile_rank(IQR)`

---

## Phase II-B: 双层 Regime (3-4h CC)

### Layer 1 (Slow): PELT — 已实现
- 日度收益率 → 宏观 regime 断点
- 用途: 长期风险环境, position sizing 基准

### Layer 2 (Fast): Event Detection — 待实现
- 小时级/5分钟级数据
- 方法: 滚动 z-score (|return| > 3σ) + CUSUM (累积偏移)
- 输出: event flag (短期异常)

### 整合
```
TimesFM quantile (基础)
  → PELT regime correction (慢层: 按 regime 调整 IQR 宽度)
  → Event detection override (快层: 事件触发时强制高 uncertainty)
  → Final risk signals
```

---

## Phase II-C: VaR 1% + EVT (2-3h CC)

### 问题
P10 离 P01 太远, conformal correction 2.34 过大

### 方案: Extreme Value Theory
- 用 Generalized Pareto Distribution 建模尾部
- 从 P10 外推到 P01/P05
- `scipy.stats.genpareto`, ~100 LOC
- 金融行业标准做法

---

## Phase II-D: 宏观数据整合 (4-5h CC)

### 问题
PyTorch TimesFM 不支持 covariates

### 方案 C+D 组合 (推荐, 不依赖 JAX)

**Stage 1:** TimesFM 做纯价格预测 (已验证有效)

**Stage 2:** 宏观数据修正 TimesFM quantile 输出
- 训练 XGBoost/Linear 模型
- 输入: TimesFM quantiles + 当前宏观指标 (S&P500 z, VIX, DXY, Treasury 10Y, BTC vol, hash rate)
- 输出: 修正后的 quantile
- 不改 TimesFM, 灵活可解释

**Stage 3:** Regime-conditional correction
- 按 PELT regime 对 quantile 做不同修正
- High-vol regime → 放大 IQR
- Low-vol regime → 收窄 IQR

### 备选: JAX 路径
- JAX TimesFM 原生支持 `forecast_with_covariates()`
- 需要: `pip install timesfm[jax]` + JAX CUDA 配置
- 可以直接传入宏观指标作为 dynamic covariates

---

## Phase II-E: Fine-tune (未知时长)

### 路径 C: 从 PR #223 安装 Finetuner
```bash
pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@refs/pull/223/head"
python -c "import timesfm; print([a for a in dir(timesfm) if 'inetun' in a.lower()])"
```

### 路径 B: 探查模型内部结构
```bash
python -c "
import timesfm, torch
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
print('Type:', type(m))
print('Children:', [n for n,_ in m.named_children()])
print('Is nn.Module:', isinstance(m, torch.nn.Module))
x = torch.randn(1, 512)
try:
    out = m(x)
    print('Forward works:', type(out))
except Exception as e:
    print('Forward failed:', e)
"
```

### 路径 A: JAX fine-tune
- JAX 版本可能有更完整的训练 API
- 用 `jax.grad` + `optax` 做梯度下降

---

## 执行顺序

| Phase | 内容 | 依赖 | 估时 (CC) |
|-------|------|------|-----------|
| II-A | 信号定义修复 | 无 | 2h |
| II-B | 双层 regime | II-A | 3-4h |
| II-C | VaR 1% EVT | 无 | 2-3h |
| II-D | 宏观数据整合 | II-A | 4-5h |
| II-E | Fine-tune 探索 | 无 | 未知 |

II-A 和 II-C 可以并行。II-B 和 II-D 可以并行。II-E 独立。
