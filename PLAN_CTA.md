# CTA + TimesFM 仓位管理研究计划

## 核心思路

```
CTA 策略 → 方向信号（做多/做空/空仓）
TimesFM  → 仓位大小（风险高时小仓，风险低时大仓）
合并      → 智能仓位管理的 CTA
```

CTA 解决 "买什么、什么时候买"，TimesFM 解决 "买多少"。

## 背景

### 已验证的 TimesFM 能力

| 能力 | 表现 | 用于 CTA |
|------|------|---------|
| 方向预测 | 46.4% daily（没用） | ❌ 不用于方向 |
| Quantile 校准 | 1.7% 偏差（优秀） | ✅ 仓位管理 |
| VaR 5% | 4.0% breach (hourly) | ✅ 风险预算 |
| VaR 1% | 0.6% breach (with EVT) | ✅ 极端风险保护 |
| Uncertainty ratio | 有效 | ✅ 波动率调整 |

### 为什么 CTA + 仓位管理

纯 CTA 的问题：
- 趋势策略在震荡市大幅回撤（仓位没调整）
- 波动率爆发时同样仓位 → 亏损放大
- 没有风险预算 → 不知道 "该下多大注"

加入 TimesFM 仓位管理后：
- 高 VaR 时自动减仓 → 熊市保护
- 低 VaR 时自动加仓 → 牛市跟上
- Anomaly 触发时紧急减仓 → 黑天鹅保护
- 不改变 CTA 的方向判断 → 不干扰 alpha

## Phase 1: 基础框架（1-2 周）

### 1.1 选择 CTA 基础策略

需要你提供：你现有的 CTA 策略是什么？

如果没有现成的，我们可以从经典策略开始：

**方案 A: 双均线趋势跟踪（最简单）**
```
- 快线 MA(20) 上穿慢线 MA(60) → 做多
- 快线下穿慢线 → 平仓（或做空）
- 简单、经典、容易理解和验证
```

**方案 B: Breakout（突破策略）**
```
- 价格突破 N 天最高点 → 做多
- 价格跌破 N 天最低点 → 做空
- Turtle Trading 的核心逻辑
```

**方案 C: 动量策略**
```
- 过去 N 天收益率排名 → 做多收益最高的
- 跨资产轮动：BTC vs ETH vs SOL
```

**方案 D: 你已有的策略**
```
- 直接接入你现有的信号源
- 只需要输出 +1（多）/ 0（空仓）/ -1（空）
```

### 1.2 仓位管理模块设计

```python
class TimesFMPositionManager:
    """用 TimesFM 信号管理 CTA 仓位大小"""

    def get_position_size(self, cta_signal, asset, risk_budget=0.02):
        """
        输入:
          cta_signal: +1 (多) / 0 (空仓) / -1 (空)
          asset: 'btc', 'eth', etc.
          risk_budget: 每日最大可接受亏损 (默认 2%)

        输出:
          position_size: 0.0 ~ 1.0 (占总资金比例)

        逻辑:
          1. 获取 TimesFM VaR 5%
          2. base_size = risk_budget / |VaR 5%|
          3. 乘以 cta_signal 的方向
          4. 如果 anomaly_flag → 减半
          5. 如果 price < MA(50) 且 cta_signal > 0 → 减 30%（逆势保护）
          6. clip to [0, max_position]
        """
```

### 1.3 回测框架

```
输入:
  - CTA 信号序列 (daily/hourly)
  - 资产价格数据
  - TimesFM 模型 (v8 fine-tuned)

处理:
  每个 rebalance 时间点:
    1. CTA 信号 → 方向
    2. TimesFM forecast → VaR, uncertainty, anomaly
    3. Position Manager → 仓位大小
    4. 执行交易（含手续费）

输出:
  - 净值曲线
  - Sharpe / Max DD / Calmar
  - 月度归因：alpha 来自 CTA，保护来自 TimesFM
  - 对比：原始 CTA vs CTA + TimesFM
```

## Phase 2: 实验设计（1 周）

### 2.1 对比实验

| 实验 | 描述 | 目的 |
|------|------|------|
| Exp 1 | 原始 CTA（固定仓位 100%）| Baseline |
| Exp 2 | CTA + 固定仓位 50% | 简单减仓效果 |
| Exp 3 | CTA + ATR 仓位管理 | 传统波动率管理 |
| Exp 4 | CTA + TimesFM VaR 仓位管理 | 核心实验 |
| Exp 5 | CTA + TimesFM + Anomaly 保护 | 完整系统 |

### 2.2 评估指标

| 指标 | 说明 |
|------|------|
| 总收益 | 绝对回报 |
| Sharpe Ratio | 风险调整回报（核心） |
| Max Drawdown | 最大回撤（TimesFM 应该显著改善） |
| Calmar Ratio | 年化收益 / 最大回撤 |
| 仓位利用率 | 平均仓位占比（太低说明太保守） |
| 保护次数 | TimesFM 减仓后市场确实下跌的次数 |
| 错过次数 | TimesFM 减仓后市场反而上涨的次数 |

### 2.3 资产范围

先从简单开始：
- **Phase 2a**: 只做 BTC（验证核心逻辑）
- **Phase 2b**: BTC + ETH（两资产分配）
- **Phase 2c**: 6 资产全部（完整组合）

## Phase 3: 优化（1-2 周）

### 3.1 Risk Budget 参数优化

```
搜索: risk_budget = [0.5%, 1%, 1.5%, 2%, 3%, 5%]
评估: 哪个 risk_budget 给出最好的 Sharpe
```

### 3.2 Rebalance 频率

```
搜索: 每小时 / 每 4 小时 / 每天 / 每周
评估: 频率 vs 交易成本 tradeoff
```

### 3.3 多策略组合

如果有多个 CTA 策略，TimesFM 可以做策略间的风险分配：
- 策略 A 不确定性高 → 分配少
- 策略 B 不确定性低 → 分配多

## Phase 4: 生产化（1 周）

### 4.1 实时信号

```
每小时运行:
  1. 拉最新数据 (fetch_*.py)
  2. CTA 信号计算
  3. TimesFM 推理 → VaR + uncertainty
  4. Position Manager → 目标仓位
  5. 输出交易指令 → 发送到交易所 / 通知
```

### 4.2 监控

- 仓位偏离报警
- VaR breach 报警
- 模型输出异常检测
- 日度 P&L 报告

## 技术架构

```
┌─────────────────────────────────────────────────┐
│  数据层                                          │
│  fetch_*.py → hourly/daily prices               │
│  macro indicators (VIX, DXY, etc.)              │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ CTA 策略  │ │ TimesFM  │ │ PELT     │
   │ 方向信号  │ │ VaR/不确│ │ Regime   │
   │ +1/0/-1  │ │ 定性信号 │ │ 状态     │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
           ┌──────────────────┐
           │ Position Manager  │
           │ 仓位 = 方向 × 大小 │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │ 执行 / 报告       │
           │ 交易指令 / 风险报告│
           └──────────────────┘
```

## 需要你确认的

1. **你有现成的 CTA 策略吗？** 还是从经典策略开始？
2. **资产范围：** 先只做 BTC？还是 BTC+ETH？
3. **交易频率：** Daily rebalance 还是 hourly？
4. **是否需要做空？** 还是只做多 + 空仓？
5. **手续费假设：** 单边多少 bps？
6. **回测时间段：** 2016-2026 全部？还是从某个时间开始？
