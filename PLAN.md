# Crypto Regime v2 — 实施计划

详细 review 见 CEO Plan: `~/.gstack/projects/sauldataman-crypto-regime/ceo-plans/2026-04-06-btc-risk-signals.md`

## 核心定位

```
不是价格预测工具，是风险仪表盘。
TimesFM 2.5 → 概率预测 → conformal 校准 → 1-day VaR / anomaly / uncertainty
```

## v1 遗留问题（全部已修复）

| # | 问题 | 修复 | 状态 |
|---|------|------|------|
| B1 | fine-tune 梯度断裂 (model.forecast() 返回 numpy) | phase2_finetune.py 重写，用 model forward 或 TimesFMFinetuner | ✅ |
| B2 | 静默吞异常 (except: pass/continue) | 全部替换为具名异常 + logging | ✅ |
| B3 | 数据 schema 不匹配 (target/xreg vs context/future) | 统一为 {context, future, dates, regime}，删除 build_timesfm_dataset.py | ✅ |
| D1 | z-score 前视偏差 (全局 StandardScaler) | rolling 252 天 z-score, std clip 1e-8 | ✅ |
| D2 | regime 标签硬编码 | ruptures PELT 替代，regime_detection.py | ✅ |
| D3 | 代码 vs 战略错位 (做方向预测) | 重新定位为风险信号系统 | ✅ |

## 架构

```
Layer 1: Regime Detection (ruptures PELT)
    输入：日度收益率
    输出：regime breakpoints + labels
    注意：BOCPD 是 Phase II 候选，Phase I 用 PELT (batch)

Layer 2: TimesFM 2.5 Probabilistic Forecast
    输入：近512天价格序列（无 covariates，PyTorch 不支持）
    自定义 quantiles: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    输出：1-day ahead 分位数预测

Layer 3: Risk Signal Generator
    输入：quantile forecasts + conformal 校准
    输出：
      - 1-day VaR (5%, 1%): conformal calibrated
      - Anomaly score: |actual - median| / IQR
      - Uncertainty ratio: IQR / |median|
      - Position sizing: 1 / uncertainty
```

## 数据

6 个核心资产: BTC, ETH, SOL, BNB, DOGE, LINK

| 频率 | BTC | ETH | SOL | BNB | DOGE | LINK |
|------|-----|-----|-----|-----|------|------|
| Daily | 2014-09 (Yahoo) | 2015-08 | 2020-04 | 2017-07 | 2013-12 | 2017-09 |
| Hourly | 2014-01 (CryptoCompare+Binance) | 2016-01 | 2020-08 | 2017-11 | 2017-01 | 2019-01 |
| 5min | 2013-10 (Kraken+Binance) | 2015-08 (Kraken+Binance) | 2020-08 | 2017-11 | 2019-07 | 2017-09 |
| 1min | 2013-10 (Kraken) | 2015-08 (Kraken) | — | — | 2019-12 (Kraken) | — |

## Temporal Split (30 天 gap 防泄漏)

```
Train:       2011-01 ~ 2022-06-01
Val:         2022-07-01 ~ 2023-06-30  (early stopping + 超参)
Calibration: 2023-07-01 ~ 2024-06-30  (conformal prediction, ~250 天)
Test:        2024-07-01 ~ 2025-03-31  (report only, 不做任何选择)
```

## 执行计划

### Phase 0: 修基础设施 — ✅ 已完成 (Mac)

- [x] Rolling 252 天 z-score 替代 StandardScaler
- [x] 统一 schema {context, future, dates, regime}，删除 build_timesfm_dataset.py
- [x] 所有 except:pass 替换为具名异常
- [x] 修 ARIMA baseline sign bug
- [x] 新增 regime_detection.py (ruptures PELT)
- [x] 数据拉取: daily (Yahoo/Binance), hourly (CryptoCompare+Binance), 5min (Kraken+Binance)
- [x] Pipeline 顺序: fetch → feature_eng → PELT → build_dataset
- [x] requirements.txt 更新

### Phase 0.5: 端到端 Smoke Test — 待跑 (DGX)

**DECISION GATE: 如果 zero-shot 已经达到 VaR breach rate 3-8%，Phase 2 边际价值低。**

- [ ] 安装 TimesFM: `pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]`
- [ ] 验证 TimesFMFinetuner API 是否存在
- [ ] Zero-shot TimesFM + conformal calibration
- [ ] 输出 1-day VaR 5% breach rate
- [ ] 脚本: `python experiments/phase05_smoke_test.py`

### Phase 1: PELT Regime Detection — ✅ 代码已写，调参中 (Mac)

- [x] regime_detection.py 实现
- [x] 集成到 run_pipeline.py
- [ ] PELT penalty 调参 (BTC 当前 0 个断点，需要降低 penalty)
- [ ] 跨资产 breakpoint 同步性分析

### Phase 2: Progressive Fine-Tuning — 待跑 (DGX, conditional on Phase 0.5)

仅在 Phase 0.5 显示 zero-shot 不够好时执行。

- [ ] Progressive: 5min → hourly → daily (3 stages)
- [ ] 6 资产价格序列 (无 covariates)
- [ ] TimesFMFinetuner 或 manual forward-pass
- [ ] Gradient norm 监控, NaN loss 检测
- [ ] 脚本: `python experiments/phase2_finetune.py`

### Phase 3: 风险信号系统 — 待跑 (DGX)

- [ ] 加载最佳模型 (progressive > daily > zero-shot)
- [ ] Conformal calibration (方法在 Phase 0.5 确定)
- [ ] 1-day VaR 5%/1%, anomaly score, uncertainty, position sizing
- [ ] 对比 fine-tuned vs zero-shot
- [ ] 脚本: `python experiments/phase3_risk_signals.py`

## 成功标准

| 指标 | 目标 |
|------|------|
| 端到端可运行 | Zero-shot + conformal 产出 VaR |
| 1-day VaR 5% breach rate | 3% ~ 8% |
| Quantile calibration 偏差 | < 5% |
| Fine-tuned > zero-shot | 如果跑了 Phase 2，校准更好 |

## 关键决策 (CEO + Eng Review 确认)

1. **先测 zero-shot + conformal** — fine-tune 可能不必要 (Claude + Codex 共识)
2. **不用 covariates** — PyTorch TimesFM 不支持 (confidence 10/10)
3. **1-day VaR** — CoinSummer 日度风险报告场景
4. **Conformal fallback 在 cal set 决定** — test set 只报告，不做选择
5. **30 天 temporal gap** — 防止滑动窗口 future 泄漏
6. **ruptures PELT** — 不是 BOCPD (BOCPD 是 Phase II online 候选)
7. **自定义 quantile levels** — [0.01, 0.05, ..., 0.99]，不用默认 deciles

## 文件结构

```
crypto-regime/
├── PLAN.md                              # 本文件
├── CLAUDE.md                            # 项目约束和上下文
├── Dockerfile                           # NGC PyTorch + TimesFM
├── requirements.txt                     # 含 ruptures, ccxt, mapie 等
├── run_pipeline.py                      # 主 pipeline: fetch→feature→PELT→build
│
├── pipeline/
│   ├── fetch_btc.py                     # BTC daily from 2010 (Yahoo)
│   ├── fetch_macro.py                   # 宏观数据 from 2010
│   ├── fetch_onchain.py                 # 链上数据
│   ├── fetch_multi_asset.py             # ETH/SOL/BNB/DOGE/LINK daily
│   ├── fetch_5min.py                    # 6 资产 5min (Binance from 2017)
│   ├── fetch_hourly.py                  # 6 资产 hourly (CryptoCompare+Binance)
│   ├── fetch_1min.py                    # CryptoCompare 1min (future use)
│   ├── merge_kraken.py                  # Kraken 早期数据合并
│   ├── feature_engineering.py           # Rolling 252d z-score + 7d lag
│   ├── regime_detection.py              # ruptures PELT
│   ├── regime_labeling.py               # 硬编码 fallback (PELT 失败时)
│   └── build_multi_asset_dataset.py     # {context, future, dates, regime}
│
├── experiments/
│   ├── phase0_baselines.py              # AR, ARIMA baselines
│   ├── phase1_regime_classifier.py      # XGBoost regime (legacy)
│   ├── phase05_smoke_test.py            # ★ Zero-shot + conformal (DGX)
│   ├── phase2_finetune.py               # ★ Progressive fine-tune (DGX)
│   ├── phase3_risk_signals.py           # ★ Risk signal output (DGX)
│   └── phase3_inference.py              # Legacy inference (deprecated)
│
├── data/raw/                            # 原始数据 (gitignored)
├── data/processed/                      # Processed 数据 (gitignored)
├── models/                              # 模型文件 (gitignored)
└── results/                             # 实验结果
```

## Deferred (Phase II+)

- Covariates in fine-tuning (需要 JAX 或 PyTorch covariate API)
- BOCPD online regime detection
- Multi-model ensemble (TimesFM + Chronos + Moirai)
- CoinSummer 日度报告 + Streamlit dashboard
- 论文写作
- 1min 数据训练
