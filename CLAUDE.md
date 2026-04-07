# Crypto Regime

## 项目目标
基于 Google TimesFM 2.5 构建 BTC 风险信号系统。**不是价格预测工具，是风险仪表盘。**

核心输出：VaR、anomaly score、regime transition signal、uncertainty-based position sizing。

## 改进方案
详见 `PLAN.md`，按 Phase 0-3 顺序执行。

## 关键上下文
- 从 `btc-regime` fork 而来，v1 有3个致命 bug 和3个设计缺陷（见 PLAN.md 顶部）
- 训练在 DGX Spark 上跑（Grace Blackwell GB10, 128GB, CUDA 13.0），用 Docker 容器（见 Dockerfile）
- 代码修改在本地 Mac 上做，改完 scp 到 DGX 跑实验

## 执行顺序

### Phase 0: 修基础设施
1. `pipeline/feature_engineering.py` — StandardScaler 全局 fit 改成 rolling 252天 z-score
2. `pipeline/build_dataset.py` — 统一输出 schema 为 `{context, future, covariates, regime, dates}`
3. 所有文件的 `except: continue/pass` 改成带 logging 的 error handling
4. 重跑 `run_pipeline.py` 生成新数据
5. 重跑 `experiments/phase0_baselines.py` 确认 baseline

### Phase 1: 修 Regime Detection
1. `pipeline/regime_detection.py`（新文件）— 用 ruptures PELT 替代硬编码断点
2. `experiments/phase1_regime.py` — 评估改为 transition detection lag + false alarm rate

### Phase 2: 修 Fine-tune（最关键）
1. `experiments/phase2_finetune.py` — 重写。先检查 TimesFM 有没有原生 fine-tune API（`dir(timesfm)` 查 Finetuner 类），有则用，没有则手写正确的 forward pass（不用 model.forecast()，用 model.backbone + model.head 保持梯度流）
2. 在 DGX Docker 容器内跑实验

### Phase 3: 风险信号系统
1. `experiments/phase3_risk_signals.py`（新文件）— quantile forecast 提取、VaR、anomaly score、uncertainty ratio
2. 评估：quantile calibration、VaR breach rate、anomaly precision

## 技术约束
- Python 3.10+
- 训练环境：`nvcr.io/nvidia/pytorch:26.02-py3` Docker 容器
- TimesFM 版本：2.5-200m（PyTorch 优先，JAX 备选）
- 数据文件（.jsonl, .parquet）不入 git，在 DGX 上用 pipeline 生成

## 不要做的事
- 不要追求 direction accuracy > 55%，日度 crypto returns 接近随机游走
- 不要用 `model.forecast()` 做训练（它返回 numpy，梯度断裂）
- 不要用 `except: continue` 吞异常
- 不要用全局 StandardScaler（前视偏差）
- 不要硬编码 regime 断点日期


## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
