# BTC Regime Data Pipeline

## 目标
根据 Fish Lab Research BTC Regime Shift 论文，构建完整数据 pipeline：
- 拉取 BTC 价格、宏观数据、链上数据
- 按论文方法做特征工程（z-score标准化、7天lag）
- 识别 regime 分区（2017-05, 2020-11, 2024-02 三个断点）
- 输出适合 TimesFM fine-tune 的数据格式

## 论文关键设定
- 数据范围：2016-01 至今
- 总样本：~3741个日度观测值
- 关键变量：
  - 宏观：S&P500(^GSPC), NASDAQ(^IXIC), Gold(GC=F), Crude Oil(CL=F), CNY/USD, EUR/USD, JPY/USD
  - FRED：DXY(DTWEXBGS), 10Y Treasury(DGS10), M2(M2SL), VIX(VIXCLS), CPI(CPIAUCSL)
  - 链上：hash rate, transaction count, mining difficulty（从Blockchain.com API）
  - 衍生：BTC日收益率, 30日实现波动率(年化), 距下次减半天数, ETF二元指标(>=2024-01-11)
- 处理：z-score标准化 + 7天滞后

## Regime 分区
- Early: 2016-01 ~ 2020-11（链上+宏观混合）
- Late: 2020-11 ~ 2024-01（纯宏观主导）
- Post-ETF: 2024-01 ~ 今（ETF后稳定期）

## 输出格式
1. `data/raw/` - 原始数据（parquet格式）
2. `data/processed/btc_full.parquet` - 合并、标准化、滞后后的完整数据集
3. `data/processed/btc_by_regime.parquet` - 加入regime标签
4. `data/timesfm_train.jsonl` - TimesFM fine-tune格式（512天滑窗，含XReg协变量）
5. `reports/data_summary.md` - 数据质量报告

## 技术要求
- Python 3.10+
- 库：yfinance, fredapi, requests, pandas, numpy, scikit-learn, pyarrow
- FRED API Key 从环境变量 FRED_API_KEY 读（如没有就跳过FRED，用Yahoo Finance替代VIX）
- Blockchain.com API 无需key，直接HTTP请求
- 代码放在 `pipeline/` 目录下，模块化设计

## 任务顺序
1. 创建项目结构和requirements.txt
2. 写 pipeline/fetch_btc.py（BTC价格）
3. 写 pipeline/fetch_macro.py（宏观数据）
4. 写 pipeline/fetch_onchain.py（链上数据）
5. 写 pipeline/feature_engineering.py（特征工程）
6. 写 pipeline/regime_labeling.py（Regime标注）
7. 写 pipeline/build_timesfm_dataset.py（TimesFM数据集）
8. 写 run_pipeline.py（主入口，依次调用上述模块）
9. 测试跑通，输出数据质量报告

完成后运行：openclaw system event --text "Done: BTC regime data pipeline ready, outputs in ~/clawd/projects/btc-regime/data/" --mode now
