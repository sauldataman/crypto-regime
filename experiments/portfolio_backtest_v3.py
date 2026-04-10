"""
Portfolio backtest v3: Risk Budget approach.

Core idea: "I'm willing to lose at most X% on a bad day (VaR 5%)"
  → max_weight = risk_budget / |asset_VaR|
  → Low VaR (calm market) → big position
  → High VaR (volatile market) → small position
  → Naturally captures bull/bear: bull markets have lower vol → bigger position

Also adds simple trend filter:
  - Price above 50-period moving average → allow long
  - Price below → reduce to minimum or go to cash

Usage:
  python experiments/portfolio_backtest_v3.py --freq daily --start 2016-01-01
  python experiments/portfolio_backtest_v3.py --freq hourly --start 2024-01-01
  python experiments/portfolio_backtest_v3.py --risk-budget 0.03  # 3% daily risk tolerance
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CONTEXT_LEN = 512
HORIZON = 1

ASSETS_CONFIG = {
    "btc": {"hourly": "data/raw/hourly/btc_1h.parquet", "daily": "data/raw/btc_price.parquet"},
    "eth": {"hourly": "data/raw/hourly/eth_1h.parquet", "daily": "data/raw/eth_price.parquet"},
}


def load_price_and_returns(asset: str, freq: str):
    """Load both price and returns."""
    path = ROOT / ASSETS_CONFIG[asset][freq]
    if not path.exists():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    price = df[close_col[0]]
    returns = np.log(price / price.shift(1)).dropna()
    return price, returns


def load_model(model_type="auto"):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    if model_type != "zero-shot":
        import torch
        for name in ["timesfm_v7_progressive_best.pt", "timesfm_v4_progressive_best.pt",
                     "timesfm_progressive_best.pt"]:
            ckpt = ROOT / "models" / name
            if ckpt.exists():
                m.model.load_state_dict(torch.load(ckpt, weights_only=True))
                logger.info("Loaded: %s", ckpt.name)
                break
    m.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN, max_horizon=HORIZON,
        normalize_inputs=True, use_continuous_quantile_head=True,
    ))
    return m


def get_var_signal(model, returns, loc):
    if loc < CONTEXT_LEN:
        return None
    context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
    try:
        point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
        qf = quantile_fc[0][0]
        n_q = len(qf)
        offset = 1 if n_q >= 10 else 0
        q10 = float(qf[0 + offset])
        q50 = float(qf[4 + offset]) if 4 + offset < n_q else 0
        q90 = float(qf[8 + offset]) if 8 + offset < n_q else 0
        return {"var_5pct": q10, "median": q50, "iqr": max(q90 - q10, 1e-10)}
    except (RuntimeError, ValueError, IndexError):
        return None


def run_backtest(model, btc_price, btc_returns, eth_price, eth_returns,
                  start, end, risk_budget, rebalance_every, freq, trend_period=50):
    """
    Risk Budget strategy:
      weight = risk_budget / |VaR 5%|
      + trend filter: only long if price > MA(trend_period)
    """
    common_start = pd.Timestamp(start)
    common_end = pd.Timestamp(end)

    btc_r = btc_returns[(btc_returns.index >= common_start) & (btc_returns.index <= common_end)]
    eth_r = eth_returns[(eth_returns.index >= common_start) & (eth_returns.index <= common_end)]
    common_dates = btc_r.index.intersection(eth_r.index).sort_values()

    step = 6 if freq == "hourly" else 1
    eval_dates = common_dates[::step]

    logger.info("Backtest: %s to %s, %d steps, risk_budget=%.1f%%, trend_ma=%d",
                eval_dates[0].date(), eval_dates[-1].date(),
                len(eval_dates), risk_budget * 100, trend_period)

    # Pre-compute moving averages for trend filter
    btc_ma = btc_price.rolling(trend_period).mean()
    eth_ma = eth_price.rolling(trend_period).mean()

    # NAV tracking
    strat_nav = [1.0]
    bench_btc_nav = [1.0]
    bench_eth_nav = [1.0]
    bench_equal_nav = [1.0]

    # Current weights
    w_btc = 0.0
    w_eth = 0.0
    w_cash = 1.0

    allocation_log = []
    rebal_count = 0

    # Fallback VaR (use if model signal unavailable)
    # Use rolling 30-period std * 1.65 as VaR 5% proxy
    btc_vol = btc_returns.rolling(30).std()
    eth_vol = eth_returns.rolling(30).std()

    t0 = time.time()

    for i, date in enumerate(eval_dates):
        btc_ret = float(btc_r.loc[date]) if date in btc_r.index else 0
        eth_ret = float(eth_r.loc[date]) if date in eth_r.index else 0

        # Portfolio return
        strat_ret = w_btc * btc_ret + w_eth * eth_ret
        strat_nav.append(strat_nav[-1] * np.exp(strat_ret))
        bench_btc_nav.append(bench_btc_nav[-1] * np.exp(btc_ret))
        bench_eth_nav.append(bench_eth_nav[-1] * np.exp(eth_ret))
        bench_equal_nav.append(bench_equal_nav[-1] * np.exp(0.5 * btc_ret + 0.5 * eth_ret))

        # Rebalance
        if (i + 1) % rebalance_every == 0:
            rebal_count += 1

            # Get VaR signals
            btc_loc = btc_returns.index.get_loc(date) if date in btc_returns.index else -1
            eth_loc = eth_returns.index.get_loc(date) if date in eth_returns.index else -1

            btc_sig = get_var_signal(model, btc_returns, btc_loc) if btc_loc >= CONTEXT_LEN else None
            eth_sig = get_var_signal(model, eth_returns, eth_loc) if eth_loc >= CONTEXT_LEN else None

            # VaR values (absolute)
            if btc_sig:
                btc_var = abs(btc_sig["var_5pct"])
            elif date in btc_vol.index and not np.isnan(btc_vol.loc[date]):
                btc_var = float(btc_vol.loc[date]) * 1.65
            else:
                btc_var = 0.03  # default 3%

            if eth_sig:
                eth_var = abs(eth_sig["var_5pct"])
            elif date in eth_vol.index and not np.isnan(eth_vol.loc[date]):
                eth_var = float(eth_vol.loc[date]) * 1.65
            else:
                eth_var = 0.04

            # Risk budget allocation
            # "With risk_budget% max loss tolerance, how much can I hold?"
            btc_max_weight = min(risk_budget / max(btc_var, 0.001), 1.0)
            eth_max_weight = min(risk_budget / max(eth_var, 0.001), 1.0)

            # Trend filter: only go long if price > MA
            btc_trend_ok = True
            eth_trend_ok = True
            if date in btc_ma.index and date in btc_price.index:
                ma_val = btc_ma.loc[date]
                price_val = btc_price.loc[date]
                if not np.isnan(ma_val) and not np.isnan(price_val):
                    btc_trend_ok = price_val > ma_val
            if date in eth_ma.index and date in eth_price.index:
                ma_val = eth_ma.loc[date]
                price_val = eth_price.loc[date]
                if not np.isnan(ma_val) and not np.isnan(price_val):
                    eth_trend_ok = price_val > ma_val

            # Apply trend filter
            if not btc_trend_ok:
                btc_max_weight *= 0.2  # reduce to 20% if below MA
            if not eth_trend_ok:
                eth_max_weight *= 0.2

            # Total crypto can't exceed 100%
            total_crypto = btc_max_weight + eth_max_weight
            if total_crypto > 1.0:
                scale = 1.0 / total_crypto
                btc_max_weight *= scale
                eth_max_weight *= scale

            w_btc = btc_max_weight
            w_eth = eth_max_weight
            w_cash = max(0, 1.0 - w_btc - w_eth)

            if rebal_count % max(1, len(eval_dates) // rebalance_every // 20) == 0:
                logger.info("  [%s] BTC:%.0f%% ETH:%.0f%% Cash:%.0f%% | VaR: BTC=%.2f%% ETH=%.2f%% | Trend: BTC=%s ETH=%s",
                            date.date(), w_btc*100, w_eth*100, w_cash*100,
                            btc_var*100, eth_var*100,
                            "UP" if btc_trend_ok else "DOWN",
                            "UP" if eth_trend_ok else "DOWN")

                allocation_log.append({
                    "date": str(date.date()),
                    "w_btc": round(w_btc, 3), "w_eth": round(w_eth, 3), "w_cash": round(w_cash, 3),
                    "btc_var": round(btc_var, 5), "eth_var": round(eth_var, 5),
                    "btc_trend": btc_trend_ok, "eth_trend": eth_trend_ok,
                })

        if (i + 1) % 5000 == 0:
            logger.info("  Step %d/%d (%.1f sec)", i + 1, len(eval_dates), time.time() - t0)

    # Metrics
    strat_nav = np.array(strat_nav)
    bench_btc_nav = np.array(bench_btc_nav)
    bench_eth_nav = np.array(bench_eth_nav)
    bench_equal_nav = np.array(bench_equal_nav)

    def calc_metrics(nav, name):
        rets = np.diff(np.log(nav))
        total = (nav[-1] / nav[0] - 1) * 100
        ann_f = (252 * 24 / step) if freq == "hourly" else 252
        n_years = len(rets) / ann_f if ann_f > 0 else 1
        ann_ret = total / n_years if n_years > 0 else 0
        vol = np.std(rets) * np.sqrt(ann_f) * 100 if len(rets) > 0 else 0
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(ann_f) if np.std(rets) > 0 else 0
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        max_dd = float(np.min(dd)) * 100
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        return {
            "name": name,
            "total_return_pct": round(total, 2),
            "ann_return_pct": round(ann_ret, 2),
            "ann_vol_pct": round(vol, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar": round(calmar, 3),
            "final_nav": round(float(nav[-1]), 4),
        }

    s = calc_metrics(strat_nav, "Risk Budget + Trend")
    b_eq = calc_metrics(bench_equal_nav, "Equal Weight BTC+ETH")
    b_btc = calc_metrics(bench_btc_nav, "100% BTC")
    b_eth = calc_metrics(bench_eth_nav, "100% ETH")

    print("\n" + "=" * 75)
    print(f"  PORTFOLIO BACKTEST v3: Risk Budget + Trend Filter")
    print(f"  Period: {start} to {end} ({freq})")
    print(f"  Risk budget: {risk_budget*100:.1f}% max daily loss (VaR 5%)")
    print(f"  Trend filter: {trend_period}-period moving average")
    print("=" * 75)
    print()
    print(f"  {'Metric':<22} {'Strategy':>12} {'BTC+ETH EW':>12} {'100% BTC':>12} {'100% ETH':>12}")
    print("  " + "-" * 70)

    for key, label in [
        ("total_return_pct", "Total Return"),
        ("ann_return_pct", "Ann. Return"),
        ("ann_vol_pct", "Ann. Volatility"),
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown"),
        ("calmar", "Calmar Ratio"),
        ("final_nav", "Final NAV ($1)"),
    ]:
        unit = "%" if "pct" in key else ("$" if "nav" in key else "")
        print(f"  {label:<22} {s[key]:>11.2f}{unit} {b_eq[key]:>11.2f}{unit} {b_btc[key]:>11.2f}{unit} {b_eth[key]:>11.2f}{unit}")

    print()

    # Highlight
    all_m = [s, b_eq, b_btc, b_eth]
    best_sharpe = max(all_m, key=lambda x: x["sharpe"])
    least_dd = max(all_m, key=lambda x: x["max_drawdown_pct"])
    best_calmar = max(all_m, key=lambda x: x["calmar"])
    print(f"  Best Sharpe:  {best_sharpe['name']} ({best_sharpe['sharpe']:.3f})")
    print(f"  Least DD:     {least_dd['name']} ({least_dd['max_drawdown_pct']:.1f}%)")
    print(f"  Best Calmar:  {best_calmar['name']} ({best_calmar['calmar']:.3f})")
    print("=" * 75)

    return {
        "config": {"start": start, "end": end, "freq": freq,
                    "risk_budget": risk_budget, "trend_period": trend_period},
        "strategy": s, "benchmark_ew": b_eq, "benchmark_btc": b_btc, "benchmark_eth": b_eth,
        "allocation_log": allocation_log,
        "nav": {
            "strategy": strat_nav[::max(1, len(strat_nav)//500)].tolist(),
            "btc": bench_btc_nav[::max(1, len(bench_btc_nav)//500)].tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", choices=["hourly", "daily"], default="daily")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--model", choices=["auto", "zero-shot"], default="auto")
    parser.add_argument("--risk-budget", type=float, default=0.02,
                        help="Max acceptable daily loss at VaR 5%% level (default 2%%)")
    parser.add_argument("--trend-period", type=int, default=50,
                        help="Moving average period for trend filter (default 50)")
    parser.add_argument("--rebal", type=int, default=None)
    args = parser.parse_args()

    rebal = args.rebal or (24 if args.freq == "hourly" else 1)

    logger.info("=" * 60)
    logger.info("Portfolio Backtest v3: Risk Budget + Trend")
    logger.info("  Risk budget: %.1f%%, Trend MA: %d", args.risk_budget * 100, args.trend_period)
    logger.info("=" * 60)

    model = load_model(args.model)

    btc_price, btc_returns = load_price_and_returns("btc", args.freq)
    eth_price, eth_returns = load_price_and_returns("eth", args.freq)
    logger.info("BTC: %d rows, ETH: %d rows", len(btc_returns), len(eth_returns))

    result = run_backtest(model, btc_price, btc_returns, eth_price, eth_returns,
                           args.start, args.end, args.risk_budget, rebal, args.freq,
                           args.trend_period)

    out_path = RESULTS / "portfolio_backtest_v3.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("\nSaved: %s", out_path)


if __name__ == "__main__":
    main()
