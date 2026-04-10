"""
Portfolio backtest v2: BTC + ETH + Cash, VaR-driven allocation.

Strategy:
  - 3 positions: BTC, ETH, Cash (risk-free)
  - Every rebalance:
    1. Get VaR 5% for BTC and ETH
    2. Low VaR (small loss risk) → high crypto weight
    3. High VaR (large loss risk) → shift to cash
    4. VaR breach → emergency shift to mostly cash
  - Cash = 1 - BTC_weight - ETH_weight (earns 0%)

Allocation formula:
  risk_score = |VaR 5%|  (bigger = more risky)
  raw_weight = 1 / risk_score  (inverse risk)
  crypto_total = min(0.95, base_crypto * risk_adjustment)
  BTC:ETH ratio by inverse risk
  Cash = 1 - crypto_total

Usage:
  python experiments/portfolio_backtest_v2.py
  python experiments/portfolio_backtest_v2.py --freq daily
  python experiments/portfolio_backtest_v2.py --start 2024-07-01 --end 2025-12-31
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

# Strategy parameters
BASE_CRYPTO_ALLOCATION = 0.80  # 80% crypto / 20% cash in normal conditions
MAX_CRYPTO = 0.95              # never more than 95% in crypto
MIN_CRYPTO = 0.10              # always at least 10% in crypto (don't fully exit)
EMERGENCY_CRYPTO = 0.20        # after VaR breach, drop to 20% crypto


def load_returns(asset: str, freq: str) -> pd.Series:
    path = ROOT / ASSETS_CONFIG[asset][freq]
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        return pd.Series(dtype=float)
    close = df[close_col[0]]
    return np.log(close / close.shift(1)).dropna()


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


def get_var_signal(model, returns: pd.Series, loc: int) -> dict:
    """Get VaR signal from TimesFM."""
    if loc < CONTEXT_LEN:
        return None
    context = returns.iloc[loc - CONTEXT_LEN:loc].values.tolist()
    try:
        point_fc, quantile_fc = model.forecast(horizon=HORIZON, inputs=[context])
        qf = quantile_fc[0][0]
        n_q = len(qf)
        offset = 1 if n_q >= 10 else 0

        q10 = float(qf[0 + offset])
        q50 = float(qf[4 + offset]) if 4 + offset < n_q else float(qf[n_q // 2])
        q90 = float(qf[8 + offset]) if 8 + offset < n_q else float(qf[-1])
        iqr = q90 - q10

        return {
            "var_5pct": q10,   # P10 as VaR 5% proxy (before conformal)
            "median": q50,
            "iqr": max(iqr, 1e-10),
            "point": float(point_fc[0][0]),
        }
    except (RuntimeError, ValueError, IndexError):
        return None


def run_backtest(model, btc_returns, eth_returns, start, end,
                  rebalance_every=24, freq="hourly"):
    """Run BTC+ETH+Cash portfolio backtest."""
    common_start = pd.Timestamp(start)
    common_end = pd.Timestamp(end)

    btc_r = btc_returns[(btc_returns.index >= common_start) & (btc_returns.index <= common_end)]
    eth_r = eth_returns[(eth_returns.index >= common_start) & (eth_returns.index <= common_end)]
    common_dates = btc_r.index.intersection(eth_r.index).sort_values()

    logger.info("Backtest: %s to %s, %d steps, rebal every %d",
                common_dates[0].date(), common_dates[-1].date(),
                len(common_dates), rebalance_every)

    step = 6 if freq == "hourly" else 1
    eval_dates = common_dates[::step]

    # Strategy NAV
    strat_nav = [1.0]
    # Benchmark 1: Equal weight BTC+ETH (no cash)
    bench_equal_nav = [1.0]
    # Benchmark 2: 100% BTC buy-and-hold
    bench_btc_nav = [1.0]

    # Current allocation
    w_btc = 0.40
    w_eth = 0.40
    w_cash = 0.20

    # State tracking
    btc_breach_count = 0
    eth_breach_count = 0
    recovery_timer = 0

    allocation_log = []
    t0 = time.time()

    for i, date in enumerate(eval_dates):
        # Actual returns
        btc_ret = float(btc_r.loc[date]) if date in btc_r.index else 0
        eth_ret = float(eth_r.loc[date]) if date in eth_r.index else 0

        # Strategy return (cash earns 0)
        strat_ret = w_btc * btc_ret + w_eth * eth_ret + w_cash * 0
        strat_nav.append(strat_nav[-1] * np.exp(strat_ret))

        # Benchmark returns
        bench_equal_nav.append(bench_equal_nav[-1] * np.exp(0.5 * btc_ret + 0.5 * eth_ret))
        bench_btc_nav.append(bench_btc_nav[-1] * np.exp(btc_ret))

        # Rebalance?
        if (i + 1) % rebalance_every == 0:
            btc_sig = get_var_signal(model, btc_returns, btc_returns.index.get_loc(date)
                                      if date in btc_returns.index else -1)
            eth_sig = get_var_signal(model, eth_returns, eth_returns.index.get_loc(date)
                                      if date in eth_returns.index else -1)

            if btc_sig and eth_sig:
                btc_var = abs(btc_sig["var_5pct"])
                eth_var = abs(eth_sig["var_5pct"])
                btc_iqr = btc_sig["iqr"]
                eth_iqr = eth_sig["iqr"]

                # Check VaR breaches (actual return < VaR)
                btc_breached = btc_ret < btc_sig["var_5pct"]
                eth_breached = eth_ret < eth_sig["var_5pct"]

                if btc_breached:
                    btc_breach_count += 1
                if eth_breached:
                    eth_breach_count += 1

                # Emergency mode: recent VaR breach → shift to cash
                if btc_breached or eth_breached:
                    recovery_timer = 5  # stay defensive for 5 rebalance periods

                if recovery_timer > 0:
                    # Defensive: mostly cash
                    crypto_total = EMERGENCY_CRYPTO
                    recovery_timer -= 1
                else:
                    # Normal mode: allocate by inverse risk
                    # Higher VaR (more risk) → lower allocation
                    avg_var = (btc_var + eth_var) / 2
                    # Scale: if avg_var is small → more crypto, if large → less
                    # Reference: typical hourly VaR 5% ~ 0.005-0.01
                    risk_factor = avg_var / 0.007  # normalize to typical level
                    crypto_total = BASE_CRYPTO_ALLOCATION / max(risk_factor, 0.5)
                    crypto_total = np.clip(crypto_total, MIN_CRYPTO, MAX_CRYPTO)

                # Split BTC:ETH by inverse IQR (lower uncertainty = more weight)
                btc_inv = 1.0 / btc_iqr
                eth_inv = 1.0 / eth_iqr
                total_inv = btc_inv + eth_inv

                w_btc = crypto_total * (btc_inv / total_inv)
                w_eth = crypto_total * (eth_inv / total_inv)
                w_cash = 1.0 - w_btc - w_eth

                if (i + 1) % (rebalance_every * 50) == 0:
                    logger.info("  [%s] BTC:%.1f%% ETH:%.1f%% Cash:%.1f%% | VaR: BTC=%.3f%% ETH=%.3f%%",
                                date.date(), w_btc*100, w_eth*100, w_cash*100,
                                btc_var*100, eth_var*100)

                allocation_log.append({
                    "date": str(date),
                    "w_btc": round(w_btc, 4),
                    "w_eth": round(w_eth, 4),
                    "w_cash": round(w_cash, 4),
                    "btc_var": round(btc_var, 6),
                    "eth_var": round(eth_var, 6),
                    "defensive": recovery_timer > 0,
                })

        if (i + 1) % 5000 == 0:
            logger.info("  Step %d/%d (%.1f sec)", i + 1, len(eval_dates), time.time() - t0)

    # Convert to arrays
    strat_nav = np.array(strat_nav)
    bench_equal_nav = np.array(bench_equal_nav)
    bench_btc_nav = np.array(bench_btc_nav)

    def metrics(nav, name):
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

    strat_m = metrics(strat_nav, "TimesFM VaR-Weighted (BTC+ETH+Cash)")
    bench_eq = metrics(bench_equal_nav, "Equal-Weight (50% BTC + 50% ETH)")
    bench_btc_m = metrics(bench_btc_nav, "100% BTC Buy&Hold")

    # Print results
    print("\n" + "=" * 70)
    print("  PORTFOLIO BACKTEST: BTC + ETH + Cash")
    print(f"  Period: {start} to {end} ({freq})")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Strategy':>15} {'BTC+ETH EW':>15} {'100% BTC':>15}")
    print("-" * 70)

    for key, label in [
        ("total_return_pct", "Total Return"),
        ("ann_return_pct", "Ann. Return"),
        ("ann_vol_pct", "Ann. Volatility"),
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown"),
        ("calmar", "Calmar Ratio"),
        ("final_nav", "Final NAV ($1)"),
    ]:
        s = strat_m[key]
        e = bench_eq[key]
        b = bench_btc_m[key]
        unit = "%" if "pct" in key else ("$" if "nav" in key else "")
        print(f"  {label:<23} {s:>14.2f}{unit} {e:>14.2f}{unit} {b:>14.2f}{unit}")

    print()
    print(f"  VaR breaches: BTC={btc_breach_count}, ETH={eth_breach_count}")
    print(f"  Rebalance count: {len(allocation_log)}")

    # Highlight winner
    print()
    all_m = [strat_m, bench_eq, bench_btc_m]
    best_sharpe = max(all_m, key=lambda x: x["sharpe"])
    least_dd = max(all_m, key=lambda x: x["max_drawdown_pct"])  # least negative
    print(f"  Best Sharpe:    {best_sharpe['name']} ({best_sharpe['sharpe']:.3f})")
    print(f"  Least Drawdown: {least_dd['name']} ({least_dd['max_drawdown_pct']:.1f}%)")
    print("=" * 70)

    output = {
        "config": {
            "start": start, "end": end, "freq": freq,
            "rebalance_every": rebalance_every,
            "base_crypto": BASE_CRYPTO_ALLOCATION,
            "assets": ["btc", "eth", "cash"],
        },
        "strategy": strat_m,
        "benchmark_equal_weight": bench_eq,
        "benchmark_btc_only": bench_btc_m,
        "var_breaches": {"btc": btc_breach_count, "eth": eth_breach_count},
        "allocation_log_sample": allocation_log[::max(1, len(allocation_log)//50)],
        "nav_series": {
            "strategy": strat_nav[::max(1, len(strat_nav)//500)].tolist(),
            "benchmark_ew": bench_equal_nav[::max(1, len(bench_equal_nav)//500)].tolist(),
            "benchmark_btc": bench_btc_nav[::max(1, len(bench_btc_nav)//500)].tolist(),
        },
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--model", choices=["auto", "zero-shot"], default="auto")
    parser.add_argument("--rebal", type=int, default=None)
    args = parser.parse_args()

    rebal = args.rebal or (24 if args.freq == "hourly" else 1)

    logger.info("=" * 60)
    logger.info("Portfolio Backtest v2: BTC + ETH + Cash")
    logger.info("=" * 60)

    model = load_model(args.model)

    btc_returns = load_returns("btc", args.freq)
    eth_returns = load_returns("eth", args.freq)
    logger.info("BTC: %d rows, ETH: %d rows", len(btc_returns), len(eth_returns))

    result = run_backtest(model, btc_returns, eth_returns,
                           args.start, args.end, rebal, args.freq)

    out_path = RESULTS / "portfolio_backtest_v2.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("\nSaved: %s", out_path)


if __name__ == "__main__":
    main()
