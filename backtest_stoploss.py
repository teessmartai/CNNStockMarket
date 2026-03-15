#!/usr/bin/env python3
"""
backtest_stoploss.py — Sweep stop-loss levels and find the optimal one.

Runs a walk-forward backtest on the trained model, simulating T+5 trades
with a range of stop-loss thresholds.  Prints a comparison table so you
can pick the stop-loss that gives the best risk-adjusted return.

Usage:
    python backtest_stoploss.py                             # defaults
    python backtest_stoploss.py --ticker AAOI --window 128
    python backtest_stoploss.py --model models/AAOI_w128_h5/best_model.pt
    python backtest_stoploss.py --confidence 0.60           # only trade when >60% confident
    python backtest_stoploss.py --capital 10000             # start with $10k

Output:
    experiments/backtest_<ticker>_<timestamp>.csv   — per-trade log
    experiments/backtest_<ticker>_<timestamp>.txt   — summary table
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import normalize
from src.models.cnn import StockCNN
from src.utils.config import CONV_CHANNELS, KERNEL_SIZE, FC_HIDDEN, NUM_CHANNELS

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stop-loss sweep backtest")
    p.add_argument("--ticker",     default="AAOI",  help="Ticker (default: AAOI)")
    p.add_argument("--window",     type=int, default=128,  help="Window size used during training (default: 128)")
    p.add_argument("--horizon",    type=int, default=5,    help="Prediction horizon in days (default: 5)")
    p.add_argument("--model",      default=None,    help="Path to best_model.pt (auto-detected if omitted)")
    p.add_argument("--years",      type=float, default=5.0, help="Years of data to backtest over (default: 5)")
    p.add_argument("--confidence", type=float, default=0.55, help="Min confidence to enter a trade (default: 0.55)")
    p.add_argument("--capital",    type=float, default=100_000, help="Starting capital (default: 100000)")
    p.add_argument("--commission", type=float, default=0.001, help="Commission per side as fraction (default: 0.001 = 0.1%%)")
    p.add_argument("--stop-levels", nargs="+", type=float,
                   default=[0.02, 0.03, 0.05, 0.08, 0.10, 0.15, None],
                   help="Stop-loss levels to sweep, e.g. 0.03 = 3%%. None = no stop loss.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, window_size: int) -> StockCNN:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    ckpt  = torch.load(path, map_location="cpu")
    model = StockCNN(
        window_size   = window_size,
        num_channels  = NUM_CHANNELS,
        conv_channels = CONV_CHANNELS,
        kernel_size   = KERNEL_SIZE,
        fc_hidden     = FC_HIDDEN,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def find_model(ticker: str, window: int, horizon: int) -> Path:
    tag  = f"{ticker}_w{window}_h{horizon}"
    path = PROJECT_ROOT / "models" / tag / "best_model.pt"
    if path.exists():
        return path
    # Fallback: any .pt in models/
    candidates = sorted(PROJECT_ROOT.glob("models/**/*.pt"))
    if candidates:
        print(f"  Hint: found {len(candidates)} model(s) in models/: {[str(c) for c in candidates[:3]]}")
    raise FileNotFoundError(
        f"No model at {path}.\n"
        f"Run:  python train_experiment.py --ticker {ticker} --window {window} --horizon {horizon}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Prediction engine
# ══════════════════════════════════════════════════════════════════════════════

def predict_at(model, df_norm: pd.DataFrame, idx: int, window: int):
    """Return (bullish_prob, bearish_prob) at index idx."""
    if idx < window:
        return None
    window_data = df_norm.iloc[idx - window: idx].values   # (window, 5)
    x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # (1, window, 5)
    with torch.no_grad():
        probs = model(x).squeeze().numpy()
    return float(probs[1]), float(probs[0])   # bullish, bearish


# ══════════════════════════════════════════════════════════════════════════════
# Single-level simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate(
    model,
    df: pd.DataFrame,
    df_norm: pd.DataFrame,
    window: int,
    horizon: int,
    confidence_threshold: float,
    stop_loss: float | None,        # e.g. 0.05 = 5%, None = no stop
    commission: float,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Walk forward through df, trade on BUY signals, exit at T+horizon or stop loss.

    Returns a DataFrame of completed trades.
    """
    prices = df["Close"].values
    dates  = df.index.tolist()
    n      = len(df)

    trades       = []
    capital      = initial_capital
    in_position  = False
    entry_idx    = entry_price = entry_date = None
    hold_to      = None

    for i in range(window, n - horizon):
        # If in a position, check stop loss and horizon exit
        if in_position:
            current_price = prices[i]
            pnl_pct = (current_price - entry_price) / entry_price

            # Stop-loss check
            if stop_loss is not None and pnl_pct <= -stop_loss:
                exit_price  = current_price * (1 - commission)
                ret         = (exit_price - entry_price) / entry_price
                pnl         = capital * ret
                capital    += pnl
                horizon_ret = (prices[entry_idx + horizon] - entry_price) / entry_price if (entry_idx + horizon) < n else 0
                trades.append({
                    "entry_date":   entry_date,
                    "exit_date":    dates[i],
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "return_pct":   ret,
                    "exit_reason":  "stop_loss",
                    "horizon_ret":  horizon_ret,
                })
                in_position = False
                continue

            # Horizon exit
            if i >= hold_to:
                exit_price = prices[i] * (1 - commission)
                ret        = (exit_price - entry_price) / entry_price
                pnl        = capital * ret
                capital   += pnl
                trades.append({
                    "entry_date":   entry_date,
                    "exit_date":    dates[i],
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "return_pct":   ret,
                    "exit_reason":  "horizon",
                    "horizon_ret":  ret,
                })
                in_position = False
                continue

        # If not in position, check for new BUY signal
        if not in_position:
            result = predict_at(model, df_norm, i, window)
            if result is None:
                continue
            bullish_prob, bearish_prob = result

            if bullish_prob >= confidence_threshold:
                entry_price  = prices[i] * (1 + commission)
                entry_idx    = i
                entry_date   = dates[i]
                hold_to      = i + horizon
                in_position  = True

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics from a trades DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {k: 0 for k in ("n_trades", "win_rate", "avg_ret",
                                "sharpe", "max_drawdown", "total_ret",
                                "stop_triggered_pct")}
    r = trades["return_pct"]
    n = len(trades)

    win_rate   = (r > 0).mean()
    avg_ret    = r.mean()
    sharpe     = (r.mean() / r.std() * (252 / 5) ** 0.5) if r.std() > 0 else 0
    total_ret  = (1 + r).prod() - 1

    # Drawdown on compounded equity curve
    equity = initial_capital * (1 + r).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    stop_pct = (trades["exit_reason"] == "stop_loss").mean() if "exit_reason" in trades else 0

    return {
        "n_trades":          n,
        "win_rate":          win_rate,
        "avg_ret":           avg_ret,
        "sharpe":            sharpe,
        "max_drawdown":      max_dd,
        "total_ret":         total_ret,
        "stop_triggered_pct": stop_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Locate model
    model_path = args.model or str(find_model(args.ticker, args.window, args.horizon))
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path, args.window)
    print(f"  ✅  Loaded")

    # Fetch data
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=int(args.years * 365.25))).strftime("%Y-%m-%d")
    print(f"\nFetching {args.ticker}  {start_date} → {end_date} ...")
    df = fetch_stock_data(args.ticker, start_date, end_date)
    if df.empty:
        print(f"No data for {args.ticker}")
        sys.exit(1)
    df_norm = normalize(df)
    print(f"  {len(df)} trading days")

    # ── Stop-loss sweep ───────────────────────────────────────────────────────
    print(f"\nSweeping stop-loss levels: {args.stop_levels}")
    print(f"Confidence threshold: {args.confidence_threshold if hasattr(args, 'confidence_threshold') else args.confidence:.0%}\n")

    rows = []
    all_trades = {}

    for sl in args.stop_levels:
        label = f"{sl:.0%}" if sl is not None else "none"
        t0 = time.time()
        trades = simulate(
            model              = model,
            df                 = df,
            df_norm            = df_norm,
            window             = args.window,
            horizon            = args.horizon,
            confidence_threshold = args.confidence,
            stop_loss          = sl,
            commission         = args.commission,
            initial_capital    = args.capital,
        )
        elapsed = time.time() - t0
        m       = calc_metrics(trades, args.capital)
        all_trades[label] = trades

        rows.append({
            "Stop Loss":     label,
            "# Trades":      m["n_trades"],
            "Win Rate":      f"{m['win_rate']:.1%}",
            "Avg Return":    f"{m['avg_ret']:.2%}",
            "Sharpe":        f"{m['sharpe']:.2f}",
            "Max Drawdown":  f"{m['max_drawdown']:.1%}",
            "Total Return":  f"{m['total_ret']:.1%}",
            "Stop Hits":     f"{m['stop_triggered_pct']:.1%}",
        })
        print(f"  SL={label:>5}  trades={m['n_trades']:>4}  win={m['win_rate']:.1%}  "
              f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
              f"total={m['total_ret']:+.1%}  ({elapsed:.1f}s)")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base      = EXPERIMENTS_DIR / f"backtest_{args.ticker}_{timestamp}"

    txt_path = base.with_suffix(".txt")
    csv_path = base.with_suffix(".csv")

    table_str = "\n" + "=" * 75 + "\n"
    table_str += f"  Stop-Loss Sweep — {args.ticker}  window={args.window}  horizon=T+{args.horizon}\n"
    table_str += f"  Confidence threshold: {args.confidence:.0%}  |  Data: {start_date} → {end_date}\n"
    table_str += "=" * 75 + "\n"
    table_str += summary.to_string(index=False) + "\n"
    table_str += "=" * 75 + "\n"
    table_str += "\nHow to read this:\n"
    table_str += "  Sharpe  — risk-adjusted return (higher = better, >1 is good, >2 is great)\n"
    table_str += "  Win Rate — % of trades that closed profitable\n"
    table_str += "  Max DD  — worst peak-to-trough equity drop (smaller magnitude = safer)\n"
    table_str += "  Stop Hits — % of trades exited via stop loss (vs. reaching T+5)\n"
    table_str += "  Recommendation: pick the row with the highest Sharpe and acceptable Max DD\n"

    print(table_str)

    with open(txt_path, "w") as f:
        f.write(table_str)

    # Also save per-trade CSVs for the best stop loss
    # (pick highest Sharpe)
    sharpe_vals = [float(r["Sharpe"]) for r in rows]
    best_idx    = int(np.argmax(sharpe_vals))
    best_sl     = rows[best_idx]["Stop Loss"]
    best_trades = all_trades[best_sl]
    if not best_trades.empty:
        best_trades.to_csv(csv_path, index=False)
        print(f"\n  Best stop loss by Sharpe: {best_sl}")
        print(f"  Trades saved to: {csv_path}")

    print(f"  Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
