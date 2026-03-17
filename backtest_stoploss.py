#!/usr/bin/env python3
"""
backtest_stoploss.py — Sweep stop-loss levels and find the optimal one.

Key optimization: predictions are computed ONCE for all windows, then
stop-loss logic is applied as fast numpy operations per level.
Total inference time = O(N), not O(N × num_levels).

Usage:
    python backtest_stoploss.py                             # defaults
    python backtest_stoploss.py --ticker AAOI --window 128
    python backtest_stoploss.py --confidence 0.60
    python backtest_stoploss.py --capital 10000

Output:
    experiments/backtest_<ticker>_<timestamp>.txt   — summary table
    experiments/backtest_<ticker>_<timestamp>.csv   — best stop-loss trades
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


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",     default="AAOI")
    p.add_argument("--window",     type=int,   default=128)
    p.add_argument("--horizon",    type=int,   default=5)
    p.add_argument("--model",      default=None)
    p.add_argument("--years",      type=float, default=5.0)
    p.add_argument("--confidence", type=float, default=0.55,
                   help="Min bullish prob to enter trade (default 0.55)")
    p.add_argument("--capital",    type=float, default=100_000)
    p.add_argument("--commission", type=float, default=0.001)
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(path: str, window: int) -> StockCNN:
    ckpt  = torch.load(path, map_location="cpu")
    model = StockCNN(
        window_size=window, num_channels=NUM_CHANNELS,
        conv_channels=CONV_CHANNELS, kernel_size=KERNEL_SIZE,
        fc_hidden=FC_HIDDEN,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def find_model(ticker, window, horizon):
    p = PROJECT_ROOT / "models" / f"{ticker}_w{window}_h{horizon}" / "best_model.pt"
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"No model at {p}. Run train_experiment.py first.")


# ── Step 1: batch inference on ALL windows ────────────────────────────────────
def precompute_predictions(model, df_norm: pd.DataFrame, window: int) -> np.ndarray:
    """
    Build all windows at once and run inference in batches.
    Returns float32 array of shape (N, 2) — [bearish_prob, bullish_prob]
    for indices window..len(df_norm)-1.
    """
    data  = df_norm.values.astype(np.float32)   # (T, 5)
    n     = len(data)
    valid = n - window                           # number of valid windows

    print(f"  Building {valid} windows and running batch inference...")
    t0 = time.time()

    # Stack all windows: shape (valid, window, 5)
    windows = np.stack([data[i: i + window] for i in range(valid)])
    X = torch.tensor(windows)   # (valid, window, 5)

    BATCH = 256
    probs_list = []
    with torch.no_grad():
        for start in range(0, valid, BATCH):
            batch = X[start: start + BATCH]
            out   = model(batch)                 # (B, 2)
            probs_list.append(out.numpy())

    probs = np.concatenate(probs_list, axis=0)  # (valid, 2)
    print(f"  Done in {time.time()-t0:.1f}s  — shape {probs.shape}")
    return probs   # index 0 = bearish, 1 = bullish


# ── Step 2: simulate trades for one stop-loss level ──────────────────────────
def simulate(
    probs:      np.ndarray,     # (valid, 2)  — row i corresponds to df row i+window
    prices:     np.ndarray,     # full close-price array, len = window + valid
    dates,                      # full date index
    window:     int,
    horizon:    int,
    confidence: float,
    stop_loss:  float | None,
    commission: float,
) -> pd.DataFrame:
    """
    Walk forward; enter on BUY signals, exit at T+horizon or stop loss.
    probs row i  ↔  prices/dates index (i + window).
    """
    trades = []
    n_probs      = len(probs)
    in_position  = False
    entry_price  = entry_date = hold_to = None

    for i in range(n_probs - horizon):
        abs_idx    = i + window            # position in the full price array
        curr_price = prices[abs_idx]

        if in_position:
            pnl_pct = (curr_price - entry_price) / entry_price

            # Stop-loss exit
            if stop_loss is not None and pnl_pct <= -stop_loss:
                exit_p = curr_price * (1 - commission)
                future_abs = min(entry_abs + horizon, len(prices) - 1)
                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   dates[abs_idx],
                    "entry_price": entry_price,
                    "exit_price":  exit_p,
                    "return_pct":  (exit_p - entry_price) / entry_price,
                    "exit_reason": "stop_loss",
                    "if_held_ret": (prices[future_abs] - entry_price) / entry_price,
                })
                in_position = False
                continue

            # Horizon exit
            if abs_idx >= hold_to:
                exit_p = curr_price * (1 - commission)
                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   dates[abs_idx],
                    "entry_price": entry_price,
                    "exit_price":  exit_p,
                    "return_pct":  (exit_p - entry_price) / entry_price,
                    "exit_reason": "horizon",
                    "if_held_ret": (exit_p - entry_price) / entry_price,
                })
                in_position = False
                continue

        if not in_position:
            bullish_prob = probs[i, 1]
            if bullish_prob >= confidence:
                entry_price = curr_price * (1 + commission)
                entry_abs   = abs_idx
                entry_date  = dates[abs_idx]
                hold_to     = abs_idx + horizon
                in_position = True

    return pd.DataFrame(trades)


# ── Step 3: metrics ───────────────────────────────────────────────────────────
def metrics(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return dict(n=0, win_rate=0, avg_ret=0, sharpe=0, max_dd=0,
                    total_ret=0, stop_pct=0)
    r = trades["return_pct"]
    equity = capital * (1 + r).cumprod()
    peak   = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    # Annualised Sharpe: scale from 5-day periods to annual
    periods_per_year = 252 / 5
    sharpe = (r.mean() / r.std() * periods_per_year ** 0.5) if r.std() > 0 else 0
    return dict(
        n        = len(r),
        win_rate = (r > 0).mean(),
        avg_ret  = r.mean(),
        sharpe   = sharpe,
        max_dd   = max_dd,
        total_ret= (1 + r).prod() - 1,
        stop_pct = (trades["exit_reason"] == "stop_loss").mean(),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    model_path = args.model or find_model(args.ticker, args.window, args.horizon)
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path, args.window)
    ckpt_meta = torch.load(model_path, map_location="cpu")
    val_acc   = float(ckpt_meta.get("val_accuracy", 0))
    print(f"  Epoch {ckpt_meta.get('epoch','?')}  val_acc {val_acc:.1%}")

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=int(args.years * 365.25))).strftime("%Y-%m-%d")
    print(f"\nFetching {args.ticker}  {start_date} → {end_date}...")
    df = fetch_stock_data(args.ticker, start_date, end_date)
    if df.empty:
        print("No data."); sys.exit(1)
    df_norm  = normalize(df)
    prices   = df["Close"].values
    dates    = df.index.tolist()
    print(f"  {len(df)} trading days")

    # ── One-time inference pass ───────────────────────────────────────────────
    probs = precompute_predictions(model, df_norm, args.window)

    # Quick sanity: what % of days would trigger an entry at this threshold?
    entry_rate = (probs[:, 1] >= args.confidence).mean()
    print(f"  Entry rate at >{args.confidence:.0%} confidence: {entry_rate:.1%} of days")

    # ── Sweep stop-loss levels ────────────────────────────────────────────────
    stop_levels = [None, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    print(f"\nSimulating {len(stop_levels)} stop-loss levels...\n")

    rows = []
    best_trades = pd.DataFrame()
    best_sharpe = -999

    for sl in stop_levels:
        trades = simulate(
            probs=probs, prices=prices, dates=dates,
            window=args.window, horizon=args.horizon,
            confidence=args.confidence,
            stop_loss=sl, commission=args.commission,
        )
        m = metrics(trades, args.capital)
        label = f"{sl:.0%}" if sl is not None else "none"
        rows.append({
            "Stop Loss":  label,
            "# Trades":   m["n"],
            "Win Rate":   f"{m['win_rate']:.1%}",
            "Avg/Trade":  f"{m['avg_ret']:+.2%}",
            "Sharpe":     f"{m['sharpe']:.2f}",
            "Max DD":     f"{m['max_dd']:.1%}",
            "Total Ret":  f"{m['total_ret']:+.1%}",
            "Stop Hits":  f"{m['stop_pct']:.1%}",
        })
        star = ""
        if m["sharpe"] > best_sharpe and m["n"] > 5:
            best_sharpe = m["sharpe"]
            best_trades = trades
            star = "  ← best Sharpe"
        print(f"  SL={label:>5}  trades={m['n']:>3}  win={m['win_rate']:.1%}  "
              f"avg={m['avg_ret']:+.2%}  sharpe={m['sharpe']:.2f}  "
              f"maxDD={m['max_dd']:.1%}  total={m['total_ret']:+.1%}{star}")

    # ── Summary table ─────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(rows)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = EXPERIMENTS_DIR / f"backtest_{args.ticker}_{ts}"

    legend = (
        "\nKey:\n"
        "  Sharpe   — risk-adjusted return annualised to 252-day year (>1 good, >2 great)\n"
        "  Win Rate — % of trades that closed in profit\n"
        "  Max DD   — worst peak→trough equity drop (closer to 0 = safer)\n"
        "  Stop Hits— % of trades exited via stop loss (vs. reaching T+5 horizon)\n"
        "  Total Ret— compounded return over full backtest period\n"
        "→ Pick the row with the highest Sharpe and a Max DD you can live with.\n"
    )

    header = (
        f"\n{'='*72}\n"
        f"  Stop-Loss Sweep — {args.ticker}  "
        f"window={args.window}d  horizon=T+{args.horizon}d  "
        f"confidence>{args.confidence:.0%}\n"
        f"  Data: {start_date} → {end_date}  ({len(df)} trading days)\n"
        f"{'='*72}\n"
    )

    output = header + df_summary.to_string(index=False) + "\n" + "="*72 + legend

    print(output)

    with open(str(base) + ".txt", "w") as f:
        f.write(output)

    if not best_trades.empty:
        best_trades.to_csv(str(base) + ".csv", index=False)

    print(f"\n  Saved: {base}.txt")
    if not best_trades.empty:
        print(f"  Saved: {base}.csv  (best stop-loss trades)")


if __name__ == "__main__":
    main()
