#!/usr/bin/env python3
"""
train_experiment.py — Run a training experiment on a single ticker.

Usage:
    python train_experiment.py                         # defaults: AAOI, window=128, horizon=5
    python train_experiment.py --ticker AAPL           # different ticker
    python train_experiment.py --window 256            # larger window
    python train_experiment.py --horizon 30            # predict T+30
    python train_experiment.py --years 3               # fewer years of data
    python train_experiment.py --lr 5e-4 --dropout 0.3

Outputs:
    models/<ticker>_w<window>_h<horizon>/best_model.pt — best checkpoint
    experiments/<ticker>_w<window>_h<horizon>_<timestamp>.log  — full log
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import prepare_data
from src.data.dataset import StockDataset
from src.models.cnn import StockCNN
from src.training.trainer import Trainer
from src.utils.config import CONV_CHANNELS, KERNEL_SIZE, FC_HIDDEN, NUM_CHANNELS

# ── Experiment output dir ─────────────────────────────────────────────────────
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def setup_logging(log_path: Path):
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN on a single stock ticker")
    p.add_argument("--ticker",        default="AAOI",  help="Stock ticker  (default: AAOI)")
    p.add_argument("--window",        type=int,   default=128,   help="Window size in days  (default: 128)")
    p.add_argument("--horizon",       type=int,   default=5,     help="Prediction horizon in days  (default: 5)")
    p.add_argument("--years",         type=float, default=5.0,   help="Years of history to fetch  (default: 5)")
    p.add_argument("--epochs",        type=int,   default=100,   help="Max epochs  (default: 100)")
    p.add_argument("--patience",      type=int,   default=10,    help="Early-stopping patience  (default: 10)")
    p.add_argument("--lr",            type=float, default=1e-3,  help="Learning rate  (default: 1e-3)")
    p.add_argument("--weight-decay",  type=float, default=1e-5,  help="Weight decay  (default: 1e-5)")
    p.add_argument("--dropout",       type=float, default=0.4,   help="Dropout  (default: 0.4)")
    p.add_argument("--batch",         type=int,   default=128,   help="Batch size  (default: 128)")
    return p.parse_args()


def main():
    args = parse_args()

    tag       = f"{args.ticker}_w{args.window}_h{args.horizon}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = EXPERIMENTS_DIR / f"{tag}_{timestamp}.log"

    log = setup_logging(log_path)

    log.info("=" * 60)
    log.info(f"  CNN Stock Experiment: {tag}")
    log.info("=" * 60)
    log.info(f"  Ticker:       {args.ticker}")
    log.info(f"  Window:       {args.window} days")
    log.info(f"  Horizon:      T+{args.horizon} days")
    log.info(f"  Data:         {args.years} years")
    log.info(f"  LR:           {args.lr}   WD: {args.weight_decay}")
    log.info(f"  Dropout:      {args.dropout}   Batch: {args.batch}")
    log.info(f"  Max epochs:   {args.epochs}  (patience={args.patience})")
    log.info(f"  Log:          {log_path}")
    log.info("")

    device = torch.device("cpu")

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=int(args.years * 365.25))).strftime("%Y-%m-%d")

    log.info(f"Fetching {args.ticker}  {start_date} → {end_date} ...")
    df = fetch_stock_data(args.ticker, start_date, end_date)

    if df.empty:
        log.error(f"No data for {args.ticker}. Check the ticker symbol.")
        sys.exit(1)

    log.info(f"  {len(df)} trading days of OHLCV data")

    # ── 2. Preprocess + windows + split ───────────────────────────────────────
    log.info(f"Building windows (size={args.window}, horizon={args.horizon}, overlapping) ...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        df,
        window_size = args.window,
        horizon     = args.horizon,
        stride      = 1,           # overlapping windows
    )

    if len(X_train) == 0:
        log.error("No training samples — use more data (--years) or a smaller window (--window).")
        sys.exit(1)

    bullish_pct = y_train.mean() * 100
    log.info(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    log.info(f"  Label balance (train): {bullish_pct:.1f}% bullish")

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    train_ds = StockDataset(X_train, y_train)
    val_ds   = StockDataset(X_val,   y_val)
    test_ds  = StockDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = StockCNN(
        window_size   = args.window,
        num_channels  = NUM_CHANNELS,
        conv_channels = CONV_CHANNELS,
        kernel_size   = KERNEL_SIZE,
        fc_hidden     = FC_HIDDEN,
        dropout       = args.dropout,
    )
    log.info(f"  Model: {model.count_parameters():,} parameters")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    checkpoint_dir = PROJECT_ROOT / "models" / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        learning_rate  = args.weight_decay,
        weight_decay   = args.weight_decay,
        device         = device,
        checkpoint_dir = checkpoint_dir,
    )
    # Fix: set LR correctly (Trainer takes lr separately)
    for g in trainer.optimizer.param_groups:
        g["lr"] = args.lr

    log.info("\nStarting training ...\n")
    t0 = time.time()

    metrics = trainer.train(
        num_epochs              = args.epochs,
        early_stopping_patience = args.patience,
    )

    elapsed = time.time() - t0
    log.info(f"\nTraining finished in {elapsed / 60:.1f} min")

    # ── 6. Test evaluation ────────────────────────────────────────────────────
    # Load best model for test evaluation
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded best checkpoint for test evaluation")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_correct = test_total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            out    = model(x)
            loss   = criterion(out, y)
            preds  = out.argmax(dim=1)
            test_correct  += (preds == y).sum().item()
            test_loss_sum += loss.item() * len(y)
            test_total    += len(y)

    test_acc  = test_correct / test_total
    test_loss = test_loss_sum / test_total

    # Pull best val accuracy from metrics history
    val_acc_history = metrics.history.get("val_accuracy", [])
    best_val_acc    = max(val_acc_history) if val_acc_history else 0.0
    epochs_run      = len(metrics.history.get("train_loss", []))

    # ── 7. Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  RESULTS")
    log.info("=" * 60)
    log.info(f"  Ticker:        {args.ticker}")
    log.info(f"  Window / Horizon:  {args.window}d / T+{args.horizon}d")
    log.info(f"  Training data: {args.years}y  ({len(df)} trading days)")
    log.info(f"  Samples:       {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    log.info(f"  Epochs run:    {epochs_run}")
    log.info(f"  Best val acc:  {best_val_acc:.1%}")
    log.info(f"  Test acc:      {test_acc:.1%}")
    log.info(f"  Test loss:     {test_loss:.4f}")
    log.info(f"  Runtime:       {elapsed / 60:.1f} min")
    log.info(f"  Model saved:   {best_ckpt}")
    log.info(f"  Log:           {log_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
