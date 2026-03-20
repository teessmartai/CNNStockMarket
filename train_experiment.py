#!/usr/bin/env python3
"""
train_experiment.py — Multi-stock CNN training with auto-resume.

The model is UNIVERSAL: trained on many stocks simultaneously so it learns
general price-action patterns (not memorised single-stock moves).  At
inference time a single model runs against any ticker.

Training can be interrupted and resumed at any time — checkpoints are saved
every N epochs and on every new best-val-loss.  If a checkpoint exists in the
model dir, training resumes from it automatically (pass --reset to force a
fresh start).

Usage:
    # Train on a preset basket:
    python train_experiment.py --preset tech
    python train_experiment.py --preset sp500-sample
    python train_experiment.py --preset finance

    # Train on specific tickers:
    python train_experiment.py --tickers AAPL MSFT NVDA AMD GOOGL

    # Resume an interrupted run (auto-detected):
    python train_experiment.py --preset tech           # just re-run same command

    # Force fresh start (ignore existing checkpoint):
    python train_experiment.py --preset tech --reset

    # Tune hyperparameters:
    python train_experiment.py --preset tech --lr 5e-4 --window 128 --horizon 5

Outputs:
    models/<run_name>/best_model.pt         — best checkpoint (resume-safe)
    models/<run_name>/checkpoint_epoch_N.pt — periodic saves every 5 epochs
    models/<run_name>/final_model.pt        — last epoch checkpoint
    experiments/<run_name>_<ts>.log         — full training log
"""

import argparse
import hashlib
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_multiple_stocks, fetch_stock_data
from src.data.preprocessor import combine_multiple_stocks, train_val_test_split
from src.data.dataset import StockDataset
from src.models.cnn import StockCNN
from src.training.trainer import Trainer
from src.utils.config import (
    CONV_CHANNELS, KERNEL_SIZE, FC_HIDDEN, NUM_CHANNELS,
    VAL_RATIO, TEST_RATIO,
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# ── Stock presets ─────────────────────────────────────────────────────────────

# S&P 500 fallback list — used when live fetch fails.
# Covers all 11 GICS sectors, ~490 of the 503 constituents as of early 2026.
_SP500_FALLBACK = [
    # Information Technology
    "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","ACN","ADBE","CSCO",
    "INTC","QCOM","TXN","AMAT","LRCX","KLAC","MRVL","MU","CDNS","SNPS",
    "APH","GLW","TEL","STX","WDC","HPQ","HPE","IBM","CTSH","CDW",
    "ANET","KEYS","FSLR","ENPH","NOW","PANW","CRWD","FTNT","ZS","ADSK",
    "EPAM","GDDY","GEN","AKAM","VRSN","TDY","JNPR","NTAP","ZBRA","TRMB",
    # Communication Services
    "GOOGL","GOOG","META","NFLX","CMCSA","DIS","CHTR","T","VZ","TMUS",
    "EA","TTWO","WBD","LYV","OMC","IPG","NWS","NWSA","FOXA","FOX",
    # Consumer Discretionary
    "AMZN","TSLA","HD","MCD","NKE","LOW","TJX","SBUX","BKNG","ORLY",
    "AZO","EBAY","CMG","DHI","LEN","PHM","CCL","RCL","HLT","MAR",
    "DKNG","EXPE","ULTA","DPZ","YUM","QSR","BBY","ROST","APTV","BWA",
    "F","GM","HAS","MHK","LEA","NVR","TOL","KMX","AN","GPC",
    # Consumer Staples
    "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","KMB",
    "GIS","K","CPB","HRL","MKC","SJM","CAG","ADM","BG","STZ",
    "EL","CHD","CLX","COTY","HSY","MNST","KHC","TSN","WBA","DG","DLTR",
    # Health Care
    "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","BMY","MDT",
    "EW","SYK","BSX","BDX","BAX","IQV","ZBH","HOLX","ISRG","RMD",
    "MTD","COO","CRL","TECH","DGX","LH","PKI","VRTX","REGN","BIIB",
    "AMGN","GILD","MRNA","ILMN","A","CI","CVS","ELV","CNC","HUM",
    "MOH","CAH","COR","MCK","DVA","DRI","HCA",
    # Financials
    "BRK-B","JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW",
    "CME","ICE","SPGI","MCO","V","MA","PYPL","FI","FIS","GPN",
    "CBOE","BX","APO","KKR","ARES","AMP","BEN","IVZ","TROW","STT",
    "BK","TFC","USB","PNC","COF","DFS","SYF","AIG","MET","PRU",
    "AFL","ALL","CB","TRV","PGR","HIG","AJG","AON","MMC","WTW",
    "BR","CPAY","NDAQ","MKTX","MDB","FLT","WEX","CINF","EG","RE",
    # Energy
    "XOM","CVX","COP","EOG","SLB","PXD","MPC","VLO","PSX","HES",
    "OXY","DVN","FANG","APA","HAL","BKR","CTRA","EQT","OKE","WMB",
    "KMI","TRGP","LNG","ET","EPD",
    # Industrials
    "CAT","DE","GE","HON","MMM","RTX","LMT","NOC","GD","BA",
    "ITW","EMR","ETN","PH","ROK","AME","DOV","IR","XYL","IEX",
    "ROP","FTV","CARR","OTIS","TT","JCI","CSGP","CSX","UNP","NSC",
    "FDX","UPS","GPN","EFX","VRSK","CTAS","ADP","PAYX","BR","LDOS",
    "LII","MAS","SNA","TDG","HWM","SPX","AXON","CPRT","DAL","UAL",
    "AAL","LUV","EXPD","JBHT","ODFL","CHRW","WSM","URI","BLDR",
    # Materials
    "LIN","APD","SHW","ECL","DOW","DD","PPG","IFF","LYB","CF",
    "MOS","NUE","STLD","FCX","NEM","AEM","ALB","CTVA","PKG","IP",
    "WRK","SEE","AVY","SON","AMCR","VMC","MLM","CRH","FMC","RPM",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","ED","XEL",
    "WEC","ES","ETR","FE","DTE","CNP","CMS","LNT","ATO","EVRG",
    "AEE","CEG","EIX","PPL","NI","AWK","PNW",
    # Real Estate
    "AMT","PLD","EQIX","CCI","SPG","PSA","EQR","AVB","DLR","ESS",
    "ARE","CPT","MAA","UDR","BXP","CBRE","VTR","WELL","VICI","O",
    "SBAC","IRM","WY","EGP","FR","NNN","ADC","GLPI","LAMR",
]

# Large-cap stable: low beta, dividend payers, consistent fundamentals.
# These move less than the market — ideal for testing lower-drawdown strategies.
_LARGECAP_STABLE = [
    # Consumer Staples (the most defensive sector)
    "KO","PEP","PG","CL","KMB","GIS","K","CPB","HRL","MKC","HSY","CLX",
    # Healthcare (steady demand, pricing power)
    "JNJ","ABT","MDT","BMY","MRK","SYK","BDX","BAX","EW",
    # Utilities (regulated, dividend-focused)
    "NEE","SO","DUK","AEP","D","ED","XEL","WEC","ES","AWK","FE","DTE",
    # Financials (large established, consistent)
    "BRK-B","JPM","V","MA","AON","MMC","AXP",
    # Industrials (blue-chip, 50+ year history)
    "MMM","HON","EMR","ITW","CAT","GE","DE",
    # Energy majors (dividend paying, stable)
    "XOM","CVX",
    # Materials (essential, pricing power)
    "APD","ECL","SHW","LIN",
    # Telecom
    "VZ","T",
]

def _fetch_sp500_live() -> list[str]:
    """
    Fetch current S&P 500 tickers from GitHub datasets repo.
    Falls back to hardcoded list if unavailable.
    """
    try:
        import urllib.request, csv, io
        url  = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode()
        reader  = csv.DictReader(io.StringIO(text))
        tickers = [row["Symbol"].replace(".", "-") for row in reader]
        print(f"  [sp500] Fetched {len(tickers)} live tickers from GitHub datasets")
        return tickers
    except Exception as e:
        print(f"  [sp500] Live fetch failed ({e}); using hardcoded fallback ({len(_SP500_FALLBACK)} tickers)")
        return _SP500_FALLBACK

PRESETS: dict[str, list[str] | str] = {
    "tech": [
        "AAPL","MSFT","NVDA","AMD","GOOGL","META","AMZN","INTC","QCOM",
        "AMAT","LRCX","KLAC","MRVL","AVGO","TXN","MU","AAOI","SMCI",
        "CRWD","PANW","NOW","ADBE","CRM","ADSK","CDNS","SNPS","ANET",
    ],
    "finance": [
        "JPM","GS","BAC","MS","WFC","C","BLK","AXP","SCHW","USB",
        "PNC","TFC","COF","DFS","SYF","V","MA","CME","ICE","SPGI",
    ],
    "largecap-stable": _LARGECAP_STABLE,
    "sp500":           "dynamic",     # resolved at runtime via _fetch_sp500_live()
    "sp500-sample": [                 # kept for quick iteration / testing
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN",
        "JPM","GS","BAC","BLK",
        "JNJ","UNH","PFE","ABBV","MRK",
        "TSLA","HD","MCD","SBUX","NKE",
        "XOM","CVX","COP","SLB",
        "CAT","BA","GE","HON","MMM",
        "NEE","AMT","PLD",
    ],
    "single": [],
}

def resolve_tickers(preset: str | None, tickers: list[str] | None) -> list[str]:
    """Return the final ticker list given CLI args."""
    if tickers:
        return [t.upper() for t in tickers]
    val = PRESETS[preset]
    if val == "dynamic":
        return _fetch_sp500_live()
    return list(val)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(__name__)


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-stock CNN training with auto-resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Stock selection
    stock = p.add_mutually_exclusive_group(required=True)
    stock.add_argument("--preset",
                       choices=["tech","finance","largecap-stable","sp500","sp500-sample"],
                       help="Use a predefined stock basket (sp500 fetches live list ~503 stocks)")
    stock.add_argument("--tickers", nargs="+",             help="Explicit list of tickers")

    # Data
    p.add_argument("--window",   type=int,   default=128,  help="Window size in trading days (default: 128)")
    p.add_argument("--horizon",  type=int,   default=5,    help="Prediction horizon in days (default: 5)")
    p.add_argument("--years",    type=float, default=5.0,  help="Years of history per stock (default: 5)")
    p.add_argument("--stride",   type=int,   default=1,    help="Window stride — 1=overlapping (default: 1)")

    # Training
    p.add_argument("--epochs",   type=int,   default=100,  help="Max epochs (default: 100)")
    p.add_argument("--patience", type=int,   default=15,   help="Early-stopping patience (default: 15)")
    p.add_argument("--ckpt-every", type=int, default=5,    help="Save checkpoint every N epochs (default: 5)")
    p.add_argument("--batch",    type=int,   default=256,  help="Batch size (default: 256)")

    # Hyperparameters
    p.add_argument("--lr",           type=float, default=1e-4,  help="Learning rate (default: 1e-4)")
    p.add_argument("--weight-decay", type=float, default=1e-5,  help="Weight decay (default: 1e-5)")
    p.add_argument("--dropout",      type=float, default=0.4,   help="Dropout (default: 0.4)")
    p.add_argument("--scheduler",    type=str,   default="plateau",
                   choices=["plateau", "cosine", "none"],
                   help="LR scheduler: plateau (default), cosine, or none")
    p.add_argument("--optimizer",    type=str,   default="adam",
                   choices=["adam", "adamw"],
                   help="Optimizer: adam (default) or adamw")
    p.add_argument("--norm",         type=str,   default="minmax",
                   choices=["minmax", "logreturns"],
                   help="Input normalization: minmax per-window (default) or logreturns")

    # Control
    p.add_argument("--reset", action="store_true", help="Ignore existing checkpoint, start fresh")
    p.add_argument("--run-name", default=None, help="Override auto-generated run name")

    return p.parse_args()


# ── Run name ──────────────────────────────────────────────────────────────────
def make_run_name(tickers: list[str], window: int, horizon: int, preset: str | None) -> str:
    if preset and preset != "single":
        return f"multi_{preset}_w{window}_h{horizon}"
    # For custom ticker lists: use sorted hash to get stable name
    tag = "_".join(sorted(tickers)[:4])
    if len(tickers) > 4:
        h = hashlib.md5("".join(sorted(tickers)).encode()).hexdigest()[:6]
        tag += f"_+{len(tickers)-4}_{h}"
    return f"multi_{tag}_w{window}_h{horizon}"


# ── Graceful interrupt ────────────────────────────────────────────────────────
_interrupted = False

def _handle_sigterm(signum, frame):
    global _interrupted
    print("\n[SIGTERM received] Finishing current epoch then saving checkpoint...")
    _interrupted = True

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT,  _handle_sigterm)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve tickers
    tickers = resolve_tickers(args.preset, args.tickers)
    if not tickers:
        print("No tickers resolved. Use --tickers or a valid --preset.")
        sys.exit(1)

    run_name = args.run_name or make_run_name(tickers, args.window, args.horizon, args.preset)
    ckpt_dir = PROJECT_ROOT / "models" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = EXPERIMENTS_DIR / f"{run_name}_{ts}.log"
    log      = setup_logging(log_path)

    log.info("=" * 65)
    log.info(f"  Multi-Stock CNN Training — {run_name}")
    log.info("=" * 65)
    log.info(f"  Tickers ({len(tickers)}): {', '.join(tickers)}")
    log.info(f"  Window: {args.window}d   Horizon: T+{args.horizon}d   Stride: {args.stride}")
    log.info(f"  Data: {args.years} years per stock")
    log.info(f"  LR: {args.lr}   WD: {args.weight_decay}   Dropout: {args.dropout}")
    log.info(f"  Scheduler: {args.scheduler}   Optimizer: {args.optimizer}   Norm: {args.norm}")
    log.info(f"  Batch: {args.batch}   Epochs: {args.epochs}   Patience: {args.patience}")
    log.info(f"  Checkpoint dir: {ckpt_dir}")
    log.info(f"  Log: {log_path}")
    log.info("")

    # ── Device selection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"  Device: GPU — {gpu_name} ({torch.cuda.device_count()} device(s))")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("  Device: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        log.info("  Device: CPU")
    log.info("")

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=int(args.years * 365.25))).strftime("%Y-%m-%d")

    log.info(f"Fetching {len(tickers)} stocks  {start_date} → {end_date}...")
    stock_data = fetch_multiple_stocks(tickers, start_date, end_date, use_cache=True)

    if not stock_data:
        log.error("No data fetched for any ticker. Exiting.")
        sys.exit(1)

    log.info(f"  Successfully fetched {len(stock_data)}/{len(tickers)} stocks")
    if len(stock_data) < len(tickers):
        skipped = set(tickers) - set(stock_data.keys())
        log.warning(f"  Skipped (no data): {', '.join(skipped)}")

    # ── 2. Build combined dataset ─────────────────────────────────────────────
    log.info(f"\nBuilding windows (size={args.window}, horizon={args.horizon}, stride={args.stride}, norm={args.norm})...")
    X_all, y_all = combine_multiple_stocks(
        stock_data,
        window_size   = args.window,
        horizon       = args.horizon,
        stride        = args.stride,
        normalization = args.norm,
    )

    # Per-stock breakdown
    for ticker, df in stock_data.items():
        from src.data.preprocessor import create_sliding_windows
        try:
            X_t, y_t = create_sliding_windows(df, args.window, args.horizon, args.stride)
            log.info(f"    {ticker:8s}: {len(X_t):>5} windows")
        except Exception:
            pass

    n_total = len(X_all)
    log.info(f"\n  Total samples: {n_total:,}")
    log.info(f"  Label balance: {y_all.mean():.1%} bullish  /  {1-y_all.mean():.1%} bearish")

    if n_total < 200:
        log.error(f"Only {n_total} samples — too few. Add more stocks or years.")
        sys.exit(1)

    # ── 3. Train / val / test split ───────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X_all, y_all, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )
    log.info(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    dataset     = StockDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.long))
    val_dataset = StockDataset(torch.tensor(X_val,   dtype=torch.float32),
                               torch.tensor(y_val,   dtype=torch.long))
    test_dataset= StockDataset(torch.tensor(X_test,  dtype=torch.float32),
                               torch.tensor(y_test,  dtype=torch.long))

    pin = device.type == "cuda"
    train_loader = DataLoader(dataset,      batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,  batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=pin)

    # ── 4. Model + trainer ────────────────────────────────────────────────────
    model = StockCNN(
        window_size   = args.window,
        num_channels  = NUM_CHANNELS,
        conv_channels = CONV_CHANNELS,
        kernel_size   = KERNEL_SIZE,
        fc_hidden     = FC_HIDDEN,
        dropout       = args.dropout,
    )
    log.info(f"\n  Model: {model.count_parameters():,} parameters")

    log.info(f"  Scheduler: {args.scheduler}   Optimizer: {args.optimizer}   Norm: {args.norm}")
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        learning_rate  = args.lr,
        weight_decay   = args.weight_decay,
        device         = device,
        checkpoint_dir = ckpt_dir,
        scheduler      = args.scheduler,
        num_epochs     = args.epochs,
        optimizer_type = args.optimizer,
    )

    # ── 5. Resume from checkpoint if available ────────────────────────────────
    start_epoch = 0
    resume_ckpt = ckpt_dir / "best_model.pt"

    # Also check for latest periodic checkpoint (may be further along than best)
    periodic = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if periodic:
        last_periodic = periodic[-1]
        best_epoch    = torch.load(resume_ckpt,    map_location="cpu").get("epoch", -1) if resume_ckpt.exists() else -1
        periodic_epoch= torch.load(last_periodic,  map_location="cpu").get("epoch", -1)
        resume_ckpt   = last_periodic if periodic_epoch > best_epoch else resume_ckpt

    if not args.reset and resume_ckpt.exists():
        log.info(f"\nResuming from checkpoint: {resume_ckpt.name}")
        start_epoch = trainer.load_checkpoint(resume_ckpt)

        # Guard: if checkpoint has a corrupt/zero best_val_loss, reset it so the
        # first real epoch is always eligible to save.  A val_loss of 0.0 means
        # no real epoch has ever been saved (e.g. checkpoint was written on init
        # before any training), and keeping it would prevent any future saves.
        if trainer.best_val_loss <= 0.0 or not (0.0 < trainer.best_val_loss < 100.0):
            log.warning(
                f"  Checkpoint best_val_loss={trainer.best_val_loss:.4f} looks corrupt — "
                "resetting to inf so first real epoch can save."
            )
            trainer.best_val_loss = float("inf")

        log.info(f"  Resuming from epoch {start_epoch + 1}  "
                 f"(best val loss so far: {trainer.best_val_loss:.4f})")
    elif args.reset:
        log.info("\n--reset flag set: starting fresh (ignoring any existing checkpoints)")
    else:
        log.info("\nNo checkpoint found — starting fresh")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    log.info(f"\nStarting training from epoch {start_epoch + 1} / {args.epochs}...\n")
    t0 = time.time()

    # Patch the training loop to honour SIGTERM: save on interrupt
    # We call train() which internally loops epochs; on SIGTERM we save
    try:
        metrics = trainer.train(
            num_epochs              = args.epochs,
            early_stopping_patience = args.patience,
            checkpoint_interval     = args.ckpt_every,
            start_epoch             = start_epoch,
        )
    except KeyboardInterrupt:
        log.info("\nInterrupted. Saving emergency checkpoint...")
        trainer.save_checkpoint("interrupted.pt", extra_config={
            "tickers": tickers, "window": args.window,
            "horizon": args.horizon, "run_name": run_name,
        })
        log.info(f"  Saved to {ckpt_dir}/interrupted.pt")
        log.info("  Re-run the same command to resume.")
        sys.exit(0)

    elapsed = time.time() - t0
    log.info(f"\nTraining finished in {elapsed/60:.1f} min")

    # ── 7. Final test evaluation ──────────────────────────────────────────────
    log.info("\nEvaluating on held-out test set...")
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    criterion    = nn.CrossEntropyLoss()
    test_correct = test_total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y  = x.to(device), y.to(device)
            out   = model(x)
            loss  = criterion(out, y)
            preds = out.argmax(dim=1)
            test_correct  += (preds == y).sum().item()
            test_loss_sum += loss.item() * len(y)
            test_total    += len(y)

    test_acc  = test_correct / test_total
    test_loss = test_loss_sum / test_total

    val_history = metrics.history.get("val_accuracy", [])
    best_val    = max(val_history) if val_history else 0.0
    epochs_run  = len(metrics.history.get("train_loss", []))

    # ── 8. Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  RESULTS")
    log.info("=" * 65)
    log.info(f"  Stocks trained on: {len(stock_data)} ({', '.join(list(stock_data.keys())[:5])}{'...' if len(stock_data)>5 else ''})")
    log.info(f"  Window / Horizon:  {args.window}d / T+{args.horizon}d")
    log.info(f"  Total samples:     {n_total:,}  ({len(X_train):,} train)")
    log.info(f"  Epochs run:        {epochs_run + start_epoch}")
    log.info(f"  Best val acc:      {best_val:.1%}")
    log.info(f"  Test accuracy:     {test_acc:.1%}")
    log.info(f"  Test loss:         {test_loss:.4f}")
    log.info(f"  Runtime:           {elapsed/60:.1f} min")
    log.info(f"  Model:             {best_ckpt}")
    log.info(f"  Log:               {log_path}")
    log.info("=" * 65)
    log.info("\nTo run backtests against any ticker with this model:")
    log.info(f"  python backtest_stoploss.py --ticker AAOI --model {best_ckpt} --window {args.window}")


if __name__ == "__main__":
    main()
