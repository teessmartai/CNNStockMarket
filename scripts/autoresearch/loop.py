#!/usr/bin/env python3
"""
AutoResearch loop for CNNStockMarket.

Commands:
  status   — show current best, history, and pending change
  run      — push current train_experiment.py to Kaggle and start a run
  collect  — download latest kernel output, parse result, update results.json
  revert   — undo last change (restore train_experiment.py to last kept commit)

Typical workflow:
  1. AI agent reads program.md + results.json, edits train_experiment.py
  2. python loop.py run --hypothesis "Lower LR to 5e-4 — fix epoch-2 peak"
  3. Wait ~20 min
  4. python loop.py collect --run-id run_002_lower_lr
  5. Review: if good, python loop.py keep; if bad, python loop.py revert
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent.parent.resolve()
RESULTS_FILE = REPO_ROOT / "experiments" / "results.json"
KAGGLE_TOKEN = "KGAT_6b6cb4995fa85455d4045d57523e70e7"
KERNEL_SLUG  = "tassistant/cnn-stock-market-training"
DATASET_DIR  = Path("/tmp/kaggle-code-dataset")
KERNEL_DIR   = Path("/tmp/kaggle-setup/kernel")
VENV_KAGGLE  = REPO_ROOT / "venv" / "bin" / "kaggle"


def kaggle(*args):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    result = subprocess.run(
        [str(VENV_KAGGLE)] + list(args),
        capture_output=True, text=True, env=env
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(results):
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")


def next_run_id(results):
    n = len(results) + 1
    return f"run_{n:03d}"


def git(*args, cwd=None):
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
        cwd=cwd or REPO_ROOT
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def rebuild_dataset():
    """Rebuild the Kaggle code dataset from current repo state."""
    import zipfile

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    # Copy files
    for name in ["train_experiment.py", "backtest_stoploss.py", "kaggle_config.yaml"]:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, DATASET_DIR / name)

    # Zip src/ and data/
    for folder in ["src", "data"]:
        src_dir = REPO_ROOT / folder
        if src_dir.exists():
            zip_path = DATASET_DIR / f"{folder}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in src_dir.rglob("*"):
                    if "__pycache__" not in str(f) and f.suffix != ".pyc":
                        zf.write(f, f.relative_to(REPO_ROOT / folder))

    # Metadata
    meta = {
        "title": "CNN Stock Market Code and Data",
        "id": "tassistant/cnn-stock-code",
        "licenses": [{"name": "other"}]
    }
    (DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    print("Uploading dataset to Kaggle...")
    out, err, rc = kaggle("datasets", "version", "-p", str(DATASET_DIR),
                          "--dir-mode", "zip", "-m",
                          f"autoresearch update {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(out or err)
    return rc == 0


def push_kernel():
    """Push kernel to Kaggle."""
    out, err, rc = kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    print(out or err)
    return rc == 0


def parse_log(log_text):
    """Extract key metrics from experiment log."""
    result = {}
    m = re.search(r"Best val acc:\s+([\d.]+)%", log_text)
    if m:
        result["best_val_acc"] = round(float(m.group(1)) / 100, 4)
    m = re.search(r"Test accuracy:\s+([\d.]+)%", log_text)
    if m:
        result["test_acc"] = round(float(m.group(1)) / 100, 4)
    m = re.search(r"Test loss:\s+([\d.]+)", log_text)
    if m:
        result["test_loss"] = float(m.group(1))
    m = re.search(r"Epochs run:\s+(\d+)", log_text)
    if m:
        result["epochs_run"] = int(m.group(1))
    m = re.search(r"Runtime:\s+([\d.]+) min", log_text)
    if m:
        result["runtime_min"] = float(m.group(1))
    result["early_stopped"] = "early stop" in log_text.lower() or result.get("epochs_run", 100) < 100
    return result


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_status(args):
    results = load_results()
    if not results:
        print("No runs yet.")
        return

    kept = [r for r in results if r.get("verdict") == "kept"]
    best = max(kept, key=lambda r: r["results"]["test_acc"]) if kept else None

    print(f"\n{'='*60}")
    print(f"  AutoResearch Status — {len(results)} runs, {len(kept)} kept")
    print(f"{'='*60}")
    if best:
        print(f"\n  🏆 Best so far:  {best['run_id']}")
        print(f"     val_acc:  {best['results']['best_val_acc']*100:.1f}%")
        print(f"     test_acc: {best['results']['test_acc']*100:.1f}%")
        print(f"     config:   lr={best['config']['lr']}, dropout={best['config']['dropout']}, "
              f"batch={best['config']['batch_size']}, scheduler={best['config']['scheduler']}")

    print(f"\n  Recent runs:")
    for r in results[-5:]:
        verdict_icon = "✅" if r.get("verdict") == "kept" else "❌"
        print(f"  {verdict_icon} {r['run_id']:30s}  "
              f"val={r['results']['best_val_acc']*100:.1f}%  "
              f"test={r['results']['test_acc']*100:.1f}%  "
              f"({r['results'].get('runtime_min', '?')} min)")
    print()


def cmd_run(args):
    hypothesis = args.hypothesis or "no hypothesis provided"
    print(f"\n── AutoResearch Run ──")
    print(f"Hypothesis: {hypothesis}")

    # Rebuild dataset with current code
    if not rebuild_dataset():
        print("Dataset upload failed. Aborting.")
        sys.exit(1)

    # Push kernel
    time.sleep(5)  # Let dataset processing start
    if not push_kernel():
        print("Kernel push failed. Aborting.")
        sys.exit(1)

    print(f"\nKernel running. Check: https://www.kaggle.com/code/{KERNEL_SLUG}")
    print(f"Run 'python loop.py collect --run-id <id> --hypothesis \"{hypothesis}\"' when done.")


def cmd_collect(args):
    run_id = args.run_id or f"{next_run_id(load_results())}_unknown"
    hypothesis = args.hypothesis or "no hypothesis"

    print("Waiting for kernel to complete...")
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

    for attempt in range(60):  # up to 60 min
        out, _, _ = kaggle("kernels", "status", KERNEL_SLUG)
        print(f"  [{attempt+1}] {out}")
        if "COMPLETE" in out.upper():
            break
        if "ERROR" in out.upper():
            print("Kernel errored.")
            break
        time.sleep(60)

    # Download output
    out_dir = Path(f"/tmp/autoresearch_collect_{int(time.time())}")
    out_dir.mkdir(parents=True)
    kaggle("kernels", "output", KERNEL_SLUG, "-p", str(out_dir))

    # Find experiment log
    log_files = list(out_dir.glob("**/experiments/*.log"))
    if not log_files:
        print("No experiment log found in output.")
        sys.exit(1)

    log_text = log_files[0].read_text()
    metrics = parse_log(log_text)
    print(f"\n── Results ──")
    print(f"  val_acc:  {metrics.get('best_val_acc', 0)*100:.1f}%")
    print(f"  test_acc: {metrics.get('test_acc', 0)*100:.1f}%")
    print(f"  epochs:   {metrics.get('epochs_run', '?')}")
    print(f"  runtime:  {metrics.get('runtime_min', '?')} min")

    # Get kernel version
    out, _, _ = kaggle("kernels", "list", "--mine")
    kv_match = re.search(r"version[^\d]*(\d+)", out, re.IGNORECASE)
    kernel_version = int(kv_match.group(1)) if kv_match else None

    # Get GPU name from log
    gpu_match = re.search(r"Device: GPU — ([^\n(]+)", log_text)
    device = gpu_match.group(1).strip() if gpu_match else "unknown"

    # Get config from log
    def log_val(key, pattern, cast=str):
        m = re.search(pattern, log_text)
        return cast(m.group(1)) if m else None

    # Build entry
    results = load_results()
    prev = next((r for r in reversed(results) if r.get("verdict") == "kept"), None)

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kaggle_kernel_version": kernel_version,
        "device": device,
        "hypothesis": hypothesis,
        "changes_from_previous": args.changes or "see git diff",
        "config": prev["config"].copy() if prev else {},
        "results": metrics,
        "verdict": "pending",
        "notes": args.notes or ""
    }

    results.append(entry)
    save_results(results)
    print(f"\nResult saved as '{run_id}' (verdict: pending)")
    print(f"Run 'python loop.py keep' or 'python loop.py revert'")


def cmd_keep(args):
    results = load_results()
    pending = [r for r in results if r.get("verdict") == "pending"]
    if not pending:
        print("No pending result to keep.")
        return
    entry = pending[-1]
    entry["verdict"] = "kept"
    save_results(results)

    # Commit results.json + train_experiment.py
    git("add", "experiments/results.json", "train_experiment.py",
        "src/training/trainer.py")
    msg = f"autoresearch: keep {entry['run_id']} — val={entry['results']['best_val_acc']*100:.1f}% test={entry['results']['test_acc']*100:.1f}%"
    git("commit", "-m", msg)
    git("push", "origin", "main")
    print(f"✅ Kept {entry['run_id']} and pushed to main.")


def cmd_revert(args):
    results = load_results()
    pending = [r for r in results if r.get("verdict") == "pending"]
    if not pending:
        print("No pending result to revert.")
        return
    entry = pending[-1]
    entry["verdict"] = "discarded"
    save_results(results)

    # Reset train_experiment.py to last commit
    git("checkout", "HEAD", "--", "train_experiment.py", "src/training/trainer.py")
    git("add", "experiments/results.json")
    git("commit", "-m", f"autoresearch: discard {entry['run_id']} — val={entry['results']['best_val_acc']*100:.1f}% test={entry['results']['test_acc']*100:.1f}%")
    git("push", "origin", "main")
    print(f"❌ Discarded {entry['run_id']}, train_experiment.py reverted.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoResearch loop for CNNStockMarket")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status")

    p_run = sub.add_parser("run")
    p_run.add_argument("--hypothesis", "-H", type=str, help="What are we testing?")

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--run-id", type=str)
    p_collect.add_argument("--hypothesis", "-H", type=str)
    p_collect.add_argument("--changes", type=str)
    p_collect.add_argument("--notes", type=str)

    sub.add_parser("keep")
    sub.add_parser("revert")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "keep":
        cmd_keep(args)
    elif args.command == "revert":
        cmd_revert(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
