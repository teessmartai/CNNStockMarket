#!/usr/bin/env python3
"""
AutoResearch loop for CNNStockMarket.

Commands:
  status          — show queue, slots, and current best
  run             — manually push current train_experiment.py to Kaggle
  collect         — download latest kernel output, parse, update results.json
  keep / revert   — mark last pending result as kept or discarded
  daemon          — poll both GPU slots, auto-launch from queue, auto-collect

Queue:
  Edit experiments/queue.json to add experiments.
  The daemon picks them up automatically.
  Format: {"id": "run_NNN_desc", "hypothesis": "...", "cmd_args": "...", "status": "pending"}
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent.parent.resolve()
RESULTS_FILE = REPO_ROOT / "experiments" / "results.json"
QUEUE_FILE   = REPO_ROOT / "experiments" / "queue.json"
SLOTS_FILE   = REPO_ROOT / "experiments" / "slot_state.json"
KAGGLE_TOKEN = "KGAT_6b6cb4995fa85455d4045d57523e70e7"
KERNEL_SLUG  = "tassistant/cnn-stock-market-training"
# Same kernel, multiple versions run in parallel — no second slug needed
SLOTS = {
    "a": KERNEL_SLUG,
    "b": KERNEL_SLUG,
}
BASE_CMD     = "--preset largecap-stable --window 128 --horizon 5 --years 5 --norm logreturns"
DATASET_DIR  = Path("/tmp/kaggle-code-dataset")
KERNEL_DIRS  = {
    "a": Path("/tmp/kaggle-setup/kernel"),
    "b": Path("/tmp/kaggle-setup/kernel"),
}
VENV_KAGGLE  = REPO_ROOT / "venv" / "bin" / "kaggle"
POLL_INTERVAL = 300  # seconds between daemon polls


# ── Helpers ────────────────────────────────────────────────────────────────

def kaggle(*args):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    result = subprocess.run(
        [str(VENV_KAGGLE)] + list(args),
        capture_output=True, text=True, env=env
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def git(*args, cwd=None):
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
        cwd=cwd or REPO_ROOT
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def ts():
    return datetime.now(timezone.utc).isoformat()


def current_best_test_acc():
    results = load_json(RESULTS_FILE, [])
    kept = [r for r in results if r.get("verdict") == "kept"]
    if not kept:
        return 0.0
    return max(r["results"].get("test_acc", 0) for r in kept)


def next_run_number():
    results = load_json(RESULTS_FILE, [])
    queue   = load_json(QUEUE_FILE, [])
    return len(results) + 1


# ── Dataset / kernel push ─────────────────────────────────────────────────

def rebuild_dataset(message="autoresearch update"):
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    for name in ["train_experiment.py", "backtest_stoploss.py", "kaggle_config.yaml"]:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, DATASET_DIR / name)

    for folder in ["src", "data"]:
        src_dir = REPO_ROOT / folder
        if src_dir.exists():
            zp = DATASET_DIR / f"{folder}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in src_dir.rglob("*"):
                    if "__pycache__" not in str(f) and f.suffix != ".pyc":
                        zf.write(f, f.relative_to(REPO_ROOT / folder))

    save_json(DATASET_DIR / "dataset-metadata.json", {
        "title": "CNN Stock Market Code and Data",
        "id": "tassistant/cnn-stock-code",
        "licenses": [{"name": "other"}]
    })

    out, err, rc = kaggle("datasets", "version", "-p", str(DATASET_DIR),
                          "--dir-mode", "zip", "-m", message)
    if rc != 0:
        print(f"  Dataset upload error: {err}")
    return rc == 0


def patch_runner_cmd(slot, cmd_args):
    """Inject cmd_args into the runner_launcher.py for the given slot."""
    runner = KERNEL_DIRS[slot] / "runner_launcher.py"
    text = runner.read_text()
    new_cmd = f"python train_experiment.py {BASE_CMD} {cmd_args}".strip()
    # Replace the _train_cmd = "..." line cleanly (avoids duplicate flags)
    text = re.sub(
        r'_train_cmd = "python train_experiment\.py[^"]*"',
        f'_train_cmd = "{new_cmd}"',
        text
    )
    runner.write_text(text)


def push_kernel(slot):
    out, err, rc = kaggle("kernels", "push", "-p", str(KERNEL_DIRS[slot]))
    print(f"  [{slot}] {out or err}")
    # Extract version number: "Kernel version 20 successfully pushed"
    m = re.search(r"version (\d+)", out)
    version = int(m.group(1)) if m else None
    return rc == 0, version


def download_log_for_version(version: int) -> Optional[Path]:
    """Download experiment_result.log for a specific kernel version via kagglehub."""
    import os as _os
    env = _os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    out_dir = Path(f"/tmp/autoresearch_v{version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "experiment_result.log"
    if dest.exists():
        return dest  # cached

    result = subprocess.run(
        [str(VENV_KAGGLE.parent / "python3"), "-c",
         f"""
import os, kagglehub
os.environ['KAGGLE_API_TOKEN'] = '{KAGGLE_TOKEN}'
path = kagglehub.notebook_output_download(
    '{KERNEL_SLUG}/versions/{version}',
    path='experiment_result.log',
    output_dir='{out_dir}',
    force_download=True
)
print('OK:', path)
"""],
        capture_output=True, text=True, env=env
    )
    if "OK:" in result.stdout and dest.exists():
        return dest
    return None


def download_latest_log(tag: str):
    """Fallback: download latest completed version's log via kaggle CLI."""
    out_dir = Path(f"/tmp/autoresearch_latest_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out, err, rc = kaggle("kernels", "output", KERNEL_SLUG,
                          "-p", str(out_dir), "--file-pattern", ".log$", "-o")
    if rc != 0:
        return None
    logs = list(out_dir.glob("**/experiments/*.log"))
    return logs[0] if logs else None


def log_matches_slot(log_path: Path, slot_state: dict) -> bool:
    """Check if a log file's config matches the experiment launched in this slot."""
    if not log_path or not slot_state:
        return False
    cmd_args = slot_state.get("cmd_args", "")
    text = log_path.read_text()
    # Each flag leaves a trace in the log — check key flags
    checks = {
        "--optimizer adamw": "Optimizer: adamw" in text,
        "--residual":        "Residual: True" in text,
        "--no-batchnorm":    "BatchNorm: False" in text,
        "--norm logreturns": "Norm: logreturns" in text,
        "--norm minmax":     "Norm: minmax" in text,
    }
    for flag, present in checks.items():
        if flag in cmd_args and not present:
            return False
        if flag not in cmd_args and present and flag in ("--optimizer adamw", "--residual", "--no-batchnorm"):
            return False
    return True


def slot_version_done(slot_state: dict):
    """Return 'complete', 'error', or 'running' for a tracked slot."""
    if slot_state is None:
        return "free"
    version = slot_state.get("version")
    if not version:
        return "running"

    # Try version-specific download via kagglehub — only succeeds when complete
    log = download_log_for_version(version)
    if log:
        return "complete"

    # Fall back to overall status to detect errors
    out, _, _ = kaggle("kernels", "status", KERNEL_SLUG)
    if "ERROR" in out.upper():
        return "error"
    return "running"


# ── Result parsing ─────────────────────────────────────────────────────────

def parse_log(log_path: Path):
    if not log_path or not log_path.exists():
        return None
    log_text = log_path.read_text()

    def extract(pattern, cast=float):
        m = re.search(pattern, log_text)
        return cast(m.group(1)) if m else None

    gpu_match = re.search(r"Device: GPU — ([^\n(]+)", log_text)
    return {
        "device": gpu_match.group(1).strip() if gpu_match else "unknown",
        "best_val_acc":  round((extract(r"Best val acc:\s+([\d.]+)%") or 0) / 100, 4),
        "test_acc":      round((extract(r"Test accuracy:\s+([\d.]+)%") or 0) / 100, 4),
        "test_loss":     extract(r"Test loss:\s+([\d.]+)") or 0,
        "epochs_run":    int(extract(r"Epochs run:\s+(\d+)") or 0),
        "runtime_min":   extract(r"Runtime:\s+([\d.]+) min") or 0,
        "early_stopped": True,
    }


def collect_result(slug, slot):
    out_dir = Path(f"/tmp/autoresearch_{slot}_{int(time.time())}")
    out_dir.mkdir(parents=True)

    out, err, rc = kaggle("kernels", "output", slug,
                          "-p", str(out_dir), "--file-pattern", ".log$")
    if rc != 0:
        print(f"  Output download error: {err}")
        return None

    log_files = list(out_dir.glob("**/experiments/*.log"))
    if not log_files:
        return None

    log_text = log_files[0].read_text()

    def extract(pattern, cast=float):
        m = re.search(pattern, log_text)
        return cast(m.group(1)) if m else None

    gpu_match = re.search(r"Device: GPU — ([^\n(]+)", log_text)
    return {
        "device": gpu_match.group(1).strip() if gpu_match else "unknown",
        "best_val_acc":  round((extract(r"Best val acc:\s+([\d.]+)%") or 0) / 100, 4),
        "test_acc":      round((extract(r"Test accuracy:\s+([\d.]+)%") or 0) / 100, 4),
        "test_loss":     extract(r"Test loss:\s+([\d.]+)") or 0,
        "epochs_run":    int(extract(r"Epochs run:\s+(\d+)") or 0),
        "runtime_min":   extract(r"Runtime:\s+([\d.]+) min") or 0,
        "early_stopped": True,
    }


# ── Slot state ─────────────────────────────────────────────────────────────

def load_slots():
    return load_json(SLOTS_FILE, {"a": None, "b": None})


def save_slots(slots):
    save_json(SLOTS_FILE, slots)


def commit_results(message):
    git("add",
        "experiments/results.json",
        "experiments/queue.json",
        "experiments/slot_state.json",
        "train_experiment.py",
        "src/training/trainer.py",
        "src/models/cnn.py")
    git("commit", "-m", message)
    git("push", "origin", "main")


# ── Daemon ─────────────────────────────────────────────────────────────────

def daemon_tick():
    slots     = load_slots()
    queue     = load_json(QUEUE_FILE, [])
    results   = load_json(RESULTS_FILE, [])
    best_test = current_best_test_acc()

    pending   = [e for e in queue if e["status"] == "pending"]
    changed   = False

    for slot_id, slug in SLOTS.items():
        slot_run = slots.get(slot_id)

        # ── Collect finished run ───────────────────────────────────────────
        status = slot_version_done(slot_run) if slot_run else "free"
        if slot_run and status in ("complete", "error"):
            run_id = slot_run["run_id"]
            print(f"  [{slot_id}] {run_id} → {status}")

            metrics = None
            if status == "complete":
                version = slot_run.get("version")
                log_path = download_log_for_version(version)
                metrics = parse_log(log_path) if log_path else None

            # Find matching pending result entry (or create if manually seeded)
            entry = next((r for r in results if r["run_id"] == run_id
                          and r.get("verdict") == "pending"), None)
            if entry is None and metrics:
                entry = {
                    "run_id":                run_id,
                    "timestamp":             slot_run.get("started_at", ts()),
                    "kaggle_kernel_version": slot_run.get("version"),
                    "device":                metrics.pop("device", "unknown"),
                    "hypothesis":            slot_run.get("run_id", ""),
                    "changes_from_previous": slot_run.get("cmd_args", ""),
                    "config":                {"cmd_args": slot_run.get("cmd_args", "")},
                    "results":               {},
                    "verdict":               "pending",
                    "notes":                 "",
                }
                results.append(entry)

            if entry and metrics:
                entry["results"] = metrics
                entry["device"]  = metrics.pop("device", "unknown")
                improved = metrics.get("test_acc", 0) > best_test
                entry["verdict"] = "kept" if improved else "discarded"
                entry["notes"]   = (
                    f"test_acc {metrics.get('test_acc',0)*100:.1f}% "
                    f"{'> IMPROVED ✅' if improved else '≤ no improvement ❌'} "
                    f"vs best {best_test*100:.1f}%"
                )
                if improved:
                    best_test = metrics["test_acc"]
                print(f"    val={entry['results']['best_val_acc']*100:.1f}%  "
                      f"test={entry['results']['test_acc']*100:.1f}%  "
                      f"→ {entry['verdict']}")
            elif entry:
                entry["verdict"] = "discarded"
                entry["notes"]   = f"kernel {status} — no output collected"
                print(f"    No metrics collected ({status})")

            # Mark queue entry done
            for qe in queue:
                if qe["id"] == run_id:
                    qe["status"] = entry["verdict"] if entry else "error"
                    break

            slots[slot_id] = None
            changed = True

        # ── Launch next pending experiment ─────────────────────────────────
        if status in ("free",) and pending:
            exp = pending.pop(0)
            print(f"  [{slot_id}] Launching {exp['id']}: {exp['cmd_args']}")

            results.append({
                "run_id":                exp["id"],
                "timestamp":             ts(),
                "kaggle_kernel_version": None,
                "device":                "pending",
                "hypothesis":            exp["hypothesis"],
                "changes_from_previous": exp["cmd_args"],
                "config":                {"cmd_args": exp["cmd_args"]},
                "results":               {},
                "verdict":               "pending",
                "notes":                 "",
            })

            patch_runner_cmd(slot_id, exp["cmd_args"])
            ok, version = push_kernel(slot_id)
            if ok:
                slots[slot_id] = {"slot": slot_id, "run_id": exp["id"],
                                   "version": version, "started_at": ts()}
                exp["status"] = "running"
                changed = True
            else:
                results.pop()
                print(f"    Push failed — will retry next tick")

    if changed:
        save_json(RESULTS_FILE, results)
        save_json(QUEUE_FILE, queue)
        save_slots(slots)
        commit_results("autoresearch: daemon tick")
        print("  Committed results + queue state")


def cmd_daemon(args):
    print(f"\n🔄 AutoResearch daemon started (poll every {POLL_INTERVAL//60} min)")
    print(f"   Queue: {QUEUE_FILE}")
    print(f"   Add experiments to queue.json at any time — picked up next tick\n")

    # Rebuild dataset once at start
    print("Rebuilding dataset...")
    rebuild_dataset()

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tick")
        try:
            daemon_tick()
        except Exception as e:
            print(f"  ERROR in tick: {e}")
            import traceback; traceback.print_exc()

        queue = load_json(QUEUE_FILE, [])
        slots = load_slots()
        running = [s for s in slots.values() if s is not None]
        pending = [e for e in queue if e["status"] == "pending"]

        if not running and not pending:
            print("\n✅ Queue empty and no runs active — daemon done.")
            break

        print(f"  Running: {len(running)} | Pending: {len(pending)}")
        time.sleep(POLL_INTERVAL)


# ── Other commands ─────────────────────────────────────────────────────────

def cmd_status(args):
    results = load_json(RESULTS_FILE, [])
    queue   = load_json(QUEUE_FILE, [])
    slots   = load_slots()

    kept = [r for r in results if r.get("verdict") == "kept"]
    best = max(kept, key=lambda r: r["results"].get("test_acc", 0)) if kept else None

    print(f"\n{'='*65}")
    print(f"  AutoResearch — {len(results)} runs  |  {len(kept)} kept")
    print(f"{'='*65}")

    if best:
        print(f"\n  🏆 Best: {best['run_id']}")
        print(f"     val={best['results']['best_val_acc']*100:.1f}%  "
              f"test={best['results']['test_acc']*100:.1f}%  "
              f"cfg: {best['config'].get('cmd_args', 'baseline')}")

    print(f"\n  Active slots:")
    for sid, slug in SLOTS.items():
        state = slots.get(sid)
        status = slot_version_done(state) if state else "free"
        run = state["run_id"] if state else "—"
        print(f"    [{sid}] v{state.get('version','?') if state else '-':3}  {status:10s}  {run}")

    print(f"\n  Recent results:")
    for r in results[-6:]:
        icon = "✅" if r.get("verdict") == "kept" else ("⏳" if r.get("verdict") == "pending" else "❌")
        ta = r["results"].get("test_acc", 0)
        va = r["results"].get("best_val_acc", 0)
        print(f"  {icon} {r['run_id']:35s}  val={va*100:.1f}%  test={ta*100:.1f}%")

    pending = [e for e in queue if e["status"] == "pending"]
    print(f"\n  Queue ({len(pending)} pending):")
    for e in queue:
        icon = {"pending": "⏳", "running": "🔄", "kept": "✅", "discarded": "❌", "error": "💥"}.get(e["status"], "?")
        print(f"  {icon} {e['id']:35s}  {e['cmd_args']}")
    print()


def cmd_run(args):
    hypothesis = args.hypothesis or "manual run"
    print(f"Rebuilding dataset...")
    rebuild_dataset()
    slot = args.slot or "a"
    if args.cmd_args:
        patch_runner_cmd(slot, args.cmd_args)
    push_kernel(slot)
    print(f"Kernel pushed to slot {slot}.")


def cmd_add(args):
    queue = load_json(QUEUE_FILE, [])
    entry = {
        "id":         args.id,
        "hypothesis": args.hypothesis or "",
        "cmd_args":   args.cmd_args,
        "status":     "pending",
    }
    queue.append(entry)
    save_json(QUEUE_FILE, queue)
    git("add", "experiments/queue.json")
    git("commit", "-m", f"autoresearch: queue {args.id}")
    git("push", "origin", "main")
    print(f"Added {args.id} to queue.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AutoResearch loop")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status")

    d = sub.add_parser("daemon", help="Poll slots and auto-launch from queue")
    d.add_argument("--once", action="store_true", help="Run one tick then exit")

    r = sub.add_parser("run")
    r.add_argument("--hypothesis", "-H")
    r.add_argument("--cmd-args",   type=str, default="")
    r.add_argument("--slot",       choices=["a", "b"], default="a")

    a = sub.add_parser("add", help="Add experiment to queue")
    a.add_argument("id")
    a.add_argument("--cmd-args",   required=True)
    a.add_argument("--hypothesis", "-H", default="")

    args = p.parse_args()
    {
        "status":  cmd_status,
        "daemon":  cmd_daemon,
        "run":     cmd_run,
        "add":     cmd_add,
    }.get(args.command, lambda a: p.print_help())(args)


if __name__ == "__main__":
    main()
