"""
AutoResearch Strategist — repo-specific LLM experiment generator.

Called by loop.py when the experiment queue runs empty.  Reads the
current results, builds a research summary, calls Claude to propose
the next batch of experiments, validates them, and writes to queue.json.

Usage (standalone):
    python scripts/autoresearch/strategist.py [--dry-run] [--max N]

Called from loop.py:
    from scripts.autoresearch.strategist import run_strategy
    added = run_strategy()   # returns number of experiments queued
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.parent.resolve()
RESULTS_FILE = REPO_ROOT / "experiments" / "results.json"
QUEUE_FILE   = REPO_ROOT / "experiments" / "queue.json"

# ── Anthropic key — read from OpenClaw auth profile ───────────────────────────
_AUTH_PROFILE = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
_MODEL        = "claude-haiku-4-5"   # fast + cheap for strategy calls

def _get_api_key() -> str:
    try:
        data = json.loads(_AUTH_PROFILE.read_text())
        for profile in data.get("profiles", {}).values():
            if profile.get("provider") == "anthropic" and profile.get("token"):
                return profile["token"]
    except Exception:
        pass
    # Fallback: env var
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    raise RuntimeError(
        "No Anthropic API key found. "
        "Expected in ~/.openclaw/agents/main/agent/auth-profiles.json "
        "or ANTHROPIC_API_KEY env var."
    )


# ── Research context (repo-specific knowledge) ────────────────────────────────
RESEARCH_GOAL = """
Predict NEXT-DAY (h1, --horizon 1) S&P 500 stock price direction (up/down binary).
FOCUS: h1 ONLY. Do NOT propose h5 or any other horizon.
PRIMARY METRIC: test accuracy (higher is better, baseline random = 50%).
SECONDARY GOAL: regime-agnostic — model should work across market regimes (pre/post COVID).

IMPORTANT — DATA LEAKAGE WARNING:
Configs using stride=1 + shuffle-split have INFLATED accuracy due to data leakage:
overlapping windows (stride=1) end up in both train and test after shuffling, so the
model effectively sees test data during training.  Any result >70% from stride=1 +
shuffle-split is NOT trustworthy.  The TRUE best validated result is 68.8% from
stride=133 (non-overlapping windows) which has zero leakage.  Do NOT propose
stride=1 + shuffle-split combinations.  Safe options:
  - stride >= horizon (non-overlapping)
  - stride=1 with chronological split (no --shuffle-split)
  - stride=5 with --shuffle-split (low but nonzero leakage, acceptable)
""".strip()

AVAILABLE_FLAGS = """
--preset [sp500 | largecap-stable]   Stock universe (sp500=402 tickers, largecap-stable=55)
--horizon [1..20]                    Prediction horizon in trading days (h1=next day, h5=1 week)
--window [5..256]                    Lookback window in trading days
--stride [1..500]                    Step between windows (133=non-overlapping for w20h5)
--years [3..20]                      Years of historical data
--shuffle-split                      Shuffle train/val/test split (reduces regime bias)
--arch [cnn | lstm | transformer | tcn]   Model architecture
--features [none | all | returns,gap,volatility,rsi,macd,bbands | subset+xrank]
                                     Feature engineering (xrank=cross-sectional rank norm)
--samples-per-param [10..200]        Controls model size: larger = smaller model (default 100)
--norm [logreturns | minmax]         Input normalisation
--dropout [0.0..0.6]                 Dropout rate
--lr [1e-5..1e-3]                    Learning rate
--optimizer [adam | adamw]
--residual                           Residual connections in CNN
--no-batchnorm                       Disable BatchNorm
""".strip()

MAX_EXPERIMENTS_PER_BATCH = 6


# ── Result summariser ─────────────────────────────────────────────────────────

def _build_summary() -> str:
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    queue   = json.loads(QUEUE_FILE.read_text())   if QUEUE_FILE.exists()   else []

    # All completed runs with real metrics
    completed = [
        r for r in results
        if r.get("verdict") in ("kept", "discarded")
        and r.get("results", {}).get("test_acc", 0) > 0.0
    ]
    kept = [r for r in completed if r["verdict"] == "kept"]

    def _is_leaky(run_id: str, queue: list) -> bool:
        """stride=1 + shuffle-split = data leakage → inflated accuracy."""
        cmd = next((e.get("cmd_args", "") for e in queue if e["id"] == run_id), "")
        # Use word-boundary regex so --stride 133 is NOT matched by stride=1 check
        stride_1 = bool(re.search(r"--stride\s+1\b", cmd))
        return stride_1 and "--shuffle-split" in cmd

    # Best result — exclude leaky configs
    clean = [r for r in completed if not _is_leaky(r["run_id"], queue)]
    best  = max(clean, key=lambda r: r["results"]["test_acc"], default=None)

    # Already-tested configs (for deduplication)
    tested_ids = {r["run_id"] for r in results}
    tested_ids |= {e["id"] for e in queue}

    lines = [
        f"TOTAL RUNS COMPLETED: {len(completed)}  "
        f"(leaky excluded: {len(completed)-len(clean)} with stride=1+shuffle)",
        f"TRUE BEST (no leakage): {best['results']['test_acc']*100:.1f}% test  "
        f"val={best['results']['best_val_acc']*100:.1f}%  "
        f"run={best['run_id']}  "
        f"cfg: {next((e['cmd_args'] for e in queue if e['id']==best['run_id']), 'see results')}"
        if best else "TRUE BEST: none yet",
        "",
        "TOP 10 CLEAN RUNS (stride >= horizon OR chronological split, no leakage):",
    ]
    for r in sorted(clean, key=lambda r: r["results"]["test_acc"], reverse=True)[:10]:
        ta = r["results"]["test_acc"] * 100
        cmd = next((e["cmd_args"] for e in queue if e["id"] == r["run_id"]), "")
        hyp = r.get("hypothesis", "")
        lines.append(f"  {ta:5.1f}%  {r['run_id']}")
        if cmd:
            lines.append(f"         cmd: {cmd}")
        if hyp:
            lines.append(f"         hyp: {hyp}")

    lines += ["", "ALL TESTED EXPERIMENT IDs (do not repeat these):"]
    lines += [f"  {eid}" for eid in sorted(tested_ids)]

    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an automated ML research strategist.
Your job: propose the NEXT batch of experiments to run, based on prior results.

RESEARCH GOAL:
{RESEARCH_GOAL}

AVAILABLE TRAIN FLAGS:
{AVAILABLE_FLAGS}

RULES:
1. Propose exactly {MAX_EXPERIMENTS_PER_BATCH} experiments (no more, no less).
2. Each experiment must test ONE clear hypothesis about what might improve accuracy.
3. Build on patterns from the results — don't repeat what's already been tried.
4. Vary one or two dimensions at a time; don't change everything at once.
5. NEVER reuse an experiment id or config that's already been tested.
6. Prefer regime-agnostic configs (--shuffle-split, --years 20, or stable post-COVID windows).
7. Use --samples-per-param 10 when using --stride 133 with --years 7 (few samples → bigger model needed).
8. Experiment IDs must follow pattern: run_p<phase>_<NNN>_<short_description>
   Use the next available phase number based on existing IDs.
9. HORIZON: ALWAYS use --horizon 1. NEVER propose --horizon 5 or any other value. h1 only.

OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
[
  {{
    "id": "run_p5_001_...",
    "hypothesis": "One sentence explaining what this tests and why it might help.",
    "cmd_args": "--preset sp500 --horizon 5 ..."
  }},
  ...
]""".strip()


def _call_claude(summary: str) -> str:
    api_key = _get_api_key()
    payload = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Here are the current experiment results:\n\n{summary}\n\n"
                    f"Propose the next {MAX_EXPERIMENTS_PER_BATCH} experiments."
                ),
            }
        ],
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


# ── Validation ────────────────────────────────────────────────────────────────

def _valid_cmd_args(cmd_args: str) -> tuple[bool, str]:
    """Basic sanity checks on proposed cmd_args."""
    known_flags = {
        "--preset", "--horizon", "--window", "--stride", "--years",
        "--shuffle-split", "--arch", "--features", "--samples-per-param",
        "--norm", "--dropout", "--lr", "--optimizer", "--residual",
        "--no-batchnorm", "--end-date", "--seed", "--epochs",
        "--target-params", "--clip-grad", "--patience", "--ckpt-every",
        "--batch", "--weight-decay", "--scheduler", "--reset",
    }
    tokens = cmd_args.split()
    for t in tokens:
        if t.startswith("--"):
            # Allow abbreviated flags like --no-batchnorm
            flag = re.match(r"(--[\w-]+)", t)
            if flag and flag.group(1) not in known_flags:
                return False, f"Unknown flag: {flag.group(1)}"
    if "--horizon" in cmd_args:
        m = re.search(r"--horizon\s+(\d+)", cmd_args)
        if m and not (1 <= int(m.group(1)) <= 20):
            return False, f"horizon out of range: {m.group(1)}"
    return True, "ok"


def _parse_llm_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response (handles markdown fences)."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first [ ... ] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in LLM response:\n{text[:400]}")
    return json.loads(m.group(0))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_strategy(dry_run: bool = False, max_experiments: int = MAX_EXPERIMENTS_PER_BATCH) -> int:
    """
    Run the strategy loop: summarise results, call LLM, validate, queue.

    Returns:
        Number of experiments added to the queue.
    """
    print("\n🧠 AutoStrategy: analysing results...")
    summary = _build_summary()
    print(summary[:600] + ("..." if len(summary) > 600 else ""))

    print(f"\n🧠 Calling Claude ({_MODEL}) for next experiment batch...")
    try:
        raw = _call_claude(summary)
    except urllib.error.HTTPError as e:
        print(f"  API error: {e.code} {e.reason}")
        body = e.read().decode()
        print(f"  Body: {body[:300]}")
        return 0
    except Exception as e:
        print(f"  Strategy call failed: {e}")
        return 0

    print(f"  Raw response:\n{raw[:800]}")

    try:
        proposals = _parse_llm_response(raw)
    except Exception as e:
        print(f"  Failed to parse LLM response: {e}")
        return 0

    # Load existing IDs for deduplication
    queue   = json.loads(QUEUE_FILE.read_text())   if QUEUE_FILE.exists()   else []
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    existing_ids = {e["id"] for e in queue} | {r["run_id"] for r in results}

    validated = []
    for p in proposals[:max_experiments]:
        exp_id   = p.get("id", "").strip()
        cmd_args = p.get("cmd_args", "").strip()
        hyp      = p.get("hypothesis", "").strip()

        if not exp_id or not cmd_args:
            print(f"  ⚠️  Skipping malformed proposal: {p}")
            continue
        if exp_id in existing_ids:
            print(f"  ⚠️  Skipping duplicate id: {exp_id}")
            continue
        ok, reason = _valid_cmd_args(cmd_args)
        if not ok:
            print(f"  ⚠️  Invalid cmd_args for {exp_id}: {reason}")
            continue

        validated.append({
            "id":          exp_id,
            "hypothesis":  hyp,
            "cmd_args":    cmd_args,
            "status":      "pending",
            "source":      "autostrategy",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  ✅ {exp_id}: {hyp[:70]}")

    if not validated:
        print("  No valid experiments proposed — stopping.")
        return 0

    if dry_run:
        print(f"\n[dry-run] Would add {len(validated)} experiments (not writing).")
        return len(validated)

    queue.extend(validated)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2) + "\n")
    print(f"\n✅ AutoStrategy added {len(validated)} experiments to queue.")
    return len(validated)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AutoResearch strategist — propose next experiments")
    p.add_argument("--dry-run", action="store_true", help="Print proposals without writing to queue")
    p.add_argument("--max",     type=int, default=MAX_EXPERIMENTS_PER_BATCH, help="Max experiments to add")
    args = p.parse_args()

    n = run_strategy(dry_run=args.dry_run, max_experiments=args.max)
    sys.exit(0 if n > 0 else 1)
