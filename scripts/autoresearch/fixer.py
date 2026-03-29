"""
AutoResearch Fixer — autonomous bug repair for failed Kaggle runs.

Called by loop.py when classify_failure() returns 'bug'.
Extracts the traceback, identifies affected files, calls Claude to propose
a minimal patch, applies + verifies it, commits, and signals loop.py to retry.

Usage (standalone for testing):
    python scripts/autoresearch/fixer.py <log_path>

Returns exit code 0 if fix was applied, 1 if fix was skipped/failed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_DIRS  = ["src", "train_experiment.py", "scripts/autoresearch/runner_launcher.py"]

# Files the fixer is allowed to modify (relative to REPO_ROOT)
ALLOWED_PREFIXES = ("src/", "train_experiment.py")

# Max lines changed — refuse larger patches (too risky to auto-apply)
MAX_PATCH_LINES = 40

_AUTH_PROFILE = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
_MODEL        = "claude-sonnet-4-5"   # use a smarter model for code fixes


def _get_api_key() -> str:
    try:
        data = json.loads(_AUTH_PROFILE.read_text())
        for profile in data.get("profiles", {}).values():
            if profile.get("provider") == "anthropic" and profile.get("token"):
                return profile["token"]
    except Exception:
        pass
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    raise RuntimeError("No Anthropic API key found.")


def _call_claude(prompt: str) -> str:
    api_key = _get_api_key()
    payload = {
        "model": _MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
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
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


# ── Source file collection ────────────────────────────────────────────────────

def _collect_source(traceback_text: str) -> dict[str, str]:
    """
    Find source files mentioned in the traceback and return their contents.
    Also always include train_experiment.py as it's commonly involved.
    """
    files: dict[str, str] = {}
    # Extract filenames from traceback: File "path/to/foo.py", line N
    for m in re.finditer(r'File "([^"]+\.py)"', traceback_text):
        path = Path(m.group(1))
        # Make relative to REPO_ROOT if possible
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            # Try stripping common Kaggle prefixes
            for prefix in ["/kaggle/working/", "/tmp/kaggle-setup/kernel/"]:
                if str(path).startswith(prefix):
                    rel = Path(str(path).replace(prefix, ""))
                    break
            else:
                continue
        abs_path = REPO_ROOT / rel
        if abs_path.exists() and abs_path.suffix == ".py":
            files[str(rel)] = abs_path.read_text(errors="replace")

    # Always include train_experiment.py
    te = REPO_ROOT / "train_experiment.py"
    if te.exists() and "train_experiment.py" not in files:
        files["train_experiment.py"] = te.read_text(errors="replace")

    return files


# ── Claude prompt ─────────────────────────────────────────────────────────────

def _build_prompt(traceback_text: str, source_files: dict[str, str]) -> str:
    src_block = "\n\n".join(
        f"### {name}\n```python\n{content[:6000]}\n```"
        for name, content in source_files.items()
    )
    return f"""You are an automated bug fixer for a PyTorch ML training pipeline.
A Kaggle kernel run failed with the following traceback:

```
{traceback_text}
```

Here are the relevant source files:

{src_block}

Your job: propose a MINIMAL fix. Output ONLY valid JSON (no markdown, no explanation outside the JSON):

{{
  "file": "relative/path/to/file.py",
  "old_text": "exact substring to replace (copy verbatim from the source)",
  "new_text": "replacement text",
  "explanation": "one sentence describing what was wrong and what was fixed"
}}

Rules:
- old_text must be an EXACT verbatim substring of the file content shown above
- Keep the fix minimal — change as few lines as possible
- Only modify files under src/ or train_experiment.py
- Do NOT touch scripts/autoresearch/ files
- If the bug requires a large refactor, or you are uncertain, return: {{"file": null}}
"""


# ── Patch application ─────────────────────────────────────────────────────────

def _apply_patch(file_rel: str, old_text: str, new_text: str) -> tuple[bool, str]:
    """Apply the patch. Returns (success, reason)."""
    abs_path = REPO_ROOT / file_rel

    # Safety checks
    if not any(file_rel.startswith(p) for p in ALLOWED_PREFIXES):
        return False, f"File {file_rel!r} not in allowed paths"
    if not abs_path.exists():
        return False, f"File not found: {abs_path}"

    original = abs_path.read_text()
    if old_text not in original:
        return False, f"old_text not found verbatim in {file_rel}"

    patch_lines = abs(new_text.count("\n") - old_text.count("\n")) + \
                  max(new_text.count("\n"), old_text.count("\n"))
    if patch_lines > MAX_PATCH_LINES:
        return False, f"Patch too large ({patch_lines} lines > {MAX_PATCH_LINES} limit)"

    patched = original.replace(old_text, new_text, 1)

    # Syntax check
    try:
        ast.parse(patched)
    except SyntaxError as e:
        return False, f"Patched file has syntax error: {e}"

    abs_path.write_text(patched)
    return True, "ok"


def _git_commit(message: str) -> bool:
    result = subprocess.run(
        ["git", "add", "-A"],
        cwd=REPO_ROOT, capture_output=True
    )
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.returncode == 0


# ── Main entry point ──────────────────────────────────────────────────────────

def attempt_fix(log_path: Path, run_id: str) -> bool:
    """
    Try to auto-fix the bug in log_path.

    Returns True if a fix was successfully applied and committed.
    Returns False if no fix could be determined (caller should fall back).
    """
    print(f"\n🔧 AutoFixer: analysing bug in {run_id}...")

    if not log_path or not log_path.exists():
        print("  No log available — cannot fix.")
        return False

    log_text = log_path.read_text(errors="replace")

    # Extract traceback
    tb_match = re.search(
        r"(Traceback \(most recent call last\).*?)(?=\n[A-Z]|\Z)",
        log_text, re.DOTALL
    )
    if not tb_match:
        print("  No traceback found in log.")
        return False
    traceback_text = tb_match.group(1).strip()
    print(f"  Traceback:\n{traceback_text[:400]}")

    source_files = _collect_source(traceback_text)
    if not source_files:
        print("  No matching source files found.")
        return False

    print(f"  Source files: {list(source_files.keys())}")
    print(f"  Calling Claude ({_MODEL}) for fix...")

    try:
        raw = _call_claude(_build_prompt(traceback_text, source_files))
    except Exception as e:
        print(f"  Claude call failed: {e}")
        return False

    print(f"  Raw response: {raw[:400]}")

    # Parse JSON — strip markdown fences if present
    raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if not m:
        print("  Could not parse JSON from response.")
        return False

    try:
        fix = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return False

    if not fix.get("file"):
        print(f"  Claude declined to fix: uncertain or too complex.")
        return False

    file_rel  = fix["file"]
    old_text  = fix.get("old_text", "")
    new_text  = fix.get("new_text", "")
    expl      = fix.get("explanation", "")

    print(f"  Fix proposed: {file_rel}")
    print(f"  Explanation: {expl}")

    ok, reason = _apply_patch(file_rel, old_text, new_text)
    if not ok:
        print(f"  ❌ Patch rejected: {reason}")
        return False

    committed = _git_commit(
        f"fix(autofixer): {expl[:72]}\n\n"
        f"Auto-fixed bug in run {run_id}.\n"
        f"File: {file_rel}\n"
        f"Fixer model: {_MODEL}"
    )
    if committed:
        print(f"  ✅ Fix applied and committed: {expl}")
    else:
        print(f"  ✅ Fix applied (commit skipped — nothing staged or already committed)")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fixer.py <log_path> [run_id]")
        sys.exit(1)
    log_p  = Path(sys.argv[1])
    run_id = sys.argv[2] if len(sys.argv) > 2 else "manual"
    success = attempt_fix(log_p, run_id)
    sys.exit(0 if success else 1)
