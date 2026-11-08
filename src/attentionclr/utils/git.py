"""Git provenance, robust to cwd / mount boundaries."""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../attentionCLR

# src/attentionclr/utils/git.py
def git_sha_and_dirty(repo_root: Path | None = None) -> tuple[str, bool]:
    cwd = repo_root or _REPO_ROOT
    if not (Path(cwd) / ".git").exists():
        return "unknown", True
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(cwd)
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=str(cwd)
        ).strip())
        return sha, dirty
    except Exception:
        return "unknown", True