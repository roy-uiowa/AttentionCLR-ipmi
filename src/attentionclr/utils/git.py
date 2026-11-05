"""Git provenance, robust to cwd / mount boundaries."""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../attentionCLR


def git_sha_and_dirty(repo_root: Path | None = None) -> tuple[str, bool]:
    cwd = repo_root or _REPO_ROOT
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(cwd), "status", "--porcelain"], text=True
            ).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", True
