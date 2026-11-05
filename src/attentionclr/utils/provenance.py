"""Capture run provenance. Written per run, JSON, includes git SHA + config hash."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from attentionclr.utils.git import git_sha_and_dirty


def capture_provenance(config: dict, run_dir: Path) -> dict:
    sha, dirty = git_sha_and_dirty()
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    info = {
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": config_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info