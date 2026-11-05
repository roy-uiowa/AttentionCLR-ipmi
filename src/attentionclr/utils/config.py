"""Config loading with shallow `defaults:` inheritance.

`defaults:` is a list of paths relative to the *consuming* config's directory.
Merge order: deepest default first, then the consuming config on top.

This is shallow merge (dict.update at top level). Nested dicts are NOT merged;
if you need to override one nested key, specify the full nested dict in the
consuming config. This is deliberate: implicit deep merges are how config
bugs get born. If a nested section is repeated, the consuming config replaces
it wholesale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load `path`, resolving `defaults:` recursively. Returns a flat dict.

    Cycle detection: a config that (transitively) defaults to itself raises.
    """
    path = Path(path).resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"Config inheritance cycle: {path}")
    _seen = _seen | {path}

    data = _load_yaml(path)
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]

    merged: dict[str, Any] = {}
    for d in defaults:
        d_path = (path.parent / d).resolve()
        if not d_path.exists():
            raise FileNotFoundError(f"{path}: default config not found: {d_path}")
        merged.update(load_config(d_path, _seen=_seen))

    merged.update(data)
    return merged


def merge_subconfig(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Inline a sub-config referenced by path (relative to cwd or config dir).

    Sub-configs are files like configs/models/resnext50.yaml; they are NOT
    discovered via `defaults:`. Instead the parent config names them by path
    under a section key, and this helper loads and inlines them.

    Example: cfg["model"] = "../models/resnext50.yaml"  →  inlined dict.
    If cfg["model"] is already a dict, it's returned unchanged (allows
    overriding a referenced sub-config with an inline block).
    """
    ref = cfg.get(key)
    if isinstance(ref, dict):
        return ref
    if isinstance(ref, str):
        p = Path(ref).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Sub-config {key!r} not found: {p}")
        sub = load_config(p)
        # The parent config's keys under `key` may still be present inline
        # alongside the path string — not supported here; keep it clean:
        return sub
    return {}
