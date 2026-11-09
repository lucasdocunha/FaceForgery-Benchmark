"""Portable JSON/YAML configuration shared by experimental entry points."""

from __future__ import annotations

import os
import json
from pathlib import Path
import re

import yaml


PATH_KEYS = {
    "path", "root", "manifest", "checkpoint", "run", "run_dir", "output", "output_dir",
    "cache", "cache_root", "calibration", "ae_run", "spatial_run", "latent_run", "metric_run",
    "weights", "backbone_weights", "lpips_state_path", "pretrained_path", "landmarks",
    "projection_run", "label_mask", "landmark_cache",
}
PATH_CONTAINERS = {"caches", "target_caches", "sources"}


def resolve_values(value, base, *, key=None, paths=False):
    """Expand declared variables and resolve path fields from the config directory."""
    if isinstance(value, dict):
        # Sources are typed records: only their explicit path fields are paths.
        # Cache mappings instead allow target names to map directly to paths.
        return {name: resolve_values(item, base, key=name,
                                     paths=key in PATH_CONTAINERS and key != "sources")
                for name, item in value.items()}
    if isinstance(value, list):
        return [resolve_values(item, base, key=key, paths=paths or key in PATH_CONTAINERS) for item in value]
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded):
        raise ValueError(f"Unset environment variable in config: {value}")
    is_path = (paths or key in PATH_KEYS or
               bool(key and key.endswith(("_manifest", "_root", "_cache", "_path"))))
    if is_path and expanded:
        result = Path(expanded).expanduser()
        return str(result if result.is_absolute() else Path(base) / result)
    return expanded


def read_document(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text()
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Config must be a JSON or YAML mapping")
    return resolve_values(value, path.parent)
