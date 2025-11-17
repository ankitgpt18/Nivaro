from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if suffix == ".json":
            return json.load(handle)

    raise ValueError(f"Unsupported config format: {suffix}")


def resolve_path(base: str | Path, relative: str | Path) -> Path:
    """Resolve a relative path against a base directory."""
    base_path = Path(base)
    rel_path = Path(relative)
    if rel_path.is_absolute():
        return rel_path
    return (base_path / rel_path).resolve()

