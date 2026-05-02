"""Load and validate run config YAML against schema (`RunConfigModel`)."""

from __future__ import annotations

from pathlib import Path

import yaml

from book_companion.schema.validation import validate_run_config


def load_run_config(path: str | Path) -> dict:
    """Read YAML file and return validated initial state dict for `run_graph_once`"""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("run config YAML must be a mapping at the root")
    cfg = validate_run_config(raw)
    return cfg.to_initial_state()
