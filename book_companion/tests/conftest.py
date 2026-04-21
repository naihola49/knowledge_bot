"""Test hooks: load repo-root .env.dev so HF_TOKEN is available for integration tests."""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _load_env_file(repo_root / ".env.dev")
