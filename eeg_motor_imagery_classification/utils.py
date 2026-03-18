"""General utility helpers for the EEG project."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(value):
    """Recursively convert NumPy-friendly objects into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path: str | Path, payload) -> Path:
    """Write JSON data with stable formatting."""

    output_path = Path(path)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2))
    return output_path


def write_text(path: str | Path, content: str) -> Path:
    """Write plain text content."""

    output_path = Path(path)
    output_path.write_text(content)
    return output_path
