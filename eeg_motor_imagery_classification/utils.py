"""General utility helpers for the EEG project."""

from __future__ import annotations

import importlib
import json
import platform
import sys
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
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path: str | Path, payload) -> Path:
    """Write JSON data with stable formatting (UTF-8 on every platform)."""

    output_path = Path(path)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return output_path


def read_json(path: str | Path):
    """Read a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, content: str) -> Path:
    """Write plain text content as UTF-8 (the Windows default code page is not UTF-8)."""

    output_path = Path(path)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_csv(path: str | Path, header: list[str], rows: list[list[object]]) -> Path:
    """Write a small CSV file without quoting (values must not contain commas)."""

    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in row))
    return write_text(path, "\n".join(lines) + "\n")


def _package_version(name: str) -> str:
    try:
        return str(getattr(importlib.import_module(name), "__version__", "unknown"))
    except Exception:  # pragma: no cover - optional dependency missing
        return "not installed"


def environment_info(device: str | None = None) -> dict[str, object]:
    """Describe the software and hardware a result was produced with."""

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "sklearn", "mne", "moabb", "pyriemann", "torch", "matplotlib")
        },
    }
    try:
        import torch

        info["torch_device_requested"] = device or "auto"
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    except ImportError:  # pragma: no cover
        pass
    return info
