"""Dataset loading helpers built around MOABB and MNE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eeg_motor_imagery_classification.config import DatasetConfig

try:
    from moabb.datasets import BNCI2014_001
except ImportError:  # pragma: no cover - handled at runtime by callers
    BNCI2014_001 = None


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata attached to extracted epochs arrays."""

    subject_id: int
    session_names: tuple[str, ...]
    run_names: tuple[str, ...]
    sfreq: float
    ch_names: tuple[str, ...]
    event_id: dict[str, int]


SubjectRun = dict[str, Any]
SubjectSessions = dict[str, dict[str, Any]]


def _require_moabb() -> None:
    if BNCI2014_001 is None:
        raise ImportError(
            "moabb is required to load BNCI2014_001. "
            "Install the project dependencies before running data loaders."
        )


def load_bnci_subject_runs(subject_id: int) -> SubjectSessions:
    """Return the raw MOABB session/run mapping for one subject."""

    _require_moabb()
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject_id])
    return data[subject_id]


def load_subject_bundle(subject_id: int, config: DatasetConfig | None = None) -> tuple[SubjectSessions, DatasetMetadata]:
    """Load one subject and attach lightweight metadata."""

    _require_moabb()
    _ = config or DatasetConfig()
    sessions = load_bnci_subject_runs(subject_id)

    session_names = tuple(sessions.keys())
    run_names = tuple(
        f"{session_name}/{run_name}"
        for session_name, runs in sessions.items()
        for run_name in runs.keys()
    )
    first_run = next(iter(next(iter(sessions.values())).values()))

    metadata = DatasetMetadata(
        subject_id=subject_id,
        session_names=session_names,
        run_names=run_names,
        sfreq=float(first_run.info["sfreq"]),
        ch_names=tuple(first_run.ch_names),
        event_id={},
    )
    return sessions, metadata

