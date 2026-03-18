"""Epoch extraction and canonical NumPy conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mne

from eeg_motor_imagery_classification.config import PreprocessingConfig
from eeg_motor_imagery_classification.data.loaders import DatasetMetadata, SubjectSessions, load_subject_bundle
from eeg_motor_imagery_classification.data.preprocessing import preprocess_runs


@dataclass(frozen=True)
class EpochsBundle:
    """Canonical array representation used by downstream models."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: DatasetMetadata
    epochs: mne.Epochs


@dataclass(frozen=True)
class MultiSubjectEpochsBundle:
    """Canonical arrays aggregated across multiple subjects."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: dict[int, DatasetMetadata]


def _extract_target_event_id(event_id: dict[str, int], config: PreprocessingConfig) -> dict[str, int]:
    missing = [event_name for event_name in config.event_names if event_name not in event_id]
    if missing:
        raise ValueError(f"Required events missing from run annotations: {missing}")
    return {event_name: event_id[event_name] for event_name in config.event_names}


def extract_subject_epochs(
    subject_id: int,
    sessions: SubjectSessions,
    config: PreprocessingConfig | None = None,
) -> EpochsBundle:
    """Preprocess all runs for one subject and return canonical arrays."""

    cfg = config or PreprocessingConfig()
    epoch_list: list[mne.Epochs] = []
    used_session_names: list[str] = []
    used_run_names: list[str] = []

    for session_name, run_name, raw in preprocess_runs(sessions, config=cfg):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        target_event_id = _extract_target_event_id(event_id, cfg)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=target_event_id,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        # Drop annotations before concatenation to avoid noisy MNE warnings.
        epochs.set_annotations(None)
        epoch_list.append(epochs)
        used_session_names.append(session_name)
        used_run_names.append(f"{session_name}/{run_name}")

    if not epoch_list:
        raise ValueError(f"No usable epochs found for subject {subject_id}")

    merged = mne.concatenate_epochs(epoch_list, add_offset=True, verbose=False)
    X, y = extract_epochs_array(merged, event_names=cfg.event_names)
    groups = np.full(shape=(len(y),), fill_value=subject_id, dtype=np.int64)

    metadata = DatasetMetadata(
        subject_id=subject_id,
        session_names=tuple(used_session_names),
        run_names=tuple(used_run_names),
        sfreq=float(merged.info["sfreq"]),
        ch_names=tuple(merged.ch_names),
        event_id={name: idx for idx, name in enumerate(cfg.event_names)},
    )
    return EpochsBundle(X=X, y=y, groups=groups, metadata=metadata, epochs=merged)


def extract_epochs_array(epochs: mne.Epochs, event_names: tuple[str, ...] = ("left_hand", "right_hand")) -> tuple[np.ndarray, np.ndarray]:
    """Convert MNE epochs to the project's NumPy format and zero-based labels."""

    X = epochs.get_data(copy=True).astype(np.float32)
    raw_labels = epochs.events[:, -1]
    target_event_id = {name: epochs.event_id[name] for name in event_names}
    zero_based_lookup = {event_code: index for index, event_code in enumerate(target_event_id.values())}
    y = np.asarray([zero_based_lookup[label] for label in raw_labels], dtype=np.int64)
    return X, y


def load_all_subject_epochs(
    subjects: tuple[int, ...],
    config: PreprocessingConfig | None = None,
) -> MultiSubjectEpochsBundle:
    """Load and extract canonical arrays for multiple subjects."""

    bundles = []
    metadata: dict[int, DatasetMetadata] = {}
    for subject_id in subjects:
        sessions, _ = load_subject_bundle(subject_id)
        bundle = extract_subject_epochs(subject_id, sessions, config=config)
        bundles.append(bundle)
        metadata[subject_id] = bundle.metadata

    if not bundles:
        raise ValueError("At least one subject is required.")

    return MultiSubjectEpochsBundle(
        X=np.concatenate([bundle.X for bundle in bundles], axis=0),
        y=np.concatenate([bundle.y for bundle in bundles], axis=0),
        groups=np.concatenate([bundle.groups for bundle in bundles], axis=0),
        metadata=metadata,
    )
