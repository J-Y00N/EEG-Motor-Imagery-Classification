"""Reusable split protocols for fair EEG evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class SplitBundle:
    """Indices defining one evaluation split."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    calibration_idx: np.ndarray | None = None


def stratified_within_subject_split(
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitBundle:
    """Create one reproducible stratified split for a single subject."""

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    return SplitBundle(train_idx=train_idx, test_idx=test_idx)


def leave_one_subject_out(groups: np.ndarray) -> list[SplitBundle]:
    """Create LOSO splits from subject IDs."""

    splits: list[SplitBundle] = []
    for subject_id in np.unique(groups):
        test_idx = np.flatnonzero(groups == subject_id)
        train_idx = np.flatnonzero(groups != subject_id)
        splits.append(SplitBundle(train_idx=train_idx, test_idx=test_idx))
    return splits


def transfer_split(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    target_subject: int,
    calibration_size: float = 0.5,
    random_state: int = 42,
) -> SplitBundle:
    """Split source subjects from one target subject plus target calibration/eval."""

    source_idx = np.flatnonzero(groups != target_subject)
    target_idx = np.flatnonzero(groups == target_subject)
    if len(target_idx) == 0:
        raise ValueError(f"Target subject {target_subject} is not present in groups")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - calibration_size,
        random_state=random_state,
    )
    cal_local_idx, eval_local_idx = next(splitter.split(np.zeros(len(target_idx)), y[target_idx]))
    calibration_idx = target_idx[cal_local_idx]
    test_idx = target_idx[eval_local_idx]
    return SplitBundle(train_idx=source_idx, test_idx=test_idx, calibration_idx=calibration_idx)

