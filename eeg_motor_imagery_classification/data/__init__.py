"""Canonical data pipeline for EEG motor imagery experiments."""

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.data.epochs import (
    EpochsBundle,
    MultiSubjectEpochsBundle,
    extract_epochs_array,
    extract_subject_epochs,
    load_all_subject_epochs,
)
from eeg_motor_imagery_classification.data.loaders import (
    DatasetMetadata,
    SubjectRun,
    SubjectSessions,
    load_bnci_subject_runs,
    load_subject_bundle,
)
from eeg_motor_imagery_classification.data.preprocessing import preprocess_runs
from eeg_motor_imagery_classification.data.splits import (
    SplitBundle,
    leave_one_subject_out,
    stratified_within_subject_split,
    transfer_split,
)

__all__ = [
    "DatasetMetadata",
    "EpochDataset",
    "EpochsBundle",
    "MultiSubjectEpochsBundle",
    "SplitBundle",
    "SubjectRun",
    "SubjectSessions",
    "extract_epochs_array",
    "extract_subject_epochs",
    "leave_one_subject_out",
    "load_bnci_subject_runs",
    "load_all_subject_epochs",
    "load_subject_bundle",
    "preprocess_runs",
    "stratified_within_subject_split",
    "transfer_split",
]
