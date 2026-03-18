"""Shared constants for the EEG motor imagery project."""

PROJECT_NAME = "EEG Motor Imagery Classification"
DATASET_NAME = "BNCI2014_001"
DEFAULT_SUBJECTS = tuple(range(1, 10))
EVENT_NAMES = ("left_hand", "right_hand")
LABEL_TO_INDEX = {label: index for index, label in enumerate(EVENT_NAMES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
