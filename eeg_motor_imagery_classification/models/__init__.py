"""Model builders for EEG experiments."""

from eeg_motor_imagery_classification.models.classical import (
    DEFAULT_FBCSP_BANDS,
    build_csp_pipeline,
    build_fbcsp_pipeline,
    build_raw_power_pipeline,
)
from eeg_motor_imagery_classification.models.eegnet import EEGNet
from eeg_motor_imagery_classification.models.riemann import build_riemann_tangent_pipeline

__all__ = [
    "DEFAULT_FBCSP_BANDS",
    "EEGNet",
    "build_csp_pipeline",
    "build_fbcsp_pipeline",
    "build_riemann_tangent_pipeline",
    "build_raw_power_pipeline",
]
