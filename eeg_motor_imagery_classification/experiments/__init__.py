"""Experiment entry points."""

from eeg_motor_imagery_classification.experiments.baseline_classical import run_classical_within_subject_cv
from eeg_motor_imagery_classification.experiments.baseline_eegnet import run_eegnet_within_subject_cv
from eeg_motor_imagery_classification.experiments.baseline_riemann import run_riemann_within_subject_cv
from eeg_motor_imagery_classification.experiments.loso import run_classical_loso, run_eegnet_loso
from eeg_motor_imagery_classification.experiments.riemann_protocols import (
    run_riemann_loso,
    run_riemann_transfer,
    run_riemann_transfer_repeated_sweep,
    run_riemann_transfer_sweep,
)
from eeg_motor_imagery_classification.experiments.transfer import (
    run_classical_transfer_fbcsp,
    run_classical_transfer_fbcsp_repeated_sweep,
    run_classical_transfer_fbcsp_sweep,
    run_eegnet_transfer,
    run_eegnet_transfer_repeated_sweep,
    run_eegnet_transfer_sweep,
)

__all__ = [
    "run_classical_loso",
    "run_classical_transfer_fbcsp",
    "run_classical_transfer_fbcsp_repeated_sweep",
    "run_classical_transfer_fbcsp_sweep",
    "run_classical_within_subject_cv",
    "run_eegnet_loso",
    "run_eegnet_transfer",
    "run_eegnet_transfer_repeated_sweep",
    "run_eegnet_transfer_sweep",
    "run_eegnet_within_subject_cv",
    "run_riemann_loso",
    "run_riemann_transfer",
    "run_riemann_transfer_repeated_sweep",
    "run_riemann_transfer_sweep",
    "run_riemann_within_subject_cv",
]
