"""Cross-session (session 1 -> session 2) evaluation within each subject.

This is the standard protocol for BNCI 2014-001 / BCI Competition IV 2a: the model is fit on
the first recording session of a subject and tested on the second session, recorded on a
different day. Unlike shuffled within-subject CV it never mixes trials of the same session
between training and testing.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.evaluation.metrics import compute_classification_metrics
from eeg_motor_imagery_classification.evaluation.protocols import SubjectResult, summarize_subject_results
from eeg_motor_imagery_classification.models import (
    EEGNet,
    build_csp_pipeline,
    build_fbcsp_pipeline,
    build_raw_power_pipeline,
    build_riemann_tangent_pipeline,
)
from eeg_motor_imagery_classification.train import (
    TrainingConfig,
    build_train_validation_datasets,
    fit_model,
    predict_model,
)


def cross_session_splits(
    groups: np.ndarray,
    sessions: np.ndarray,
    *,
    train_session: int = 0,
    test_session: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return (subject, train_idx, test_idx) for every subject that has both sessions."""

    if sessions is None:
        raise ValueError("Cross-session evaluation requires per-trial session labels.")
    splits = []
    for subject_id in np.unique(groups):
        train_idx = np.flatnonzero((groups == subject_id) & (sessions == train_session))
        test_idx = np.flatnonzero((groups == subject_id) & (sessions == test_session))
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Subject {subject_id} is missing session {train_session} or {test_session}.")
        splits.append((int(subject_id), train_idx, test_idx))
    return splits


def _run_sklearn_cross_session(pipelines: dict[str, object], X, y, groups, sessions) -> dict[str, dict[str, object]]:
    subject_results = {name: [] for name in pipelines}
    runtimes = {name: 0.0 for name in pipelines}
    for subject_id, train_idx, test_idx in cross_session_splits(groups, sessions):
        for name, pipeline in pipelines.items():
            start = time.perf_counter()
            model = clone(pipeline).fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            runtimes[name] += time.perf_counter() - start
            subject_results[name].append(
                SubjectResult(label=f"S{subject_id}", metrics=compute_classification_metrics(y[test_idx], y_pred))
            )
    outputs = {name: summarize_subject_results(results) for name, results in subject_results.items()}
    for name in outputs:
        outputs[name]["runtime_seconds"] = float(runtimes[name])
    return outputs


def run_classical_cross_session(X, y, groups, sessions, *, sfreq: float) -> dict[str, dict[str, object]]:
    """Raw power, CSP, and FBCSP trained on session 1 and tested on session 2."""

    pipelines = {
        "raw_power": build_raw_power_pipeline(),
        "csp": build_csp_pipeline(),
        "fbcsp": build_fbcsp_pipeline(sfreq=sfreq),
    }
    return _run_sklearn_cross_session(pipelines, X, y, groups, sessions)


def run_riemann_cross_session(X, y, groups, sessions) -> dict[str, object]:
    """Tangent-space model trained on session 1 and tested on session 2."""

    return _run_sklearn_cross_session({"riemann": build_riemann_tangent_pipeline()}, X, y, groups, sessions)["riemann"]


def run_eegnet_cross_session(
    X,
    y,
    groups,
    sessions,
    *,
    training_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """EEGNet trained on session 1 (validation split taken from session 1) and tested on session 2."""

    cfg = training_config or TrainingConfig()
    subject_results: list[SubjectResult] = []
    training_histories: list[dict[str, object]] = []
    runtime_seconds = 0.0
    for subject_id, train_idx, test_idx in cross_session_splits(groups, sessions):
        train_dataset, val_dataset = build_train_validation_datasets(X[train_idx], y[train_idx], config=cfg)
        test_dataset = EpochDataset(X[test_idx], y[test_idx], scaler=train_dataset.scaler)
        start = time.perf_counter()
        model = EEGNet(n_channels=X.shape[1], n_times=X.shape[2], n_classes=len(np.unique(y)))
        training = fit_model(model, train_dataset, validation_dataset=val_dataset, config=cfg)
        y_pred = predict_model(training.model, test_dataset, config=cfg)
        runtime_seconds += time.perf_counter() - start
        training_histories.append({"label": f"S{subject_id}", **training.history})
        subject_results.append(
            SubjectResult(label=f"S{subject_id}", metrics=compute_classification_metrics(y[test_idx], y_pred))
        )
    result = summarize_subject_results(subject_results)
    result["runtime_seconds"] = float(runtime_seconds)
    result["training_histories"] = training_histories
    return result
