"""LOSO with a training set matched in size to cross-session evaluation.

Cross-session models are trained on one session of the test subject (144 trials) and tested on
that subject's second session. To separate the source of the training data (the same subject vs
other subjects) from its amount, this protocol trains on 144 trials drawn from the eight other
subjects (9 trials per class per subject) and tests on the same second-session trials of the
held-out subject. Only where the training data come from differs between the two protocols.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.evaluation.metrics import compute_classification_metrics
from eeg_motor_imagery_classification.evaluation.protocols import SubjectResult, summarize_subject_results
from eeg_motor_imagery_classification.experiments.repeated import average_over_seeds
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

TRIALS_PER_CLASS_PER_SOURCE = 9  # 8 source subjects x 2 classes x 9 = 144 trials, as in one session


def matched_loso_splits(
    y: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    *,
    seed: int,
    per_class_per_subject: int = TRIALS_PER_CLASS_PER_SOURCE,
    test_session: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(held-out subject, train_idx, test_idx) with a class- and subject-balanced training subsample."""

    if sessions is None:
        raise ValueError("Matched LOSO requires per-trial session labels.")
    subjects = np.unique(groups)
    splits = []
    for test_subject in subjects:
        train_parts = []
        for source in subjects:
            if source == test_subject:
                continue
            rng = np.random.default_rng([seed, int(test_subject), int(source)])
            for class_id in np.unique(y):
                candidates = np.flatnonzero((groups == source) & (y == class_id))
                if len(candidates) < per_class_per_subject:
                    raise ValueError(f"Subject {source} has fewer than {per_class_per_subject} trials of class {class_id}.")
                train_parts.append(rng.choice(candidates, size=per_class_per_subject, replace=False))
        train_idx = np.sort(np.concatenate(train_parts))
        test_idx = np.flatnonzero((groups == test_subject) & (sessions == test_session))
        splits.append((int(test_subject), train_idx, test_idx))
    return splits


def _run_sklearn_once(pipelines, X, y, groups, sessions, seed: int) -> dict[str, dict[str, object]]:
    subject_results = {name: [] for name in pipelines}
    runtimes = {name: 0.0 for name in pipelines}
    for subject_id, train_idx, test_idx in matched_loso_splits(y, groups, sessions, seed=seed):
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


def _run_sklearn_over_seeds(pipelines, X, y, groups, sessions, seeds) -> dict[str, dict[str, object]]:
    per_seed = {seed: _run_sklearn_once(pipelines, X, y, groups, sessions, seed) for seed in seeds}
    return {name: average_over_seeds({f"seed_{seed}": per_seed[seed][name] for seed in seeds}) for name in pipelines}


def run_classical_matched_loso(X, y, groups, sessions, *, sfreq: float, seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    """Raw power, CSP, and FBCSP; each seed draws a different training subsample."""

    pipelines = {
        "raw_power": build_raw_power_pipeline(),
        "csp": build_csp_pipeline(),
        "fbcsp": build_fbcsp_pipeline(sfreq=sfreq),
    }
    return _run_sklearn_over_seeds(pipelines, X, y, groups, sessions, seeds)


def run_riemann_matched_loso(X, y, groups, sessions, *, seeds: tuple[int, ...]) -> dict[str, object]:
    """Tangent-space model; each seed draws a different training subsample."""

    return _run_sklearn_over_seeds({"riemann": build_riemann_tangent_pipeline()}, X, y, groups, sessions, seeds)["riemann"]


def run_eegnet_matched_loso(X, y, groups, sessions, *, training_config: TrainingConfig | None = None) -> dict[str, object]:
    """EEGNet; the training seed also selects the training subsample (use with run_over_training_seeds)."""

    cfg = training_config or TrainingConfig()
    subject_results: list[SubjectResult] = []
    training_histories: list[dict[str, object]] = []
    runtime_seconds = 0.0
    for subject_id, train_idx, test_idx in matched_loso_splits(y, groups, sessions, seed=cfg.seed):
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
