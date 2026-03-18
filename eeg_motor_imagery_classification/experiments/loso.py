"""Leave-one-subject-out evaluation protocols."""

from __future__ import annotations

import time
import numpy as np
from sklearn.base import clone

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.data.splits import leave_one_subject_out
from eeg_motor_imagery_classification.evaluation.metrics import compute_classification_metrics
from eeg_motor_imagery_classification.evaluation.protocols import SubjectResult, summarize_subject_results
from eeg_motor_imagery_classification.models import (
    EEGNet,
    build_csp_pipeline,
    build_fbcsp_pipeline,
    build_raw_power_pipeline,
)
from eeg_motor_imagery_classification.train import (
    TrainingConfig,
    build_train_validation_datasets,
    fit_model,
    predict_model,
)


def run_classical_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    sfreq: float,
) -> dict[str, dict[str, object]]:
    """Evaluate classical baselines under strict LOSO subject splits."""

    pipelines = {
        "raw_power": build_raw_power_pipeline(),
        "csp": build_csp_pipeline(),
        "fbcsp": build_fbcsp_pipeline(sfreq=sfreq),
    }
    subject_results = {name: [] for name in pipelines}
    runtimes = {name: 0.0 for name in pipelines}

    for split in leave_one_subject_out(groups):
        subject_id = int(groups[split.test_idx][0])
        X_train, X_test = X[split.train_idx], X[split.test_idx]
        y_train, y_test = y[split.train_idx], y[split.test_idx]

        for name, pipeline in pipelines.items():
            start = time.perf_counter()
            model = clone(pipeline)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            runtimes[name] += time.perf_counter() - start
            subject_results[name].append(
                SubjectResult(
                    label=f"S{subject_id}",
                    metrics=compute_classification_metrics(y_test, y_pred),
                )
            )

    outputs = {name: summarize_subject_results(results) for name, results in subject_results.items()}
    for name in outputs:
        outputs[name]["runtime_seconds"] = float(runtimes[name])
    return outputs


def run_eegnet_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    training_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Evaluate EEGNet under strict LOSO subject splits."""

    cfg = training_config or TrainingConfig()
    subject_results: list[SubjectResult] = []
    runtime_seconds = 0.0
    training_histories: list[dict[str, object]] = []

    for split in leave_one_subject_out(groups):
        subject_id = int(groups[split.test_idx][0])
        X_train, X_test = X[split.train_idx], X[split.test_idx]
        y_train, y_test = y[split.train_idx], y[split.test_idx]

        train_dataset, val_dataset = build_train_validation_datasets(X_train, y_train, config=cfg)
        test_dataset = EpochDataset(X_test, y_test, scaler=train_dataset.scaler)

        start = time.perf_counter()
        model = EEGNet(
            n_channels=X.shape[1],
            n_times=X.shape[2],
            n_classes=len(np.unique(y)),
        )
        training = fit_model(model, train_dataset, validation_dataset=val_dataset, config=cfg)
        y_pred = predict_model(training.model, test_dataset, config=cfg)
        runtime_seconds += time.perf_counter() - start
        training_histories.append({"label": f"S{subject_id}", **training.history})
        subject_results.append(
            SubjectResult(
                label=f"S{subject_id}",
                metrics=compute_classification_metrics(y_test, y_pred),
            )
        )

    result = summarize_subject_results(subject_results)
    result["runtime_seconds"] = float(runtime_seconds)
    result["training_histories"] = training_histories
    return result
