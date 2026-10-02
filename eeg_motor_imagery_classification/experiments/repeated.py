"""Repeat a per-subject protocol over several seeds and average per subject."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from eeg_motor_imagery_classification.evaluation.metrics import FoldMetrics
from eeg_motor_imagery_classification.evaluation.protocols import SubjectResult, summarize_subject_results
from eeg_motor_imagery_classification.train import TrainingConfig


def average_over_seeds(seed_runs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Average per-subject rows of several seed runs (same row labels) into one result.

    The returned ``summary`` aggregates subjects (n = number of subjects). Per-seed results are
    kept under ``seed_runs`` and the between-seed SD of each subject under ``per_subject_seed_sd``.
    """

    if not seed_runs:
        raise ValueError("At least one seed run is required.")
    labels = [str(row["label"]) for row in next(iter(seed_runs.values()))["rows"]]
    averaged: list[SubjectResult] = []
    seed_sd: dict[str, float] = {}
    for label in labels:
        rows = [next(row for row in run["rows"] if str(row["label"]) == label) for run in seed_runs.values()]
        accuracy = np.asarray([float(row["accuracy"]) for row in rows])
        averaged.append(
            SubjectResult(
                label=label,
                metrics=FoldMetrics(
                    accuracy=float(accuracy.mean()),
                    balanced_accuracy=float(np.mean([float(row["balanced_accuracy"]) for row in rows])),
                    macro_f1=float(np.mean([float(row["macro_f1"]) for row in rows])),
                    confusion_matrix=np.sum([np.asarray(row["confusion_matrix"]) for row in rows], axis=0),
                ),
            )
        )
        seed_sd[label] = float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0

    combined = summarize_subject_results(averaged)
    combined["seeds"] = [int(key.split("_", 1)[1]) for key in seed_runs]
    combined["per_subject_seed_sd"] = seed_sd
    combined["seed_runs"] = seed_runs
    combined["runtime_seconds"] = float(sum(float(run.get("runtime_seconds", 0.0)) for run in seed_runs.values()))
    return combined


def run_over_training_seeds(
    run_fn: Callable[..., dict[str, object]],
    *args,
    seeds: tuple[int, ...],
    training_config: TrainingConfig | None = None,
    **kwargs,
) -> dict[str, object]:
    """Run ``run_fn`` once per training seed and average the per-subject rows over seeds.

    Only the seed in the training configuration changes between repeats; evaluation splits
    (CV folds, held-out subjects, sessions) stay fixed, so the seed-averaged rows remain paired
    with the deterministic classical baselines.
    """

    if not seeds:
        raise ValueError("At least one seed is required.")
    base_cfg = training_config or TrainingConfig()
    seed_runs: dict[str, dict[str, object]] = {}
    histories: list[dict[str, object]] = []
    for seed in seeds:
        result = run_fn(*args, training_config=replace(base_cfg, seed=seed), **kwargs)
        for history in result.pop("training_histories", []) or []:
            histories.append({**history, "label": f"seed_{seed}:{history.get('label', '')}"})
        seed_runs[f"seed_{seed}"] = result

    combined = average_over_seeds(seed_runs)
    combined["training_histories"] = histories
    return combined
