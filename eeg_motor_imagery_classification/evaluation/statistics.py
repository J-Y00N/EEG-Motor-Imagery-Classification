"""Simple statistical helpers for repeated transfer comparisons."""

from __future__ import annotations

import numpy as np


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 10000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estimate a two-sided paired sign-flip permutation p-value."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired samples must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional.")

    diffs = x - y
    observed = float(np.mean(diffs))
    if len(diffs) == 0:
        raise ValueError("At least one paired observation is required.")

    rng = np.random.default_rng(random_state)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, len(diffs)))
    null_distribution = (signs * diffs).mean(axis=1)
    p_value = float((np.sum(np.abs(null_distribution) >= abs(observed)) + 1) / (n_resamples + 1))

    return {
        "n_pairs": int(len(diffs)),
        "mean_difference": observed,
        "abs_mean_difference": float(abs(observed)),
        "p_value": p_value,
        "wins_x": int(np.sum(diffs > 0.0)),
        "wins_y": int(np.sum(diffs < 0.0)),
        "ties": int(np.sum(np.isclose(diffs, 0.0))),
    }


def compare_paired_result_rows(
    result_a: dict[str, object],
    result_b: dict[str, object],
    *,
    metric: str = "accuracy",
    n_resamples: int = 10000,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Compare two result dictionaries that share the same row labels."""

    rows_a = result_a.get("rows")
    rows_b = result_b.get("rows")
    if not isinstance(rows_a, list) or not isinstance(rows_b, list):
        raise ValueError("Both results must contain 'rows' lists.")

    map_a = {str(row["label"]): float(row[metric]) for row in rows_a}
    map_b = {str(row["label"]): float(row[metric]) for row in rows_b}
    labels = tuple(sorted(set(map_a) & set(map_b)))
    if not labels:
        raise ValueError("No shared row labels were found for paired comparison.")

    x = np.asarray([map_a[label] for label in labels], dtype=float)
    y = np.asarray([map_b[label] for label in labels], dtype=float)
    stats = paired_permutation_test(x, y, n_resamples=n_resamples, random_state=random_state)
    return {"metric": metric, "labels": labels, **stats}
