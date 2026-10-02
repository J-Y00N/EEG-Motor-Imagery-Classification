"""Paired statistical comparisons between models evaluated on the same subjects."""

from __future__ import annotations

import itertools

import numpy as np

# Exact sign-flip enumeration is used up to this many pairs (2**16 = 65536 sign vectors).
MAX_EXACT_PAIRS = 16


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 10000,
    random_state: int = 42,
    exact: bool | None = None,
) -> dict[str, float | int | bool]:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Each pair must be an independent unit (e.g. one target subject). Repeated seeds of the
    same subject should be averaged before calling this function; otherwise the p-value is
    deflated by pseudo-replication. With ``exact=None`` all ``2**n`` sign vectors are
    enumerated when ``n <= MAX_EXACT_PAIRS``, otherwise a Monte Carlo estimate is used.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired samples must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional.")
    if len(x) == 0:
        raise ValueError("At least one paired observation is required.")

    diffs = x - y
    n = len(diffs)
    observed = float(np.mean(diffs))
    use_exact = n <= MAX_EXACT_PAIRS if exact is None else exact

    if use_exact:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
        null_distribution = (signs * diffs).mean(axis=1)
        p_value = float(np.mean(np.abs(null_distribution) >= abs(observed) - 1e-12))
    else:
        rng = np.random.default_rng(random_state)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
        null_distribution = (signs * diffs).mean(axis=1)
        p_value = float((np.sum(np.abs(null_distribution) >= abs(observed) - 1e-12) + 1) / (n_resamples + 1))

    return {
        "n_pairs": int(n),
        "mean_difference": observed,
        "abs_mean_difference": float(abs(observed)),
        "p_value": p_value,
        "exact": bool(use_exact),
        "wins_x": int(np.sum(diffs > 0.0)),
        "wins_y": int(np.sum(diffs < 0.0)),
        "ties": int(np.sum(np.isclose(diffs, 0.0))),
    }


def paired_difference_interval(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """t-distribution confidence interval of the mean paired difference x - y."""

    from scipy import stats

    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * diffs.std(ddof=1) / np.sqrt(n))
    return float(diffs.mean() - half), float(diffs.mean() + half)


def holm_correction(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""

    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


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


def compare_models_pairwise(
    per_unit_scores: dict[str, dict[str, float]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Paired tests for model pairs over shared units, Holm-adjusted within this family.

    ``per_unit_scores`` maps model name -> {unit label (e.g. subject) -> score}.
    """

    comparisons: list[dict[str, object]] = []
    for model_a, model_b in pairs:
        if model_a not in per_unit_scores or model_b not in per_unit_scores:
            continue
        labels = sorted(set(per_unit_scores[model_a]) & set(per_unit_scores[model_b]))
        if not labels:
            continue
        x = np.asarray([per_unit_scores[model_a][label] for label in labels], dtype=float)
        y = np.asarray([per_unit_scores[model_b][label] for label in labels], dtype=float)
        ci_low, ci_high = paired_difference_interval(x, y)
        comparisons.append({"model_a": model_a, "model_b": model_b, "units": labels, **paired_permutation_test(x, y),
                            "diff_ci95_low": ci_low, "diff_ci95_high": ci_high})

    adjusted = holm_correction([item["p_value"] for item in comparisons])
    for item, p_holm in zip(comparisons, adjusted, strict=True):
        item["p_holm"] = float(p_holm)
    return comparisons
