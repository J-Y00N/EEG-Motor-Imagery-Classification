"""Statistical model comparisons computed from saved experiment outputs.

Units of analysis are subjects: per-subject scores for within-subject CV and LOSO, and
per-target scores (seeds averaged within each target) for transfer. Paired sign-flip tests
are exact for up to 16 units, and p-values are Holm-adjusted within each protocol family.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from eeg_motor_imagery_classification.evaluation.statistics import compare_models_pairwise
from eeg_motor_imagery_classification.figures import (
    SHOT_LABELS,
    describe,
    load_saved_results,
    split_model_results,
    subject_accuracies,
    transfer_setting_table,
)
from eeg_motor_imagery_classification.utils import ensure_directory, write_json, write_text

TRANSFER_PAIRS = [("EEGNet", "Riemann"), ("Riemann", "FBCSP"), ("EEGNet", "FBCSP")]


def _protocol_statistics(models: dict[str, dict[str, object]]) -> dict[str, object]:
    scores = {name: subject_accuracies(result) for name, result in models.items()}
    scores = {name: values for name, values in scores.items() if values}
    descriptives = {name: describe(list(values.values())) for name, values in scores.items()}
    comparisons = compare_models_pairwise(scores, list(combinations(scores, 2)))
    return {"descriptives": descriptives, "comparisons": comparisons, "per_subject": scores}


def _transfer_statistics(transfer_models: dict[str, dict[str, object]]) -> dict[str, object]:
    tables = {name: transfer_setting_table(result) for name, result in transfer_models.items()}
    descriptives = {
        name: {setting: table[setting]["stats"] for setting in SHOT_LABELS if setting in table}
        for name, table in tables.items()
    }
    flat_scores: dict[str, dict[str, float]] = {}
    flat_pairs: list[tuple[str, str]] = []
    for setting in SHOT_LABELS:
        for model_a, model_b in TRANSFER_PAIRS:
            if setting in tables.get(model_a, {}) and setting in tables.get(model_b, {}):
                key_a, key_b = f"{model_a}@{setting}", f"{model_b}@{setting}"
                flat_scores[key_a] = tables[model_a][setting]["per_target"]
                flat_scores[key_b] = tables[model_b][setting]["per_target"]
                flat_pairs.append((key_a, key_b))
    comparisons = compare_models_pairwise(flat_scores, flat_pairs)
    for item in comparisons:
        item["setting"] = str(item["model_a"]).split("@")[1]
        item["model_a"] = str(item["model_a"]).split("@")[0]
        item["model_b"] = str(item["model_b"]).split("@")[0]
    return {"descriptives": descriptives, "comparisons": comparisons}


def _descriptive_lines(descriptives: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| Label | n | Mean | SD | 95% CI |", "|---|---:|---:|---:|---:|"]
    for name, st in descriptives.items():
        lines.append(f"| {name} | {st['n']} | {st['mean']:.4f} | {st['std']:.4f} | [{st['ci_low']:.4f}, {st['ci_high']:.4f}] |")
    return lines


def _comparison_lines(comparisons: list[dict[str, object]], *, with_setting: bool = False) -> list[str]:
    head = "| Setting | A | B | n | Mean diff (A-B) | Wins A/B | p (exact) | p (Holm) |" if with_setting else \
        "| A | B | n | Mean diff (A-B) | Wins A/B | p (exact) | p (Holm) |"
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for item in comparisons:
        cells = [str(item["model_a"]), str(item["model_b"]), str(item["n_pairs"]), f"{item['mean_difference']:+.4f}",
                 f"{item['wins_x']}/{item['wins_y']}", f"{item['p_value']:.4f}", f"{item['p_holm']:.4f}"]
        if with_setting:
            cells.insert(0, str(item["setting"]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def export_statistics(*, project_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Compute descriptive statistics and paired model comparisons; write JSON and Markdown."""

    root = Path(project_root)
    out = ensure_directory(output_dir)
    loaded = load_saved_results(root / "outputs")

    result: dict[str, object] = {"sources": {key: str(path) for key, (_payload, path) in loaded.items()}}
    lines = ["# Statistical Comparisons", "",
             "Units are subjects (transfer: target subjects, seeds averaged within target). SD uses ddof=1, "
             "CIs use the t distribution, paired sign-flip tests are exact, and p-values are Holm-adjusted within each family.", ""]

    for protocol, title in (("within", "Within-Subject CV"), ("loso", "LOSO")):
        models = split_model_results(loaded, protocol)
        if not models:
            continue
        stats = _protocol_statistics(models)
        result[protocol] = stats
        lines += [f"## {title}", "", *_descriptive_lines(stats["descriptives"]), "", *_comparison_lines(stats["comparisons"]), ""]

    transfer_models = {name: loaded[key][0] for name, key in (("FBCSP", "transfer_fbcsp"), ("Riemann", "transfer_riemann"), ("EEGNet", "transfer_eegnet")) if key in loaded}
    if transfer_models:
        stats = _transfer_statistics(transfer_models)
        result["transfer"] = stats
        lines += ["## Transfer (per shot setting, across target subjects)", ""]
        for name, per_setting in stats["descriptives"].items():
            lines += [f"### {name}", "", *_descriptive_lines(per_setting), ""]
        lines += ["### Paired comparisons", "", *_comparison_lines(stats["comparisons"], with_setting=True), ""]

    lines += ["## Sources", "", *[f"- `{key}`: `{path}`" for key, path in result["sources"].items()], ""]
    write_json(out / "statistics.json", result)
    write_text(out / "statistics.md", "\n".join(lines))
    return {"statistics_json": str(out / "statistics.json"), "statistics_md": str(out / "statistics.md"), "sources": result["sources"]}
