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

# Pre-specified confirmatory comparisons (docs/analysis_plan.md). They are tested as one family:
# exact paired sign-flip tests on subject-level accuracy, Holm-adjusted across these tests only.
PRIMARY_COMPARISONS: list[tuple[str, str, str]] = [
    ("cross_session", "EEGNet", "FBCSP + LDA"),
    ("cross_session", "Riemann + Tangent Space + LDA", "FBCSP + LDA"),
    ("loso", "EEGNet", "Riemann + Tangent Space + LDA"),
]
# Exploratory: does a model difference change between protocols? For each subject the paired
# difference (A - B) under the first protocol is compared with the same difference under the second.
INTERACTION_COMPARISONS: list[tuple[str, str, str, str]] = [
    ("EEGNet", "Riemann + Tangent Space + LDA", "loso", "cross_session"),
    ("EEGNet", "FBCSP + LDA", "loso", "cross_session"),
]
PROTOCOL_TITLES = {"within": "Within-Subject CV (sessions pooled)", "loso": "LOSO", "cross_session": "Cross-Session (session 1 -> 2)"}


def _primary_statistics(loaded: dict[str, tuple[dict[str, object], Path]]) -> list[dict[str, object]]:
    scores: dict[str, dict[str, float]] = {}
    pairs: list[tuple[str, str]] = []
    for protocol, model_a, model_b in PRIMARY_COMPARISONS:
        models = split_model_results(loaded, protocol)
        if model_a not in models or model_b not in models:
            continue
        key_a, key_b = f"{model_a}@{protocol}", f"{model_b}@{protocol}"
        scores[key_a] = subject_accuracies(models[model_a])
        scores[key_b] = subject_accuracies(models[model_b])
        pairs.append((key_a, key_b))
    comparisons = compare_models_pairwise(scores, pairs)
    for item in comparisons:
        item["setting"] = PROTOCOL_TITLES[str(item["model_a"]).split("@")[1]]
        item["model_a"] = str(item["model_a"]).split("@")[0]
        item["model_b"] = str(item["model_b"]).split("@")[0]
    return comparisons


def _interaction_statistics(loaded: dict[str, tuple[dict[str, object], Path]]) -> list[dict[str, object]]:
    scores: dict[str, dict[str, float]] = {}
    pairs: list[tuple[str, str]] = []
    for model_a, model_b, first, second in INTERACTION_COMPARISONS:
        diffs = {}
        for protocol in (first, second):
            models = split_model_results(loaded, protocol)
            if model_a not in models or model_b not in models:
                break
            a, b = subject_accuracies(models[model_a]), subject_accuracies(models[model_b])
            diffs[protocol] = {unit: a[unit] - b[unit] for unit in sorted(set(a) & set(b))}
        else:
            key = f"{model_a} - {model_b}"
            scores[f"{key}@{first}"] = diffs[first]
            scores[f"{key}@{second}"] = diffs[second]
            pairs.append((f"{key}@{first}", f"{key}@{second}"))
    comparisons = compare_models_pairwise(scores, pairs)
    for item in comparisons:
        item["setting"] = str(item["model_a"]).split("@")[0]
        item["model_a"] = PROTOCOL_TITLES[str(item["model_a"]).split("@")[1]]
        item["model_b"] = PROTOCOL_TITLES[str(item["model_b"]).split("@")[1]]
    return comparisons


def _environment_lines(loaded: dict[str, tuple[dict[str, object], Path]]) -> list[str]:
    # The torch device only applies to EEGNet; classical and Riemannian pipelines always run on the CPU.
    lines = ["| Result | Python | torch | Torch device (EEGNet only) | CUDA device | Platform |", "|---|---|---|---|---|---|"]
    for key, (payload, _path) in loaded.items():
        env = payload.get("environment")
        if not isinstance(env, dict):
            lines.append(f"| {key} | not recorded | | | | |")
            continue
        packages = env.get("packages", {})
        device = (payload.get("config") or {}).get("device", env.get("torch_device_requested", "?"))
        lines.append(f"| {key} | {env.get('python')} | {packages.get('torch')} | {device} | {env.get('cuda_device_name') or '-'} | {env.get('platform')} |")
    return lines


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


def _per_subject_lines(per_subject: dict[str, dict[str, float]]) -> list[str]:
    models = list(per_subject)
    labels = sorted({label for scores in per_subject.values() for label in scores}, key=lambda x: (len(x), x))
    lines = ["| Subject | " + " | ".join(models) + " |", "|---|" + "|".join(["---:"] * len(models)) + "|"]
    for label in labels:
        cells = [f"{per_subject[m][label]:.4f}" if label in per_subject[m] else "-" for m in models]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _runtime_lines(loaded: dict[str, tuple[dict[str, object], Path]]) -> list[str]:
    lines = ["| Result | Model | Runtime (s) |", "|---|---|---:|"]
    for key, (payload, _path) in loaded.items():
        if key.endswith("_classical"):
            for name in ("raw_power", "csp", "fbcsp"):
                runtime = payload.get(name, {}).get("runtime_seconds") if isinstance(payload.get(name), dict) else None
                if runtime is not None:
                    lines.append(f"| {key} | {name} | {float(runtime):.1f} |")
        elif payload.get("runtime_seconds") is not None:
            lines.append(f"| {key} | - | {float(payload['runtime_seconds']):.1f} |")
    return lines


def _comparison_lines(comparisons: list[dict[str, object]], *, with_setting: bool = False) -> list[str]:
    head = "| Setting | A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |" if with_setting else \
        "| A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |"
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for item in comparisons:
        cells = [str(item["model_a"]), str(item["model_b"]), str(item["n_pairs"]), f"{item['mean_difference']:+.4f}",
                 f"[{item.get('diff_ci95_low', float('nan')):+.4f}, {item.get('diff_ci95_high', float('nan')):+.4f}]",
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

    primary = _primary_statistics(loaded)
    result["primary"] = primary
    lines += ["## Primary (pre-specified) comparisons", "",
              f"Family of {len(PRIMARY_COMPARISONS)} tests defined in docs/analysis_plan.md; Holm adjustment across this family only. "
              f"{len(primary)} of {len(PRIMARY_COMPARISONS)} comparisons have results available.", "",
              *_comparison_lines(primary, with_setting=True), ""]

    for protocol, title in PROTOCOL_TITLES.items():
        models = split_model_results(loaded, protocol)
        if not models:
            continue
        stats = _protocol_statistics(models)
        result[protocol] = stats
        lines += [f"## {title}", "", *_descriptive_lines(stats["descriptives"]), "", *_comparison_lines(stats["comparisons"]), "",
                  "Per-subject accuracy:", "", *_per_subject_lines(stats["per_subject"]), ""]
        seed_sd = models.get("EEGNet", {}).get("per_subject_seed_sd")
        if isinstance(seed_sd, dict) and seed_sd:
            seeds = ", ".join(str(seed) for seed in models["EEGNet"].get("seeds", []))
            lines += [f"EEGNet accuracy is averaged over training seeds {seeds}; between-seed SD per subject:", "",
                      "| " + " | ".join(seed_sd) + " |", "|" + "|".join(["---:"] * len(seed_sd)) + "|",
                      "| " + " | ".join(f"{value:.4f}" for value in seed_sd.values()) + " |", ""]

    transfer_models = {name: loaded[key][0] for name, key in (("FBCSP", "transfer_fbcsp"), ("Riemann", "transfer_riemann"), ("EEGNet", "transfer_eegnet")) if key in loaded}
    if transfer_models:
        stats = _transfer_statistics(transfer_models)
        result["transfer"] = stats
        lines += ["## Transfer (per shot setting, across target subjects)", ""]
        for name, per_setting in stats["descriptives"].items():
            lines += [f"### {name}", "", *_descriptive_lines(per_setting), ""]
        lines += ["### Paired comparisons", "", *_comparison_lines(stats["comparisons"], with_setting=True), ""]

    interaction = _interaction_statistics(loaded)
    if interaction:
        result["interaction"] = interaction
        lines += ["## Exploratory: model difference between protocols", "",
                  "Per subject, the accuracy difference of the listed model pair under protocol A is compared with the same "
                  "difference under protocol B (exact paired sign-flip test, Holm across these tests). Not pre-specified.", "",
                  *_comparison_lines(interaction, with_setting=True), ""]

    lines += ["## Runtime (wall-clock fit + predict, machine-specific)", "", *_runtime_lines(loaded), ""]
    lines += ["## Environment", "", *_environment_lines(loaded), ""]
    lines += ["## Sources", "", *[f"- `{key}`: `{path}`" for key, path in result["sources"].items()], ""]
    write_json(out / "statistics.json", result)
    write_text(out / "statistics.md", "\n".join(lines))
    return {"statistics_json": str(out / "statistics.json"), "statistics_md": str(out / "statistics.md"), "sources": result["sources"]}
