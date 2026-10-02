"""Figure and summary-table exports for report-friendly assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eeg_motor_imagery_classification.evaluation.protocols import summarize_transfer_across_targets
from eeg_motor_imagery_classification.utils import ensure_directory, write_text

# Saved-result locations, in priority order. The first existing file wins, so writing a new
# run to the first (canonical) name always takes precedence over older variants.
RESULT_SOURCES: dict[str, tuple[str, ...]] = {
    "within_classical": ("within_subject_classical", "repro_classical_within"),
    "within_riemann": ("within_subject_riemann", "repro_riemann_within"),
    "within_eegnet": ("within_subject_eegnet", "within_subject_eegnet_es50", "within_subject_eegnet_e30", "repro_eegnet_within"),
    "loso_classical": ("loso_classical", "repro_classical_loso"),
    "loso_riemann": ("loso_riemann", "repro_riemann_loso"),
    "loso_eegnet": ("loso_eegnet", "loso_eegnet_es50", "loso_eegnet_e30", "repro_eegnet_loso"),
    "cross_session_classical": ("cross_session_classical",),
    "cross_session_riemann": ("cross_session_riemann",),
    "cross_session_eegnet": ("cross_session_eegnet",),
    "loso_matched_classical": ("loso_matched_classical",),
    "loso_matched_riemann": ("loso_matched_riemann",),
    "loso_matched_eegnet": ("loso_matched_eegnet",),
    "transfer_fbcsp": (
        "transfer_classical_all_targets_seed42_43",
        "transfer_classical_all_targets_seed42_43_v2",
        "repro_transfer_classical_seed42_43",
    ),
    "transfer_riemann": (
        "transfer_riemann_all_targets_seed42_43_v2",
        "transfer_riemann_all_targets_seed42_43",
        "repro_transfer_riemann_seed42_43",
    ),
    "transfer_eegnet": (
        "transfer_eegnet_all_targets_seed42_43",
        "transfer_eegnet_all_targets_seed42_43_es50",
        "transfer_eegnet_all_targets_seed42_43_e30",
        "repro_transfer_eegnet_seed42_43",
    ),
}

SHOT_LABELS = ["zero_shot", "5_shot", "10_shot", "20_shot", "30_shot"]


def _load_json_if_exists(path: str | Path) -> dict[str, object] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def load_saved_results(outputs_dir: str | Path) -> dict[str, tuple[dict[str, object], Path]]:
    """Load every available saved result as {key: (payload, source path)}."""

    outputs = Path(outputs_dir)
    loaded: dict[str, tuple[dict[str, object], Path]] = {}
    for key, candidates in RESULT_SOURCES.items():
        for candidate in candidates:
            path = outputs / candidate / "result.json"
            payload = _load_json_if_exists(path)
            if payload is not None:
                loaded[key] = (payload, path)
                break
    return loaded


def split_model_results(loaded: dict[str, tuple[dict[str, object], Path]], protocol: str) -> dict[str, dict[str, object]]:
    """Return {display model name: result} for "within", "loso", "cross_session", or "loso_matched"."""

    models: dict[str, dict[str, object]] = {}
    classical = loaded.get(f"{protocol}_classical")
    if classical is not None:
        for key, name in (("raw_power", "Raw Power + LDA"), ("csp", "CSP + LDA"), ("fbcsp", "FBCSP + LDA")):
            if key in classical[0]:
                models[name] = classical[0][key]
    if f"{protocol}_riemann" in loaded:
        models["Riemann + Tangent Space + LDA"] = loaded[f"{protocol}_riemann"][0]
    if f"{protocol}_eegnet" in loaded:
        models["EEGNet"] = loaded[f"{protocol}_eegnet"][0]
    return models


def subject_accuracies(result: dict[str, object]) -> dict[str, float]:
    """Per-subject accuracies {label: accuracy} from a saved result (empty if absent)."""

    rows = result.get("rows")
    if not isinstance(rows, list):
        return {}
    return {str(row["label"]): float(row["accuracy"]) for row in rows}


def describe(values: list[float] | np.ndarray) -> dict[str, float]:
    """Mean, sample std (ddof=1), and t-based 95% CI of independent units."""

    from scipy import stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def transfer_setting_table(result: dict[str, object]) -> dict[str, dict[str, object]]:
    """{setting: {"stats": describe(...), "per_target": {...}, "confusion": ...}} across targets."""

    table: dict[str, dict[str, object]] = {}
    if isinstance(result.get("targets"), dict) and result["targets"]:
        for setting, summary in summarize_transfer_across_targets(result).items():
            per_target = {str(row["label"]): float(row["accuracy"]) for row in summary["rows"]}
            table[setting] = {
                "stats": describe(list(per_target.values())),
                "per_target": per_target,
                "confusion": np.asarray(summary["summary"]["confusion_matrix_sum"]),
            }
        return table
    # Legacy fallback: only per-setting means are available (no between-target spread).
    aggregate = result.get("aggregate_by_setting", {})
    for row in aggregate.get("rows", []) if isinstance(aggregate, dict) else []:
        table[str(row["label"])] = {
            "stats": {"n": 1, "mean": float(row["accuracy"]), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
            "per_target": {},
            "confusion": None,
        }
    return table


def _rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def _save_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers), *[",".join(row) for row in rows]]
    write_text(path, "\n".join(lines) + "\n")


def _save_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    write_text(path, _rows_to_markdown(headers, rows))


def _save_bar_chart(
    path: Path,
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str = "Accuracy",
    color: str = "#2f5d50",
    errors: list[float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(labels))
    yerr = None if errors is None else np.nan_to_num(np.asarray(errors, dtype=float))
    bars = ax.bar(positions, values, color=color, width=0.65, yerr=yerr, capsize=4, error_kw={"elinewidth": 1.1})
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    offsets = yerr if yerr is not None else np.zeros(len(values))
    for bar, value, offset in zip(bars, values, offsets, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + offset + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    if errors is not None:
        ax.text(0.99, 0.02, "error bars: SD across subjects", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_transfer_curve(
    path: Path,
    shot_labels: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    intervals: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.25))
    palette = {
        "FBCSP": "#7a8b99",
        "Riemann": "#2f5d50",
        "EEGNet": "#c26d3a",
    }
    x = np.arange(len(shot_labels))
    for model_name, values in series.items():
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=model_name,
            color=palette.get(model_name, "#444444"),
        )
        if intervals and model_name in intervals:
            low, high = intervals[model_name]
            ax.fill_between(x, low, high, color=palette.get(model_name, "#444444"), alpha=0.15, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(shot_labels)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    *,
    title: str,
    labels: tuple[str, str] = ("Left", "Right"),
) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}\n({int(matrix[i, j])})",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.55 else "#1f2933",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_learning_curve(path: Path, histories: list[dict[str, object]], *, title: str) -> None:
    import matplotlib.pyplot as plt

    if not histories:
        raise ValueError("At least one training history is required.")

    max_epochs = max(int(history.get("epochs_ran", 0)) for history in histories)
    if max_epochs <= 0:
        raise ValueError("Training histories must contain at least one epoch.")

    def stack_metric(key: str) -> np.ndarray:
        stacked = np.full((len(histories), max_epochs), np.nan, dtype=float)
        for row_idx, history in enumerate(histories):
            values = history.get(key, [])
            if not isinstance(values, list):
                raise ValueError(f"Training history field '{key}' must be a list.")
            if values:
                stacked[row_idx, : len(values)] = np.asarray(values, dtype=float)
        return stacked

    train = stack_metric("train_loss")
    val = stack_metric("val_loss")
    epochs = np.arange(1, max_epochs + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, np.nanmean(train, axis=0), color="#c26d3a", linewidth=2.2, label="Train loss")
    if np.isfinite(val).any():
        ax.plot(epochs, np.nanmean(val, axis=0), color="#2f5d50", linewidth=2.2, label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pipeline_figure(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str = "#24313f", fs: int = 11) -> tuple[float, float, float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.4,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#13202b")
        return (x, y, w, h)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#41566b",
            )
        )

    def poly_arrow(points: list[tuple[float, float]]) -> None:
        for idx in range(len(points) - 2):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            ax.plot([x1, x2], [y1, y2], color="#41566b", linewidth=1.5)
        arrow(*points[-2], *points[-1])

    dataset = box(0.05, 0.76, 0.16, 0.13, "BNCI2014_001\nLeft vs Right MI", fc="#dceef8")
    preproc = box(0.27, 0.76, 0.22, 0.13, "Canonical Preprocessing\n50 Hz notch\n8-32 Hz band-pass\n0.0-4.0 s epochs", fc="#e9f4df")
    arrays = box(0.55, 0.76, 0.16, 0.13, "Canonical Arrays\nX, y, groups", fc="#f8efd8")
    metrics = box(0.77, 0.76, 0.17, 0.13, "Evaluation Metrics\nAccuracy\nBalanced Accuracy\nMacro F1", fc="#f3e3ef")

    classical = box(0.08, 0.50, 0.20, 0.14, "Classical Branch\nRaw Power + LDA\nCSP + LDA\nFBCSP + LDA", fc="#eef3f7")
    geometric = box(0.38, 0.50, 0.20, 0.14, "Geometric Branch\nCovariance\nTangent Space\nLDA", fc="#e4f1eb")
    deep = box(0.68, 0.50, 0.18, 0.14, "Deep Branch\nEEGNet", fc="#f7e9dd")

    shared = box(0.34, 0.255, 0.28, 0.125, "Shared Evaluation Regimes\nWithin / LOSO: all models\nTransfer: FBCSP, Riemann, EEGNet", fc="#eef4fb", fs=10)
    within = box(0.72, 0.30, 0.22, 0.085, "Within-Subject CV", fc="#ecf5fb")
    loso = box(0.72, 0.185, 0.22, 0.085, "LOSO", fc="#ecf5fb")
    transfer = box(0.72, 0.04, 0.22, 0.115, "Cross-Subject Transfer\nzero/few-shot\nrepeated seeds", fc="#ecf5fb", fs=10)

    def right_mid(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = b
        return (x + w, y + h / 2)

    def left_mid(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, _, h = b
        return (x, y + h / 2)

    def top_mid(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = b
        return (x + w / 2, y + h)

    def bottom_mid(b: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, _ = b
        return (x + w / 2, y)

    arrow(*right_mid(dataset), *left_mid(preproc))
    arrow(*right_mid(preproc), *left_mid(arrays))
    arrow(*right_mid(arrays), *left_mid(metrics))

    branch_bus_y = 0.68
    array_bottom = bottom_mid(arrays)
    for target in (classical, geometric, deep):
        tx, ty = top_mid(target)
        poly_arrow([array_bottom, (array_bottom[0], branch_bus_y), (tx, branch_bus_y), (tx, ty)])

    merge_trunk_x = top_mid(shared)[0]
    merge_bus_y = 0.44
    shared_top = top_mid(shared)
    for source in (classical, geometric, deep):
        sx, sy = bottom_mid(source)
        ax.plot([sx, sx], [sy, merge_bus_y], color="#41566b", linewidth=1.5)
    left_merge_x = bottom_mid(classical)[0]
    right_merge_x = bottom_mid(deep)[0]
    ax.plot([left_merge_x, right_merge_x], [merge_bus_y, merge_bus_y], color="#41566b", linewidth=1.5)
    poly_arrow([(merge_trunk_x, merge_bus_y), (merge_trunk_x, shared_top[1]), shared_top])

    shared_right = right_mid(shared)
    branch_x = 0.69
    within_y = left_mid(within)[1]
    loso_y = left_mid(loso)[1]
    transfer_y = left_mid(transfer)[1]
    # Shared -> protocols: use a Z-shaped lead-in, then branch from the same x-position.
    poly_arrow([shared_right, (branch_x, shared_right[1]), (branch_x, within_y), (left_mid(within)[0], within_y)])
    ax.plot([branch_x, branch_x], [within_y, transfer_y], color="#41566b", linewidth=1.5)
    for target_box in (loso, transfer):
        tx, ty = left_mid(target_box)
        arrow(branch_x, ty, tx, ty)

    ax.text(0.5, 0.94, "EEG Motor Imagery Evaluation Pipeline", ha="center", va="center", fontsize=15, weight="bold", color="#13202b")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_grid(path: Path, matrices: dict[str, np.ndarray], *, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = ("Left", "Right")
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.1 * len(matrices), 3.4), squeeze=False)
    for ax, (setting, matrix) in zip(axes[0], matrices.items(), strict=True):
        matrix = np.asarray(matrix, dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{normalized[i, j]:.2f}\n({int(matrix[i, j])})", ha="center", va="center",
                        color="white" if normalized[i, j] > 0.55 else "#1f2933", fontsize=8)
        ax.set_title(setting, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def _export_protocol(
    assets: Path,
    generated: dict[str, object],
    models: dict[str, dict[str, object]],
    *,
    prefix: str,
    title: str,
    color: str,
) -> None:
    if not models:
        return
    headers = ["Model", "Subjects", "Accuracy Mean", "Accuracy SD", "Accuracy CI95 Low", "Accuracy CI95 High"]
    rows: list[list[str]] = []
    means: list[float] = []
    errors: list[float] = []
    for name, result in models.items():
        per_subject = subject_accuracies(result)
        if per_subject:
            described = describe(list(per_subject.values()))
        else:  # legacy result without per-subject rows
            summary = result.get("summary", {})
            described = {"n": int(summary.get("n", 0)), "mean": float(summary["accuracy_mean"]),
                         "std": float(summary.get("accuracy_std", float("nan"))), "ci_low": float("nan"), "ci_high": float("nan")}
        rows.append([name, str(described["n"]), _fmt(described["mean"]), _fmt(described["std"]),
                     _fmt(described["ci_low"]), _fmt(described["ci_high"])])
        means.append(described["mean"])
        errors.append(described["std"])
    _save_csv(assets / f"{prefix}_summary.csv", headers, rows)
    _save_markdown_table(assets / f"{prefix}_summary.md", headers, rows)
    _save_bar_chart(assets / f"{prefix}_accuracy.png", list(models), means, title=title, color=color, errors=errors)
    generated[f"{prefix}_table"] = str(assets / f"{prefix}_summary.md")
    generated[f"{prefix}_figure"] = str(assets / f"{prefix}_accuracy.png")


def _export_report_assets_impl(root: Path, assets: Path) -> dict[str, object]:
    generated: dict[str, object] = {}
    _save_pipeline_figure(assets / "evaluation_pipeline.png")
    generated["pipeline_figure"] = str(assets / "evaluation_pipeline.png")

    loaded = load_saved_results(root / "outputs")
    generated["sources"] = {key: str(path) for key, (_payload, path) in loaded.items()}
    generated["missing_sources"] = [key for key in RESULT_SOURCES if key not in loaded]

    within_models = split_model_results(loaded, "within")
    loso_models = split_model_results(loaded, "loso")
    cross_session_models = split_model_results(loaded, "cross_session")
    _export_protocol(assets, generated, within_models, prefix="within_subject", title="Within-Subject CV Accuracy (sessions pooled)", color="#2f5d50")
    _export_protocol(assets, generated, loso_models, prefix="loso", title="LOSO Accuracy", color="#385f8c")
    _export_protocol(assets, generated, cross_session_models, prefix="cross_session", title="Cross-Session Accuracy (session 1 -> 2)", color="#6b4c8a")
    _export_protocol(assets, generated, split_model_results(loaded, "loso_matched"), prefix="loso_matched",
                     title="LOSO, 144 Training Trials (tested on session 2)", color="#8a6b2f")

    for protocol, prefix, label in (("within", "within_subject", "Within-Subject"), ("loso", "loso", "LOSO"), ("cross_session", "cross_session", "Cross-Session")):
        eegnet = loaded.get(f"{protocol}_eegnet")
        histories = eegnet[0].get("training_histories") if eegnet else None
        if isinstance(histories, list) and histories:
            _save_learning_curve(assets / f"{prefix}_eegnet_learning_curve.png", histories, title=f"{label} EEGNet Learning Curve")
            generated[f"{prefix}_eegnet_learning_curve"] = str(assets / f"{prefix}_eegnet_learning_curve.png")

    fbcsp_within = within_models.get("FBCSP + LDA", {}).get("summary", {})
    if "confusion_matrix_sum" in fbcsp_within:
        _save_confusion_matrix(assets / "within_subject_fbcsp_confusion.png", np.asarray(fbcsp_within["confusion_matrix_sum"]),
                               title="Within-Subject FBCSP Confusion Matrix")
        generated["within_subject_confusion"] = str(assets / "within_subject_fbcsp_confusion.png")
    eegnet_loso = loso_models.get("EEGNet", {}).get("summary", {})
    if "confusion_matrix_sum" in eegnet_loso:
        _save_confusion_matrix(assets / "loso_eegnet_confusion.png", np.asarray(eegnet_loso["confusion_matrix_sum"]),
                               title="LOSO EEGNet Confusion Matrix")
        generated["loso_confusion"] = str(assets / "loso_eegnet_confusion.png")

    # Transfer: statistics per shot setting with the target subject as the unit
    # (repeated seeds are averaged within each target before aggregation).
    transfer_models = {name: loaded[key][0] for name, key in (("FBCSP", "transfer_fbcsp"), ("Riemann", "transfer_riemann"), ("EEGNet", "transfer_eegnet")) if key in loaded}
    transfer_tables = {name: transfer_setting_table(result) for name, result in transfer_models.items()}
    if transfer_tables:
        headers = ["Model", "Setting", "Targets", "Accuracy Mean", "Accuracy SD", "Accuracy CI95 Low", "Accuracy CI95 High"]
        rows = []
        for name, table in transfer_tables.items():
            for setting in SHOT_LABELS:
                if setting in table:
                    st = table[setting]["stats"]
                    rows.append([name, setting, str(st["n"]), _fmt(st["mean"]), _fmt(st["std"]), _fmt(st["ci_low"]), _fmt(st["ci_high"])])
        _save_csv(assets / "transfer_repeated_summary.csv", headers, rows)
        _save_markdown_table(assets / "transfer_repeated_summary.md", headers, rows)
        generated["transfer_table"] = str(assets / "transfer_repeated_summary.md")

        series = {name: [table[s]["stats"]["mean"] for s in SHOT_LABELS] for name, table in transfer_tables.items() if all(s in table for s in SHOT_LABELS)}
        intervals = {
            name: ([table[s]["stats"]["ci_low"] for s in SHOT_LABELS], [table[s]["stats"]["ci_high"] for s in SHOT_LABELS])
            for name, table in transfer_tables.items()
            if all(s in table and np.isfinite(table[s]["stats"]["ci_low"]) for s in SHOT_LABELS)
        }
        if series:
            _save_transfer_curve(assets / "transfer_repeated_accuracy.png", SHOT_LABELS, series,
                                 title="Transfer Accuracy (mean across targets, 95% CI band)", intervals=intervals)
            generated["transfer_figure"] = str(assets / "transfer_repeated_accuracy.png")

        for name, key in (("Riemann", "transfer_riemann_confusion"), ("EEGNet", "transfer_confusion")):
            if name not in transfer_tables:
                continue
            path = assets / f"transfer_repeated_{name.lower()}_confusion.png"
            per_setting = {s: transfer_tables[name][s]["confusion"] for s in SHOT_LABELS
                           if s in transfer_tables[name] and transfer_tables[name][s]["confusion"] is not None}
            if per_setting:
                _save_confusion_grid(path, per_setting, title=f"{name} Transfer Confusion Matrices by Setting")
            else:
                pooled = transfer_models[name]["aggregate_by_setting"]["summary"]["confusion_matrix_sum"]
                _save_confusion_matrix(path, np.asarray(pooled), title=f"{name} Transfer Confusion (all settings pooled)")
            generated[key] = str(path)

    return generated


def export_report_assets(
    *,
    project_root: str | Path,
    assets_dir: str | Path,
) -> dict[str, object]:
    """Export report-ready tables and figures from saved experiment outputs."""

    import matplotlib.pyplot as plt

    # Use matplotlib defaults so figures do not depend on the caller's rcParams/style.
    with plt.style.context("default"):
        return _export_report_assets_impl(Path(project_root), ensure_directory(assets_dir))
