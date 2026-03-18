"""Figure and summary-table exports for report-friendly assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eeg_motor_imagery_classification.utils import ensure_directory, write_text


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _load_json_if_exists(path: str | Path) -> dict[str, object] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text())


def _load_first_existing_json(*paths: str | Path) -> dict[str, object] | None:
    for path in paths:
        payload = _load_json_if_exists(path)
        if payload is not None:
            return payload
    return None


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
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, color=color, width=0.65)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_transfer_curve(
    path: Path,
    shot_labels: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
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

    shared = box(0.36, 0.27, 0.24, 0.10, "Shared Evaluation Regimes\nAll model families use all protocols", fc="#eef4fb", fs=10)
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


def export_report_assets(
    *,
    project_root: str | Path,
    assets_dir: str | Path,
) -> dict[str, str]:
    """Export report-ready tables and figures from saved experiment outputs."""

    root = Path(project_root)
    assets = ensure_directory(assets_dir)
    outputs = root / "outputs"

    generated: dict[str, str] = {}
    _save_pipeline_figure(assets / "evaluation_pipeline.png")
    generated["pipeline_figure"] = str(assets / "evaluation_pipeline.png")

    within_classical = _load_json_if_exists(outputs / "within_subject_classical" / "result.json")
    within_riemann = _load_json_if_exists(outputs / "within_subject_riemann" / "result.json")
    within_eegnet = _load_first_existing_json(
        outputs / "within_subject_eegnet_es50" / "result.json",
        outputs / "within_subject_eegnet_e30" / "result.json",
    )
    loso_classical = _load_json_if_exists(outputs / "loso_classical" / "result.json")
    loso_riemann = _load_json_if_exists(outputs / "loso_riemann" / "result.json")
    loso_eegnet = _load_first_existing_json(
        outputs / "loso_eegnet_es50" / "result.json",
        outputs / "loso_eegnet_e30" / "result.json",
    )
    transfer_repeated_classical = _load_json_if_exists(outputs / "transfer_classical_all_targets_seed42_43" / "result.json")
    transfer_repeated_riemann = _load_json_if_exists(outputs / "transfer_riemann_all_targets_seed42_43_v2" / "result.json")
    transfer_repeated_eegnet = _load_first_existing_json(
        outputs / "transfer_eegnet_all_targets_seed42_43_es50" / "result.json",
        outputs / "transfer_eegnet_all_targets_seed42_43_e30" / "result.json",
    )

    within_headers = ["Model", "Accuracy Mean", "Accuracy Std"]
    within_rows: list[list[str]] = []
    if within_classical is not None:
        within_rows.extend(
            [
                ["Raw Power + LDA", f"{within_classical['raw_power']['summary']['accuracy_mean']:.4f}", f"{within_classical['raw_power']['summary']['accuracy_std']:.4f}"],
                ["CSP + LDA", f"{within_classical['csp']['summary']['accuracy_mean']:.4f}", f"{within_classical['csp']['summary']['accuracy_std']:.4f}"],
                ["FBCSP + LDA", f"{within_classical['fbcsp']['summary']['accuracy_mean']:.4f}", f"{within_classical['fbcsp']['summary']['accuracy_std']:.4f}"],
            ]
        )
    if within_riemann is not None:
        within_rows.append(
            ["Riemann + Tangent Space + LDA", f"{within_riemann['summary']['accuracy_mean']:.4f}", f"{within_riemann['summary']['accuracy_std']:.4f}"]
        )
    if within_eegnet is not None:
        within_rows.append(["EEGNet", f"{within_eegnet['summary']['accuracy_mean']:.4f}", f"{within_eegnet['summary']['accuracy_std']:.4f}"])
    if within_rows:
        _save_csv(assets / "within_subject_summary.csv", within_headers, within_rows)
        _save_markdown_table(assets / "within_subject_summary.md", within_headers, within_rows)
        _save_bar_chart(
            assets / "within_subject_accuracy.png",
            [row[0] for row in within_rows],
            [float(row[1]) for row in within_rows],
            title="Within-Subject Accuracy",
        )
        generated["within_subject_table"] = str(assets / "within_subject_summary.md")
        generated["within_subject_figure"] = str(assets / "within_subject_accuracy.png")
    if within_eegnet is not None:
        histories = within_eegnet.get("training_histories")
        if isinstance(histories, list) and histories:
            _save_learning_curve(
                assets / "within_subject_eegnet_learning_curve.png",
                histories,
                title="Within-Subject EEGNet Learning Curve",
            )
            generated["within_subject_eegnet_learning_curve"] = str(assets / "within_subject_eegnet_learning_curve.png")
    if within_classical is not None:
        _save_confusion_matrix(
            assets / "within_subject_fbcsp_confusion.png",
            np.asarray(within_classical["fbcsp"]["summary"]["confusion_matrix_sum"]),
            title="Within-Subject FBCSP Confusion Matrix",
        )
        generated["within_subject_confusion"] = str(assets / "within_subject_fbcsp_confusion.png")

    loso_headers = ["Model", "Accuracy Mean", "Accuracy Std"]
    loso_rows: list[list[str]] = []
    if loso_classical is not None:
        loso_rows.extend(
            [
                ["Raw Power + LDA", f"{loso_classical['raw_power']['summary']['accuracy_mean']:.4f}", f"{loso_classical['raw_power']['summary']['accuracy_std']:.4f}"],
                ["CSP + LDA", f"{loso_classical['csp']['summary']['accuracy_mean']:.4f}", f"{loso_classical['csp']['summary']['accuracy_std']:.4f}"],
                ["FBCSP + LDA", f"{loso_classical['fbcsp']['summary']['accuracy_mean']:.4f}", f"{loso_classical['fbcsp']['summary']['accuracy_std']:.4f}"],
            ]
        )
    if loso_riemann is not None:
        loso_rows.append(
            ["Riemann + Tangent Space + LDA", f"{loso_riemann['summary']['accuracy_mean']:.4f}", f"{loso_riemann['summary']['accuracy_std']:.4f}"]
        )
    if loso_eegnet is not None:
        loso_rows.append(["EEGNet", f"{loso_eegnet['summary']['accuracy_mean']:.4f}", f"{loso_eegnet['summary']['accuracy_std']:.4f}"])
    if loso_rows:
        _save_csv(assets / "loso_summary.csv", loso_headers, loso_rows)
        _save_markdown_table(assets / "loso_summary.md", loso_headers, loso_rows)
        _save_bar_chart(
            assets / "loso_accuracy.png",
            [row[0] for row in loso_rows],
            [float(row[1]) for row in loso_rows],
            title="LOSO Accuracy",
            color="#385f8c",
        )
        generated["loso_table"] = str(assets / "loso_summary.md")
        generated["loso_figure"] = str(assets / "loso_accuracy.png")
    if loso_eegnet is not None:
        histories = loso_eegnet.get("training_histories")
        if isinstance(histories, list) and histories:
            _save_learning_curve(
                assets / "loso_eegnet_learning_curve.png",
                histories,
                title="LOSO EEGNet Learning Curve",
            )
            generated["loso_eegnet_learning_curve"] = str(assets / "loso_eegnet_learning_curve.png")
    if loso_eegnet is not None:
        _save_confusion_matrix(
            assets / "loso_eegnet_confusion.png",
            np.asarray(loso_eegnet["summary"]["confusion_matrix_sum"]),
            title="LOSO EEGNet Confusion Matrix",
        )
        generated["loso_confusion"] = str(assets / "loso_eegnet_confusion.png")

    transfer_headers = ["Model", "Accuracy Mean", "Accuracy CI95 Low", "Accuracy CI95 High"]
    transfer_rows: list[list[str]] = []
    if transfer_repeated_classical is not None:
        transfer_rows.append(
            [
                "FBCSP",
                f"{transfer_repeated_classical['aggregate_by_setting']['summary']['accuracy_mean']:.4f}",
                f"{transfer_repeated_classical['aggregate_by_setting']['summary']['accuracy_ci95_low']:.4f}",
                f"{transfer_repeated_classical['aggregate_by_setting']['summary']['accuracy_ci95_high']:.4f}",
            ]
        )
    if transfer_repeated_riemann is not None:
        transfer_rows.append(
            [
                "Riemann",
                f"{transfer_repeated_riemann['aggregate_by_setting']['summary']['accuracy_mean']:.4f}",
                f"{transfer_repeated_riemann['aggregate_by_setting']['summary']['accuracy_ci95_low']:.4f}",
                f"{transfer_repeated_riemann['aggregate_by_setting']['summary']['accuracy_ci95_high']:.4f}",
            ]
        )
    if transfer_repeated_eegnet is not None:
        transfer_rows.append(
            [
                "EEGNet",
                f"{transfer_repeated_eegnet['aggregate_by_setting']['summary']['accuracy_mean']:.4f}",
                f"{transfer_repeated_eegnet['aggregate_by_setting']['summary']['accuracy_ci95_low']:.4f}",
                f"{transfer_repeated_eegnet['aggregate_by_setting']['summary']['accuracy_ci95_high']:.4f}",
            ]
        )
    if transfer_rows:
        _save_csv(assets / "transfer_repeated_summary.csv", transfer_headers, transfer_rows)
        _save_markdown_table(assets / "transfer_repeated_summary.md", transfer_headers, transfer_rows)
        generated["transfer_table"] = str(assets / "transfer_repeated_summary.md")

    shot_labels = ["zero_shot", "5_shot", "10_shot", "20_shot", "30_shot"]
    transfer_series: dict[str, list[float]] = {}
    if transfer_repeated_classical is not None:
        transfer_series["FBCSP"] = [
            float(next(row["accuracy"] for row in transfer_repeated_classical["aggregate_by_setting"]["rows"] if row["label"] == label))
            for label in shot_labels
        ]
    if transfer_repeated_riemann is not None:
        transfer_series["Riemann"] = [
            float(next(row["accuracy"] for row in transfer_repeated_riemann["aggregate_by_setting"]["rows"] if row["label"] == label))
            for label in shot_labels
        ]
    if transfer_repeated_eegnet is not None:
        transfer_series["EEGNet"] = [
            float(next(row["accuracy"] for row in transfer_repeated_eegnet["aggregate_by_setting"]["rows"] if row["label"] == label))
            for label in shot_labels
        ]
    if transfer_series:
        _save_transfer_curve(
            assets / "transfer_repeated_accuracy.png",
            shot_labels,
            transfer_series,
            title="Repeated-Seed Transfer Accuracy",
        )
        generated["transfer_figure"] = str(assets / "transfer_repeated_accuracy.png")
    if transfer_repeated_riemann is not None:
        _save_confusion_matrix(
            assets / "transfer_repeated_riemann_confusion.png",
            np.asarray(transfer_repeated_riemann["aggregate_by_setting"]["summary"]["confusion_matrix_sum"]),
            title="Repeated-Seed Transfer Riemann Confusion Matrix",
        )
        generated["transfer_riemann_confusion"] = str(assets / "transfer_repeated_riemann_confusion.png")
    if transfer_repeated_eegnet is not None:
        _save_confusion_matrix(
            assets / "transfer_repeated_eegnet_confusion.png",
            np.asarray(transfer_repeated_eegnet["aggregate_by_setting"]["summary"]["confusion_matrix_sum"]),
            title="Repeated-Seed Transfer EEGNet Confusion Matrix",
        )
        generated["transfer_confusion"] = str(assets / "transfer_repeated_eegnet_confusion.png")

    return generated
