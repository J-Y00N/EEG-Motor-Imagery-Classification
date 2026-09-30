"""Exploratory analysis exports built on the canonical EEG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import torch

from eeg_motor_imagery_classification.config import PreprocessingConfig
from eeg_motor_imagery_classification.constants import INDEX_TO_LABEL
from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.data.epochs import extract_epochs_array
from eeg_motor_imagery_classification.data.loaders import load_subject_bundle
from eeg_motor_imagery_classification.data.preprocessing import preprocess_runs
from eeg_motor_imagery_classification.features import LogVarianceVectorizer
from eeg_motor_imagery_classification.models import (
    EEGNet,
    build_csp_pipeline,
    build_fbcsp_pipeline,
    build_riemann_tangent_pipeline,
)
from eeg_motor_imagery_classification.train import TrainingConfig, train_model
from eeg_motor_imagery_classification.utils import ensure_directory, write_json

# EDA epochs are cut wider than the classification window (0-4 s after the cue) so that the
# ERD/ERS reference interval and the displayed task interval stay clear of the Morlet
# wavelet edge effects at both ends of each epoch.
EDA_TMIN = -1.5  # s relative to the cue (= 0.5 s after the fixation cross / warning tone)
EDA_TMAX_PAD = 0.5  # s appended after the task window
ERDS_BASELINE = (-1.0, -0.2)  # pre-cue reference interval (s)
ERDS_DISPLAY = (-1.0, 4.0)  # interval shown in ERDS figures (s)


def baseline_percent_change(power: np.ndarray, times: np.ndarray, baseline: tuple[float, float] = ERDS_BASELINE) -> np.ndarray:
    """Classical ERD/ERS: average power over trials first, then express it relative to baseline.

    ``power`` has trials on axis 0 and time on the last axis. Averaging per-trial ratios
    instead (E[P/B]) is biased upwards because the short single-trial baseline B is noisy, which
    produces a spurious sustained "power increase" even for data without any task effect.
    """

    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if not np.any(baseline_mask):
        raise ValueError(f"No time points inside the ERDS baseline window {baseline}.")
    mean_power = power.mean(axis=0)
    reference = mean_power[..., baseline_mask].mean(axis=-1, keepdims=True)
    return 100.0 * (mean_power - reference) / reference


def _log_variance_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(LogVarianceVectorizer().transform(X))


def _plot_psd(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    sfreq = bundle.metadata.sfreq
    fig, ax = plt.subplots(figsize=(7.5, 4.25))
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        class_data = bundle.X[bundle.y == class_id]
        freqs, psd = welch(class_data, fs=sfreq, nperseg=min(256, class_data.shape[-1]), axis=-1)
        mean_psd_uv2 = psd.mean(axis=(0, 1)) * 1e12  # V^2/Hz -> uV^2/Hz
        ax.plot(freqs, 10.0 * np.log10(np.maximum(mean_psd_uv2, np.finfo(float).tiny)), linewidth=2.0, label=class_name)
    ax.set_xlim(0.0, 40.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re 1 $\\mu$V$^2$/Hz)")
    ax.set_title(f"Subject {bundle.metadata.subject_id} PSD by Class")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pca(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    features = _log_variance_features(bundle.X)
    projection = PCA(n_components=2, random_state=42).fit_transform(features)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        mask = bundle.y == class_id
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=18,
            alpha=0.75,
            label=class_name,
            color=palette.get(int(class_id), "#444444"),
        )
    ax.set_title(f"Subject {bundle.metadata.subject_id} PCA of Channel Log-Variance")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_tsne(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    features = _log_variance_features(bundle.X)
    perplexity = max(5, min(30, (bundle.X.shape[0] - 1) // 3))
    projection = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=42,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        mask = bundle.y == class_id
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=18,
            alpha=0.8,
            label=class_name,
            color=palette.get(int(class_id), "#444444"),
        )
    ax.set_title(f"Subject {bundle.metadata.subject_id} t-SNE of Channel Log-Variance")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    info = mne.pick_info(bundle.epochs.info.copy(), eeg_picks)
    tiny = np.finfo(float).tiny
    X_uv = bundle.X[:, eeg_picks, :].astype(np.float64) * 1e6
    left_power = np.log(np.maximum(X_uv[bundle.y == 0].var(axis=-1).mean(axis=0), tiny))
    right_power = np.log(np.maximum(X_uv[bundle.y == 1].var(axis=-1).mean(axis=0), tiny))
    diff_power = left_power - right_power

    # Shared data-driven scale for the two class maps; symmetric scale for the difference map.
    shared_vlim = (float(min(left_power.min(), right_power.min())), float(max(left_power.max(), right_power.max())))
    diff_abs = float(np.max(np.abs(diff_power)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    panels = [
        (left_power, "Reds", shared_vlim, "Left MI log-var ($\\mu$V$^2$)"),
        (right_power, "Reds", shared_vlim, "Right MI log-var ($\\mu$V$^2$)"),
        (diff_power, "RdBu_r", (-diff_abs, diff_abs), "Left - Right (log ratio)"),
    ]
    for ax, (values, cmap, vlim, title) in zip(axes, panels, strict=True):
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=vlim)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
    fig.suptitle(f"Subject {bundle.metadata.subject_id} Channel Topography", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _select_erds_channels(ch_names: tuple[str, ...]) -> list[str]:
    preferred = [name for name in ("C3", "C4", "Cz") if name in ch_names]
    if preferred:
        return preferred[:2] if len(preferred) >= 2 else preferred
    return list(ch_names[:2])


def _compute_morlet_power(
    epochs: mne.Epochs,
    *,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    decim: int,
):
    return epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        decim=decim,
        verbose=False,
    )


def _plot_erds(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    channel_names = _select_erds_channels(bundle.epochs.ch_names)
    epochs = bundle.epochs.copy().pick(channel_names)
    freqs = np.arange(8.0, 31.0, 2.0)
    n_cycles = freqs / 2.0
    power = _compute_morlet_power(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=4,
    )

    times = power.times
    display_mask = (times >= ERDS_DISPLAY[0]) & (times <= ERDS_DISPLAY[1])
    shown_times = times[display_mask]

    fig, axes = plt.subplots(
        len(channel_names),
        2,
        figsize=(9.2, 3.6 * len(channel_names)),
        squeeze=False,
        constrained_layout=True,
    )
    class_labels = [label for _, label in sorted(INDEX_TO_LABEL.items())]
    cmap = "RdBu_r"
    im = None

    for row_idx, channel_name in enumerate(channel_names):
        for col_idx, class_id in enumerate(np.unique(bundle.y)):
            class_mask = bundle.y == class_id
            class_power = power.data[class_mask, row_idx, :, :]
            mean_power = baseline_percent_change(class_power, times)[:, display_mask]

            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                mean_power,
                aspect="auto",
                origin="lower",
                extent=[shown_times[0], shown_times[-1], freqs[0], freqs[-1]],
                cmap=cmap,
                vmin=-50,
                vmax=50,
            )
            ax.set_title(f"{channel_name} - {class_labels[int(class_id)]}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

    if im is None:
        raise ValueError("ERDS plot could not be created because no image handle was produced.")
    fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.02, label="Power change (%)")
    fig.suptitle(f"Subject {bundle.metadata.subject_id} ERDS-style Time-Frequency Maps", y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fix_pattern_sign(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if abs(values.min()) > abs(values.max()):
        return -values
    return values


def _plot_classical_patterns(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    info = mne.pick_info(bundle.epochs.info.copy(), eeg_picks)
    X_eeg = bundle.X[:, eeg_picks, :]

    csp_pipeline = build_csp_pipeline(n_components=4)
    csp_pipeline.fit(X_eeg, bundle.y)
    csp = csp_pipeline.named_steps["csp"]
    csp_patterns = np.asarray(csp.patterns_)

    fbcsp_pipeline = build_fbcsp_pipeline(
        sfreq=bundle.metadata.sfreq,
        n_components=4,
        k_best=8,
    )
    fbcsp_pipeline.fit(X_eeg, bundle.y)
    fbcsp = fbcsp_pipeline.named_steps["fbcsp"]
    first_band = fbcsp.bands[0]
    last_band = fbcsp.bands[-1]
    first_band_pattern = np.asarray(fbcsp.csp_models_[0].patterns_)
    last_band_pattern = np.asarray(fbcsp.csp_models_[-1].patterns_)

    fig, axes = plt.subplots(2, 2, figsize=(8.75, 6.75))
    maps = [
        ("CSP comp 1", _fix_pattern_sign(csp_patterns[0])),
        ("CSP comp 4", _fix_pattern_sign(csp_patterns[min(3, csp_patterns.shape[0] - 1)])),
        (f"FBCSP {first_band[0]:.0f}-{first_band[1]:.0f}Hz", _fix_pattern_sign(first_band_pattern[0])),
        (f"FBCSP {last_band[0]:.0f}-{last_band[1]:.0f}Hz", _fix_pattern_sign(last_band_pattern[0])),
    ]
    for ax, (title, values) in zip(axes.ravel(), maps, strict=True):
        mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r")
        ax.set_title(title)
    fig.suptitle(f"Subject {bundle.metadata.subject_id} Classical Spatial Patterns", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_riemann_3d(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    X_eeg = bundle.X[:, eeg_picks, :]
    covariances = Covariances(estimator="oas").fit_transform(X_eeg)
    tangent = TangentSpace(metric="riemann").fit_transform(covariances)
    projection = PCA(n_components=3, random_state=42).fit_transform(tangent)

    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        mask = bundle.y == class_id
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            projection[mask, 2],
            s=22,
            alpha=0.78,
            color=palette.get(int(class_id), "#444444"),
            label=class_name,
        )
    ax.set_title(f"Subject {bundle.metadata.subject_id} Riemann Tangent-Space View")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _out_of_fold_scores(pipeline, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LDA decision scores for each trial from a model that never saw that trial."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_predict(pipeline, X, y, cv=cv, method="decision_function")
    return np.asarray(scores).reshape(-1)


def _plot_riemann_lda_distribution(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    X_eeg = bundle.X[:, eeg_picks, :]
    scores = _out_of_fold_scores(build_riemann_tangent_pipeline(), X_eeg, bundle.y)
    if np.ndim(scores) > 1:
        scores = np.asarray(scores).reshape(-1)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    bins = np.linspace(float(np.min(scores)) - 0.2, float(np.max(scores)) + 0.2, 24)
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        class_scores = scores[bundle.y == class_id]
        ax.hist(
            class_scores,
            bins=bins,
            alpha=0.55,
            density=True,
            label=class_name,
            color=palette.get(int(class_id), "#444444"),
        )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_title(f"Subject {bundle.metadata.subject_id} Riemann + LDA Scores (out-of-fold, 5-fold CV)")
    ax.set_xlabel("LDA decision score")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_csp_lda_distribution(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    X_eeg = bundle.X[:, eeg_picks, :]
    scores = _out_of_fold_scores(build_csp_pipeline(n_components=4), X_eeg, bundle.y)
    if np.ndim(scores) > 1:
        scores = np.asarray(scores).reshape(-1)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    bins = np.linspace(float(np.min(scores)) - 0.2, float(np.max(scores)) + 0.2, 24)
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        class_scores = scores[bundle.y == class_id]
        ax.hist(
            class_scores,
            bins=bins,
            alpha=0.55,
            density=True,
            label=class_name,
            color=palette.get(int(class_id), "#444444"),
        )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_title(f"Subject {bundle.metadata.subject_id} CSP + LDA Scores (out-of-fold, 5-fold CV)")
    ax.set_xlabel("LDA decision score")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_eegnet_saliency_topomap(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    info = mne.pick_info(bundle.epochs.info.copy(), eeg_picks)
    X_eeg = bundle.X[:, eeg_picks, :]
    dataset = EpochDataset(X_eeg, bundle.y, fit_scaler=True)
    model = EEGNet(
        n_channels=X_eeg.shape[1],
        n_times=X_eeg.shape[2],
        n_classes=len(np.unique(bundle.y)),
    )
    model = train_model(
        model,
        dataset,
        config=TrainingConfig(epochs=15, batch_size=64, learning_rate=1e-3, seed=42, deterministic=True),
    )
    model.eval()
    device = next(model.parameters()).device

    X_tensor = torch.from_numpy(dataset.X).to(device)
    y_tensor = torch.from_numpy(dataset.y).to(device)
    X_tensor.requires_grad_(True)
    logits = model(X_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, class_id in zip(axes, sorted(np.unique(bundle.y)), strict=True):
        model.zero_grad(set_to_none=True)
        class_mask = y_tensor == int(class_id)
        selected_logits = logits[class_mask, int(class_id)].sum()
        selected_logits.backward(retain_graph=True)
        saliency = X_tensor.grad[class_mask].detach().abs().mean(dim=(0, 2)).cpu().numpy()
        X_tensor.grad.zero_()
        mne.viz.plot_topomap(saliency, info, axes=ax, show=False, cmap="Reds")
        ax.set_title(f"{INDEX_TO_LABEL[int(class_id)]} saliency")

    fig.suptitle(f"Subject {bundle.metadata.subject_id} EEGNet Saliency Topomaps (model trained on all trials)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_csp_feature_projection(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    X_eeg = bundle.X[:, eeg_picks, :]
    pipeline = build_csp_pipeline(n_components=4)
    pipeline.fit(X_eeg, bundle.y)
    csp_features = pipeline.named_steps["csp"].transform(X_eeg)

    fig, ax = plt.subplots(figsize=(6.6, 5.1))
    palette = {0: "#2f5d50", 1: "#c26d3a"}
    for class_id in np.unique(bundle.y):
        class_name = INDEX_TO_LABEL[int(class_id)]
        mask = bundle.y == class_id
        ax.scatter(
            csp_features[mask, 0],
            csp_features[mask, 1],
            s=18,
            alpha=0.75,
            color=palette.get(int(class_id), "#444444"),
            label=class_name,
        )
    ax.set_title(f"Subject {bundle.metadata.subject_id} CSP Feature Projection (in-sample)")
    ax.set_xlabel("CSP component 1")
    ax.set_ylabel("CSP component 2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_sensorimotor_erds_summary(bundle, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    available = [name for name in ("C3", "C4") if name in bundle.epochs.ch_names]
    if len(available) < 2:
        fallback = [name for name in bundle.epochs.ch_names if name.startswith("C")]
        available = fallback[:2] if len(fallback) >= 2 else list(bundle.epochs.ch_names[:2])

    epochs = bundle.epochs.copy().pick(available)
    freqs = np.arange(8.0, 31.0, 1.0)
    n_cycles = freqs / 2.0
    power = _compute_morlet_power(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=4,
    )

    times = power.times
    display_mask = (times >= ERDS_DISPLAY[0]) & (times <= ERDS_DISPLAY[1])

    mu_mask = (freqs >= 8.0) & (freqs <= 12.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)

    def _band_percent_change(class_id: int, channel_idx: int, band_mask: np.ndarray) -> np.ndarray:
        class_mask = bundle.y == class_id
        class_power = power.data[class_mask, channel_idx, :, :][:, band_mask, :].mean(axis=1)
        return baseline_percent_change(class_power, times)[display_mask]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, sharey=True)
    channel_to_col = {name: idx for idx, name in enumerate(available[:2])}
    class_names = {0: "Left-hand imagery", 1: "Right-hand imagery"}
    band_colors = {"Mu (8-12Hz)": "#2f5d50", "Beta (13-30Hz)": "#c26d3a"}

    for row_idx, class_id in enumerate(sorted(np.unique(bundle.y))):
        for channel_name, col_idx in channel_to_col.items():
            channel_idx = available.index(channel_name)
            ax = axes[row_idx, col_idx]
            mu_curve = _band_percent_change(int(class_id), channel_idx, mu_mask)
            beta_curve = _band_percent_change(int(class_id), channel_idx, beta_mask)

            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
            ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle=":")
            ax.axvspan(*ERDS_BASELINE, color="#999999", alpha=0.12, linewidth=0)
            ax.plot(times[display_mask], mu_curve, color=band_colors["Mu (8-12Hz)"], linewidth=2.2, label="Mu (8-12Hz)")
            ax.plot(times[display_mask], beta_curve, color=band_colors["Beta (13-30Hz)"], linewidth=2.2, label="Beta (13-30Hz)")
            ax.set_title(f"{class_names[int(class_id)]}\n{channel_name}", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power change (%)")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.set_xlim(*ERDS_DISPLAY)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        f"Subject {bundle.metadata.subject_id} Sensorimotor ERD/ERS Summary",
        y=1.03,
        fontsize=14,
    )
    fig.subplots_adjust(top=0.82, hspace=0.35, wspace=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _collect_sensorimotor_curves(bundle) -> dict[tuple[int, str, str], np.ndarray]:
    available = [name for name in ("C3", "C4") if name in bundle.epochs.ch_names]
    if len(available) < 2:
        fallback = [name for name in bundle.epochs.ch_names if name.startswith("C")]
        available = fallback[:2] if len(fallback) >= 2 else list(bundle.epochs.ch_names[:2])

    epochs = bundle.epochs.copy().pick(available)
    freqs = np.arange(8.0, 31.0, 1.0)
    n_cycles = freqs / 2.0
    power = _compute_morlet_power(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=4,
    )
    times = power.times
    display_mask = (times >= ERDS_DISPLAY[0]) & (times <= ERDS_DISPLAY[1])

    mu_mask = (freqs >= 8.0) & (freqs <= 12.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
    curves: dict[tuple[int, str, str], np.ndarray] = {}
    for class_id in sorted(np.unique(bundle.y)):
        class_mask = bundle.y == class_id
        for channel_name in available[:2]:
            channel_idx = available.index(channel_name)
            class_power = power.data[class_mask, channel_idx, :, :]
            for band_name, band_mask in (("mu", mu_mask), ("beta", beta_mask)):
                band_power = class_power[:, band_mask, :].mean(axis=1)
                curves[(int(class_id), channel_name, band_name)] = baseline_percent_change(band_power, times)[display_mask]
    curves[(-1, "times", "times")] = times[display_mask]
    return curves


def export_grand_average_sensorimotor_erds(
    *,
    subject_ids: tuple[int, ...],
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    bundles = [_build_eda_bundle(subject_id, config=config) for subject_id in subject_ids]
    curve_maps = [_collect_sensorimotor_curves(bundle) for bundle in bundles]
    times = curve_maps[0][(-1, "times", "times")]
    output_path = ensure_directory(output_dir)
    figure_path = output_path / "grand_average_sensorimotor_erds.png"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, sharey=True)
    class_names = {0: "Left-hand imagery", 1: "Right-hand imagery"}
    band_colors = {"mu": "#2f5d50", "beta": "#c26d3a"}
    channels = ("C3", "C4")
    if not all((0, channel, "mu") in curve_maps[0] for channel in channels):
        channels = tuple(
            name
            for name in sorted({key[1] for key in curve_maps[0] if key[0] in (0, 1) and key[2] in ("mu", "beta")})
            if name not in {"times"}
        )[:2]

    for row_idx, class_id in enumerate((0, 1)):
        for col_idx, channel_name in enumerate(channels):
            ax = axes[row_idx, col_idx]
            mu_curves = np.stack([curve_map[(class_id, channel_name, "mu")] for curve_map in curve_maps], axis=0)
            beta_curves = np.stack([curve_map[(class_id, channel_name, "beta")] for curve_map in curve_maps], axis=0)
            mu_mean = mu_curves.mean(axis=0)
            beta_mean = beta_curves.mean(axis=0)
            mu_sem = mu_curves.std(axis=0, ddof=0) / np.sqrt(mu_curves.shape[0])
            beta_sem = beta_curves.std(axis=0, ddof=0) / np.sqrt(beta_curves.shape[0])

            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
            ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle=":")
            ax.axvspan(*ERDS_BASELINE, color="#999999", alpha=0.12, linewidth=0)
            ax.plot(times, mu_mean, color=band_colors["mu"], linewidth=2.2, label="Mu (8-12Hz)")
            ax.fill_between(times, mu_mean - mu_sem, mu_mean + mu_sem, color=band_colors["mu"], alpha=0.18)
            ax.plot(times, beta_mean, color=band_colors["beta"], linewidth=2.2, label="Beta (13-30Hz)")
            ax.fill_between(times, beta_mean - beta_sem, beta_mean + beta_sem, color=band_colors["beta"], alpha=0.18)
            ax.set_title(f"{class_names[class_id]}\n{channel_name}", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power change (%)")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.set_xlim(times[0], times[-1])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Grand-average Sensorimotor ERD/ERS Summary", y=1.03, fontsize=14)
    fig.subplots_adjust(top=0.82, hspace=0.35, wspace=0.22)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "subjects": list(subject_ids),
        "artifact": str(figure_path),
        "n_subjects": len(subject_ids),
    }
    write_json(output_path / "grand_average_sensorimotor_erds.json", summary)
    return summary


def _build_eda_bundle(subject_id: int, config: PreprocessingConfig | None = None):
    cfg = config or PreprocessingConfig()
    sessions, metadata = load_subject_bundle(subject_id)
    epoch_list: list[mne.Epochs] = []
    for _, _, raw in preprocess_runs(sessions, config=cfg):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        target_event_id = {event_name: event_id[event_name] for event_name in cfg.event_names}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=target_event_id,
            tmin=EDA_TMIN,
            tmax=cfg.tmax + EDA_TMAX_PAD,
            baseline=None,
            preload=True,
            verbose=False,
        )
        epochs.set_annotations(None)
        epoch_list.append(epochs)

    merged = mne.concatenate_epochs(epoch_list, add_offset=True, verbose=False)
    X_full, y = extract_epochs_array(merged, event_names=cfg.event_names)
    sfreq = float(merged.info["sfreq"])
    start = int(round((cfg.tmin - EDA_TMIN) * sfreq))
    stop = start + int(round((cfg.tmax - cfg.tmin) * sfreq)) + 1
    X_task = X_full[:, :, start:stop]  # same 0-4 s task window as the classification arrays
    bundle_metadata = SimpleNamespace(
        subject_id=subject_id,
        sfreq=float(merged.info["sfreq"]),
        ch_names=tuple(merged.ch_names),
        session_names=metadata.session_names,
        run_names=metadata.run_names,
    )
    return SimpleNamespace(
        X=X_task,
        y=y,
        epochs=merged,
        metadata=bundle_metadata,
    )


def export_subject_eda_assets(
    *,
    subject_id: int,
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
) -> dict[str, object]:
    """Export basic subject-level EDA assets from canonical epochs."""

    bundle = _build_eda_bundle(subject_id, config=config)

    output_path = ensure_directory(output_dir)
    psd_path = output_path / f"subject_{subject_id}_psd.png"
    pca_path = output_path / f"subject_{subject_id}_pca.png"
    tsne_path = output_path / f"subject_{subject_id}_tsne.png"
    topomap_path = output_path / f"subject_{subject_id}_topomap.png"
    erds_path = output_path / f"subject_{subject_id}_erds.png"
    patterns_path = output_path / f"subject_{subject_id}_classical_patterns.png"
    erds_summary_path = output_path / f"subject_{subject_id}_sensorimotor_erds.png"
    riemann_3d_path = output_path / f"subject_{subject_id}_riemann_3d.png"
    csp_lda_path = output_path / f"subject_{subject_id}_csp_lda_distribution.png"
    csp_projection_path = output_path / f"subject_{subject_id}_csp_projection.png"
    riemann_lda_path = output_path / f"subject_{subject_id}_riemann_lda_distribution.png"
    eegnet_saliency_path = output_path / f"subject_{subject_id}_eegnet_saliency_topomap.png"

    _plot_psd(bundle, psd_path)
    _plot_pca(bundle, pca_path)
    _plot_tsne(bundle, tsne_path)
    _plot_topomap(bundle, topomap_path)
    _plot_erds(bundle, erds_path)
    _plot_classical_patterns(bundle, patterns_path)
    _plot_sensorimotor_erds_summary(bundle, erds_summary_path)
    _plot_riemann_3d(bundle, riemann_3d_path)
    _plot_riemann_lda_distribution(bundle, riemann_lda_path)
    _plot_csp_lda_distribution(bundle, csp_lda_path)
    _plot_csp_feature_projection(bundle, csp_projection_path)
    _plot_eegnet_saliency_topomap(bundle, eegnet_saliency_path)

    summary = {
        "subject_id": subject_id,
        "n_epochs": int(bundle.X.shape[0]),
        "n_channels": int(bundle.X.shape[1]),
        "n_times": int(bundle.X.shape[2]),
        "sfreq": float(bundle.metadata.sfreq),
        "class_counts": {
            INDEX_TO_LABEL[int(class_id)]: int(np.sum(bundle.y == class_id)) for class_id in np.unique(bundle.y)
        },
        "artifacts": {
            "psd": str(psd_path),
            "pca": str(pca_path),
            "tsne": str(tsne_path),
            "topomap": str(topomap_path),
            "erds": str(erds_path),
            "classical_patterns": str(patterns_path),
            "sensorimotor_erds": str(erds_summary_path),
            "riemann_3d": str(riemann_3d_path),
            "riemann_lda_distribution": str(riemann_lda_path),
            "csp_lda_distribution": str(csp_lda_path),
            "csp_projection": str(csp_projection_path),
            "eegnet_saliency_topomap": str(eegnet_saliency_path),
        },
    }
    write_json(output_path / f"subject_{subject_id}_eda_summary.json", summary)
    return json.loads((output_path / f"subject_{subject_id}_eda_summary.json").read_text())
