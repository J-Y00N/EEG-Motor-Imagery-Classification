"""Exploratory analysis exports built on the canonical EEG pipeline."""

from __future__ import annotations

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
from eeg_motor_imagery_classification.evaluation.statistics import holm_correction, paired_permutation_test
from eeg_motor_imagery_classification.utils import ensure_directory, read_json, write_csv, write_json, write_text

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


def _plot_topomap(bundle, output_path: Path, values_csv: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    eeg_picks = mne.pick_types(bundle.epochs.info, eeg=True, exclude="bads")
    info = mne.pick_info(bundle.epochs.info.copy(), eeg_picks)
    tiny = np.finfo(float).tiny
    X_uv = bundle.X[:, eeg_picks, :].astype(np.float64) * 1e6
    left_power = np.log(np.maximum(X_uv[bundle.y == 0].var(axis=-1).mean(axis=0), tiny))
    right_power = np.log(np.maximum(X_uv[bundle.y == 1].var(axis=-1).mean(axis=0), tiny))
    diff_power = left_power - right_power
    if values_csv is not None:
        names = [info.ch_names[i] for i in range(len(diff_power))]
        write_csv(values_csv, ["channel", "left_log_var_uv2", "right_log_var_uv2", "left_minus_right"],
                  [[name, float(a), float(b), float(c)] for name, a, b, c in zip(names, left_power, right_power, diff_power, strict=True)])

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


def _plot_eegnet_saliency_topomap(bundle, output_path: Path, device: str | None = None) -> None:
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
        config=TrainingConfig(epochs=15, batch_size=64, learning_rate=1e-3, seed=42, deterministic=True, device=device),
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


ERDS_SUMMARY_WINDOW = (0.5, 2.5)  # s after the cue; pre-specified window for mean ERD/ERS
ERDS_SEARCH_WINDOW = (0.0, 4.0)  # s after the cue; where the most negative value (peak ERD) is searched
CONTRALATERAL_CHANNEL = {0: "C4", 1: "C3"}  # left-hand imagery -> right hemisphere and vice versa
CLASS_TITLES = {0: "Left-hand imagery", 1: "Right-hand imagery"}
BAND_STYLE = {"mu": ("Mu (8-12Hz)", "#2f5d50"), "beta": ("Beta (13-30Hz)", "#c26d3a")}


def _collect_sensorimotor_curves(bundle) -> dict[tuple[int, str, str], np.ndarray]:
    """Band-averaged ERD/ERS curves {(class_id, channel, band): curve} plus {(-1, "times", "times"): times}."""

    available = [name for name in ("C3", "C4") if name in bundle.epochs.ch_names]
    if len(available) < 2:
        fallback = [name for name in bundle.epochs.ch_names if name.startswith("C")]
        available = fallback[:2] if len(fallback) >= 2 else list(bundle.epochs.ch_names[:2])

    epochs = bundle.epochs.copy().pick(available)
    freqs = np.arange(8.0, 31.0, 1.0)
    power = _compute_morlet_power(epochs, freqs=freqs, n_cycles=freqs / 2.0, decim=4)
    times = power.times
    display_mask = (times >= ERDS_DISPLAY[0]) & (times <= ERDS_DISPLAY[1])
    band_masks = {"mu": (freqs >= 8.0) & (freqs <= 12.0), "beta": (freqs >= 13.0) & (freqs <= 30.0)}

    curves: dict[tuple[int, str, str], np.ndarray] = {}
    for class_id in sorted(np.unique(bundle.y)):
        class_mask = bundle.y == class_id
        for channel_idx, channel_name in enumerate(available[:2]):
            class_power = power.data[class_mask, channel_idx, :, :]
            for band_name, band_mask in band_masks.items():
                band_power = class_power[:, band_mask, :].mean(axis=1)
                curves[(int(class_id), channel_name, band_name)] = baseline_percent_change(band_power, times)[display_mask]
    curves[(-1, "times", "times")] = times[display_mask]
    return curves


def erds_summary_rows(curves: dict[tuple[int, str, str], np.ndarray], unit: str) -> list[dict[str, object]]:
    """Window mean, peak ERD (most negative value), and its latency for every class/channel/band curve."""

    times = curves[(-1, "times", "times")]
    window = (times >= ERDS_SUMMARY_WINDOW[0]) & (times <= ERDS_SUMMARY_WINDOW[1])
    search = (times >= ERDS_SEARCH_WINDOW[0]) & (times <= ERDS_SEARCH_WINDOW[1])
    rows = []
    for (class_id, channel, band), curve in curves.items():
        if class_id < 0:
            continue
        peak_index = int(np.argmin(np.where(search, curve, np.inf)))
        rows.append(
            {
                "unit": unit,
                "class": INDEX_TO_LABEL[class_id],
                "channel": channel,
                "band": band,
                "side": "contralateral" if CONTRALATERAL_CHANNEL.get(class_id) == channel else "ipsilateral",
                "window_mean_pct": float(curve[window].mean()),
                "peak_erd_pct": float(curve[peak_index]),
                "peak_latency_s": float(times[peak_index]),
            }
        )
    return rows


SUMMARY_COLUMNS = ["unit", "class", "channel", "band", "side", "window_mean_pct", "peak_erd_pct", "peak_latency_s"]


def _summary_markdown(rows: list[dict[str, object]], title: str) -> list[str]:
    lines = [f"## {title}", "",
             f"Window mean over {ERDS_SUMMARY_WINDOW[0]}-{ERDS_SUMMARY_WINDOW[1]} s; peak = most negative value in "
             f"{ERDS_SEARCH_WINDOW[0]}-{ERDS_SEARCH_WINDOW[1]} s (percent change from the {ERDS_BASELINE} s reference).", "",
             "| Class | Channel | Band | Side | Window mean (%) | Peak ERD (%) | Peak latency (s) |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['class']} | {row['channel']} | {row['band']} | {row['side']} | "
                     f"{row['window_mean_pct']:.1f} | {row['peak_erd_pct']:.1f} | {row['peak_latency_s']:.2f} |")
    return lines + [""]


def _plot_erds_curves(curves, output_path: Path, *, title: str, sem=None) -> None:
    import matplotlib.pyplot as plt

    times = curves[(-1, "times", "times")]
    channels = sorted({key[1] for key in curves if key[0] >= 0})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, sharey=True)
    for row_idx, class_id in enumerate((0, 1)):
        for col_idx, channel_name in enumerate(channels[:2]):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
            ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle=":")
            ax.axvspan(*ERDS_BASELINE, color="#999999", alpha=0.12, linewidth=0)
            for band, (label, color) in BAND_STYLE.items():
                key = (class_id, channel_name, band)
                ax.plot(times, curves[key], color=color, linewidth=2.2, label=label)
                if sem is not None:
                    ax.fill_between(times, curves[key] - sem[key], curves[key] + sem[key], color=color, alpha=0.18)
            ax.set_title(f"{CLASS_TITLES[class_id]}\n{channel_name}", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power change (%)")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.set_xlim(*ERDS_DISPLAY)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, y=1.03, fontsize=14)
    fig.subplots_adjust(top=0.82, hspace=0.35, wspace=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_curves_csv(path: Path, curves, sem=None) -> None:
    times = curves[(-1, "times", "times")]
    header = ["time_s", "class", "channel", "band", "percent_change"] + (["sem"] if sem is not None else [])
    rows = []
    for key, curve in curves.items():
        if key[0] < 0:
            continue
        for index, time_value in enumerate(times):
            row = [float(time_value), INDEX_TO_LABEL[key[0]], key[1], key[2], float(curve[index])]
            if sem is not None:
                row.append(float(sem[key][index]))
            rows.append(row)
    write_csv(path, header, rows)


def export_sensorimotor_erds(bundle, output_dir: Path) -> dict[str, str]:
    """Figure, curves (CSV), and summary (CSV + Markdown) of one subject's mu/beta ERD/ERS."""

    subject_id = bundle.metadata.subject_id
    curves = _collect_sensorimotor_curves(bundle)
    figure = output_dir / f"subject_{subject_id}_sensorimotor_erds.png"
    _plot_erds_curves(curves, figure, title=f"Subject {subject_id} Sensorimotor ERD/ERS Summary")
    curves_csv = output_dir / f"subject_{subject_id}_erds_curves.csv"
    _write_curves_csv(curves_csv, curves)
    rows = erds_summary_rows(curves, f"S{subject_id}")
    summary_csv = output_dir / f"subject_{subject_id}_erds_summary.csv"
    write_csv(summary_csv, SUMMARY_COLUMNS, [[row[c] for c in SUMMARY_COLUMNS] for row in rows])
    summary_md = output_dir / f"subject_{subject_id}_erds_summary.md"
    write_text(summary_md, "\n".join(_summary_markdown(rows, f"Subject {subject_id} sensorimotor ERD/ERS")))
    return {"sensorimotor_erds": str(figure), "erds_curves_csv": str(curves_csv),
            "erds_summary_csv": str(summary_csv), "erds_summary_md": str(summary_md)}


def _lateralization_tests(per_subject_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Contralateral vs ipsilateral window-mean ERD/ERS, paired over subjects, Holm across class x band."""

    tests = []
    for class_id, class_name in sorted(INDEX_TO_LABEL.items()):
        for band in BAND_STYLE:
            contra = {r["unit"]: r["window_mean_pct"] for r in per_subject_rows
                      if r["class"] == class_name and r["band"] == band and r["side"] == "contralateral"}
            ipsi = {r["unit"]: r["window_mean_pct"] for r in per_subject_rows
                    if r["class"] == class_name and r["band"] == band and r["side"] == "ipsilateral"}
            units = sorted(set(contra) & set(ipsi))
            if not units:
                continue
            result = paired_permutation_test(np.array([contra[u] for u in units]), np.array([ipsi[u] for u in units]))
            tests.append({"class": class_name, "band": band, **result})
    for test, p_holm in zip(tests, holm_correction([t["p_value"] for t in tests]), strict=True):
        test["p_holm"] = float(p_holm)
    return tests


def export_grand_average_sensorimotor_erds(
    *,
    subject_ids: tuple[int, ...],
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
) -> dict[str, object]:
    """Grand-average ERD/ERS figure plus exact per-subject and grand-average summaries."""

    curve_maps = [_collect_sensorimotor_curves(_build_eda_bundle(subject_id, config=config)) for subject_id in subject_ids]
    output_path = ensure_directory(output_dir)
    times = curve_maps[0][(-1, "times", "times")]
    keys = [key for key in curve_maps[0] if key[0] >= 0]
    n = len(curve_maps)
    mean_curves = {key: np.mean([curves[key] for curves in curve_maps], axis=0) for key in keys}
    sem_curves = {
        key: (np.std([curves[key] for curves in curve_maps], axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(times)
        for key in keys
    }
    mean_curves[(-1, "times", "times")] = times

    figure_path = output_path / "grand_average_sensorimotor_erds.png"
    _plot_erds_curves(mean_curves, figure_path, title=f"Grand-average Sensorimotor ERD/ERS (n = {n}, band = SEM)", sem=sem_curves)
    _write_curves_csv(output_path / "grand_average_erds_curves.csv", mean_curves, sem=sem_curves)

    per_subject_rows = [row for subject_id, curves in zip(subject_ids, curve_maps, strict=True)
                        for row in erds_summary_rows(curves, f"S{subject_id}")]
    write_csv(output_path / "grand_average_erds_per_subject.csv", SUMMARY_COLUMNS,
              [[row[c] for c in SUMMARY_COLUMNS] for row in per_subject_rows])
    grand_rows = erds_summary_rows(mean_curves, "grand_average")
    tests = _lateralization_tests(per_subject_rows)

    lines = ["# Grand-average sensorimotor ERD/ERS", "", f"Subjects: {', '.join(f'S{s}' for s in subject_ids)} (n = {n}).", ""]
    lines += _summary_markdown(grand_rows, "Grand-average curve")
    lines += ["## Per-subject window means (mean +- SEM across subjects)", "",
              "| Class | Channel | Band | Side | Mean (%) | SEM (%) |", "|---|---|---|---|---:|---:|"]
    for key in keys:
        class_name = INDEX_TO_LABEL[key[0]]
        values = np.array([r["window_mean_pct"] for r in per_subject_rows
                           if r["class"] == class_name and r["channel"] == key[1] and r["band"] == key[2]])
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        side = "contralateral" if CONTRALATERAL_CHANNEL.get(key[0]) == key[1] else "ipsilateral"
        lines.append(f"| {class_name} | {key[1]} | {key[2]} | {side} | {values.mean():.1f} | {sem:.1f} |")
    lines += ["", "## Lateralization: contralateral vs ipsilateral window mean", "",
              "Exact paired sign-flip test across subjects; Holm adjustment across the class x band tests.", "",
              "| Class | Band | n | Mean diff contra-ipsi (%) | Subjects contra < ipsi | p (exact) | p (Holm) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for test in tests:
        lines.append(f"| {test['class']} | {test['band']} | {test['n_pairs']} | {test['mean_difference']:.1f} | "
                     f"{test['wins_y']}/{test['n_pairs']} | {test['p_value']:.4f} | {test['p_holm']:.4f} |")
    summary_md = output_path / "grand_average_erds_summary.md"
    write_text(summary_md, "\n".join(lines) + "\n")

    summary = {
        "subjects": list(subject_ids),
        "n_subjects": n,
        "artifact": str(figure_path),
        "curves_csv": str(output_path / "grand_average_erds_curves.csv"),
        "per_subject_csv": str(output_path / "grand_average_erds_per_subject.csv"),
        "summary_md": str(summary_md),
        "lateralization_tests": tests,
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
    device: str | None = None,
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
    topomap_csv = output_path / f"subject_{subject_id}_topomap_values.csv"
    riemann_3d_path = output_path / f"subject_{subject_id}_riemann_3d.png"
    csp_lda_path = output_path / f"subject_{subject_id}_csp_lda_distribution.png"
    csp_projection_path = output_path / f"subject_{subject_id}_csp_projection.png"
    riemann_lda_path = output_path / f"subject_{subject_id}_riemann_lda_distribution.png"
    eegnet_saliency_path = output_path / f"subject_{subject_id}_eegnet_saliency_topomap.png"

    _plot_psd(bundle, psd_path)
    _plot_pca(bundle, pca_path)
    _plot_tsne(bundle, tsne_path)
    _plot_topomap(bundle, topomap_path, values_csv=topomap_csv)
    _plot_erds(bundle, erds_path)
    _plot_classical_patterns(bundle, patterns_path)
    erds_outputs = export_sensorimotor_erds(bundle, output_path)
    _plot_riemann_3d(bundle, riemann_3d_path)
    _plot_riemann_lda_distribution(bundle, riemann_lda_path)
    _plot_csp_lda_distribution(bundle, csp_lda_path)
    _plot_csp_feature_projection(bundle, csp_projection_path)
    _plot_eegnet_saliency_topomap(bundle, eegnet_saliency_path, device=device)

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
            "topomap_values_csv": str(topomap_csv),
            **erds_outputs,
            "riemann_3d": str(riemann_3d_path),
            "riemann_lda_distribution": str(riemann_lda_path),
            "csp_lda_distribution": str(csp_lda_path),
            "csp_projection": str(csp_projection_path),
            "eegnet_saliency_topomap": str(eegnet_saliency_path),
        },
    }
    write_json(output_path / f"subject_{subject_id}_eda_summary.json", summary)
    return read_json(output_path / f"subject_{subject_id}_eda_summary.json")
