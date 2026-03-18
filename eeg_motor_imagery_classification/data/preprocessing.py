"""Preprocessing helpers shared across models and protocols."""

from __future__ import annotations

from collections.abc import Iterable

import mne

from eeg_motor_imagery_classification.config import PreprocessingConfig
from eeg_motor_imagery_classification.data.loaders import SubjectSessions


def _pick_relevant_channels(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    picked = raw.copy()
    if config.eeg_channels_only:
        picks = mne.pick_types(picked.info, eeg=True, eog=False, stim=True, exclude="bads")
    else:
        picks = mne.pick_types(picked.info, eeg=True, eog=True, stim=True, exclude="bads")
    picked.pick(picks)
    return picked


def _apply_ica_if_requested(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    if not config.apply_ica:
        return raw

    raw_for_ica = raw.copy().filter(
        l_freq=config.ica_l_freq,
        h_freq=None,
        fir_design="firwin",
        verbose=False,
    )
    ica = mne.preprocessing.ICA(
        n_components=config.ica_components,
        random_state=config.random_state,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_for_ica, verbose=False)

    eog_names = [ch for ch in ("EOG1", "EOG2", "EOG3") if ch in raw.ch_names]
    if eog_names:
        eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name=eog_names, verbose=False)
        ica.exclude = eog_indices
        cleaned = raw.copy()
        ica.apply(cleaned, verbose=False)
        return cleaned
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, config: PreprocessingConfig | None = None) -> mne.io.BaseRaw:
    """Apply the project's canonical preprocessing to one raw run."""

    cfg = config or PreprocessingConfig()
    processed = _pick_relevant_channels(raw, cfg)
    processed.notch_filter(cfg.notch_freq, verbose=False)
    processed.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose=False)

    if cfg.apply_average_reference:
        processed.set_eeg_reference("average", projection=False, verbose=False)

    processed = _apply_ica_if_requested(processed, cfg)
    processed.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return processed


def preprocess_runs(sessions: SubjectSessions, config: PreprocessingConfig | None = None) -> list[tuple[str, str, mne.io.BaseRaw]]:
    """Flatten and preprocess every run for one subject."""

    cfg = config or PreprocessingConfig()
    preprocessed_runs: list[tuple[str, str, mne.io.BaseRaw]] = []
    for session_name, runs in sessions.items():
        for run_name, raw in runs.items():
            preprocessed_runs.append((session_name, run_name, preprocess_raw(raw, cfg)))
    return preprocessed_runs


def iter_preprocessed_runs(
    sessions: SubjectSessions,
    config: PreprocessingConfig | None = None,
) -> Iterable[tuple[str, str, mne.io.BaseRaw]]:
    """Yield preprocessed runs lazily when batch materialization is not needed."""

    for item in preprocess_runs(sessions, config=config):
        yield item

