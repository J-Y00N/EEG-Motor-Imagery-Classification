"""Filter-bank CSP feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class FrequencyBand:
    """One band-pass range used in filter-bank processing."""

    low: float
    high: float


class BandPassFilter(BaseEstimator, TransformerMixin):
    """Apply one MNE band-pass filter to epoched EEG arrays."""

    def __init__(self, sfreq: float, l_freq: float, h_freq: float) -> None:
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BandPassFilter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return mne.filter.filter_data(
            X,
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            verbose=False,
        )


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Apply CSP to multiple fixed bands and concatenate their features."""

    def __init__(
        self,
        sfreq: float,
        bands: tuple[tuple[float, float], ...],
        n_components: int = 4,
        cov_est: str = "epoch",
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.cov_est = cov_est
        self.filters_: list[BandPassFilter] = []
        self.csp_models_: list[CSP] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSP":
        self.filters_ = []
        self.csp_models_ = []
        for low, high in self.bands:
            band_filter = BandPassFilter(self.sfreq, low, high)
            X_band = band_filter.transform(X)
            csp = CSP(
                n_components=self.n_components,
                reg=None,
                log=True,
                norm_trace=False,
                cov_est=self.cov_est,
            )
            csp.fit(X_band, y)
            self.filters_.append(band_filter)
            self.csp_models_.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for band_filter, csp in zip(self.filters_, self.csp_models_, strict=True):
            X_band = band_filter.transform(X)
            features.append(csp.transform(X_band))
        return np.concatenate(features, axis=1)
