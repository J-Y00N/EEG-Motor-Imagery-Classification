"""Classical baselines rebuilt on top of the canonical data pipeline."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_motor_imagery_classification.features import FilterBankCSP, LogVarianceVectorizer

DEFAULT_FBCSP_BANDS: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)


def _build_lda() -> LinearDiscriminantAnalysis:
    """Create a numerically safer LDA for small or degenerate folds."""

    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


class SafeFeatureSelector(BaseEstimator, TransformerMixin):
    """Select top-k features while guarding against degenerate score arrays."""

    def __init__(self, k: int = 8) -> None:
        self.k = k
        self.indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafeFeatureSelector":
        scores, _ = f_classif(X, y)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        scores[~np.isfinite(scores)] = -np.inf

        n_features = X.shape[1]
        if n_features == 0:
            raise ValueError("SafeFeatureSelector received zero features.")

        k = max(1, min(self.k, n_features))
        order = np.argsort(scores)[::-1]
        self.indices_ = np.sort(order[:k])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.indices_ is None:
            raise ValueError("SafeFeatureSelector must be fit before transform.")
        return X[:, self.indices_]


def build_raw_power_pipeline() -> Pipeline:
    """Baseline using channel-wise log variance and LDA."""

    return Pipeline(
        steps=[
            ("logvar", LogVarianceVectorizer()),
            ("scaler", StandardScaler()),
            ("lda", _build_lda()),
        ]
    )


def build_csp_pipeline(n_components: int = 4) -> Pipeline:
    """Baseline CSP + LDA pipeline."""

    return Pipeline(
        steps=[
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg=None,
                    log=True,
                    norm_trace=False,
                    cov_est="epoch",
                ),
            ),
            ("lda", _build_lda()),
        ]
    )


def build_fbcsp_pipeline(
    sfreq: float,
    *,
    bands: tuple[tuple[float, float], ...] = DEFAULT_FBCSP_BANDS,
    n_components: int = 4,
    k_best: int = 8,
) -> Pipeline:
    """Filter-bank CSP + train-fold feature selection + LDA."""

    return Pipeline(
        steps=[
            (
                "fbcsp",
                FilterBankCSP(
                    sfreq=sfreq,
                    bands=bands,
                    n_components=n_components,
                ),
            ),
            ("select", SafeFeatureSelector(k=k_best)),
            ("lda", _build_lda()),
        ]
    )
