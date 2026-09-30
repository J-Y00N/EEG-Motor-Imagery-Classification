"""CSP-adjacent feature transforms used in classical pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogVarianceVectorizer(BaseEstimator, TransformerMixin):
    """Convert epoched signals into log-variance feature vectors."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogVarianceVectorizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Signals are in volts (MOABB converts uV -> V), so band-limited variances are
        # typically ~1e-11 V^2. A fixed floor such as 1e-10 clips every channel to the same
        # constant, so only guard against exact zeros here.
        variances = np.var(np.asarray(X, dtype=np.float64), axis=-1)
        return np.log(np.maximum(variances, np.finfo(np.float64).tiny))

