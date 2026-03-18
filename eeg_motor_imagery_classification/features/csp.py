"""CSP-adjacent feature transforms used in classical pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogVarianceVectorizer(BaseEstimator, TransformerMixin):
    """Convert epoched signals into log-variance feature vectors."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogVarianceVectorizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        variances = np.var(X, axis=-1)
        return np.log(np.maximum(variances, 1e-10))

