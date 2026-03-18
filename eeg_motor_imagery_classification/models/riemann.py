"""Riemannian tangent-space baselines for EEG classification."""

from __future__ import annotations

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_lda() -> LinearDiscriminantAnalysis:
    """Create a stable LDA backend for tangent-space features."""

    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def build_riemann_tangent_pipeline(cov_estimator: str = "oas") -> Pipeline:
    """Build a covariance -> tangent space -> LDA baseline."""

    return Pipeline(
        steps=[
            ("covariances", Covariances(estimator=cov_estimator)),
            ("tangent", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("lda", _build_lda()),
        ]
    )

