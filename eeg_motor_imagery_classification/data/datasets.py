"""Minimal dataset helpers for array-based EEG experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional until training is wired in
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback stub so imports still work before torch is installed."""

        pass


@dataclass(frozen=True)
class ChannelwiseScaler:
    """Channel-wise standardization stats fit on training data only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def fit_channelwise_scaler(X: np.ndarray, eps: float = 1e-6) -> ChannelwiseScaler:
    """Fit one scaler shared across trials and time for each channel."""

    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ChannelwiseScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


class EpochDataset(Dataset):
    """Torch-friendly dataset with leakage-safe channel scaling."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        scaler: ChannelwiseScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        if fit_scaler and scaler is not None:
            raise ValueError("Use either fit_scaler=True or provide an existing scaler, not both.")

        self.scaler = fit_channelwise_scaler(X) if fit_scaler else scaler
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        self.X = X_scaled.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        sample = self.X[index]
        label = self.y[index]
        if torch is None:
            return sample, label
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)

