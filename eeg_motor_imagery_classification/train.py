"""Reusable training helpers for EEG deep learning experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

from eeg_motor_imagery_classification.data.datasets import EpochDataset


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters shared across EEGNet experiments."""

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str | None = None
    seed: int = 42
    deterministic: bool = True
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    min_epochs: int = 10
    min_delta: float = 1e-4


@dataclass(frozen=True)
class TrainingResult:
    """Trained model together with epoch-level optimization history."""

    model: nn.Module
    history: dict[str, object]


def default_device() -> str:
    """Pick the best available torch device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def _build_loader(dataset, config: TrainingConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def build_train_validation_datasets(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: TrainingConfig | None = None,
) -> tuple[EpochDataset, EpochDataset | None]:
    """Create leakage-safe train/validation datasets for one split."""

    cfg = config or TrainingConfig()
    validation_split = float(cfg.validation_split)
    if validation_split <= 0.0:
        return EpochDataset(X, y, fit_scaler=True), None

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or np.min(class_counts) < 2:
        return EpochDataset(X, y, fit_scaler=True), None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_split,
            random_state=cfg.seed,
            stratify=y,
        )
    except ValueError:
        return EpochDataset(X, y, fit_scaler=True), None

    train_dataset = EpochDataset(X_train, y_train, fit_scaler=True)
    val_dataset = EpochDataset(X_val, y_val, scaler=train_dataset.scaler)
    return train_dataset, val_dataset


@torch.no_grad()
def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    total_loss = 0.0
    total_samples = 0
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def fit_model(
    model: nn.Module,
    train_dataset,
    *,
    validation_dataset=None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train one torch model and return both weights and optimization history."""

    cfg = config or TrainingConfig()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device or default_device())
    train_loader = _build_loader(train_dataset, cfg, shuffle=True)
    eval_train_loader = _build_loader(train_dataset, cfg, shuffle=False)
    val_loader = _build_loader(validation_dataset, cfg, shuffle=False) if validation_dataset is not None else None

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_epoch = 0
    epochs_without_improvement = 0
    best_val_loss = float("inf")
    best_state_dict = deepcopy(model.state_dict())
    early_stopped = False

    for epoch in range(cfg.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_loss = _evaluate_loss(model, eval_train_loader, criterion, device)
        train_loss_history.append(float(train_loss))

        if val_loader is not None:
            val_loss = _evaluate_loss(model, val_loader, criterion, device)
            val_loss_history.append(float(val_loss))

            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss = float(val_loss)
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state_dict = deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            if cfg.early_stopping and (epoch + 1) >= cfg.min_epochs and epochs_without_improvement >= cfg.patience:
                early_stopped = True
                break
        else:
            best_epoch = epoch + 1
            best_state_dict = deepcopy(model.state_dict())

    if val_loader is not None:
        model.load_state_dict(best_state_dict)

    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "epochs_ran": len(train_loss_history),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "used_validation": bool(val_loader is not None),
    }
    return TrainingResult(model=model, history=history)


def train_model(
    model: nn.Module,
    train_dataset,
    *,
    validation_dataset=None,
    config: TrainingConfig | None = None,
) -> nn.Module:
    """Backward-compatible wrapper that returns only the trained model."""

    return fit_model(model, train_dataset, validation_dataset=validation_dataset, config=config).model


@torch.no_grad()
def predict_model(model: nn.Module, dataset, *, config: TrainingConfig | None = None) -> np.ndarray:
    """Run inference on one dataset and return NumPy class predictions."""

    cfg = config or TrainingConfig()
    device = torch.device(cfg.device or default_device())
    loader = _build_loader(dataset, cfg, shuffle=False)

    model = model.to(device)
    model.eval()

    predictions = []
    for inputs, _labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)
