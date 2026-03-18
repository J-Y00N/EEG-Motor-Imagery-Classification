"""Project configuration objects for EEG motor imagery experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Canonical preprocessing choices shared across experiments."""

    notch_freq: float = 50.0
    l_freq: float = 8.0
    h_freq: float = 32.0
    tmin: float = 0.0
    tmax: float = 4.0
    apply_average_reference: bool = False
    apply_ica: bool = False
    ica_l_freq: float = 1.0
    ica_components: int = 15
    random_state: int = 42
    event_names: tuple[str, str] = ("left_hand", "right_hand")
    eeg_channels_only: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset source and subject-selection options."""

    dataset_name: str = "BNCI2014_001"
    subjects: tuple[int, ...] = tuple(range(1, 10))
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

