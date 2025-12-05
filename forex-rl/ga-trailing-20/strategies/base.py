"""Core interfaces and shared configs for trailing-stop entrance strategies."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np


class DatasetSplit(Protocol):
    """Minimal protocol for iterables that yield training/eval batches."""

    def __iter__(self) -> Any:  # pragma: no cover - protocol stub
        ...


class TrailingStopSimulator(Protocol):
    """Protocol for the per-bar trailing-stop simulator/backtester."""

    def evaluate(self, strategy: "Strategy", split: DatasetSplit) -> Dict[str, Any]:  # pragma: no cover - stub
        ...


@dataclass
class RegularizationConfig:
    """Common L1/L2 knobs shared across strategies."""

    l1: float = 0.0
    l2: float = 0.0
    complexity_penalty: float = 0.0  # e.g., tree depth or param-count penalty weight


@dataclass
class CrossValidationConfig:
    """Rolling-origin cross validation parameters for time-series splits."""

    n_splits: int = 5
    min_train_bars: int = 5_000
    val_bars: int = 1_000
    gap_bars: int = 0  # optional gap between train/val to reduce leakage
    shuffle: bool = False  # default to chronological splits; override only for diagnostics


@dataclass
class StrategyArtifact:
    """Metadata persisted alongside learned parameters."""

    name: str
    version: str
    feature_names: Optional[list[str]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Strategy(abc.ABC):
    """Abstract base class for entrance decision-makers."""

    def __init__(
        self,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
    ) -> None:
        self.regularization = regularization or RegularizationConfig()
        self.cv = cv or CrossValidationConfig()
        self.artifact: Optional[StrategyArtifact] = None

    @abc.abstractmethod
    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        """Train the strategy using provided data splits and simulator."""

    @abc.abstractmethod
    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        """Execute rolling-origin CV and return aggregate metrics."""

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return per-instrument entrance logits/probabilities for a batch."""

    @abc.abstractmethod
    def save(self, path: str) -> StrategyArtifact:
        """Persist learned params + metadata; returns stored artifact."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "Strategy":
        """Hydrate strategy from disk."""

    def set_artifact(self, artifact: StrategyArtifact) -> None:
        """Helper for subclasses to register artifact metadata."""

        self.artifact = artifact
