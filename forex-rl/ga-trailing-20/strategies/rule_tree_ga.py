"""Genetic-programmed rule-tree strategy scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .base import (
    CrossValidationConfig,
    DatasetSplit,
    RegularizationConfig,
    Strategy,
    StrategyArtifact,
    TrailingStopSimulator,
)


@dataclass
class RuleTreeGAConfig:
    """Hyper-parameters for rule-tree evolution."""

    population: int = 128
    generations: int = 50
    max_depth: int = 5
    max_conditions: int = 32
    mutation_prob: float = 0.2
    crossover_prob: float = 0.6
    elitism_frac: float = 0.1
    threshold_init_range: tuple[float, float] = (-1.0, 1.0)


class RuleTreeGAStrategy(Strategy):
    """Evolves AND/OR rule trees on normalized features."""

    def __init__(
        self,
        config: Optional[RuleTreeGAConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or RuleTreeGAConfig()
        self._best_rules: Optional[Any] = None  # placeholder until GA implemented

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        raise NotImplementedError("Rule-tree GA training not implemented yet")

    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Rule-tree GA CV not implemented yet")

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Rule-tree GA predict not implemented yet")

    def save(self, path: str) -> StrategyArtifact:
        raise NotImplementedError("Rule-tree GA save not implemented yet")

    @classmethod
    def load(cls, path: str) -> "RuleTreeGAStrategy":
        raise NotImplementedError("Rule-tree GA load not implemented yet")
