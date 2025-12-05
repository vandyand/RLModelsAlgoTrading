"""Simple baseline threshold strategy.

This strategy does not train; it maps a single feature per instrument to
{-1, 0, +1} signals using fixed thresholds.
"""
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
class BaselineThresholdConfig:
    feature_index: int = 0  # index into feature vector
    enter_long: float = 0.5
    exit_long: float = 0.0
    enter_short: float = -0.5
    exit_short: float = 0.0
    version: str = "v0"


class BaselineThresholdStrategy(Strategy):
    """Deterministic threshold-based entrance strategy.

    This is mainly for end-to-end testing, not production use.
    """

    def __init__(
        self,
        config: Optional[BaselineThresholdConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or BaselineThresholdConfig()

    # ------------------------------------------------------------------
    # Strategy API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        # No learning; could be extended to scan thresholds in future.
        return None

    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        # Single-pass evaluation using current thresholds.
        result = simulator.evaluate(self, data)
        return {"metrics": getattr(result, "metrics", None)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            feats = features
        else:
            feats = features[0]
        idx = max(0, min(int(self.config.feature_index), feats.size - 1)) if feats.size > 0 else 0
        x = feats[idx] if feats.size > 0 else 0.0
        # Map scalar feature to entrance signal logit
        if x > self.config.enter_long:
            return np.array([1.0], dtype=float)
        if x < self.config.enter_short:
            return np.array([-1.0], dtype=float)
        return np.array([0.0], dtype=float)

    def save(self, path: str) -> StrategyArtifact:
        art = StrategyArtifact(
            name="baseline_threshold",
            version=self.config.version,
            feature_names=None,
            scaler_state=None,
            extra={
                "feature_index": int(self.config.feature_index),
                "enter_long": float(self.config.enter_long),
                "exit_long": float(self.config.exit_long),
                "enter_short": float(self.config.enter_short),
                "exit_short": float(self.config.exit_short),
            },
        )
        self.set_artifact(art)
        return art

    @classmethod
    def load(cls, path: str) -> "BaselineThresholdStrategy":
        # For now, ignore path and return default-config strategy.
        return cls()
