"""Strategy factory and registry for ga-trailing-20."""
from __future__ import annotations

from typing import Dict, Type

from .base import Strategy
from .gradient_nn import GradientNNStrategy
from .multihead_gated_nn import MultiHeadGatedNNStrategy
from .neuro_ga import NeuroGAStrategy
from .rule_tree_ga import RuleTreeGAStrategy
from .baseline_threshold import BaselineThresholdStrategy
from .ga_policy import GAPolicyStrategy


STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "baseline_threshold": BaselineThresholdStrategy,
    "rule_tree_ga": RuleTreeGAStrategy,
    "neuro_ga": NeuroGAStrategy,
    "gradient_nn": GradientNNStrategy,
    "multihead_gated_nn": MultiHeadGatedNNStrategy,
    "ga_policy": GAPolicyStrategy,
}


def create_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate a registered strategy by name."""

    key = name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}")
    cls = STRATEGY_REGISTRY[key]
    return cls(**kwargs)
