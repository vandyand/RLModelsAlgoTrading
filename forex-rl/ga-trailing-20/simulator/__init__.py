"""Trailing stop simulator package."""
from .config import TrailingConfig
from .costs import CostModel, CostModelConfig
from .engine import TrailingStopSimulator, SimulatorResult, TradeRecord

__all__ = [
    "TrailingConfig",
    "CostModel",
    "CostModelConfig",
    "TrailingStopSimulator",
    "SimulatorResult",
    "TradeRecord",
]
