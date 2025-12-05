"""Dataclasses and enums for trailing-stop simulation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrailingMode(str, Enum):
    ATR = "atr"
    PIP = "pip"
    TICK_VOL = "tick_vol"


class GapFillMode(str, Enum):
    BAR_OPEN = "bar_open"
    STOP_PRICE = "stop_price"


@dataclass
class TrailingConfig:
    mode: TrailingMode = TrailingMode.ATR
    atr_period: int = 20
    atr_multiplier: float = 1.0
    pip_distance: float = 25.0
    tick_vol_multiplier: float = 3.0
    min_distance_pips: float = 5.0
    min_step_pips: float = 1.5
    max_trailing_pips: float | None = None
    enforce_quantization: bool = False
    price_precision: int = 5
    gap_mode: GapFillMode = GapFillMode.BAR_OPEN
    allow_multiple_entries: bool = False
    # Optional cap on the number of instruments with open positions at any
    # given time. If None, no explicit cap is enforced beyond the strategy's
    # own gating logic.
    max_open_instruments: int | None = None

    def canonical_mode(self) -> TrailingMode:
        try:
            return TrailingMode(self.mode)
        except Exception:
            return TrailingMode.ATR
