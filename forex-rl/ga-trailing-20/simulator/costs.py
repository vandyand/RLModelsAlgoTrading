"""Transaction cost and slippage utilities."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional


def _default_spread_table() -> Dict[str, float]:
    return {
        "EUR_USD": 0.5,
        "USD_JPY": 0.7,
        "GBP_USD": 0.8,
    }


def pip_size(instrument: str) -> float:
    inst = instrument.upper()
    if inst.endswith("JPY"):
        return 0.01
    return 0.0001


@dataclass
class CostModelConfig:
    spread_mode: str = "static"  # static | dynamic (future)
    spread_table: Dict[str, float] = field(default_factory=_default_spread_table)
    default_spread_pips: float = 1.0
    spread_multiplier: float = 1.0
    commission_per_million: float = 0.0
    slippage_mode: str = "deterministic"  # deterministic | vol | stochastic
    slippage_params: Dict[str, float] = field(default_factory=lambda: {"pips": 0.2})
    financing_rate_bps: float = 0.0  # applied per day if positions held overnight


class CostModel:
    def __init__(self, cfg: CostModelConfig) -> None:
        self.cfg = cfg

    def _spread_pips(self, instrument: str) -> float:
        spread = self.cfg.spread_table.get(instrument.upper(), self.cfg.default_spread_pips)
        return float(spread) * float(self.cfg.spread_multiplier)

    def _commission(self, units: float) -> float:
        return abs(units) / 1_000_000.0 * float(self.cfg.commission_per_million)

    def entry_cost(self, instrument: str, price: float, units: float, bar_info: Optional[Dict[str, float]] = None) -> float:
        spread_cost = self._spread_pips(instrument) * pip_size(instrument) * abs(units)
        return spread_cost + self._commission(units)

    def exit_cost(self, instrument: str, price: float, units: float, bar_info: Optional[Dict[str, float]] = None) -> float:
        spread_cost = self._spread_pips(instrument) * pip_size(instrument) * abs(units)
        return spread_cost + self._commission(units)

    def apply_slippage(
        self,
        instrument: str,
        intended_price: float,
        direction: int,
        bar_info: Optional[Dict[str, float]] = None,
    ) -> float:
        signed = 1 if direction >= 0 else -1
        slip_pips = self._slippage_pips(instrument, bar_info)
        delta = slip_pips * pip_size(instrument) * signed
        return max(0.0, intended_price + delta)

    def _slippage_pips(self, instrument: str, bar_info: Optional[Dict[str, float]]) -> float:
        mode = self.cfg.slippage_mode
        params = self.cfg.slippage_params or {}
        if mode == "vol":
            atr = 0.0
            if bar_info:
                atr = float(bar_info.get("atr", 0.0))
                if atr == 0.0:
                    high = bar_info.get("high")
                    low = bar_info.get("low")
                    if high is not None and low is not None:
                        atr = abs(float(high) - float(low))
            atr = max(atr, pip_size(instrument))
            mult = float(params.get("atr_mult", 0.1))
            min_pips = float(params.get("min_pips", 0.0))
            return max(min_pips, (atr / pip_size(instrument)) * mult)
        if mode == "stochastic":
            mean = float(params.get("mean_pips", 0.0))
            std = float(params.get("std_pips", 0.2))
            limit = float(params.get("max_abs_pips", 3.0))
            sample = random.gauss(mean, std)
            return max(-limit, min(limit, sample))
        # deterministic default
        return float(params.get("pips", 0.1))

    def financing_cost(self, instrument: str, units: float, days_held: float) -> float:
        rate = float(self.cfg.financing_rate_bps) / 10_000.0
        return abs(units) * rate * days_held
