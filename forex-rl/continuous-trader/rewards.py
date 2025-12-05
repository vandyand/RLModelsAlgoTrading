from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np


@dataclass
class RewardContext:
    """Carries contextual information for reward computation.

    This struct is intentionally minimal and generic so custom reward functions can
    introspect what they need. Extend as needed later.
    """
    # Next-step/next-day returns of the traded instrument (scalar)
    inst_return: float
    # Optional realized pnl fraction for a closed trade (per episode), if available
    trade_pnl_frac: Optional[float] = None
    # Optional dictionary for arbitrary extras (e.g., costs, spread, drawdown)
    extras: Optional[Dict[str, float]] = None


# Type alias for a reward function
RewardFn = Callable[[RewardContext], float]


def sortino_like_daily(ctx: RewardContext, target: float = 0.0, dn_eps: float = 1e-8) -> float:
    """Sortino-like reward on a single-period outcome.

    r = ctx.inst_return (e.g., next-day log return contributed by the policy)
    reward = (r - target) / sqrt(mean_squared_negative_deviation + eps)
    For a single step, approximate downside as max(0, target - r).
    """
    r = float(ctx.inst_return)
    downside = max(0.0, target - r)
    denom = np.sqrt(downside * downside + dn_eps)
    return (r - target) / denom


def sharpe_like_daily(ctx: RewardContext, target: float = 0.0, sd_eps: float = 1e-8) -> float:
    r = float(ctx.inst_return)
    sd = max(abs(r - target), sd_eps)
    return (r - target) / sd


def raw_pnl(ctx: RewardContext, scale: float = 1.0) -> float:
    if ctx.trade_pnl_frac is None:
        return float(ctx.inst_return) * scale
    return float(ctx.trade_pnl_frac) * scale


class RewardRegistry:
    """Simple registry to select reward shaping by name.

    Usage:
      reg = RewardRegistry()
      fn = reg.get("sortino")
      r = fn(ctx)
    """
    def __init__(self) -> None:
        self._fns: Dict[str, RewardFn] = {
            "sortino": sortino_like_daily,
            "sharpe": sharpe_like_daily,
            "pnl": raw_pnl,
        }

    def get(self, name: str) -> RewardFn:
        key = (name or "").strip().lower()
        # Default to raw PnL if missing
        return self._fns.get(key, raw_pnl)

    def register(self, name: str, fn: RewardFn) -> None:
        self._fns[name.strip().lower()] = fn
