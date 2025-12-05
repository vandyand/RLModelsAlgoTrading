#!/usr/bin/env python3
"""Sanity-check the trailing simulator and PnL environment.

This script runs a few trivial policies on a single instrument and reports
P&L metrics under zero cost and realistic cost assumptions.

Policies:
  - AlwaysFlatStrategy: never trades.
  - AlwaysLongStrategy: always long 1 unit when flat.
  - RandomDirStrategy: random long/short 1 unit each bar when flat.

We run each policy twice:
  1) With zero costs (spread=0, slippage=0, commission=0) to check for
     structural PnL bias.
  2) With default costs from CostModelConfig to see cost impact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Sequence

import numpy as np

# Add ga-trailing-20 root to import path when run from repo
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]


class StrategyProtocol:
    def predict(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError


@dataclass
class AlwaysFlatStrategy(StrategyProtocol):
    instruments: Sequence[str]

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Return all zeros so no entries are taken.
        return np.zeros(len(self.instruments), dtype=np.float32)


@dataclass
class AlwaysLongStrategy(StrategyProtocol):
    instruments: Sequence[str]

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Emit 4N layout with gate=1, dir=+1, pos_frac=1, trail_frac=0.5.
        N = len(self.instruments)
        enter = np.ones(N, dtype=np.float32)
        dir_scores = np.ones(N, dtype=np.float32)  # interpreted as long
        pos = np.ones(N, dtype=np.float32)
        trail = np.full(N, 0.5, dtype=np.float32)
        return np.concatenate([enter, dir_scores, pos, trail], axis=0)


@dataclass
class AlwaysShortStrategy(StrategyProtocol):
    instruments: Sequence[str]

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Emit 4N layout with gate=1, dir=-1, pos_frac=1, trail_frac=0.5.
        N = len(self.instruments)
        enter = np.ones(N, dtype=np.float32)
        dir_scores = -np.ones(N, dtype=np.float32)  # interpreted as short
        pos = np.ones(N, dtype=np.float32)
        trail = np.full(N, 0.5, dtype=np.float32)
        return np.concatenate([enter, dir_scores, pos, trail], axis=0)

@dataclass
class RandomDirStrategy(StrategyProtocol):
    instruments: Sequence[str]
    seed: int = 123

    def __post_init__(self) -> None:
        self.rng = np.random.RandomState(self.seed)

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Random long/short (50/50) with gate always open and full size.
        N = len(self.instruments)
        enter = np.ones(N, dtype=np.float32)
        # Random +/-1 signals
        rand = self.rng.rand(N)
        dir_scores = np.where(rand < 0.5, -1.0, 1.0).astype(np.float32)
        pos = np.ones(N, dtype=np.float32)
        trail = np.full(N, 0.5, dtype=np.float32)
        return np.concatenate([enter, dir_scores, pos, trail], axis=0)


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def run_once(
    name: str,
    strat: StrategyProtocol,
    loader: DatasetLoader,
    train_range: tuple[str, str],
    cost_cfg: CostModelConfig,
    trailing_cfg: TrailingConfig,
) -> Dict[str, Any]:
    split_cfg = SplitConfig(train=train_range)
    train_split, _, _ = loader.split_by_dates(split_cfg)

    # Downsample to keep it cheap.
    records = []
    for idx, rec in enumerate(train_split):
        if idx >= 800:
            break
        records.append(rec)

    class SmallSplit:
        def __iter__(self) -> Iterable[Dict[str, Any]]:  # pragma: no cover - trivial
            return iter(records)

    split_small: Iterable[Dict[str, Any]] = SmallSplit()

    sim = TrailingStopSimulator(trailing_cfg, CostModel(cost_cfg))
    result = sim.evaluate(strat, split_small, record_equity=False, return_trades=True)
    m = result.metrics

    return {
        "strategy": name,
        "cum_return": float(m.cum_return),
        "sharpe": float(m.sharpe),
        "profit_factor": float(m.profit_factor),
        "trades": len(result.trades),
        "per_instrument_tim_mean": float(m.per_instrument_tim_mean),
        "per_instrument_tim_std": float(m.per_instrument_tim_std),
    }


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Debug PnL environment with trivial policies")
    ap.add_argument("--instrument", default="EUR_USD")
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    ap.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    ap.add_argument("--train", default="2025-01-01:2025-07-31")
    ap.add_argument("--max-rows", type=int, default=15000)
    ap.add_argument(
        "--trail-mode",
        default="atr",
        choices=["atr", "pip"],
        help="Trailing mode for debug runs (atr or pip).",
    )
    ap.add_argument(
        "--max-trail-pips",
        type=float,
        default=20.0,
        help="Max trailing distance in pips (for pip mode or as cap).",
    )
    args = ap.parse_args()

    inst = args.instrument.upper()
    instruments = [inst]

    loader_cfg = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity="M5",
        aux_granularities=("D",),
        normalize=True,
        feature_subset=None,
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
    )
    loader = DatasetLoader(loader_cfg)

    train_range = _parse_range(args.train)

    # Zero-cost config
    zero_cost_cfg = CostModelConfig(
        spread_mode="static",
        spread_table={},
        default_spread_pips=0.0,
        spread_multiplier=0.0,
        commission_per_million=0.0,
        slippage_mode="deterministic",
        slippage_params={"pips": 0.0},
        financing_rate_bps=0.0,
    )

    # Default cost config (as used elsewhere)
    default_cost_cfg = CostModelConfig()

    strats = [
        ("always_flat", AlwaysFlatStrategy(instruments)),
        ("always_long", AlwaysLongStrategy(instruments)),
        ("always_short", AlwaysShortStrategy(instruments)),
        ("random_dir", RandomDirStrategy(instruments, seed=123)),
    ]

    # Trailing configuration: allow experimenting with larger pip distances.
    if args.trail_mode.lower() == "pip":
        trail_cfg = TrailingConfig(
            mode="pip",
            pip_distance=float(args.max_trail_pips),
            min_distance_pips=float(args.max_trail_pips),
            max_trailing_pips=float(args.max_trail_pips),
        )
    else:
        trail_cfg = TrailingConfig(max_trailing_pips=float(args.max_trail_pips))

    print({"phase": "zero_cost", "instrument": inst, "trail_cfg": trail_cfg})
    for name, s in strats:
        res = run_once(name, s, loader, train_range, zero_cost_cfg, trail_cfg)
        print(res)

    # Random-dir distribution under zero cost
    rand_results = []
    for seed in range(10):
        s = RandomDirStrategy(instruments, seed=seed)
        rand_results.append(run_once(f"random_dir_seed{seed}", s, loader, train_range, zero_cost_cfg, trail_cfg))
    if rand_results:
        mean_cum = float(np.mean([r["cum_return"] for r in rand_results]))
        mean_pf = float(np.mean([r["profit_factor"] for r in rand_results]))
        print({"phase": "zero_cost_random_dir_summary", "instrument": inst, "mean_cum_return": mean_cum, "mean_profit_factor": mean_pf})

    print({"phase": "default_cost", "instrument": inst, "trail_cfg": trail_cfg})
    for name, s in strats:
        res = run_once(name, s, loader, train_range, default_cost_cfg, trail_cfg)
        print(res)

    # Random-dir distribution under default cost
    rand_results = []
    for seed in range(10):
        s = RandomDirStrategy(instruments, seed=seed)
        rand_results.append(run_once(f"random_dir_seed{seed}", s, loader, train_range, default_cost_cfg, trail_cfg))
    if rand_results:
        mean_cum = float(np.mean([r["cum_return"] for r in rand_results]))
        mean_pf = float(np.mean([r["profit_factor"] for r in rand_results]))
        print({"phase": "default_cost_random_dir_summary", "instrument": inst, "mean_cum_return": mean_cum, "mean_profit_factor": mean_pf})


if __name__ == "__main__":
    main()
