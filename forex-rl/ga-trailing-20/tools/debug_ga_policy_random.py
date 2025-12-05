#!/usr/bin/env python3
"""Inspect random GAPolicyStrategy genomes to see if they trade / make profit.

This bypasses GA and just samples random genomes from GAPolicyStrategy._init_genome,
then runs the simulator to inspect trades, cum_return, PF, and gross profit.

Useful to debug whether the GA search space even contains active / profitable
behaviour under a given cost model and trailing configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]
from simulator.config import TrailingMode  # type: ignore[import]
from strategies.ga_policy import GAPolicyStrategy, GAPolicyConfig  # type: ignore[import]


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Debug random GAPolicy genomes")
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    ap.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    ap.add_argument(
        "--instruments",
        default=(
            "EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,"
            "EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD"
        ),
    )
    ap.add_argument("--train", default="2025-01-01:2025-07-31")
    ap.add_argument("--max-train-bars", type=int, default=600)
    ap.add_argument("--max-rows", type=int, default=15000)
    ap.add_argument("--n-genomes", type=int, default=10)
    ap.add_argument(
        "--trail-mode",
        default="pip",
        choices=["atr", "pip"],
        help="Trailing mode (default pip for this debug)",
    )
    ap.add_argument("--min-trail-pips", type=float, default=25.0)
    ap.add_argument("--max-trail-pips", type=float, default=75.0)
    ap.add_argument("--zero-cost", action="store_true")
    args = ap.parse_args()

    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]

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
    split_cfg = SplitConfig(train=train_range)
    train_split, _, _ = loader.split_by_dates(split_cfg)

    # Downsample
    records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(train_split):
        if idx >= int(args.max_train_bars):
            break
        records.append(rec)

    class SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)

    split_small = SmallSplit()

    trail_kwargs: Dict[str, Any] = {}
    if args.trail_mode.lower() == "pip":
        trail_kwargs["mode"] = TrailingMode.PIP
        trail_kwargs["min_distance_pips"] = float(args.min_trail_pips)
        trail_kwargs["pip_distance"] = float(args.min_trail_pips)
        trail_kwargs["max_trailing_pips"] = float(args.max_trail_pips)

    if args.zero_cost:
        cost_cfg = CostModelConfig(
            spread_mode="static",
            spread_table={},
            default_spread_pips=0.0,
            spread_multiplier=0.0,
            commission_per_million=0.0,
            slippage_mode="deterministic",
            slippage_params={"pips": 0.0},
            financing_rate_bps=0.0,
        )
    else:
        cost_cfg = CostModelConfig()

    sim = TrailingStopSimulator(TrailingConfig(**trail_kwargs), CostModel(cost_cfg))

    # Instantiate strategy and infer input_dim from first record.
    pcfg = GAPolicyConfig(hidden_dims=(64,))
    strat = GAPolicyStrategy(instruments=instruments, config=pcfg)

    first = records[0]
    feats0 = np.asarray(first.get("features"), dtype=np.float32)
    input_dim = int(feats0.shape[0])

    results = []
    for i in range(int(args.n_genomes)):
        genome = strat._init_genome(input_dim)  # type: ignore[attr-defined]
        strat._input_dim = input_dim
        strat._genome = genome
        res = sim.evaluate(strat, split_small, record_equity=False, return_trades=True)
        m = res.metrics
        pnls = [float(t.metadata.get("pnl", 0.0)) for t in res.trades]
        total_pos = sum(p for p in pnls if p > 0.0)
        total_neg = sum(-p for p in pnls if p < 0.0)
        results.append(
            {
                "genome": i,
                "cum_return": float(m.cum_return),
                "profit_factor": float(m.profit_factor),
                "trades": len(res.trades),
                "tim_mean": float(m.per_instrument_tim_mean),
                "total_pos_pnl": total_pos,
                "total_neg_pnl": total_neg,
            }
        )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
