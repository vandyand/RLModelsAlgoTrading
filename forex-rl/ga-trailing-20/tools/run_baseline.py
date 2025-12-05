#!/usr/bin/env python3
"""Run a baseline threshold strategy through the trailing-stop simulator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Add ga-trailing-20 root to path when run from repo
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]
from strategies import create_strategy  # type: ignore[import]
from strategies.baseline_threshold import BaselineThresholdConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline threshold backtest")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--instruments", default="USD_PLN")
    p.add_argument("--train", default="2000-01-01:2100-01-01")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D", help="Aux granularities, e.g. 'D' or 'M5,D'")
    p.add_argument("--feature-index", type=int, default=0)
    p.add_argument("--enter-long", type=float, default=0.5)
    p.add_argument("--enter-short", type=float, default=-0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]

    cfg = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity=args.base_gran.upper(),
        aux_granularities=aux,
        normalize=True,
    )
    loader = DatasetLoader(cfg)
    start, end = [s.strip() for s in args.train.split(":", 1)]
    split_cfg = SplitConfig(train=(start, end))
    train_split, _, _ = loader.split_by_dates(split_cfg)

    # Simple baseline strategy using first feature index
    bcfg = BaselineThresholdConfig(
        feature_index=int(args.feature_index),
        enter_long=float(args.enter_long),
        enter_short=float(args.enter_short),
    )
    strat = create_strategy("baseline_threshold", config=bcfg)

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))
    result = sim.evaluate(strat, train_split, record_equity=True, return_trades=True)
    m = result.metrics
    print({
        "cum_return": m.cum_return,
        "sharpe": m.sharpe,
        "profit_factor": m.profit_factor,
        "trades": len(result.trades),
    })


if __name__ == "__main__":  # pragma: no cover
    main()
