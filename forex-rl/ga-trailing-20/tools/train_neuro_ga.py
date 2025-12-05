#!/usr/bin/env python3
"""Train a NeuroGAStrategy via GA and evaluate with the trailing-stop simulator.

This is a small driver intended for experimentation / smoke testing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add ga-trailing-20 root to import path when run from repo
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]
from strategies.neuro_ga import NeuroGAStrategy, NeuroGAConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NeuroGAStrategy and evaluate via trailing simulator")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--instruments", default="USD_PLN")
    p.add_argument("--train", default="2021-01-01:2024-01-01")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="", help="Aux granularities, e.g. 'D' or 'M5,D' (empty string for none)")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--hidden", default="32,16")
    p.add_argument("--mutation-prob", type=float, default=0.2)
    p.add_argument("--weight-sigma", type=float, default=0.05)
    p.add_argument("--bias-sigma", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--complexity-penalty", type=float, default=0.0)
    p.add_argument("--max-train-bars", type=int, default=300)
    p.add_argument("--model-path", default="ga-trailing-20/checkpoints/neuro_ga_smoke.json")
    return p.parse_args()


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]
    hidden = tuple(int(h.strip()) for h in args.hidden.split(",") if h.strip()) or (32, 16)

    cfg_loader = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity=args.base_gran.upper(),
        aux_granularities=aux,
        normalize=True,
    )

    loader = DatasetLoader(cfg_loader)
    split_cfg = SplitConfig(train=_parse_range(args.train))
    train_split, _, _ = loader.split_by_dates(split_cfg)

    # Downsample
    records = []
    for idx, rec in enumerate(train_split):
        if idx >= int(args.max_train_bars):
            break
        records.append(rec)

    class SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)

    small_split = SmallSplit()

    gcfg = NeuroGAConfig(
        population=int(args.population),
        generations=int(args.generations),
        hidden_layers=hidden,
        mutation_prob=float(args.mutation_prob),
        weight_sigma=float(args.weight_sigma),
        bias_sigma=float(args.bias_sigma),
    )
    reg = RegularizationConfig(l2=float(args.l2), complexity_penalty=float(args.complexity_penalty))
    strat = NeuroGAStrategy(config=gcfg, regularization=reg)

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(small_split, None, sim)

    # Evaluate
    result = sim.evaluate(strat, small_split, record_equity=False, return_trades=True)
    m = result.metrics
    print({
        "cum_return": m.cum_return,
        "sharpe": m.sharpe,
        "profit_factor": m.profit_factor,
        "trades": len(result.trades),
    })

    # Save
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    strat.save(str(model_path))
    print({"saved_model": str(model_path)})


if __name__ == "__main__":  # pragma: no cover
    main()
