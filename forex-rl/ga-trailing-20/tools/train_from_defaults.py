#!/usr/bin/env python3
"""Train strategies using settings from config/defaults.json.

This script is a thin orchestrator around DatasetLoader + simulator + strategies,
so you don't have to remember the long CLI argument lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Make ga-trailing-20 importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]
from strategies.gradient_nn import GradientNNStrategy, GradientNNConfig  # type: ignore[import]
from strategies.neuro_ga import NeuroGAStrategy, NeuroGAConfig  # type: ignore[import]
from strategies.multihead_gated_nn import MultiHeadGatedNNStrategy, MultiHeadGatedNNConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strategy using ga-trailing-20/config/defaults.json")
    p.add_argument("--strategy", choices=["gradient_nn", "neuro_ga", "multihead_gated_nn"], default="gradient_nn")
    p.add_argument("--config", default=str(PKG_ROOT / "config" / "defaults.json"))
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    p.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    return p.parse_args()


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def _load_defaults(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def train_gradient_nn(cfg: Dict[str, Any], raw_dir: Path, feature_dir: Path) -> None:
    loader_cfg = cfg.get("loader", {})
    tr_cfg = cfg.get("train_ranges", {})
    gn_cfg = cfg.get("gradient_nn", {})

    instruments = loader_cfg.get("instruments", ["USD_PLN"])
    base_gran = loader_cfg.get("base_granularity", "M5")
    aux = loader_cfg.get("aux_granularities", [])
    max_rows = loader_cfg.get("max_rows")

    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=raw_dir,
        feature_dir=feature_dir,
        base_granularity=str(base_gran).upper(),
        aux_granularities=[str(g).upper() for g in aux],
        normalize=bool(loader_cfg.get("normalize", True)),
        max_rows=int(max_rows) if max_rows is not None else None,
    )
    loader = DatasetLoader(lcfg)

    train_range = _parse_range(tr_cfg.get("train", "2021-01-01:2024-01-01"))
    val_range = _parse_range(tr_cfg.get("val", "2024-01-02:2025-11-01"))
    split_cfg = SplitConfig(train=train_range, val=val_range)
    train_split, val_split, _ = loader.split_by_dates(split_cfg)

    # Downsample to keep memory bounded; can be moved into config later.
    def _take(split, n: int):
        if split is None:
            return None
        records = []
        for idx, rec in enumerate(split):
            if idx >= n:
                break
            records.append(rec)
        class _SmallSplit:
            def __iter__(self):  # pragma: no cover - trivial
                return iter(records)
        return _SmallSplit()

    train_split = _take(train_split, gn_cfg.get("max_train_bars", 800))
    val_split = _take(val_split, gn_cfg.get("max_val_bars", 400))

    hidden = tuple(int(h) for h in gn_cfg.get("hidden_dims", [64, 32]))
    gcfg = GradientNNConfig(
        hidden_dims=hidden,
        dropout=float(gn_cfg.get("dropout", 0.1)),
        learning_rate=float(gn_cfg.get("learning_rate", 1e-3)),
        batch_size=int(gn_cfg.get("batch_size", 64)),
        epochs=int(gn_cfg.get("epochs", 3)),
        patience=int(gn_cfg.get("patience", 2)),
        checkpoint_every=int(gn_cfg.get("checkpoint_every", 0)),
    )
    reg = RegularizationConfig(l2=float(gn_cfg.get("l2", 0.0)))
    strat = GradientNNStrategy(config=gcfg, regularization=reg)
    ckpt_prefix = gn_cfg.get("checkpoint_prefix")
    if ckpt_prefix:
        strat.checkpoint_base = str(ckpt_prefix)

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(train_split, val_split, sim)

    if val_split is not None:
        result = sim.evaluate(strat, val_split, record_equity=False, return_trades=True)
        m = result.metrics
        print({
            "segment": "val",
            "cum_return": m.cum_return,
            "sharpe": m.sharpe,
            "profit_factor": m.profit_factor,
            "trades": len(result.trades),
        })


def train_neuro_ga(cfg: Dict[str, Any], raw_dir: Path, feature_dir: Path) -> None:
    loader_cfg = cfg.get("loader", {})
    tr_cfg = cfg.get("train_ranges", {})
    ga_cfg = cfg.get("neuro_ga", {})

    instruments = loader_cfg.get("instruments", ["USD_PLN"])
    base_gran = loader_cfg.get("base_granularity", "M5")
    aux = loader_cfg.get("aux_granularities", [])
    max_rows = loader_cfg.get("max_rows")

    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=raw_dir,
        feature_dir=feature_dir,
        base_granularity=str(base_gran).upper(),
        aux_granularities=[str(g).upper() for g in aux],
        normalize=bool(loader_cfg.get("normalize", True)),
        max_rows=int(max_rows) if max_rows is not None else None,
    )
    loader = DatasetLoader(lcfg)

    train_range = _parse_range(tr_cfg.get("train", "2021-01-01:2024-01-01"))
    split_cfg = SplitConfig(train=train_range)
    train_split, _, _ = loader.split_by_dates(split_cfg)

    # Downsample
    records = []
    max_train = int(ga_cfg.get("max_train_bars", 300))
    for idx, rec in enumerate(train_split):
        if idx >= max_train:
            break
        records.append(rec)

    class SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)

    small_split = SmallSplit()

    hidden = tuple(int(h) for h in ga_cfg.get("hidden_layers", [32]))
    gcfg = NeuroGAConfig(
        population=int(ga_cfg.get("population", 8)),
        generations=int(ga_cfg.get("generations", 3)),
        hidden_layers=hidden,
        mutation_prob=float(ga_cfg.get("mutation_prob", 0.2)),
        weight_sigma=float(ga_cfg.get("weight_sigma", 0.05)),
        bias_sigma=float(ga_cfg.get("bias_sigma", 0.02)),
        crossover_frac=float(ga_cfg.get("crossover_frac", 0.3)),
        elite_frac=float(ga_cfg.get("elite_frac", 0.1)),
    )
    reg = RegularizationConfig(
        l2=float(ga_cfg.get("l2", 0.0)),
        complexity_penalty=float(ga_cfg.get("complexity_penalty", 0.0)),
    )
    strat = NeuroGAStrategy(config=gcfg, regularization=reg)

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(small_split, None, sim)

    result = sim.evaluate(strat, small_split, record_equity=False, return_trades=True)
    m = result.metrics
    print({
        "segment": "train_ga",
        "cum_return": m.cum_return,
        "sharpe": m.sharpe,
        "profit_factor": m.profit_factor,
        "trades": len(result.trades),
    })


def train_multihead_gated_nn(cfg: Dict[str, Any], raw_dir: Path, feature_dir: Path) -> None:
    loader_cfg = cfg.get("loader", {})
    tr_cfg = cfg.get("train_ranges", {})
    mh_cfg = cfg.get("multihead_gated_nn", {})

    instruments = loader_cfg.get("instruments", [])
    base_gran = loader_cfg.get("base_granularity", "M5")
    aux = loader_cfg.get("aux_granularities", [])
    max_rows = loader_cfg.get("max_rows")

    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=raw_dir,
        feature_dir=feature_dir,
        base_granularity=str(base_gran).upper(),
        aux_granularities=[str(g).upper() for g in aux],
        normalize=bool(loader_cfg.get("normalize", True)),
        max_rows=int(max_rows) if max_rows is not None else None,
    )
    loader = DatasetLoader(lcfg)

    train_range = _parse_range(tr_cfg.get("train", "2023-01-01:2025-06-30"))
    val_range = _parse_range(tr_cfg.get("val", "2025-07-01:2025-10-01"))
    split_cfg = SplitConfig(train=train_range, val=val_range)
    train_split, val_split, _ = loader.split_by_dates(split_cfg)

    # Downsample
    def _take(split, n: int):
        if split is None:
            return None
        records = []
        for idx, rec in enumerate(split):
            if idx >= n:
                break
            records.append(rec)

        class _SmallSplit:
            def __iter__(self):  # pragma: no cover - trivial
                return iter(records)

        return _SmallSplit()

    train_split = _take(train_split, mh_cfg.get("max_train_bars", 2000))
    val_split = _take(val_split, mh_cfg.get("max_val_bars", 800))

    # Basic sanity checks on split sizes to avoid silent no-op training runs.
    def _len_split(split) -> int:
        if split is None:
            return 0
        return sum(1 for _ in split)

    n_train = _len_split(train_split)
    n_val = _len_split(val_split)
    if n_train < 2 or (val_split is not None and n_val < 2):
        print(
            {
                "type": "split_summary",
                "train_range": tr_cfg.get("train"),
                "val_range": tr_cfg.get("val"),
                "train_bars": n_train,
                "val_bars": n_val,
                "max_rows": loader_cfg.get("max_rows"),
                "reason": "insufficient records after applying date ranges and caps",
            }
        )
        return

    hidden = tuple(int(h) for h in mh_cfg.get("hidden_dims", [64, 64]))
    mcfg = MultiHeadGatedNNConfig(
        learning_rate=float(mh_cfg.get("learning_rate", 1e-3)),
        batch_size=int(mh_cfg.get("batch_size", 256)),
        epochs=int(mh_cfg.get("epochs", 8)),
        patience=int(mh_cfg.get("patience", 3)),
        checkpoint_every=int(mh_cfg.get("checkpoint_every", 0)),
        dropout=float(mh_cfg.get("dropout", 0.1)),
        active_frac=float(mh_cfg.get("active_frac", 0.02)),
        ret_scale=0.005,
        gate_ret_threshold=float(mh_cfg.get("gate_ret_threshold", 0.002)),
        max_trail_pips=20.0,
        gate_sparsity_weight=float(mh_cfg.get("gate_sparsity_weight", 0.01)),
    )
    reg = RegularizationConfig(l2=float(mh_cfg.get("l2", 1e-5)))
    strat = MultiHeadGatedNNStrategy(instruments=instruments, config=mcfg, regularization=reg)
    strat.eval_every_epoch = True
    ckpt_prefix = mh_cfg.get("checkpoint_prefix")
    if ckpt_prefix:
        strat.checkpoint_base = str(ckpt_prefix)

    # Simulator without a hard concurrency cap; rely on learned gating.
    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(train_split, val_split, sim)

    if val_split is not None:
        result = sim.evaluate(strat, val_split, record_equity=False, return_trades=True)
        m = result.metrics
        print(
            {
                "segment": "val",
                "cum_return": m.cum_return,
                "sharpe": m.sharpe,
                "profit_factor": m.profit_factor,
                "trades": len(result.trades),
                "per_instrument_tim_mean": m.per_instrument_tim_mean,
                "per_instrument_tim_std": m.per_instrument_tim_std,
            }
        )


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = _load_defaults(cfg_path)
    raw_dir = Path(args.raw_dir)
    feature_dir = Path(args.feature_dir)

    if args.strategy == "gradient_nn":
        train_gradient_nn(cfg, raw_dir, feature_dir)
    elif args.strategy == "neuro_ga":
        train_neuro_ga(cfg, raw_dir, feature_dir)
    else:
        train_multihead_gated_nn(cfg, raw_dir, feature_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
