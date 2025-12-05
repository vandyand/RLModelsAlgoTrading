#!/usr/bin/env python3
"""Train GAPolicyStrategy via GA on a fixed window and evaluate via simulator.

This is a direct, feed-forward optimization setup:
  - Define a small MLP over the current feature vector producing
    [gate_logit, dir_raw, pos_raw, trail_raw] per instrument.
  - Use a genetic algorithm to search weight space, with fitness
    cum_return - lambda_tim * per_instrument_tim_mean.
  - Evaluate the best genome on a held-out validation window.
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
from strategies.ga_policy import GAPolicyStrategy, GAPolicyConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GA-optimized multi-instrument policy and evaluate via trailing simulator")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    p.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    p.add_argument(
        "--instruments",
        default=(
            "EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,"
            "EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD"
        ),
    )
    p.add_argument("--train", default="2025-01-01:2025-07-31")
    p.add_argument("--val", default="2025-08-01:2025-10-01")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D")
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--hidden", default="64")
    p.add_argument("--lambda-tim", type=float, default=2.0)
    p.add_argument("--mutation-prob", type=float, default=0.2)
    p.add_argument("--weight-sigma", type=float, default=0.05)
    p.add_argument("--bias-sigma", type=float, default=0.02)
    p.add_argument("--crossover-frac", type=float, default=0.3)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--complexity-penalty", type=float, default=0.0)
    p.add_argument("--max-train-bars", type=int, default=600)
    p.add_argument("--max-val-bars", type=int, default=400)
    p.add_argument("--max-rows", type=int, default=15000)
    # Feature subset controls (e.g. MULTI20 top-K list)
    p.add_argument(
        "--feature-subset-file",
        default=str(PKG_ROOT / "config" / "selected_features_EUR_AUD_M5D.txt"),
        help="Optional path to a text file with one feature name per line; if set, restricts loader to this subset.",
    )
    p.add_argument(
        "--feature-top-k",
        type=int,
        default=50,
        help="If >0, use only the first K features from --feature-subset-file.",
    )
    p.add_argument(
        "--zero-cost",
        action="store_true",
        help="If set, run GA with zero transaction costs (no spread, no slippage, no commission).",
    )
    # Trailing-stop configuration
    p.add_argument(
        "--trail-mode",
        default="atr",
        choices=["atr", "pip"],
        help="Trailing mode: 'atr' (default) or 'pip'.",
    )
    p.add_argument(
        "--min-trail-pips",
        type=float,
        default=10.0,
        help="Minimum trailing-stop distance in pips (used in pip mode or as lower bound).",
    )
    p.add_argument(
        "--max-trail-pips",
        type=float,
        default=20.0,
        help="Maximum trailing-stop distance in pips (used when strategy supplies trail_frac).",
    )
    return p.parse_args()


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]

    # Optional feature subset: load names from text file (e.g. MULTI20 ranking)
    feature_subset = None
    if args.feature_subset_file:
        path = Path(args.feature_subset_file)
        if path.exists():
            names = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    name = line.strip()
                    if not name:
                        continue
                    names.append(name)
                    if args.feature_top_k and len(names) >= int(args.feature_top_k):
                        break
            feature_subset = names or None

    cfg_loader = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity=args.base_gran.upper(),
        aux_granularities=aux,
        normalize=True,
        feature_subset=feature_subset,
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
    )
    loader = DatasetLoader(cfg_loader)

    split_cfg = SplitConfig(train=_parse_range(args.train), val=_parse_range(args.val))
    train_split, val_split, _ = loader.split_by_dates(split_cfg)

    # Downsample to manageable windows for GA.
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

    train_split_small = _take(train_split, int(args.max_train_bars))
    val_split_small = _take(val_split, int(args.max_val_bars))

    # Quick guard against empty splits.
    def _len_split(s) -> int:
        if s is None:
            return 0
        return sum(1 for _ in s)

    n_train = _len_split(train_split_small)
    n_val = _len_split(val_split_small)
    if n_train < 2 or n_val < 2:
        print(
            {
                "type": "split_summary",
                "train_range": args.train,
                "val_range": args.val,
                "train_bars": n_train,
                "val_bars": n_val,
                "max_rows": args.max_rows,
                "reason": "insufficient records after applying date ranges and caps",
            }
        )
        return

    hidden = tuple(int(h.strip()) for h in str(args.hidden).split(",") if h.strip()) or (64,)
    pcfg = GAPolicyConfig(
        population=int(args.population),
        generations=int(args.generations),
        hidden_dims=hidden,
        mutation_prob=float(args.mutation_prob),
        weight_sigma=float(args.weight_sigma),
        bias_sigma=float(args.bias_sigma),
        crossover_frac=float(args.crossover_frac),
        elite_frac=float(args.elite_frac),
        lambda_tim=float(args.lambda_tim),
    )
    reg = RegularizationConfig(
        l2=float(args.l2),
        complexity_penalty=float(args.complexity_penalty),
    )
    strat = GAPolicyStrategy(instruments=instruments, config=pcfg, regularization=reg)

    from simulator.config import TrailingMode  # type: ignore[import]

    trail_kwargs = {}
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

    # Train via GA
    strat.fit(train_split_small, None, sim)

    # Evaluate on validation window
    result = sim.evaluate(strat, val_split_small, record_equity=False, return_trades=True)
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


if __name__ == "__main__":  # pragma: no cover
    main()

