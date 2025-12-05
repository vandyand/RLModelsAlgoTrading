#!/usr/bin/env python3
"""Train a GradientNNStrategy on aligned dataset splits and run a backtest.

This is a minimal training driver for experimentation / smoke testing.
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
from strategies.gradient_nn import GradientNNStrategy, GradientNNConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GradientNNStrategy and evaluate via trailing simulator")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--instruments", default="USD_PLN")
    # Defaults target the most recent few years rather than extreme ranges.
    p.add_argument("--train", default="2021-01-01:2024-01-01")
    p.add_argument("--val", default="2024-01-02:2025-11-01")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D", help="Aux granularities, e.g. 'D' or 'M5,D' (empty string for none)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--hidden", default="128,64")
    p.add_argument("--model-path", default="ga-trailing-20/checkpoints/gradient_nn.pt")
    # Feature subset controls (e.g. MULTI20 top-K list)
    p.add_argument(
        "--feature-subset-file",
        default="",
        help="Optional path to a text file with one feature name per line; if set, restricts loader to this subset.",
    )
    p.add_argument(
        "--feature-top-k",
        type=int,
        default=0,
        help="If >0, use only the first K features from --feature-subset-file.",
    )
    p.add_argument("--max-train-bars", type=int, default=500, help="Limit number of training bars for memory/speed")
    p.add_argument("--max-val-bars", type=int, default=300, help="Limit number of validation bars for memory/speed")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows loaded per instrument")
    p.add_argument("--checkpoint-prefix", default="", help="Optional prefix path for per-epoch checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N epochs (0=disabled)")
    # Lightweight time-series cross-validation: evaluate trained model on rolling folds.
    p.add_argument(
        "--cv-splits",
        type=int,
        default=0,
        help="If >0, evaluate the trained model on N rolling validation folds and report metrics.",
    )
    p.add_argument(
        "--cv-val-bars",
        type=int,
        default=500,
        help="Number of bars per validation fold when using --cv-splits.",
    )
    p.add_argument(
        "--cv-gap-bars",
        type=int,
        default=0,
        help="Gap bars between training/validation when constructing rolling folds.",
    )
    return p.parse_args()


def _parse_range(s: str) -> tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]
    hidden = tuple(int(h.strip()) for h in args.hidden.split(",") if h.strip()) or (128, 64)

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

    # Downsample to a manageable number of records for smoke runs
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

    train_split = _take(train_split, int(args.max_train_bars))
    val_split = _take(val_split, int(args.max_val_bars))

    gcfg = GradientNNConfig(
        hidden_dims=hidden,
        learning_rate=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=2,
        checkpoint_every=int(args.checkpoint_every),
    )
    reg = RegularizationConfig(l2=float(args.l2))
    strat = GradientNNStrategy(config=gcfg, regularization=reg)
    if args.checkpoint_prefix:
        strat.checkpoint_base = args.checkpoint_prefix

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    # Train
    strat.fit(train_split, val_split, sim)

    # Evaluate on validation segment
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

    # Save model if training actually produced a network
    model_path = Path(args.model_path)
    if getattr(strat, "_model", None) is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        strat.save(str(model_path))
        print({"saved_model": str(model_path)})
    else:
        print({"saved_model": None, "reason": "no training data or early exit"}) 

    # Optional lightweight cross-validation: evaluate trained model on rolling folds
    if getattr(strat, "_model", None) is not None and int(args.cv_splits) > 0:
        folds = loader.make_rolling_folds(
            n_splits=int(args.cv_splits),
            val_bars=int(args.cv_val_bars),
            gap_bars=int(args.cv_gap_bars),
        )
        cv_results = []
        for i, fold in enumerate(folds):
            res = sim.evaluate(strat, fold, record_equity=False, return_trades=True)
            m = res.metrics
            cv_results.append(
                {
                    "fold": i,
                    "cum_return": m.cum_return,
                    "sharpe": m.sharpe,
                    "profit_factor": m.profit_factor,
                    "trades": len(res.trades),
                }
            )
        if cv_results:
            avg_sharpe = sum(r["sharpe"] for r in cv_results) / max(1, len(cv_results))
            print({"cv_results": cv_results, "cv_avg_sharpe": avg_sharpe})


if __name__ == "__main__":  # pragma: no cover
    main()
