#!/usr/bin/env python3
"""Train a MultiHeadGatedNNStrategy on aligned dataset splits and run a backtest.

This script trains a multi-instrument neural strategy that emits per-instrument
discrete entrance signals {-1,0,+1} using quantile-based gating to control
time-in-market, then evaluates it via the trailing-stop simulator.
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
from strategies.multihead_gated_nn import MultiHeadGatedNNStrategy, MultiHeadGatedNNConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MultiHeadGatedNNStrategy and evaluate via trailing simulator")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--instruments", required=True, help="Comma-separated list of instruments (e.g. DEFAULT_OANDA_20)")
    p.add_argument("--train", default="2025-06-26:2025-09-15")
    p.add_argument("--val", default="2025-09-16:2025-10-01")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D", help="Aux granularities, e.g. 'D' or 'M5,D' (empty string for none)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--hidden", default="256,128")
    p.add_argument("--active-frac", type=float, default=0.10, help="Target non-flat fraction via quantile gating")
    p.add_argument("--model-path", default="ga-trailing-20/checkpoints/multihead_gated_nn.pt")
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
    p.add_argument(
        "--feature-lags",
        default="0",
        help="Comma-separated non-negative lags in bars (e.g. '0,7' or '0,7,49'). 0 = current bar only.",
    )
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
    p.add_argument("--max-train-bars", type=int, default=2000, help="Limit number of training bars for memory/speed")
    p.add_argument("--max-val-bars", type=int, default=800, help="Limit number of validation bars for memory/speed")
    p.add_argument("--max-rows", type=int, default=20000, help="Optional cap on rows loaded per instrument")
    p.add_argument("--checkpoint-prefix", default="", help="Optional prefix path for per-epoch checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N epochs (0=disabled)")
    # Per-epoch simulator evaluation (enabled by default; can be disabled)
    p.add_argument(
        "--no-epoch-eval",
        action="store_true",
        help="Disable per-epoch backtest evaluation on the validation split.",
    )
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
        default=400,
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
    hidden = tuple(int(h.strip()) for h in args.hidden.split(",") if h.strip()) or (256, 128)
    try:
        feature_lags = [int(tok.strip()) for tok in str(args.feature_lags).split(",") if tok.strip()]
    except Exception:
        feature_lags = [0]

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

    # Downsample to a manageable number of records
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

    # Basic sanity checks on split sizes to avoid silent no-op training runs.
    def _len_split(split) -> int:
        if split is None:
            return 0
        return sum(1 for _ in split)

    n_train = _len_split(train_split_small)
    n_val = _len_split(val_split_small)
    if n_train < 2 or (val_split_small is not None and n_val < 2):
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

    mcfg = MultiHeadGatedNNConfig(
        learning_rate=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=3,
        checkpoint_every=int(args.checkpoint_every),
        active_frac=float(args.active_frac),
        feature_lags=tuple(feature_lags),
    )
    reg = RegularizationConfig(l2=float(args.l2))
    strat = MultiHeadGatedNNStrategy(instruments=instruments, config=mcfg, regularization=reg)
    # Enable per-epoch evaluation by default; the flag allows users to turn it off.
    strat.eval_every_epoch = not bool(args.no_epoch_eval)
    if args.checkpoint_prefix:
        strat.checkpoint_base = args.checkpoint_prefix

    # Simulator without a hard concurrency cap; we rely on the gate and its
    # sparsity regularization to learn when *not* to trade. Allow overriding
    # the trailing-mode and pip distance range via CLI.
    from simulator.config import TrailingMode  # type: ignore[import]

    trail_kwargs = {}
    if args.trail_mode.lower() == "pip":
        trail_kwargs["mode"] = TrailingMode.PIP
        trail_kwargs["min_distance_pips"] = float(args.min_trail_pips)
        trail_kwargs["pip_distance"] = float(args.min_trail_pips)
        trail_kwargs["max_trailing_pips"] = float(args.max_trail_pips)
    sim = TrailingStopSimulator(TrailingConfig(**trail_kwargs), CostModel(CostModelConfig()))

    # Train
    strat.fit(train_split_small, val_split_small, sim)

    # Evaluate on validation segment
    if val_split_small is not None:
        result = sim.evaluate(strat, val_split_small, record_equity=False, return_trades=True)
        m = result.metrics
        # New edge-like score: (total_pos - total_neg) / (total_pos + total_neg)
        pnls = [float(t.metadata.get("pnl", 0.0)) for t in result.trades]
        total_pos = sum(p for p in pnls if p > 0.0)
        total_neg = sum(-p for p in pnls if p < 0.0)
        denom = total_pos + total_neg
        edge_score = 0.0 if denom <= 0.0 else (total_pos - total_neg) / denom
        print(
            {
                "segment": "val",
                "cum_return": m.cum_return,
                "sharpe": m.sharpe,
                "profit_factor": m.profit_factor,
                "trades": len(result.trades),
                "per_instrument_tim_mean": m.per_instrument_tim_mean,
                "per_instrument_tim_std": m.per_instrument_tim_std,
                "edge_score": edge_score,
            }
        )

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
                    "per_instrument_tim_mean": m.per_instrument_tim_mean,
                    "per_instrument_tim_std": m.per_instrument_tim_std,
                }
            )
        if cv_results:
            avg_sharpe = sum(r["sharpe"] for r in cv_results) / max(1, len(cv_results))
            avg_tim_mean = sum(r["per_instrument_tim_mean"] for r in cv_results) / max(1, len(cv_results))
            print({"cv_results": cv_results, "cv_avg_sharpe": avg_sharpe, "cv_avg_per_instrument_tim_mean": avg_tim_mean})


if __name__ == "__main__":  # pragma: no cover
    main()

