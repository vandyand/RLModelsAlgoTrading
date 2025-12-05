#!/usr/bin/env python3
"""Small meta-parameter search for MultiHeadGatedNNStrategy.

This script runs a *tiny* grid over a few key knobs:
  - active_frac (target non-flat fraction via quantile gating)
  - gate_sparsity_weight (penalty on average gate probability)
  - flat_reward_weight (reward for staying flat on low-|ret| bars)
  - feature_lags (temporal lags for input features)

For each configuration it:
  1) Loads aligned data via DatasetLoader.
  2) Trains a MultiHeadGatedNNStrategy on a capped train window.
  3) Evaluates on a capped validation window via the trailing simulator.
  4) Prints a concise JSON-ish summary with key metrics.

The goal is not exhaustive search, just to see if any combination clearly
improves cum_return / profit_factor / per_instrument_tim_mean.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

# Add ga-trailing-20 root to import path when run from repo
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import (  # type: ignore[import]
    CostModel,
    CostModelConfig,
    TrailingConfig,
    TrailingStopSimulator,
)
from strategies.base import RegularizationConfig  # type: ignore[import]
from strategies.multihead_gated_nn import (  # type: ignore[import]
    MultiHeadGatedNNConfig,
    MultiHeadGatedNNStrategy,
)


@dataclass
class TrialConfig:
    name: str
    active_frac: float
    gate_sparsity_weight: float
    flat_reward_weight: float
    feature_lags: Tuple[int, ...]


def _build_loader(
    instruments: Sequence[str],
    feature_subset_file: Path | None,
    feature_top_k: int,
    max_rows: int,
) -> DatasetLoader:
    # Optional feature subset
    subset: List[str] | None = None
    if feature_subset_file is not None and feature_subset_file.exists():
        names: List[str] = []
        with feature_subset_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                name = line.strip()
                if not name:
                    continue
                names.append(name)
                if feature_top_k and len(names) >= int(feature_top_k):
                    break
        subset = names or None

    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=REPO_ROOT / "continuous-trader" / "data",
        feature_dir=REPO_ROOT / "continuous-trader" / "data" / "features",
        base_granularity="M5",
        aux_granularities=("D",),
        normalize=True,
        feature_subset=subset,
        max_rows=max_rows,
    )
    return DatasetLoader(lcfg)


def _take(split, n: int):
    if split is None:
        return None
    records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(split):
        if idx >= n:
            break
        records.append(rec)

    class _SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)

    return _SmallSplit()


def run_trial(
    trial: TrialConfig,
    instruments: Sequence[str],
    train_range: Tuple[str, str],
    val_range: Tuple[str, str],
    max_train_bars: int,
    max_val_bars: int,
    max_rows: int,
    feature_subset_file: Path | None,
    feature_top_k: int,
) -> Dict[str, Any]:
    loader = _build_loader(
        instruments=instruments,
        feature_subset_file=feature_subset_file,
        feature_top_k=feature_top_k,
        max_rows=max_rows,
    )

    split_cfg = SplitConfig(train=train_range, val=val_range)
    train_split, val_split, _ = loader.split_by_dates(split_cfg)

    train_split = _take(train_split, max_train_bars)
    val_split = _take(val_split, max_val_bars)

    # Quick length checks
    def _len_split(s) -> int:
        if s is None:
            return 0
        return sum(1 for _ in s)

    n_train = _len_split(train_split)
    n_val = _len_split(val_split)
    if n_train < 2 or n_val < 2:
        return {
            "trial": asdict(trial),
            "status": "insufficient_records",
            "n_train": n_train,
            "n_val": n_val,
        }

    mcfg = MultiHeadGatedNNConfig(
        learning_rate=1e-3,
        batch_size=128,
        epochs=4,
        patience=2,
        checkpoint_every=0,
        dropout=0.1,
        active_frac=float(trial.active_frac),
        ret_scale=0.005,
        max_trail_pips=20.0,
        gate_sparsity_weight=float(trial.gate_sparsity_weight),
        flat_reward_weight=float(trial.flat_reward_weight),
        feature_lags=trial.feature_lags,
    )
    reg = RegularizationConfig(l2=1e-5)
    strat = MultiHeadGatedNNStrategy(instruments=instruments, config=mcfg, regularization=reg)
    strat.eval_every_epoch = False

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(train_split, val_split, sim)

    result = sim.evaluate(strat, val_split, record_equity=False, return_trades=True)
    m = result.metrics
    return {
        "trial": asdict(trial),
        "status": "ok",
        "metrics": {
            "cum_return": float(m.cum_return),
            "sharpe": float(m.sharpe),
            "profit_factor": float(m.profit_factor),
            "trades": len(result.trades),
            "per_instrument_tim_mean": float(m.per_instrument_tim_mean),
            "per_instrument_tim_std": float(m.per_instrument_tim_std),
        },
    }


def main() -> None:  # pragma: no cover
    instruments = [
        "EUR_USD",
        "USD_JPY",
        "GBP_USD",
        "AUD_USD",
        "USD_CHF",
        "USD_CAD",
        "NZD_USD",
        "EUR_JPY",
        "GBP_JPY",
        "EUR_GBP",
        "EUR_CHF",
        "EUR_AUD",
        "EUR_CAD",
        "GBP_CHF",
        "AUD_JPY",
        "AUD_CHF",
        "CAD_JPY",
        "NZD_JPY",
        "GBP_AUD",
        "AUD_NZD",
    ]

    # Train/val windows chosen to keep memory in check and align with previous
    # experiments.
    train_range = ("2025-01-01", "2025-07-31")
    val_range = ("2025-08-01", "2025-10-01")

    feature_subset_file = PKG_ROOT / "config" / "selected_features_EUR_AUD_M5D.txt"
    feature_top_k = 50

    # Tiny hand-picked grid of trials.
    trials: List[TrialConfig] = [
        TrialConfig(
            name="A_baseline_lag0",
            active_frac=0.05,
            gate_sparsity_weight=0.05,
            flat_reward_weight=0.01,
            feature_lags=(0,),
        ),
        TrialConfig(
            name="B_sparse_lag0",
            active_frac=0.02,
            gate_sparsity_weight=0.08,
            flat_reward_weight=0.02,
            feature_lags=(0,),
        ),
        TrialConfig(
            name="C_baseline_lag0_7",
            active_frac=0.05,
            gate_sparsity_weight=0.05,
            flat_reward_weight=0.01,
            feature_lags=(0, 7),
        ),
        TrialConfig(
            name="D_sparse_lag0_7",
            active_frac=0.02,
            gate_sparsity_weight=0.1,
            flat_reward_weight=0.02,
            feature_lags=(0, 7),
        ),
    ]

    results: List[Dict[str, Any]] = []
    for trial in trials:
        res = run_trial(
            trial=trial,
            instruments=instruments,
            train_range=train_range,
            val_range=val_range,
            max_train_bars=800,
            max_val_bars=400,
            max_rows=15000,
            feature_subset_file=feature_subset_file,
            feature_top_k=feature_top_k,
        )
        results.append(res)
        print(json.dumps(res))  # one JSON object per line for easy grepping

    # Also print a compact summary table for quick eyeballing.
    summary = []
    for r in results:
        row: Dict[str, Any] = {"name": r["trial"]["name"], "status": r["status"]}
        if r["status"] == "ok":
            m = r["metrics"]
            row.update(
                {
                    "cum_return": m["cum_return"],
                    "sharpe": m["sharpe"],
                    "profit_factor": m["profit_factor"],
                    "trades": m["trades"],
                    "tim_mean": m["per_instrument_tim_mean"],
                    "tim_std": m["per_instrument_tim_std"],
                }
            )
        summary.append(row)
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()

