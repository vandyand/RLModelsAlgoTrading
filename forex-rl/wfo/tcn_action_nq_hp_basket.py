from __future__ import annotations

"""
Run a small, hand-crafted basket of hyperparameter configurations for NQ
using the TCNActionNQAdapter (TCN backbone + MLP policy), to force exploration
of settings that are more likely to produce trades, without using GA/Optuna.

Usage (from forex-rl root):

  python3 -m wfo.tcn_action_nq_hp_basket \
      --base-config wfo/tcn_nq_2023_weeks.json \
      --plot-equity
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .tcn_action_ga_search import _eval_individual, _plot_global_equity
from .model_zoo_2023 import load_base_config


def _nq_hp_basket(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a small basket of NQ-focused hyperparameter configs.

    These are biased toward:
    - moderate lookbacks,
    - moderate horizons,
    - modest hidden size,
    - low dropout,
    - relatively loose label thresholds,
    with the goal of actually producing trades.
    """
    common: Dict[str, Any] = {
        "tcn_blocks": 1,
        "tcn_kernel": 1,
        "tcn_trade_cost_bps": float(base_cfg.get("tcn_trade_cost_bps", 1.0)),
    }

    basket: List[Dict[str, Any]] = []

    # Short horizon, more reactive
    basket.append(
        dict(
            common,
            tcn_lookback=64,
            tcn_target_horizon=8,
            tcn_hidden=16,
            tcn_dropout=0.05,
            label_long_thresh=0.0010,
            label_short_thresh=-0.0010,
        )
    )

    # Medium horizon
    basket.append(
        dict(
            common,
            tcn_lookback=96,
            tcn_target_horizon=16,
            tcn_hidden=16,
            tcn_dropout=0.05,
            label_long_thresh=0.0015,
            label_short_thresh=-0.0015,
        )
    )

    # Longer horizon, larger hidden
    basket.append(
        dict(
            common,
            tcn_lookback=128,
            tcn_target_horizon=24,
            tcn_hidden=24,
            tcn_dropout=0.05,
            label_long_thresh=0.0020,
            label_short_thresh=-0.0020,
        )
    )

    # Quantile-based labeling variants (if label_mode == "quantile")
    if str(base_cfg.get("label_mode", "bps")) == "quantile":
        basket.append(
            dict(
                common,
                tcn_lookback=64,
                tcn_target_horizon=8,
                tcn_hidden=16,
                tcn_dropout=0.05,
                label_long_thresh=float(base_cfg.get("label_long_thresh", 0.0010)),
                label_short_thresh=float(base_cfg.get("label_short_thresh", -0.0010)),
                label_long_quantile=0.8,
                label_short_quantile=0.2,
            )
        )
        basket.append(
            dict(
                common,
                tcn_lookback=96,
                tcn_target_horizon=16,
                tcn_hidden=24,
                tcn_dropout=0.05,
                label_long_thresh=float(base_cfg.get("label_long_thresh", 0.0015)),
                label_short_thresh=float(base_cfg.get("label_short_thresh", -0.0015)),
                label_long_quantile=0.85,
                label_short_quantile=0.15,
            )
        )

    return basket


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a basket of NQ-focused TCNAction hyperparameter configs"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_nq_2023_weeks.json",
        help="Path to base JSON config (geometry/instrument/etc.)",
    )
    ap.add_argument(
        "--plot-equity",
        action="store_true",
        help="Render ASCII equity curve for each run.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs in base config for these runs (e.g. 5 or 10).",
    )
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))
    if args.epochs is not None:
        base_cfg["epochs"] = int(args.epochs)

    # Force NQ instrument
    base_cfg["tcn_instrument"] = "NQ"

    basket = _nq_hp_basket(base_cfg)
    results: List[Dict[str, Any]] = []

    for i, hp in enumerate(basket, start=1):
        print(f"\n=== NQ HP basket run {i}/{len(basket)} ===", flush=True)
        print(json.dumps({"hp": hp}, indent=2), flush=True)
        score, summary, run_dir, _ = _eval_individual(base_cfg, hp, individual_id=i)
        rec = {
            "hp_index": i,
            "score": float(score),
            "summary": summary,
            "run_dir": run_dir,
        }
        results.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
        if args.plot_equity:
            _plot_global_equity(Path(run_dir))

    print("\n=== NQ HP basket summary ===", flush=True)
    for rec in results:
        print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()

