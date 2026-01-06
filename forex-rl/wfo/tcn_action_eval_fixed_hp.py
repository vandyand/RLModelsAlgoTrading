from __future__ import annotations

"""
Evaluate a fixed TCNActionAdapter hyperparameter set over an arbitrary WFO period.

Usage examples (from forex-rl root):

  # Re-run a GA winner on its original 2023 geometry
  python3 -m wfo.tcn_action_eval_fixed_hp \
      --base-config wfo/tcn_eurusd_2023_weeks.json \
      --tcn-lookback 72 \
      --tcn-target-horizon 22 \
      --tcn-hidden 40 \
      --tcn-blocks 4 \
      --tcn-kernel 3 \
      --tcn-dropout 0.06738135694257803 \
      --label-long-thresh 0.0007214131103349863 \
      --label-short-thresh -0.0006928715440570291 \
      --tcn-trade-cost-bps 1.0 \
      --epochs 3 \
      --plot-equity

  # Same hyperparams, but evaluate on 2024 instead
  python3 -m wfo.tcn_action_eval_fixed_hp \
      --base-config wfo/tcn_eurusd_2023_weeks.json \
      --start 2024-01-01 \
      --end 2024-12-31 \
      --tcn-lookback 72 \
      --tcn-target-horizon 22 \
      --tcn-hidden 40 \
      --tcn-blocks 4 \
      --tcn-kernel 3 \
      --tcn-dropout 0.06738135694257803 \
      --label-long-thresh 0.0007214131103349863 \
      --label-short-thresh -0.0006928715440570291 \
      --tcn-trade-cost-bps 1.0 \
      --epochs 3 \
      --plot-equity
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .model_zoo_2023 import load_base_config
from .tcn_action_ga_search import (
    _eval_individual,
    _plot_global_equity,
    _tcn_hp_to_adapter_kwargs,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a fixed TCNActionAdapter hyperparameter set over a WFO period"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_2023_weeks.json",
        help="Path to base JSON config (geometry/instrument/etc.)",
    )
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override WFO start date (YYYY-MM-DD). If unset, use base config.",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override WFO end date (YYYY-MM-DD). If unset, use base config.",
    )
    ap.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Override instrument (tcn_instrument) from base config, e.g. AUD_USD",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override TCN training epochs in base config.",
    )

    # Core TCNAction hyperparameters (optional: fall back to base config)
    ap.add_argument("--tcn-lookback", type=int, default=None)
    ap.add_argument("--tcn-target-horizon", type=int, default=None)
    ap.add_argument("--tcn-hidden", type=int, default=None)
    ap.add_argument("--tcn-blocks", type=int, default=None)
    ap.add_argument("--tcn-kernel", type=int, default=None)
    ap.add_argument("--tcn-dropout", type=float, default=None)
    ap.add_argument("--label-long-thresh", type=float, default=None)
    ap.add_argument("--label-short-thresh", type=float, default=None)
    ap.add_argument(
        "--tcn-trade-cost-bps",
        type=float,
        default=1.0,
        help="Per-trade cost in basis points (default: 1.0).",
    )

    ap.add_argument(
        "--plot-equity",
        action="store_true",
        help="Render ASCII equity curve for the stitched global trades.",
    )

    args = ap.parse_args()

    base_cfg: Dict[str, Any] = load_base_config(Path(args.base_config))

    # Optional overrides for timeframe, instrument, and training length.
    if args.start is not None:
        base_cfg["start"] = str(args.start)
    if args.end is not None:
        base_cfg["end"] = str(args.end)
    if args.instrument is not None:
        base_cfg["tcn_instrument"] = str(args.instrument).upper()
    if args.epochs is not None:
        base_cfg["epochs"] = int(args.epochs)

    def _cfg_or_arg(name: str, cfg_key: str) -> float | int:
        val = getattr(args, name)
        if val is not None:
            return val
        if cfg_key in base_cfg:
            return base_cfg[cfg_key]
        raise SystemExit(
            f"Missing required hyperparameter '{name}' (and '{cfg_key}' not found in base config)"
        )

    hp: Dict[str, Any] = {
        "tcn_lookback": int(_cfg_or_arg("tcn_lookback", "tcn_lookback")),
        "tcn_target_horizon": int(_cfg_or_arg("tcn_target_horizon", "tcn_target_horizon")),
        "tcn_hidden": int(_cfg_or_arg("tcn_hidden", "tcn_hidden")),
        "tcn_blocks": int(_cfg_or_arg("tcn_blocks", "tcn_blocks")),
        "tcn_kernel": int(_cfg_or_arg("tcn_kernel", "tcn_kernel")),
        "tcn_dropout": float(_cfg_or_arg("tcn_dropout", "tcn_dropout")),
        "label_long_thresh": float(_cfg_or_arg("label_long_thresh", "label_long_thresh")),
        "label_short_thresh": float(_cfg_or_arg("label_short_thresh", "label_short_thresh")),
        "tcn_trade_cost_bps": float(base_cfg.get("tcn_trade_cost_bps", args.tcn_trade_cost_bps)),
    }

    # Run a single WFO evaluation using the same worker as GA.
    score, summary, run_dir_str, _ = _eval_individual(base_cfg, hp, individual_id=0)
    run_dir = Path(run_dir_str)

    result = {
        "score": float(score),
        "hp": hp,
        "summary": summary,
        "run_dir": str(run_dir),
    }
    print(json.dumps(result), flush=True)

    if args.plot_equity:
        _plot_global_equity(run_dir)


if __name__ == "__main__":  # pragma: no cover
    main()

