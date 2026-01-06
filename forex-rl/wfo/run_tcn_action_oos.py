from __future__ import annotations

"""
Run a fixed TCN action-classifier configuration out-of-sample over a given period.

This is a thin wrapper around `TCNActionAdapter` + `run_wfo` that:
- Loads general WFO settings (date range, granularity, etc.) from a base JSON config.
- Overrides the adapter hyperparameters with a *fixed* best configuration
  discovered via `tcn_action_random_search.py`.

Usage (from forex-rl root):

  python3 -m wfo.run_tcn_action_oos \\
      --base-config wfo/tcn_eurusd_example.json \\
      --start 2023-01-01 \\
      --end 2023-12-31
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .core import WFOConfig, run_wfo
from .adapters.tcn_scalar_threshold import TCNActionAdapter


BEST_HP: Dict[str, Any] = {
    # Best configuration from recent tcn_action_random_search run
    "tcn_lookback": 48,
    "tcn_target_horizon": 4,
    "tcn_hidden": 32,
    "tcn_blocks": 2,
    "tcn_kernel": 3,
    "tcn_dropout": 0.15975472342699204,
    "label_long_thresh": 0.0009426610298733185,
    "label_short_thresh": -0.0005379716686380912,
    "tcn_trade_cost_bps": 0.5,
}


def load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Base config JSON must contain an object at the top level")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run fixed TCN action-classifier config out-of-sample via WFO"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_example.json",
        help="Path to base JSON config (used for generic WFO settings)",
    )
    ap.add_argument(
        "--start",
        default="2023-01-01",
        help="OOS WFO start date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--end",
        default="2023-12-31",
        help="OOS WFO end date (YYYY-MM-DD)",
    )
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))

    # Dates: CLI overrides base config, which in turn defaults to args
    start = str(args.start or base_cfg.get("start", "2023-01-01"))
    end = str(args.end or base_cfg.get("end", "2023-12-31"))

    train_n = float(base_cfg.get("train_n", 1.0))
    val_n = float(base_cfg.get("val_n", 1.0))
    step_n = float(base_cfg.get("step_n", 1.0))
    unit = str(base_cfg.get("unit", "months"))
    base_gran = str(base_cfg.get("base_gran", "M5"))
    out_dir = str(base_cfg.get("out_dir", "wfo/runs"))
    candle_cache_base = str(
        base_cfg.get("candle_cache_base", "http://127.0.0.1:9100")
    )

    adapter_kwargs = {
        "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
        "grans": str(base_cfg.get("grans", "M5,H1,D")),
        "base_gran": str(base_cfg.get("tcn_base_gran", base_gran)),
        "epochs": int(base_cfg.get("epochs", 5)),
        "lr": float(base_cfg.get("lr", 1e-3)),
        "candle_cache_base": candle_cache_base,
        # Fixed best hyperparameters
        "lookback_bars": int(BEST_HP["tcn_lookback"]),
        "target_horizon": int(BEST_HP["tcn_target_horizon"]),
        "hidden_channels": int(BEST_HP["tcn_hidden"]),
        "num_blocks": int(BEST_HP["tcn_blocks"]),
        "kernel_size": int(BEST_HP["tcn_kernel"]),
        "dropout": float(BEST_HP["tcn_dropout"]),
        "label_long_thresh": float(BEST_HP["label_long_thresh"]),
        "label_short_thresh": float(BEST_HP["label_short_thresh"]),
        "trade_cost_bps": float(BEST_HP["tcn_trade_cost_bps"]),
    }

    adapter = TCNActionAdapter(**adapter_kwargs)

    cfg = WFOConfig(
        start=start,
        end=end,
        train_n=train_n,
        val_n=val_n,
        step_n=step_n,
        unit=unit,
        windows_limit=int(base_cfg.get("windows_limit", 0)),
        base_gran=base_gran,
        out_dir=out_dir,
        adapter_spec="wfo.adapters.tcn_scalar_threshold:TCNActionAdapter",
        adapter_kwargs=adapter_kwargs,
        parallel=1,
        no_chart=bool(base_cfg.get("no_chart", False)),
        chart_every=int(base_cfg.get("chart_every", 1)),
        quiet=bool(base_cfg.get("quiet", False)),
        mode=str(base_cfg.get("mode", "thorough")),
    )

    run_dir = run_wfo(adapter, cfg)
    print(run_dir)


if __name__ == "__main__":  # pragma: no cover
    main()

