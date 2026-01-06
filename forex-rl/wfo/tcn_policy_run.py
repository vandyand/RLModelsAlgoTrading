from __future__ import annotations

"""Small EUR/USD TCN+MLP policy debug run (3w train, 1w test).

This mirrors the NQ debug/good-running scripts but uses EUR_USD as the
instrument while reusing the same TCNActionNQAdapter (TCN backbone + MLP
policy head).

Run from forex-rl root:

    python3 -m wfo.tcn_eurusd_policy_debug_small_run
"""

from pathlib import Path
from typing import Any, Dict, Optional

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from .core import generate_windows, _add_offset
from .core import _instantiate_adapter  # type: ignore
from .model_zoo_2023 import load_base_config


def _run_windows(base_cfg_path: str, out_jsonl: Optional[str] = None) -> None:
    # Fix random seeds for deterministic debug behavior across runs.
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_cfg: Dict[str, Any] = load_base_config(Path(base_cfg_path))

    start = pd.Timestamp(str(base_cfg.get("start")), tz="UTC")
    end = pd.Timestamp(str(base_cfg.get("end")), tz="UTC")
    train_n = float(base_cfg.get("train_n", 3.0))
    val_n = float(base_cfg.get("val_n", 1.0))
    step_n = float(base_cfg.get("step_n", 1.0))
    unit = str(base_cfg.get("unit", "weeks"))
    base_gran = str(base_cfg.get("base_gran", "M15"))

    # Optional dynamic geometry: when rolling_from_yesterday is true, ignore
    # explicit start/end in the base config and instead build `windows_limit`
    # windows that end on yesterday, using the same geometry fields.
    rolling_from_yesterday = bool(base_cfg.get("rolling_from_yesterday", False))
    win_limit_cfg = int(base_cfg.get("windows_limit", 0) or 0)
    if rolling_from_yesterday and win_limit_cfg > 0:
        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        anchor_val_end = today_utc - pd.Timedelta(seconds=1)  # yesterday 23:59:59
        total_span_units = float(train_n) + float(val_n) + float(step_n) * float(max(0, win_limit_cfg - 1))
        start = _add_offset(
            anchor_val_end + pd.Timedelta(seconds=1),
            -total_span_units,
            unit,
            base_gran,
        )
        end = anchor_val_end

    windows = generate_windows(start, end, train_n, val_n, step_n, unit=unit, base_gran=base_gran)
    if not windows:
        print("No windows generated; check start/end/geometry", flush=True)
        return

    # Respect an optional windows_limit in the debug config.
    win_limit = int(base_cfg.get("windows_limit", 0) or 0)
    if win_limit > 0:
        windows = windows[:win_limit]

    adapter_spec = str(base_cfg.get("adapter_spec", "wfo.adapters.tcn_nq_action:TCNActionNQAdapter"))

    # Minimal adapter_kwargs using the same mapping as GA (shared across windows).
    from .tcn_action_ga_search import _tcn_hp_to_adapter_kwargs

    hp = {
        "tcn_lookback": int(base_cfg["tcn_lookback"]),
        "tcn_target_horizon": int(base_cfg["tcn_target_horizon"]),
        "tcn_hidden": int(base_cfg["tcn_hidden"]),
        "tcn_blocks": int(base_cfg["tcn_blocks"]),
        "tcn_kernel": int(base_cfg["tcn_kernel"]),
        "tcn_dropout": float(base_cfg["tcn_dropout"]),
        "label_long_thresh": float(base_cfg["label_long_thresh"]),
        "label_short_thresh": float(base_cfg["label_short_thresh"]),
        "label_long_quantile": float(base_cfg.get("label_long_quantile", 0.7)),
        "label_short_quantile": float(base_cfg.get("label_short_quantile", 0.3)),
        "tcn_trade_cost_bps": float(base_cfg.get("tcn_trade_cost_bps", 1.0)),
        "nq_ret_thresh": float(base_cfg.get("nq_ret_thresh", 0.0005)),
        "nq_vol_lookback": int(base_cfg.get("nq_vol_lookback", 64)),
        "nq_z_thresh": float(base_cfg.get("nq_z_thresh", 1.0)),
        "nq_enter_long": float(base_cfg.get("nq_enter_long", 0.02)),
        "nq_exit_long": float(base_cfg.get("nq_exit_long", 0.0)),
        "nq_enter_short": float(base_cfg.get("nq_enter_short", -0.02)),
        "nq_exit_short": float(base_cfg.get("nq_exit_short", 0.0)),
    }

    adapter_kwargs = _tcn_hp_to_adapter_kwargs(base_cfg, hp)

    fp: Optional[Any] = None
    try:
        if out_jsonl:
            out_path = Path(out_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fp = out_path.open("w", encoding="utf-8")

        for wi, (tr_s, tr_e, v_s, v_e) in enumerate(windows, start=1):
            print(
                json.dumps(
                    {
                        "event": "debug_window",
                        "win": wi,
                        "train": [tr_s.isoformat(), tr_e.isoformat()],
                        "val": [v_s.isoformat(), v_e.isoformat()],
                    }
                ),
                flush=True,
            )

            adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)

            # Train window
            X_tr, C_tr = adapter.load_window(tr_s, tr_e)
            print(
                json.dumps(
                    {
                        "event": "train_loaded",
                        "win": wi,
                        "rows": int(len(X_tr)),
                        "cols": int(X_tr.shape[1]) if hasattr(X_tr, "shape") else None,
                    }
                ),
                flush=True,
            )

            model = adapter.fit(X_tr, C_tr)

            # Evaluate on train (in-sample)
            m_train = adapter.validate(model, X_tr, C_tr)
            print(
                json.dumps({"event": "metrics_train", "win": wi, **m_train}),
                flush=True,
            )

            # Validation window
            X_val, C_val = adapter.load_window(v_s, v_e)
            print(
                json.dumps(
                    {
                        "event": "val_loaded",
                        "win": wi,
                        "rows": int(len(X_val)),
                        "cols": int(X_val.shape[1]) if hasattr(X_val, "shape") else None,
                    }
                ),
                flush=True,
            )

            m_val = adapter.validate(model, X_val, C_val)
            print(
                json.dumps({"event": "metrics_val", "win": wi, **m_val}),
                flush=True,
            )

            if fp is not None:
                summary = {
                    "win": wi,
                    "train": [tr_s.isoformat(), tr_e.isoformat()],
                    "val": [v_s.isoformat(), v_e.isoformat()],
                    "metrics_train": m_train,
                    "metrics_val": m_val,
                }
                fp.write(json.dumps(summary) + "\n")
                fp.flush()
    finally:
        if fp is not None:
            fp.close()


def main() -> None:
    """Entry point.

    By default this uses the EUR/USD policy debug config, but you can point it
    at any compatible base-config JSON (e.g. AUD_USD, USD_JPY) via
    ``--base-config``.
    """
    ap = argparse.ArgumentParser(
        description="TCN+MLP policy WFO run for a single instrument",
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_policy_debug_3w1w.json",
        help="Base JSON config for geometry/hyperparams/instrument",
    )
    ap.add_argument(
        "--out-jsonl",
        default=None,
        help=(
            "Optional path to write per-window JSONL summaries with "
            "metrics_train/metrics_val for downstream analysis"
        ),
    )
    args = ap.parse_args()

    _run_windows(args.base_config, out_jsonl=args.out_jsonl)


if __name__ == "__main__":  # pragma: no cover
    main()
