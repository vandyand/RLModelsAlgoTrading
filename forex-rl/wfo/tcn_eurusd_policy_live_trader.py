from __future__ import annotations

"""
Live EUR/USD TCN+MLP policy trader (M15 candles via candle cache + OANDA).

- Trains a TCNActionNQAdapter policy on the prior N weeks (from base config).
- On each new M15 bar close, it:
  * pulls up-to-date multi-timeframe candles from the local candle cache;
  * runs the TCN+MLP policy to get a target {-1,0,+1} position;
  * compares with current OANDA position and sends a delta-units market order.

Usage (from forex-rl root, with candle cache and OANDA env vars set):

    OANDA_DEMO_ACCOUNT_ID=... OANDA_DEMO_KEY=... \\
      python3 -m wfo.tcn_eurusd_policy_live_trader

This script is intentionally simple and single-instrument (EUR_USD) to keep
it tightly aligned with the debug config `tcn_eurusd_policy_debug_3w1w.json`.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from oandapyV20 import API  # type: ignore

from .core import _instantiate_adapter  # type: ignore
from .model_zoo_2023 import load_base_config
from .tcn_action_ga_search import _tcn_hp_to_adapter_kwargs


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _current_bar_end_m15(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the timestamp of the last completed M15 bar."""
    if now is None:
        now = _utc_now()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    # Floor to 15 minutes -> last completed bar
    return now.floor("15min")


def _build_adapter_from_cfg(base_cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate a TCNActionNQAdapter and return (adapter, adapter_kwargs)."""
    adapter_spec = str(
        base_cfg.get(
            "adapter_spec",
            "wfo.adapters.tcn_nq_action:TCNActionNQAdapter",
        )
    )

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
    adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)
    return adapter, adapter_kwargs


def _train_policy(
    base_cfg: Dict[str, Any],
    as_of_bar: pd.Timestamp,
) -> Tuple[Any, Any, pd.Timestamp]:
    """Train TCN+MLP policy on the prior train window up to `as_of_bar`.

    The window geometry respects `train_n` and `unit` in the base config, so
    you can use weeks (default), days, months, or explicit step counts
    consistent with the WFO debug runners.
    """
    train_n = float(base_cfg.get("train_n", 7.0))
    unit = str(base_cfg.get("unit", "weeks"))
    base_gran = str(base_cfg.get("base_gran", base_cfg.get("tcn_base_gran", "M5")))

    if unit == "steps":
        # Interpret steps in terms of the base granularity, mirroring wfo.core._add_offset.
        steps = max(1, int(round(float(train_n))))
        if base_gran.upper() == "H1":
            delta = pd.Timedelta(hours=steps)
        elif base_gran.upper() == "D":
            delta = pd.Timedelta(days=steps)
        else:
            # Assume M5-like base; 5 minutes per bar
            delta = pd.Timedelta(minutes=5 * steps)
    elif unit == "days":
        delta = pd.Timedelta(days=float(train_n))
    elif unit == "weeks":
        delta = pd.Timedelta(weeks=float(train_n))
    else:
        # Treat months as ~30 days to stay simple and consistent with generate_windows.
        delta = pd.Timedelta(days=30.0 * float(train_n))

    train_start = as_of_bar - delta

    adapter, _adapter_kwargs = _build_adapter_from_cfg(base_cfg)

    X_tr, C_tr = adapter.load_window(train_start, as_of_bar)
    print(
        json.dumps(
            {
                "event": "live_train_loaded",
                "train_start": train_start.isoformat(),
                "train_end": as_of_bar.isoformat(),
                "rows": int(len(X_tr)),
                "cols": int(X_tr.shape[1]) if hasattr(X_tr, "shape") else None,
            }
        ),
        flush=True,
    )
    model = adapter.fit(X_tr, C_tr)
    return adapter, model, train_start


def _infer_target_position(
    adapter: Any,
    model: Any,
    start: pd.Timestamp,
    end_bar: pd.Timestamp,
) -> Tuple[float, Dict[str, Any]]:
    """Run adapter.validate on [start,end_bar] and return last position + debug."""
    X_live, C_live = adapter.load_window(start, end_bar)
    metrics = adapter.validate(model, X_live, C_live)

    pos_df = getattr(adapter, "_last_positions", None)
    if pos_df is None or pos_df.empty:
        target_pos = 0.0
        ts = end_bar
    else:
        # Last bar in closes / positions
        ts = pos_df.index[-1]
        target_pos = float(pos_df.iloc[-1, 0])

    debug = dict(metrics)
    debug.update(
        {
            "signal_ts": ts.isoformat(),
            "target_pos": target_pos,
        }
    )
    return target_pos, debug


def _get_oanda_api_and_account(env: str) -> Tuple[API, str]:
    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if env == "live":
        account_id = os.environ.get("OANDA_LIVE_ACCOUNT_ID", account_id)
        access_token = os.environ.get("OANDA_LIVE_KEY", access_token)
    if not account_id or not access_token:
        raise RuntimeError(
            "Missing OANDA credentials. "
            "Set OANDA_DEMO_ACCOUNT_ID/OANDA_DEMO_KEY (and/or live equivalents)."
        )
    api = API(access_token=access_token, environment=env)
    return api, str(account_id)


def _get_net_units(api: API, account_id: str, instrument: str) -> int:
    """Thin wrapper around online_trader.get_net_units without importing heavy deps."""
    import oandapyV20.endpoints.positions as positions  # type: ignore

    try:
        r = positions.OpenPositions(accountID=account_id)
        resp = api.request(r)
        for pos in resp.get("positions", []):
            if pos.get("instrument") != instrument:
                continue
            long_units = float(pos.get("long", {}).get("units") or 0.0)
            short_units = float(pos.get("short", {}).get("units") or 0.0)
            net = long_units + short_units
            return int(round(net))
    except Exception:
        return 0
    return 0


def _place_units_delta_order(
    api: API,
    account_id: str,
    instrument: str,
    delta_units: int,
) -> Optional[Dict[str, Any]]:
    """Use existing streamer.orders helper to place a pure delta-units market order."""
    if delta_units == 0:
        return None
    # Defer import to avoid dragging streamer deps if user runs in dry-run.
    from streamer.orders import place_market_order  # type: ignore

    try:
        result = place_market_order(
            api=api,
            account_id=account_id,
            instrument=instrument,
            units=delta_units,
            tp_pips=None,
            sl_pips=None,
            anchor=None,
            client_tag="tcn-eurusd-policy",
            client_comment="tcn_eurusd_policy_live_trader",
            fifo_safe=False,
            fifo_adjust=False,
        )
        return result
    except Exception as exc:
        return {"error": str(exc)}


def _summarize_order_result(order_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not order_result or not isinstance(order_result, dict):
        return None
    resp = order_result.get("response")
    if not isinstance(resp, dict):
        return {"error": order_result.get("error")}
    fill = resp.get("orderFillTransaction")
    create = resp.get("orderCreateTransaction")
    if isinstance(fill, dict):
        return {
            "order_id": fill.get("orderID"),
            "fill_id": fill.get("id"),
            "instrument": fill.get("instrument"),
            "units": fill.get("units"),
            "price": fill.get("price"),
            "time": fill.get("time"),
        }
    if isinstance(create, dict):
        return {
            "order_id": create.get("id"),
            "instrument": create.get("instrument"),
            "units": create.get("units"),
            "time": create.get("time"),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live EUR_USD TCN+MLP policy trader (M15, candle cache + OANDA)",
    )
    parser.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_policy_debug_3w1w.json",
        help="Base JSON config for TCN hyperparams and train_n geometry",
    )
    parser.add_argument(
        "--environment",
        choices=["practice", "live"],
        default="practice",
        help="OANDA environment for execution",
    )
    parser.add_argument(
        "--instrument",
        default="EUR_USD",
        help="Instrument to trade (must match training config)",
    )
    parser.add_argument(
        "--units-per-side",
        type=int,
        default=1000,
        help="Absolute units for full long/short (position 1 -> +units, -1 -> -units)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="How often to wake up and check for a new M15 bar",
    )
    parser.add_argument(
        "--retrain-hours",
        type=float,
        default=24.0,
        help="Retrain policy after this many wall-clock hours (0=never after startup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals but do not send live orders",
    )
    parser.add_argument(
        "--log-debug",
        action="store_true",
        help="Print verbose debug information each bar",
    )
    args = parser.parse_args()

    # Ensure deterministic behavior per process (optional).
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_cfg = load_base_config(Path(args.base_config))

    # Tie adapter candle source to candle_cache_service unless explicitly overridden.
    if "candle_cache_base" not in base_cfg:
        os.environ.setdefault("CANDLE_CACHE_BASE", "http://127.0.0.1:9100")

    api, account_id = _get_oanda_api_and_account(args.environment)
    instrument = str(args.instrument).upper()

    # Initial training as of the last completed bar.
    last_bar = _current_bar_end_m15()
    adapter, model, train_start = _train_policy(base_cfg, last_bar)
    last_retrain_ts = time.time()

    print(
        json.dumps(
            {
                "event": "live_init",
                "instrument": instrument,
                "environment": args.environment,
                "train_start": train_start.isoformat(),
                "train_end": last_bar.isoformat(),
                "units_per_side": int(args.units_per_side),
                "dry_run": bool(args.dry_run),
            }
        ),
        flush=True,
    )

    last_signal_bar = last_bar

    while True:
        try:
            now_bar = _current_bar_end_m15()
            if now_bar > last_signal_bar:
                # Optionally retrain periodically.
                if args.retrain_hours > 0.0 and (time.time() - last_retrain_ts) >= (
                    args.retrain_hours * 3600.0
                ):
                    adapter, model, train_start = _train_policy(base_cfg, now_bar)
                    last_retrain_ts = time.time()

                # Run inference on [train_start, now_bar].
                target_pos, debug = _infer_target_position(adapter, model, train_start, now_bar)

                # Map {-1,0,1} to target units.
                target_units = int(round(target_pos * args.units_per_side))
                current_units = _get_net_units(api, account_id, instrument)
                delta_units = target_units - current_units

                event = {
                    "event": "live_signal",
                    "bar_end": debug.get("signal_ts"),
                    "instrument": instrument,
                    "target_pos": float(target_pos),
                    "target_units": int(target_units),
                    "current_units": int(current_units),
                    "delta_units": int(delta_units),
                }
                event.update(
                    {
                        k: v
                        for k, v in debug.items()
                        if k
                        not in (
                            "signal_ts",
                        )
                    }
                )
                print(json.dumps(event), flush=True)

                if not args.dry_run and delta_units != 0:
                    order_result = _place_units_delta_order(
                        api=api,
                        account_id=account_id,
                        instrument=instrument,
                        delta_units=delta_units,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "live_order",
                                "instrument": instrument,
                                "delta_units": int(delta_units),
                                "summary": _summarize_order_result(order_result),
                            }
                        ),
                        flush=True,
                    )

                last_signal_bar = now_bar

            time.sleep(max(1.0, float(args.poll_seconds)))
        except KeyboardInterrupt:
            print(json.dumps({"event": "live_shutdown", "reason": "KeyboardInterrupt"}), flush=True)
            break
        except Exception as exc:
            print(json.dumps({"event": "live_error", "error": str(exc)}), flush=True)
            time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":  # pragma: no cover
    main()

