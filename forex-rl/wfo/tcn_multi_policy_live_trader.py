from __future__ import annotations

"""
Multi-instrument TCN+MLP policy live trader (M15 candles via candle cache + OANDA).

- For each instrument you configure, we:
  * load its base TCN+MLP policy config (per-instrument JSON);
  * train a `TCNActionNQAdapter` on the prior N weeks (train_n) up to "now";
  * on each new M15 bar, re-run inference for that instrument, producing a
    target {-1,0,+1} position and mapping to OANDA units;
  * compute delta-units vs current OANDA position and send a market order.

All instruments share:
  - one OANDA account / API client;
  - one candle cache service (CANDLE_CACHE_BASE);
  - one main event loop.

Usage example (from forex-rl root, with candle cache + demo creds):

    cat wfo/tcn_multi_policy_live_config.json
    {
      "environment": "practice",
      "poll_seconds": 30.0,
      "retrain_hours": 24.0,
      "instruments": [
        {
          "name": "EUR_USD",
          "base_config": "wfo/tcn_eurusd_policy_debug_3w1w.json",
          "units_per_side": 1000
        },
        {
          "name": "AUD_USD",
          "base_config": "wfo/tcn_audusd_policy_debug_3w1w.json",
          "units_per_side": 1000
        }
      ]
    }

    OANDA_DEMO_ACCOUNT_ID=... OANDA_DEMO_KEY=... \\
      python3 -m wfo.tcn_multi_policy_live_trader \\
        --config wfo/tcn_multi_policy_live_config.json

This keeps the per-instrument configs local and reusable (same JSONs you use
for the small debug runs), while giving you a single process to run everything.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from oandapyV20 import API  # type: ignore
import oandapyV20.endpoints.positions as positions  # type: ignore

from .core import _instantiate_adapter  # type: ignore
from .model_zoo_2023 import load_base_config
from .tcn_action_ga_search import _tcn_hp_to_adapter_kwargs


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


_RETRAIN_LOG_PATH = Path(__file__).resolve().parents[1] / "live_retrain.log"


def _append_retrain_log(extra: Dict[str, Any]) -> None:
    """Append a single JSON event line to the dedicated retrain log.

    This is intentionally best-effort: failures here must never impact live trading.
    """
    try:
        event: Dict[str, Any] = dict(extra)
        event.setdefault("ts", _utc_now().isoformat())
        with _RETRAIN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        # Logging is purely diagnostic; never fail trading logic because of it.
        pass


def _current_bar_end_for_gran(gran: str, now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the timestamp of the last completed bar for a given granularity.

    Granularity is an OANDA-style string (e.g. "M5", "M15", "H1", "H4", "D").
    """
    if now is None:
        now = _utc_now()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    g = gran.upper()
    # Map OANDA granularity codes to pandas frequency strings.
    freq_map = {
        "M1": "1min",
        "M5": "5min",
        "M10": "10min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H2": "2h",
        "H3": "3h",
        "H4": "4h",
        "H6": "6h",
        "H8": "8h",
        "H12": "12h",
        "D": "1D",
    }
    freq = freq_map.get(g, "15min")
    return now.floor(freq)


def _current_bar_end_m15(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Backward-compatible helper for code that assumes M15 bars."""
    return _current_bar_end_for_gran("M15", now)


def _build_adapter_from_cfg(base_cfg: Dict[str, Any]) -> Any:
    """Instantiate adapter for a base config (TCNActionNQAdapter)."""
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
    return adapter


def _train_policy(
    base_cfg: Dict[str, Any],
    as_of_bar: pd.Timestamp,
) -> Tuple[Any, Any, pd.Timestamp]:
    """Train a TCN+MLP policy for one instrument up to `as_of_bar`.

    The window geometry respects `train_n` and `unit` in the base config, just
    like the WFO debug runners, so per-instrument configs can switch between
    weeks, days, months, or explicit step counts.
    """
    train_n = float(base_cfg.get("train_n", 7.0))
    unit = str(base_cfg.get("unit", "weeks"))
    base_gran = str(base_cfg.get("base_gran", base_cfg.get("tcn_base_gran", "M5")))

    if unit == "steps":
        steps = max(1, int(round(float(train_n))))
        if base_gran.upper() == "H1":
            delta = pd.Timedelta(hours=steps)
        elif base_gran.upper() == "D":
            delta = pd.Timedelta(days=steps)
        else:
            delta = pd.Timedelta(minutes=5 * steps)
    elif unit == "days":
        delta = pd.Timedelta(days=float(train_n))
    elif unit == "weeks":
        delta = pd.Timedelta(weeks=float(train_n))
    else:
        delta = pd.Timedelta(days=30.0 * float(train_n))

    train_start = as_of_bar - delta

    adapter = _build_adapter_from_cfg(base_cfg)
    X_tr, C_tr = adapter.load_window(train_start, as_of_bar)
    print(
        json.dumps(
            {
                "event": "multi_live_train_loaded",
                "instrument": str(base_cfg.get("tcn_instrument", "")).upper(),
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
    train_start: pd.Timestamp,
    end_bar: pd.Timestamp,
) -> Tuple[float, Dict[str, Any]]:
    """Run adapter.validate on [train_start, end_bar] and return last position + debug."""
    X_live, C_live = adapter.load_window(train_start, end_bar)
    # Debug window info to understand live inference behavior.
    try:
        window_debug: Dict[str, Any] = {
            "event": "multi_live_window_debug",
            "instrument": getattr(getattr(adapter, "cfg", None), "instrument", ""),
            "train_start": train_start.isoformat(),
            "end_bar": end_bar.isoformat(),
            "closes_rows": int(len(C_live)),
            "closes_last_ts": C_live.index[-1].isoformat() if not C_live.empty else None,
            "X_rows": int(len(X_live)),
            "X_cols": int(X_live.shape[1]) if hasattr(X_live, "shape") and len(X_live.shape) > 1 else None,
        }
        print(json.dumps(window_debug), flush=True)
    except Exception:
        # Never let debug logging break trading.
        pass

    metrics = adapter.validate(model, X_live, C_live)

    pos_df = getattr(adapter, "_last_positions", None)
    if pos_df is None or pos_df.empty:
        target_pos = 0.0
        ts = end_bar
    else:
        ts = pos_df.index[-1]
        # Take the last position for that instrument column
        target_pos = float(pos_df.iloc[-1, 0])

    debug = dict(metrics)
    debug.update({"signal_ts": ts.isoformat(), "target_pos": target_pos})
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


def _get_net_units_all(api: API, account_id: str) -> Dict[str, int]:
    """Return net units per instrument from OpenPositions."""
    out: Dict[str, int] = {}
    try:
        r = positions.OpenPositions(accountID=account_id)
        resp = api.request(r)
        for pos in resp.get("positions", []):
            inst = str(pos.get("instrument") or "").upper()
            long_units = float((pos.get("long") or {}).get("units") or 0.0)
            short_units = float((pos.get("short") or {}).get("units") or 0.0)
            net = long_units + short_units
            out[inst] = int(round(net))
    except Exception:
        # On error, treat all as flat; outer logic will be conservative.
        return {}
    return out


def _place_units_delta_order(
    api: API,
    account_id: str,
    instrument: str,
    delta_units: int,
) -> Optional[Dict[str, Any]]:
    """Use existing streamer.orders helper to place a pure delta-units market order."""
    if delta_units == 0:
        return None
    # Ensure the project root (which contains the `streamer` package) is on sys.path.
    try:
        from pathlib import Path
        import sys as _sys

        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from streamer.orders import place_market_order  # type: ignore
    except Exception as exc:
        # Surface import errors as structured data but do not crash the loop.
        return {"error": f"import_error: {exc}"}

    try:
        result = place_market_order(
            api=api,
            account_id=account_id,
            instrument=instrument,
            units=delta_units,
            tp_pips=None,
            sl_pips=None,
            anchor=None,
            client_tag="tcn-multi-policy",
            client_comment="tcn_multi_policy_live_trader",
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


@dataclass
class InstrumentState:
    name: str
    base_cfg_path: Path
    units_per_side: int
    signal_gran: str
    retrain_hours: float
    retrain_unit: str
    enter_long: float
    exit_long: float
    enter_short: float
    exit_short: float
    base_cfg: Dict[str, Any]
    adapter: Any
    model: Any
    train_start: pd.Timestamp
    last_retrain_ts: float
    last_retrain_bar: pd.Timestamp
    last_signal_bar: pd.Timestamp
    last_pos: float


def _should_retrain(st: InstrumentState, now_bar: pd.Timestamp) -> bool:
    """Decide whether to retrain this instrument's policy at `now_bar`.

    Priority:
    - If a positive retrain_hours is configured (global or per-instrument),
      use time-based retraining for backward compatibility.
    - Otherwise, align retraining with the higher-level window geometry in
      the base config via `unit` ("days", "weeks", "months", etc.).
    """
    # Explicit hours override (backward compatible behavior)
    if st.retrain_hours > 0.0:
        return (time.time() - st.last_retrain_ts) >= st.retrain_hours * 3600.0

    u = st.retrain_unit.lower()
    try:
        # Drop timezone information before converting to Period to avoid
        # pandas warnings and ensure consistent UTC-based calendar logic.
        nb = now_bar.tz_convert("UTC").tz_localize(None) if now_bar.tzinfo is not None else now_bar
        lb = st.last_retrain_bar.tz_convert("UTC").tz_localize(None) if st.last_retrain_bar.tzinfo is not None else st.last_retrain_bar
        if u == "days":
            # Retrain on first bar of a new UTC day.
            return nb.normalize() > lb.normalize()
        if u == "weeks":
            # Align to weekly bars (week ending Friday).
            return nb.to_period("W-FRI") > lb.to_period("W-FRI")
        if u == "months":
            # Align to calendar months.
            return nb.to_period("M") > lb.to_period("M")
    except Exception:
        # Fallback: if any of the period conversions fail, skip event-based retrain.
        return False
    # For other units (e.g. "steps"), require explicit retrain_hours.
    return False


def _step_position(
    prev_pos: float,
    a: float,
    enter_long: float,
    exit_long: float,
    enter_short: float,
    exit_short: float,
) -> float:
    """Map continuous action `a` in (-1,1) and previous position to new position.

    This mirrors the hysteresis logic inside `TCNActionNQAdapter.validate` but
    operates on a single bar and a single scalar `prev_pos`.
    """
    new_pos = prev_pos
    if prev_pos == 0.0:
        if a >= enter_long:
            new_pos = 1.0
        elif a <= enter_short:
            new_pos = -1.0
    elif prev_pos > 0.0:
        if a <= exit_long:
            new_pos = 0.0
    else:  # prev_pos < 0.0
        if a >= exit_short:
            new_pos = 0.0
    return float(new_pos)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-instrument TCN+MLP policy live trader (M15, candle cache + OANDA)",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="JSON config describing environment and per-instrument base-configs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals but do not send live orders",
    )
    args = ap.parse_args()

    # Global randomness (optional determinism)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_path = Path(args.config)

    # Initial config load to determine environment and initial settings.
    with cfg_path.open("r", encoding="utf-8") as f:
        top_cfg = json.load(f)
    if not isinstance(top_cfg, dict):
        raise SystemExit("--config must be a JSON object")

    environment = str(top_cfg.get("environment", "practice"))
    poll_seconds = float(top_cfg.get("poll_seconds", 30.0))
    # Global default retrain interval (hours); individual instruments can
    # override this via an optional "retrain_hours" field in their config
    # entries. When omitted (default 0.0), retraining is aligned to the
    # base-config window geometry via `unit` (days/weeks/months).
    retrain_hours = float(top_cfg.get("retrain_hours", 0.0))
    instruments_cfg = top_cfg.get("instruments") or []
    if not isinstance(instruments_cfg, list) or not instruments_cfg:
        raise SystemExit("top-level 'instruments' must be a non-empty array")

    # Candle cache base: allow global override; per-instrument JSON may override as usual.
    os.environ.setdefault("CANDLE_CACHE_BASE", "http://127.0.0.1:9100")

    api, account_id = _get_oanda_api_and_account(environment)

    # Keep instrument state in a dict so we can hot-add/remove entries.
    inst_states: Dict[str, InstrumentState] = {}
    now_ts = _utc_now()

    # Bootstrap initial instruments.
    for inst_entry in instruments_cfg:
        if not isinstance(inst_entry, dict):
            continue
        name = str(inst_entry.get("name", "")).upper()
        base_config_path = inst_entry.get("base_config")
        if not name or not base_config_path:
            continue
        units_per_side = int(inst_entry.get("units_per_side", 1000))
        # Per-instrument retrain cadence (hours). When omitted (default 0.0),
        # this instrument will retrain based on its `unit` in the base config.
        inst_retrain_hours = float(inst_entry.get("retrain_hours", 0.0))
        base_cfg_path = Path(base_config_path)
        base_cfg = load_base_config(base_cfg_path)
        base_cfg["tcn_instrument"] = name

        # Decide signal granularity for this instrument. Allow an optional
        # "signal_gran" override in the base config (or future top-level
        # instrument entries); otherwise default to the model's base granularity.
        signal_gran = str(
            base_cfg.get(
                "signal_gran",
                base_cfg.get("base_gran", base_cfg.get("tcn_base_gran", "M15")),
            )
        )
        last_bar = _current_bar_end_for_gran(signal_gran, now_ts)

        retrain_unit = str(base_cfg.get("unit", "weeks"))
        enter_long = float(base_cfg.get("nq_enter_long", 0.02))
        exit_long = float(base_cfg.get("nq_exit_long", 0.0))
        enter_short = float(base_cfg.get("nq_enter_short", -0.02))
        exit_short = float(base_cfg.get("nq_exit_short", 0.0))

        adapter, model, train_start = _train_policy(base_cfg, last_bar)
        st = InstrumentState(
            name=name,
            base_cfg_path=base_cfg_path,
            units_per_side=units_per_side,
            signal_gran=signal_gran,
            retrain_hours=inst_retrain_hours,
            retrain_unit=retrain_unit,
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=enter_short,
            exit_short=exit_short,
            base_cfg=base_cfg,
            adapter=adapter,
            model=model,
            train_start=train_start,
            last_retrain_ts=time.time(),
            last_retrain_bar=last_bar,
            last_signal_bar=last_bar,
            last_pos=0.0,
        )
        inst_states[name] = st

        print(
            json.dumps(
                {
                    "event": "multi_live_init_instrument",
                    "instrument": name,
                    "environment": environment,
                    "train_start": train_start.isoformat(),
                    "train_end": last_bar.isoformat(),
                    "units_per_side": units_per_side,
                    "retrain_hours": float(inst_retrain_hours),
                }
            ),
            flush=True,
        )

        # Record the initial training window for this instrument in a dedicated
        # retrain log so we can later verify when each instrument last updated.
        try:
            _append_retrain_log(
                {
                    "event": "multi_live_retrain",
                    "kind": "initial",
                    "instrument": name,
                    "train_start": train_start.isoformat(),
                    "train_end": last_bar.isoformat(),
                    "retrain_unit": retrain_unit,
                    "retrain_hours": float(inst_retrain_hours),
                }
            )
        except Exception:
            # Never let auxiliary logging affect main initialization.
            pass

    print(
        json.dumps(
            {
                "event": "multi_live_init",
                "environment": environment,
                "instruments": list(inst_states.keys()),
                "poll_seconds": poll_seconds,
                "retrain_hours": retrain_hours,
                "dry_run": bool(args.dry_run),
            }
        ),
        flush=True,
    )

    if not inst_states:
        raise SystemExit("No valid instruments configured in --config")

    while True:
        try:
            # Reload config each loop so we can hot-add/remove instruments or tweak timing parameters.
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    top_cfg = json.load(f)
                if isinstance(top_cfg, dict):
                    poll_seconds = float(top_cfg.get("poll_seconds", poll_seconds))
                    # Update global default; individual instrument overrides are
                    # reapplied below when we walk instruments_cfg.
                    retrain_hours = float(top_cfg.get("retrain_hours", retrain_hours))
                    instruments_cfg = top_cfg.get("instruments") or []
                else:
                    instruments_cfg = []
            except Exception:
                instruments_cfg = []

            now_ts = _utc_now()
            # Trading prohibit window in US Eastern time: skip live order
            # placement (but still compute signals) between 16:30 and 18:00 ET
            # to avoid illiquid conditions around the FX daily roll.
            try:
                now_et = now_ts.tz_convert("America/New_York")
                t_et = now_et.time()
                in_prohibit_window = (
                    (t_et.hour > 16 or (t_et.hour == 16 and t_et.minute >= 30))
                    and (t_et.hour < 18)
                )
            except Exception:
                in_prohibit_window = False

            # Fetch current net units for all instruments with a single API call.
            net_units_all = _get_net_units_all(api, account_id)

            # Pause behavior: if PAUSE_LIVE_TRADER is set truthy in the
            # environment, close all open positions for configured instruments
            # and skip further trading activity while the flag remains set.
            pause_flag = os.environ.get("PAUSE_LIVE_TRADER", "")
            pause_flag_norm = str(pause_flag).strip().lower()
            is_paused = pause_flag_norm in ("1", "true", "yes", "on")
            if is_paused:
                any_delta = False
                for st in inst_states.values():
                    current_units = int(net_units_all.get(st.name, 0))
                    if current_units == 0:
                        continue
                    delta_units = -current_units
                    any_delta = True
                    order_result = None
                    if not args.dry_run:
                        # For an explicit pause, we flatten even if we are in
                        # the usual trading prohibit window; the intent is to
                        # get flat as soon as possible.
                        order_result = _place_units_delta_order(
                            api=api,
                            account_id=account_id,
                            instrument=st.name,
                            delta_units=delta_units,
                        )
                    print(
                        json.dumps(
                            {
                                "event": "multi_live_pause_flatten",
                                "instrument": st.name,
                                "current_units": current_units,
                                "delta_units": int(delta_units),
                                "summary": _summarize_order_result(order_result),
                            }
                        ),
                        flush=True,
                    )

                print(
                    json.dumps(
                        {
                            "event": "multi_live_pause_state",
                            "paused": True,
                            "flatten_attempted": bool(any_delta),
                        }
                    ),
                    flush=True,
                )

                # When paused, do not emit signals or place new orders; loop
                # continues so that unpausing (by clearing the env var and
                # restarting under systemd, or in a direct shell session)
                # cleanly resumes trading.
                time.sleep(max(1.0, poll_seconds))
                continue

            # Ensure any newly-added instruments are initialized.
            for inst_entry in instruments_cfg:
                if not isinstance(inst_entry, dict):
                    continue
                name = str(inst_entry.get("name", "")).upper()
                base_config_path = inst_entry.get("base_config")
                if not name or not base_config_path:
                    continue
                if name in inst_states:
                    # Allow units_per_side to be updated from config on the fly.
                    st = inst_states[name]
                    st.units_per_side = int(inst_entry.get("units_per_side", st.units_per_side))
                    # Also allow per-instrument retrain_hours to be tuned live;
                    # when omitted, keep the existing value (often 0.0 to use `unit`).
                    st.retrain_hours = float(inst_entry.get("retrain_hours", st.retrain_hours))
                    continue
                units_per_side = int(inst_entry.get("units_per_side", 1000))
                inst_retrain_hours = float(inst_entry.get("retrain_hours", 0.0))
                base_cfg_path = Path(base_config_path)
                base_cfg = load_base_config(base_cfg_path)
                base_cfg["tcn_instrument"] = name

                signal_gran = str(
                    base_cfg.get(
                        "signal_gran",
                        base_cfg.get("base_gran", base_cfg.get("tcn_base_gran", "M15")),
                    )
                )
                now_bar_inst = _current_bar_end_for_gran(signal_gran, now_ts)

                enter_long = float(base_cfg.get("nq_enter_long", 0.02))
                exit_long = float(base_cfg.get("nq_exit_long", 0.0))
                enter_short = float(base_cfg.get("nq_enter_short", -0.02))
                exit_short = float(base_cfg.get("nq_exit_short", 0.0))

                adapter, model, train_start = _train_policy(base_cfg, now_bar_inst)
                st = InstrumentState(
                    name=name,
                    base_cfg_path=base_cfg_path,
                    units_per_side=units_per_side,
                    signal_gran=signal_gran,
                    retrain_hours=inst_retrain_hours,
                    retrain_unit=str(base_cfg.get("unit", "weeks")),
                    enter_long=enter_long,
                    exit_long=exit_long,
                    enter_short=enter_short,
                    exit_short=exit_short,
                    base_cfg=base_cfg,
                    adapter=adapter,
                    model=model,
                    train_start=train_start,
                    last_retrain_ts=time.time(),
                    last_retrain_bar=now_bar_inst,
                    last_signal_bar=now_bar_inst,
                    last_pos=0.0,
                )
                inst_states[name] = st

                print(
                    json.dumps(
                        {
                            "event": "multi_live_add_instrument",
                            "instrument": name,
                            "train_start": train_start.isoformat(),
                            "train_end": now_bar_inst.isoformat(),
                            "units_per_side": units_per_side,
                            "retrain_hours": float(inst_retrain_hours),
                        }
                    ),
                    flush=True,
                )

            # Optionally, instruments removed from config simply stop receiving new signals.
            for st in list(inst_states.values()):
                if isinstance(instruments_cfg, list) and all(
                    (not isinstance(e, dict)) or str(e.get("name", "")).upper() != st.name
                    for e in instruments_cfg
                ):
                    # Skip trading this instrument but keep state so we don't force-close anything.
                    continue

                now_bar_inst = _current_bar_end_for_gran(st.signal_gran, now_ts)
                if now_bar_inst <= st.last_signal_bar:
                    continue

                # Periodic retraining
                if _should_retrain(st, now_bar_inst):
                    st.adapter, st.model, st.train_start = _train_policy(st.base_cfg, now_bar_inst)
                    st.last_retrain_ts = time.time()
                    st.last_retrain_bar = now_bar_inst

                    # Persist a compact retrain audit event for this instrument.
                    # This allows you to confirm that daily/weekly retrains are
                    # actually occurring, independent of the main live_trader.log.
                    try:
                        _append_retrain_log(
                            {
                                "event": "multi_live_retrain",
                                "kind": "scheduled",
                                "instrument": st.name,
                                "train_start": st.train_start.isoformat(),
                                "train_end": now_bar_inst.isoformat(),
                                "retrain_unit": st.retrain_unit,
                                "retrain_hours": float(st.retrain_hours),
                            }
                        )
                    except Exception:
                        # Diagnostics only; ignore any filesystem/logging issues.
                        pass

                    # On retrain, emit a richer metrics snapshot by running a
                    # full validate pass once for this bar.
                    try:
                        _pos, metrics_debug = _infer_target_position(
                            st.adapter,
                            st.model,
                            st.train_start,
                            now_bar_inst,
                        )
                        metrics_event: Dict[str, Any] = {
                            "event": "multi_live_metrics",
                            "bar_end": now_bar_inst.isoformat(),
                            "instrument": st.name,
                        }
                        for k in (
                            "cum_return",
                            "sharpe",
                            "bar_sharpe",
                            "sortino",
                            "max_dd",
                            "trades",
                            "time_in_mkt",
                            "win_rate",
                            "profit_factor",
                            "acts_mean",
                            "acts_std",
                            "acts_min",
                            "acts_max",
                            "acts_frac_gt_enter_long",
                            "acts_frac_lt_enter_short",
                            "pos_frac_long",
                            "pos_frac_short",
                            "pos_frac_flat",
                        ):
                            if k in metrics_debug:
                                metrics_event[k] = metrics_debug[k]
                        if "last_action" in metrics_debug:
                            metrics_event["raw_action"] = metrics_debug["last_action"]
                        print(json.dumps(metrics_event), flush=True)
                    except Exception:
                        # Metrics are purely diagnostic; never fail the loop
                        # just because a metrics event could not be produced.
                        pass

                # Run inference for this instrument: obtain metrics + raw action
                _ignored_pos, debug = _infer_target_position(st.adapter, st.model, st.train_start, now_bar_inst)
                raw_action = float(debug.get("last_action", 0.0) or 0.0)
                # Guard against NaNs/infinities to keep downstream logic stable.
                if not np.isfinite(raw_action):
                    raw_action = 0.0

                # Map raw action + previous position to new target position using
                # the same hysteresis thresholds as the adapter, but tracked
                # explicitly per instrument to avoid any ambiguity in how the
                # adapter's internal position history is interpreted.
                target_pos = _step_position(
                    prev_pos=st.last_pos,
                    a=raw_action,
                    enter_long=st.enter_long,
                    exit_long=st.exit_long,
                    enter_short=st.enter_short,
                    exit_short=st.exit_short,
                )
                st.last_pos = target_pos

                target_units = int(round(target_pos * st.units_per_side))
                current_units = int(net_units_all.get(st.name, 0))
                delta_units = target_units - current_units

                event = {
                    "event": "multi_live_signal",
                    "bar_end": now_bar_inst.isoformat(),
                    "current_timestamp": now_ts.isoformat(),
                    "instrument": st.name,
                    "target_pos": float(target_pos),
                    "target_units": int(target_units),
                    "current_units": int(current_units),
                    "delta_units": int(delta_units),
                }
                # Attach the raw policy output for this bar and whether we are
                # currently in the trading prohibit window.
                event["raw_action"] = raw_action
                event["in_prohibit_window"] = bool(in_prohibit_window)

                print(json.dumps(event), flush=True)

                if not args.dry_run and delta_units != 0 and not in_prohibit_window:
                    order_result = _place_units_delta_order(
                        api=api,
                        account_id=account_id,
                        instrument=st.name,
                        delta_units=delta_units,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "multi_live_order",
                                "instrument": st.name,
                                "delta_units": int(delta_units),
                                "summary": _summarize_order_result(order_result),
                            }
                        ),
                        flush=True,
                    )

                st.last_signal_bar = now_bar_inst

            time.sleep(max(1.0, poll_seconds))
        except KeyboardInterrupt:
            print(json.dumps({"event": "multi_live_shutdown", "reason": "KeyboardInterrupt"}), flush=True)
            break
        except Exception as exc:
            print(json.dumps({"event": "multi_live_error", "error": str(exc)}), flush=True)
            time.sleep(max(5.0, poll_seconds))


if __name__ == "__main__":  # pragma: no cover
    main()

