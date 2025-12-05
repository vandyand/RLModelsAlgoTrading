from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import importlib.util

# Ensure logs flush promptly when not attached to a TTY (e.g., nohup)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

# Local imports
from config import Config
from instruments import DEFAULT_OANDA_20
from model import MultiHeadGenome, forward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA Multi-Head live trader (runs on 5m closes + grace)")
    p.add_argument("--checkpoint", required=True, help="Path to best_genome-*.json")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"), help="Full OANDA account ID")
    p.add_argument("--account-suffix", type=int, default=None, help="Override last 3 digits of account id (e.g., 3 -> '003')")
    p.add_argument("--instruments", default=",".join(DEFAULT_OANDA_20))
    p.add_argument("--units", type=int, default=100, help="Target absolute units per instrument when Long/Short")
    p.add_argument("--grace-seconds", type=int, default=15, help="Seconds after 5m close to wait before acting")
    p.add_argument("--heartbeat-seconds", type=float, default=60.0, help="Log a heartbeat while waiting")
    p.add_argument("--lookback-days", type=int, default=2, help="Days of data to read for feature alignment")
    p.add_argument("--threshold-mode", choices=["band","absolute"], default=None)
    p.add_argument("--band-enter", type=float, default=None)
    p.add_argument("--band-exit", type=float, default=None)
    p.add_argument("--enter-long", type=float, default=None)
    p.add_argument("--exit-long", type=float, default=None)
    p.add_argument("--enter-short", type=float, default=None)
    p.add_argument("--exit-short", type=float, default=None)
    return p.parse_args()


def _next_5m_boundary(with_grace_sec: int) -> datetime:
    now = datetime.now(timezone.utc)
    # floor to current 5m boundary
    minute = (now.minute // 5) * 5
    last_close = now.replace(minute=minute, second=0, microsecond=0)
    current_grace = last_close + timedelta(seconds=with_grace_sec)
    if now < current_grace:
        # Candle just closed at last_close; act at grace for that close
        return current_grace
    # Otherwise schedule next close + grace
    next_close = last_close + timedelta(minutes=5)
    return next_close + timedelta(seconds=with_grace_sec)


def _wait_until(ts: datetime, heartbeat_secs: float = 60.0) -> None:
    next_hb = datetime.now(timezone.utc)
    while True:
        now = datetime.now(timezone.utc)
        if now >= ts:
            return
        if heartbeat_secs and now >= next_hb:
            eta = (ts - now).total_seconds()
            print(f"[heartbeat] {now.isoformat()} waiting {eta:.1f}s for next M5+grace")
            next_hb = now + timedelta(seconds=max(5.0, float(heartbeat_secs)))
        time.sleep(min(1.0, max(0.05, (ts - now).total_seconds())))


def _map_single_with_hysteresis(prev_state: int, out_val: float, *, mode: str, band_enter: float, band_exit: float,
                                 enter_long: float, exit_long: float, enter_short: float, exit_short: float) -> int:
    v = float(out_val)
    state = int(prev_state)
    if mode == "band":
        up_enter = 0.5 + band_enter
        dn_enter = 0.5 - band_enter
        up_exit = 0.5 + band_exit
        dn_exit = 0.5 - band_exit
        if state == 0:
            if v > up_enter:
                state = 1
            elif v < dn_enter:
                state = -1
        elif state == 1:
            if v < up_exit:
                state = 0
        elif state == -1:
            if v > dn_exit:
                state = 0
    else:
        if state == 0:
            if v > enter_long:
                state = 1
            elif v < enter_short:
                state = -1
        elif state == 1:
            if v < exit_long:
                state = 0
        elif state == -1:
            if v > exit_short:
                state = 0
    return int(state)


def _safe_inst(inst: str) -> str:
    # OANDA expects underscore instrument codes
    return inst.replace("/", "_")


def _load_ct_features_module():
    """Dynamically load continuous-trader/features.py as a module despite hyphen in path."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "continuous-trader", "features.py")
    if not os.path.exists(path):
        raise RuntimeError(f"missing features.py at {path}")
    spec = importlib.util.spec_from_file_location("ct_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create module spec for continuous-trader/features.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Ensure module is visible to the import system during execution (dataclasses relies on sys.modules)
    sys.modules[spec.name] = mod  # type: ignore[index]
    try:
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    return mod


def _build_live_feature_row(
    instruments: List[str],
    cfg: Config,
    environment: str,
    access_token: Optional[str],
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Fetch recent candles from OANDA and compute the selected features for a single live row.

    Returns (X_panel, now_idx) where X_panel has MultiIndex columns (instrument, feature_name_with_gran_prefix),
    and one row indexed by the latest M5 bar timestamp.
    """
    ct = _load_ct_features_module()
    # Periods needed across our feature set
    needed_periods = set([15, 45, 135, 405])
    blocks: List[pd.DataFrame] = []
    m5_timestamps: List[pd.Timestamp] = []
    for inst in instruments:
        inst_blocks: List[pd.DataFrame] = []
        base_ts: Optional[pd.Timestamp] = None
        # M5 (required for base timestamp)
        try:
            df_m5 = ct._fetch_fx_recent(inst, "M5", max(200, max(needed_periods)), environment, access_token)
        except Exception:
            df_m5 = pd.DataFrame()
        if df_m5.empty:
            # Skip this instrument if no M5 data
            continue
        base_ts = df_m5.index.max()
        m5_timestamps.append(base_ts)
        feats_m5 = ct.compute_indicator_grid(df_m5, prefix=f"M5_{inst}_", periods=sorted(list(needed_periods)))
        # Select configured M5 features
        want = [f"M5_{inst}_{name}" for name in cfg.m5_features]
        sel = [c for c in want if c in feats_m5.columns]
        m5_row = feats_m5.tail(1)[sel].copy()
        m5_row.index = pd.DatetimeIndex([base_ts])
        # Strip prefix to match training feature names and add gran prefix
        m5_row.columns = [f"M5::{c.split(f'M5_{inst}_',1)[1]}" for c in m5_row.columns]
        inst_blocks.append(m5_row)
        # H1
        try:
            df_h1 = ct._fetch_fx_recent(inst, "H1", max(200, max(needed_periods)), environment, access_token)
        except Exception:
            df_h1 = pd.DataFrame()
        if not df_h1.empty:
            feats_h1 = ct.compute_indicator_grid(df_h1, prefix=f"H1_{inst}_", periods=sorted(list(needed_periods)))
            want = [f"H1_{inst}_{name}" for name in cfg.h1_features]
            sel = [c for c in want if c in feats_h1.columns]
            h1_row = feats_h1.tail(1)[sel].copy()
            h1_row.index = pd.DatetimeIndex([base_ts])
            h1_row.columns = [f"H1::{c.split(f'H1_{inst}_',1)[1]}" for c in h1_row.columns]
            inst_blocks.append(h1_row)
        # D
        try:
            df_d = ct._fetch_fx_recent(inst, "D", max(600, max(needed_periods)), environment, access_token)
        except Exception:
            df_d = pd.DataFrame()
        if not df_d.empty:
            feats_d = ct.compute_indicator_grid(df_d, prefix=f"D_{inst}_", periods=sorted(list(needed_periods)))
            want = [f"D_{inst}_{name}" for name in cfg.d_features]
            sel = [c for c in want if c in feats_d.columns]
            d_row = feats_d.tail(1)[sel].copy()
            d_row.index = pd.DatetimeIndex([base_ts])
            d_row.columns = [f"D::{c.split(f'D_{inst}_',1)[1]}" for c in d_row.columns]
            inst_blocks.append(d_row)

        if inst_blocks:
            inst_block = pd.concat(inst_blocks, axis=1)
            inst_block.columns = pd.MultiIndex.from_product([[inst], inst_block.columns])
            blocks.append(inst_block)

    if not blocks or not m5_timestamps:
        return pd.DataFrame(), pd.Timestamp(0, tz=timezone.utc)
    X_panel = pd.concat(blocks, axis=1)
    now_idx = max(m5_timestamps)
    # All inst blocks already share base_ts per instrument; return union indexed result
    return X_panel, now_idx

def fetch_open_positions(api, account_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        import oandapyV20
        import oandapyV20.endpoints.positions as positions  # type: ignore
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for pos in resp.get("positions", []):
            inst = str(pos.get("instrument") or "")
            long_u = float((pos.get("long") or {}).get("units") or 0.0)
            short_u = float((pos.get("short") or {}).get("units") or 0.0)
            out[inst] = int(round(long_u + short_u))
    except Exception:
        pass
    return out


def submit_order(api, account_id: str, instrument: str, delta_units: int, environment: str) -> None:
    if delta_units == 0:
        return
    try:
        import oandapyV20
        import oandapyV20.endpoints.orders as orders  # type: ignore
        order = {
            "order": {
                "instrument": _safe_inst(instrument),
                "units": str(int(delta_units)),
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        resp = api.request(orders.OrderCreate(accountID=account_id, data=order))
        print(f"order: {instrument} delta={delta_units} -> ok")
    except Exception as e:
        print(f"order: {instrument} delta={delta_units} -> fail: {e}")


def load_genome(path: str) -> MultiHeadGenome:
    import json
    with open(path, "r") as f:
        d = json.load(f)
    return MultiHeadGenome.from_dict(d)


def main() -> None:
    args = parse_args()
    instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    if len(instruments) == 0:
        instruments = list(DEFAULT_OANDA_20)

    # Build config for feature loading and thresholds
    cfg = Config()
    cfg.instruments = instruments
    cfg.lookback_days = int(args.lookback_days)
    if args.threshold_mode is not None:
        cfg.threshold_mode = args.threshold_mode
    if args.band_enter is not None:
        cfg.band_enter = float(args.band_enter)
    if args.band_exit is not None:
        cfg.band_exit = float(args.band_exit)
    if args.enter_long is not None:
        cfg.enter_long = float(args.enter_long)
    if args.exit_long is not None:
        cfg.exit_long = float(args.exit_long)
    if args.enter_short is not None:
        cfg.enter_short = float(args.enter_short)
    if args.exit_short is not None:
        cfg.exit_short = float(args.exit_short)

    # Load model
    genome = load_genome(args.checkpoint)
    # Try to load sibling meta file for defaults
    meta_path = os.path.splitext(args.checkpoint)[0] + ".meta.json"
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if args.instruments == ",".join(DEFAULT_OANDA_20) and isinstance(meta.get("instruments"), list):
                instruments = [str(s) for s in meta["instruments"]]
            if args.threshold_mode is None and isinstance(meta.get("threshold_mode"), str):
                cfg.threshold_mode = str(meta["threshold_mode"]) or cfg.threshold_mode
            if args.band_enter is None and isinstance(meta.get("band_enter"), (int, float)):
                cfg.band_enter = float(meta["band_enter"]) or cfg.band_enter
            if args.band_exit is None and isinstance(meta.get("band_exit"), (int, float)):
                cfg.band_exit = float(meta["band_exit"]) or cfg.band_exit
            if args.enter_long is None and isinstance(meta.get("enter_long"), (int, float)):
                cfg.enter_long = float(meta["enter_long"]) or cfg.enter_long
            if args.exit_long is None and isinstance(meta.get("exit_long"), (int, float)):
                cfg.exit_long = float(meta["exit_long"]) or cfg.exit_long
            if args.enter_short is None and isinstance(meta.get("enter_short"), (int, float)):
                cfg.enter_short = float(meta["enter_short"]) or cfg.enter_short
            if args.exit_short is None and isinstance(meta.get("exit_short"), (int, float)):
                cfg.exit_short = float(meta["exit_short"]) or cfg.exit_short
        except Exception as e:
            print(f"Warning: failed to load meta defaults: {e}")

    # OANDA API setup
    api = None
    try:
        from oandapyV20 import API  # type: ignore
        token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("Missing OANDA_DEMO_KEY or OANDA_ACCESS_TOKEN in environment")
        api = API(access_token=token, environment=args.environment)
    except Exception:
        print("Warning: oandapyV20 not available or credentials missing; will dry-run orders.")
    # Resolve account id using suffix if provided
    if args.account_suffix is not None:
        base = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
        if not base or len(base) < 3:
            raise RuntimeError("Base account id missing or malformed; provide --account-id or set OANDA_DEMO_ACCOUNT_ID")
        account_id = base[:-3] + f"{int(args.account_suffix):03d}"
    else:
        account_id = args.account_id

    print(f"live starting with instruments={instruments} mode={cfg.threshold_mode} band=({cfg.band_enter},{cfg.band_exit}) abs=({cfg.enter_long},{cfg.exit_long},{cfg.enter_short},{cfg.exit_short})")
    # Align to next 5m boundary + grace
    last_processed_close: Optional[pd.Timestamp] = None
    while True:
        target_ts = _next_5m_boundary(args.grace_seconds)
        _wait_until(target_ts, heartbeat_secs=float(args.heartbeat_seconds))

        # Build live features by fetching fresh candles instead of static CSVs
        token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
        try:
            X_panel, now_idx = _build_live_feature_row(instruments, cfg, args.environment, token)
        except Exception as e:
            import traceback
            print(f"feature build failed: {e!r}")
            traceback.print_exc()
            continue
        if len(X_panel.index) == 0:
            continue
        # Warn if data is stale compared to wall clock
        age_sec = (datetime.now(timezone.utc) - now_idx).total_seconds()
        if age_sec > 600:
            print(f"[warn] latest live bar age {age_sec/60.0:.1f}m (last={now_idx})")
        stale_bar = last_processed_close is not None and now_idx <= last_processed_close
        if stale_bar:
            # Stale/no new bar; log explicitly so it's visible in output
            print(
                f"{datetime.now(timezone.utc).isoformat()} no-new-bar: last={last_processed_close} current={now_idx}"
            )

        # Flatten columns in same order as training
        flat_cols: List[Tuple[str, str]] = []
        for inst in instruments:
            for col in X_panel[inst].columns:
                flat_cols.append((inst, col))
        X = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
        X.columns = [f"{i}::{c}" for i, c in X.columns]

        # Select last row for inference
        x_row = X.tail(1)
        out = forward(genome, x_row)  # shape (1, H)
        out_vals = out.reshape(-1)

        # For each instrument, map to target state using current net position as prev state
        all_pos = fetch_open_positions(api, account_id) if api else {}
        print(f"{datetime.now(timezone.utc).isoformat()} close={now_idx} decisions:{' (stale)' if stale_bar else ''}")
        send_orders: List[Tuple[str, int]] = []
        for idx, inst in enumerate(instruments):
            prev_units = int(all_pos.get(_safe_inst(inst), 0)) if api else 0
            prev_state = 0
            if prev_units > 0:
                prev_state = 1
            elif prev_units < 0:
                prev_state = -1
            new_state = _map_single_with_hysteresis(
                prev_state,
                float(out_vals[idx]),
                mode=("absolute" if cfg.threshold_mode == "absolute" else "band"),
                band_enter=cfg.band_enter, band_exit=cfg.band_exit,
                enter_long=cfg.enter_long, exit_long=cfg.exit_long,
                enter_short=cfg.enter_short, exit_short=cfg.exit_short,
            )
            target_units = 0
            if new_state > 0:
                target_units = int(args.units)
            elif new_state < 0:
                target_units = -int(args.units)
            delta = int(target_units - prev_units)
            print(f"  {inst}: prev_units={prev_units:+d} prev_state={prev_state:+d} out={out_vals[idx]:.4f} -> state={new_state:+d} target={target_units:+d} delta={delta:+d}")
            if delta != 0:
                send_orders.append((inst, delta))

        # Submit orders only on fresh bar
        if stale_bar:
            print("  stale bar -> no orders sent")
        else:
            for inst, delta in send_orders:
                if api:
                    submit_order(api, account_id, inst, delta, args.environment)
                else:
                    print(f"DRYRUN order: {inst} delta={delta}")
            if not send_orders:
                print("  no orders needed (in sync)")
            last_processed_close = now_idx


if __name__ == "__main__":
    main()
