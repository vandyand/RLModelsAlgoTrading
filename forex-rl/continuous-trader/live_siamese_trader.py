#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repo paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
CT_ROOT = os.path.join(FX_ROOT, "continuous-trader")
if CT_ROOT not in sys.path:
    sys.path.append(CT_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore
from streamer.orders import place_market_order  # type: ignore
try:
    import broker_ipc  # type: ignore
except Exception:
    broker_ipc = None  # type: ignore
try:
    from slippage import linear_ramp_bps  # type: ignore
except Exception:
    linear_ramp_bps = None  # type: ignore
import features as ct_features  # type: ignore
from model import SiameseMultiGranActorCritic  # type: ignore


@dataclass
class Thresholds:
    enter_long: float = 0.7
    exit_long: float = 0.6
    enter_short: float = 0.3
    exit_short: float = 0.4


def _safe_inst(inst: str) -> str:
    return (inst or "").replace('/', '_')


def _slash_inst(inst: str) -> str:
    return (inst or "").replace('_', '/')


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _append_prices_csv(instrument: str, granularity: str, df_new: pd.DataFrame) -> str:
    base_dir = os.path.join(CT_ROOT, "data")
    _ensure_dir(base_dir)
    safe = _safe_inst(instrument)
    path = os.path.join(base_dir, f"{safe}_{granularity}.csv")
    if df_new.empty:
        return path
    df_new = df_new.copy()
    df_new = df_new[["open","high","low","close","volume"]].astype(float)
    df_new.index.name = "timestamp"
    try:
        if os.path.exists(path):
            # Append only missing
            cur = pd.read_csv(path)
            if "timestamp" in cur.columns:
                cur["timestamp"] = pd.to_datetime(cur["timestamp"], utc=True)
                cur = cur.set_index("timestamp").sort_index()
            combo = pd.concat([cur, df_new], axis=0)
            combo = combo[~combo.index.duplicated(keep='last')].sort_index()
            combo.to_csv(path, index=True)
        else:
            df_new.to_csv(path, index=True)
    except Exception as exc:
        print(json.dumps({"status": "append_prices_error", "instrument": instrument, "gran": granularity, "error": str(exc)}), flush=True)
    return path


def _read_header_numeric_cols(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    df0 = pd.read_csv(path, nrows=1)
    cols = list(df0.columns)
    if not cols:
        return []
    # First column is timestamp
    return [c for c in cols[1:]]


def _append_features_csv(instrument: str, granularity: str, feats: pd.DataFrame) -> str:
    base_dir = os.path.join(CT_ROOT, "data", "features", granularity)
    _ensure_dir(base_dir)
    safe = _safe_inst(instrument)
    path = os.path.join(base_dir, f"{safe}_{granularity}_features.csv")
    if feats.empty:
        return path
    feats = feats.copy()
    feats.index.name = "timestamp"
    try:
        if os.path.exists(path):
            feats.to_csv(path, mode='a', header=False, index=True)
        else:
            feats.to_csv(path, mode='w', header=True, index=True)
    except Exception as exc:
        print(json.dumps({"status": "append_features_error", "instrument": instrument, "gran": granularity, "error": str(exc)}), flush=True)
    return path


def _build_latest_features_for_inst(inst: str, grans: List[str], periods: List[int], environment: str, access_token: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Fetch recent candles for inst per gran and compute indicator grid; return last-row DataFrames per gran."""
    out: Dict[str, pd.DataFrame] = {}
    inst_slash = _slash_inst(inst)
    for gran in grans:
        # Choose bar counts per gran
        if gran == "M5":
            count = 600
        elif gran == "H1":
            count = 720
        else:
            count = 1200
        adapter = OandaRestCandlesAdapter(instrument=inst, granularity=gran, environment=environment, access_token=access_token)
        rows = list(adapter.fetch(count=count))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()[['open','high','low','close','volume']].astype(float)
        if gran in ("D", "W"):
            df.index = df.index.normalize()
        # Compute features over the window
        feats = ct_features.compute_indicator_grid(df, prefix=f"{inst_slash}_", periods=periods)
        # Keep only last row for online inference
        out[gran] = feats.tail(1)
    return out


def _make_flat_row_from_feats(
    feats_by_gran: Dict[str, pd.DataFrame],
    feature_dir: str,
    idx_map: Dict[str, Dict[str, List[int]]],
    instruments: List[str],
) -> Tuple[np.ndarray, int]:
    """Construct a flat row matching training layout using headers from exported features.
    Returns (flat, total_dim).
    """
    grans = list(idx_map.keys())
    # Preload expected columns per (gran, inst)
    expected_cols: Dict[Tuple[str, str], List[str]] = {}
    total_dim = 0
    for gran in grans:
        for inst in instruments:
            p = os.path.join(feature_dir, gran, f"{_safe_inst(inst)}_{gran}_features.csv")
            cols = _read_header_numeric_cols(p)
            expected_cols[(gran, inst)] = cols
            total_dim += len(cols)

    # Initialize flat vector
    flat = np.zeros((1, total_dim), dtype=np.float32)
    # Fill per gran, per inst following the same ordering as training indices
    for gran, by_inst in idx_map.items():
        for inst, idxs in by_inst.items():
            if not idxs:
                continue
            cols = expected_cols.get((gran, inst), [])
            if len(cols) != len(idxs):
                # If header mismatch, align with whatever length we have
                cols = cols[:len(idxs)]
            # Extract row values for this gran/inst
            feats_df = feats_by_gran.get(gran)
            if feats_df is None or feats_df.empty:
                # Already zeros
                continue
            # Reindex columns to expected order
            row = feats_df.reindex(columns=cols, fill_value=0.0).iloc[0].values.astype(np.float32)
            # Place into flat vector
            flat[0, idxs[: len(row)]] = row
    return flat, total_dim


def resolve_account_id(base_account_id: Optional[str], suffix: Optional[int]) -> str:
    if suffix is None:
        if not base_account_id:
            raise RuntimeError("Missing OANDA account id; set OANDA_DEMO_ACCOUNT_ID or pass --account-id")
        return str(base_account_id)
    # Override last 3 digits with suffix
    base = base_account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    if not base or len(base) < 3:
        raise RuntimeError("Base account id missing or malformed; provide --account-id or set OANDA_DEMO_ACCOUNT_ID")
    return base[:-3] + f"{int(suffix):03d}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Live M5 trader using Siamese multi-gran actor-critic and exported features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--broker", choices=["oanda","ipc"], default="oanda")
    p.add_argument("--broker-account-id", type=int, default=int(os.environ.get("BROKER_ACCOUNT_ID", "1")))
    p.add_argument("--ipc-socket", default=os.environ.get("PRAGMAGEN_IPC_SOCKET", "/run/pragmagen/pragmagen.sock"))
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    p.add_argument("--account-suffix", type=int, default=1, choices=list(range(0, 1000)), help="Override last 3 digits of account id (e.g., 3 -> '003')")
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--model-path", default=os.path.join(CT_ROOT, "checkpoints", "m5_siamese.pt"))
    p.add_argument("--feature-dir", default=os.path.join(CT_ROOT, "data", "features"))
    p.add_argument("--periods", default="5,15,45,135,405", help="Indicator lookbacks for live features")
    p.add_argument("--poll-seconds", type=float, default=30.0, help="Fallback polling interval if not aligning to M5 or on errors")
    # Alignment flags: enabled by default; --no-align-to-m5 to disable
    p.add_argument("--align-to-m5", dest="align_to_m5", action="store_true", default=True, help="Sleep until next completed M5 bar")
    p.add_argument("--no-align-to-m5", dest="align_to_m5", action="store_false", help="Disable alignment and use fixed polling")
    p.add_argument("--m5-grace-seconds", type=float, default=15.0, help="Extra seconds after the 5-minute boundary before fetching")
    p.add_argument("--thresholds", default="0.7,0.6,0.3,0.4", help="enter_long,exit_long,enter_short,exit_short")
    p.add_argument("--units", type=int, default=100, help="Absolute units when in position")
    p.add_argument("--order-cooldown", type=float, default=5.0)
    # Local broker sim controls and ramp
    p.add_argument("--sim-slippage-bps", type=float, default=float(os.environ.get("SIM_SLIPPAGE_BPS", "0")))
    p.add_argument("--sim-fee-perc", type=float, default=float(os.environ.get("SIM_FEE_PERC", "0")))
    p.add_argument("--sim-fee-fixed", type=float, default=float(os.environ.get("SIM_FEE_FIXED", "0")))
    p.add_argument("--slip-ramp-start-bps", type=float, default=float(os.environ.get("SLIP_RAMP_START_BPS", "0")))
    p.add_argument("--slip-ramp-target-bps", type=float, default=float(os.environ.get("SLIP_RAMP_TARGET_BPS", "1")))
    p.add_argument("--slip-ramp-days", type=float, default=float(os.environ.get("SLIP_RAMP_DAYS", "5")))
    p.add_argument("--slip-ramp-epoch-ts", type=float, default=float(os.environ.get("SLIP_RAMP_EPOCH_TS", "0")))
    p.add_argument("--log-heartbeat-secs", type=float, default=60.0)
    args = p.parse_args()

    api = None
    ipc_client = None
    if args.broker == "oanda":
        if not args.access_token:
            raise RuntimeError("Missing OANDA access token; set OANDA_DEMO_KEY or pass --access-token")
        account_id = resolve_account_id(args.account_id, args.account_suffix)
    else:
        if broker_ipc is None:
            raise RuntimeError("Local broker IPC client unavailable")
        ipc_client = broker_ipc.BrokerIPCClient(socket_path=args.ipc_socket)
        account_id = str(args.broker_account_id)
    ramp_epoch = time.time() if float(args.slip_ramp_epoch_ts) == 0.0 else float(args.slip_ramp_epoch_ts)

    # Load checkpoint and reconstruct model
    ckpt = torch.load(args.model_path, map_location="cpu")
    idx_map = ckpt["indices"]
    cfg_saved = ckpt.get("cfg", {})
    feature_grans = cfg_saved.get("feature_grans", ["M5", "H1", "D"]) if isinstance(cfg_saved, dict) else ["M5", "H1", "D"]

    # Prepare network
    # Compute total input dim from indices
    total_dim = 0
    for gran in idx_map:
        for inst in idx_map[gran]:
            total_dim += len(idx_map[gran][inst])
    net = SiameseMultiGranActorCritic(
        flat_input_dim=total_dim,
        indices_by_gran_inst=idx_map,
        embed_dim=64,
        hidden_per_inst=256,
        policy_hidden=512,
        value_hidden=512,
        use_attention=True,
    )
    net.load_state_dict(ckpt["model_state"])  # type: ignore
    net.eval()

    # Thresholds
    tl, xl, ts, xs = (float(x) for x in args.thresholds.split(','))
    th = Thresholds(enter_long=tl, exit_long=xl, enter_short=ts, exit_short=xs)

    # Universe for online features: only the trained instruments inside indices
    instruments = list({inst for gran in idx_map for inst in idx_map[gran].keys()})

    # Periods list
    periods = [int(x) for x in (args.periods or "").split(',') if x.strip()]

    # State
    last_m5_ts: Optional[pd.Timestamp] = None
    last_log_ts = 0.0
    last_order_time = 0.0

    while True:
        loop_start = time.time()
        try:
            # 1) Fetch latest candles, compute last completed M5 timestamp
            adapter_m5 = OandaRestCandlesAdapter(instrument=args.instrument, granularity="M5", environment=args.environment, access_token=args.access_token)
            recent = list(adapter_m5.fetch(count=2))
            if not recent:
                time.sleep(max(5.0, args.poll_seconds))
                continue
            df_m5 = pd.DataFrame(recent)
            df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
            df_m5 = df_m5.set_index('timestamp').sort_index()
            cur_last = df_m5.index[-1]
            if last_m5_ts is not None and cur_last == last_m5_ts:
                # No new closed M5 yet
                if (loop_start - last_log_ts) >= args.log_heartbeat_secs:
                    print(json.dumps({"type": "HB", "m5": str(cur_last)}), flush=True)
                    last_log_ts = loop_start
                if args.align_to_m5:
                    # Sleep until next 5-min boundary + grace
                    next_due = last_m5_ts + pd.Timedelta(minutes=5)
                    now_utc = pd.Timestamp.now(tz="UTC")
                    sleep_s = float((next_due - now_utc).total_seconds()) + float(args.m5_grace_seconds)
                    if sleep_s < 1.0:
                        sleep_s = max(1.0, args.m5_grace_seconds)
                    time.sleep(sleep_s)
                else:
                    time.sleep(max(5.0, args.poll_seconds))
                continue

            # 2) Append latest candles to price CSVs for M5/H1/D
            # Fetch compact windows per gran and append
            feats_by_gran = _build_latest_features_for_inst(args.instrument, feature_grans, periods, args.environment, args.access_token)

            # Also append prices and features rows to disk for traceability
            for gran, feats in feats_by_gran.items():
                # Append price rows: use the same window candles we just fetched inside helper
                # Re-fetch matching candles narrowly (cheap) to guarantee a consistent latest row appended
                adapter = OandaRestCandlesAdapter(instrument=args.instrument, granularity=gran, environment=args.environment, access_token=args.access_token)
                rows = list(adapter.fetch(count=10))
                if rows:
                    df_rows = pd.DataFrame(rows)
                    df_rows['timestamp'] = pd.to_datetime(df_rows['timestamp'], utc=True)
                    df_rows = df_rows.set_index('timestamp').sort_index()[['open','high','low','close','volume']].astype(float)
                    _append_prices_csv(args.instrument, gran, df_rows.tail(1))
                # Append features last row
                _append_features_csv(args.instrument, gran, feats.tail(1))

            # 3) Build flat input row using headers from exported features
            flat_row, flat_dim = _make_flat_row_from_feats(
                feats_by_gran=feats_by_gran,
                feature_dir=args.feature_dir,
                idx_map=idx_map,
                instruments=instruments,
            )
            x = torch.tensor(flat_row, dtype=torch.float32)

            # 4) Inference
            with torch.no_grad():
                a, _, _ = net(x)
            out_val = float(a.item())

            # 5/6) Map to target units with hysteresis
            # Query current position
            try:
                if args.broker == "oanda":
                    from oandapyV20 import API  # type: ignore
                    import oandapyV20.endpoints.positions as positions  # type: ignore
                    api = API(access_token=args.access_token, environment=args.environment)
                    resp = api.request(positions.OpenPositions(accountID=account_id))
                    net_units = 0
                    for pos in resp.get("positions", []):
                        if pos.get("instrument") == _safe_inst(args.instrument):
                            long_u = float((pos.get("long") or {}).get("units") or 0.0)
                            short_u = float((pos.get("short") or {}).get("units") or 0.0)
                            net_units = int(round(long_u + short_u))
                            break
                else:
                    net_units = 0
                    pr = ipc_client.get_positions(int(args.broker_account_id)) if ipc_client else None
                    plist = (pr.data if pr and pr.ok else []) or []
                    for pos in plist:
                        sym = str(pos.get("symbol",""))
                        if sym.upper() == _safe_inst(args.instrument).upper():
                            qty = pos.get("net")
                            if isinstance(qty, (int,float)):
                                net_units = int(round(float(qty)))
                            else:
                                side = str(pos.get("side",""))
                                q = float(pos.get("quantity") or pos.get("qty") or 0.0)
                                net_units = int(round(q if side.lower()=="buy" else (-q if side.lower()=="sell" else 0.0)))
                            break
            except Exception:
                net_units = 0

            target_units: Optional[int] = None
            if net_units == 0:
                if out_val > th.enter_long:
                    target_units = +int(args.units)
                elif out_val < th.enter_short:
                    target_units = -int(args.units)
                else:
                    target_units = 0
            elif net_units > 0:
                if out_val < th.exit_long:
                    target_units = 0
            else:  # net_units < 0
                if out_val > th.exit_short:
                    target_units = 0

            # 7) If target different, place order to align
            action = "HOLD"
            order_summary: Optional[Dict[str, object]] = None
            if target_units is not None and target_units != net_units and (time.time() - last_order_time) >= args.order_cooldown:
                delta = int(target_units) - int(net_units)
                try:
                    if args.broker == "oanda":
                        order = place_market_order(
                            api=api,
                            account_id=account_id,
                            instrument=_safe_inst(args.instrument),
                            units=delta,
                            tp_pips=None,
                            sl_pips=None,
                            anchor=None,
                            client_tag="ct-live",
                            client_comment="continuous-trader live siamese",
                            fifo_safe=False,
                            fifo_adjust=False,
                        )
                        action = "ORDER"
                        # summarize minimal details
                        resp = (order or {}).get("response", {})
                        create = resp.get("orderCreateTransaction", {})
                        fill = resp.get("orderFillTransaction", {})
                        order_summary = {
                            "id": create.get("id"),
                            "fill_id": fill.get("id"),
                            "price": fill.get("price"),
                            "time": fill.get("time") or create.get("time"),
                            "reason": fill.get("reason") or create.get("reason"),
                            "delta": delta,
                        }
                    else:
                        side = "buy" if delta > 0 else "sell"
                        qty = abs(int(delta))
                        cur_slip_bps = linear_ramp_bps(
                            start_bps=float(args.slip_ramp_start_bps),
                            target_bps=float(args.slip_ramp_target_bps),
                            start_epoch_ts=ramp_epoch,
                            ramp_days=float(args.slip_ramp_days),
                            now_ts=time.time(),
                        ) if linear_ramp_bps else float(args.sim_slippage_bps)
                        r = ipc_client.place_order(
                            account_id=int(args.broker_account_id),
                            symbol=_safe_inst(args.instrument),
                            side=side,
                            quantity=float(qty),
                            order_type="market",
                            limit_price=None,
                            time_in_force="GTC",
                            sim_slippage_bps=float(cur_slip_bps),
                            sim_fee_perc=float(args.sim_fee_perc),
                            sim_fee_fixed=float(args.sim_fee_fixed),
                        ) if ipc_client else None
                        action = "ORDER"
                        order_summary = getattr(r, 'data', None)
                except Exception as exc:
                    action = "ORDER_ERROR"
                    order_summary = {"error": str(exc)}
                last_order_time = time.time()

            print(json.dumps({
                "type": action if action != "HOLD" else "HB",
                "m5": str(cur_last),
                "out": out_val,
                "net": net_units,
                "target": target_units,
                "order": order_summary,
            }), flush=True)

            last_m5_ts = cur_last
            # Sleep aligned to next M5 boundary if requested
            if args.align_to_m5:
                next_due = last_m5_ts + pd.Timedelta(minutes=5)
                now_utc = pd.Timestamp.now(tz="UTC")
                sleep_s = float((next_due - now_utc).total_seconds()) + float(args.m5_grace_seconds)
                if sleep_s < 1.0:
                    sleep_s = max(1.0, args.m5_grace_seconds)
                time.sleep(sleep_s)
            else:
                time.sleep(max(5.0, args.poll_seconds))
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as exc:
            print(json.dumps({"type": "ERROR", "error": str(exc)}), flush=True)
            time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    main()
