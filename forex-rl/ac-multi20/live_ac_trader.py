from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
FX_ROOT = os.path.dirname(os.path.dirname(__file__))
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
try:
    import broker_ipc  # type: ignore
except Exception:
    broker_ipc = None  # type: ignore
try:
    from slippage import linear_ramp_bps  # type: ignore
except Exception:
    linear_ramp_bps = None  # type: ignore

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure stdout/stderr flush in nohup/systemd
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass

from feature_select import M5_FEATURES, H1_FEATURES, D_FEATURES


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.policy(h)
        v = self.value(h).squeeze(-1)
        return h, mu, v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actor-Critic Multi-20 live trader (M5 closes + grace)")
    p.add_argument("--model", required=True, help="Path to AC checkpoint (.pt) from ac-multi20 trainer")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    p.add_argument("--broker", choices=["oanda","ipc"], default="oanda")
    p.add_argument("--broker-account-id", type=int, default=int(os.environ.get("BROKER_ACCOUNT_ID", "1")))
    p.add_argument("--ipc-socket", default=os.environ.get("PRAGMAGEN_IPC_SOCKET", "/run/pragmagen/pragmagen.sock"))
    p.add_argument("--account-suffix", type=int, default=4, help="Override last 3 digits of account id (e.g., 4 -> '004')")
    p.add_argument("--instruments", default="", help="Comma-separated OANDA instruments; default loads from checkpoint meta")
    p.add_argument("--max-units", type=int, default=100, help="Absolute units when action=+/-1")
    p.add_argument("--grace-seconds", type=int, default=15)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--lookback-days", type=int, default=2)
    p.add_argument("--grans", default="", help="Override granularities (e.g., 'H1,D'); default from model meta")
    p.add_argument("--enter-long", type=float, default=None)
    p.add_argument("--exit-long", type=float, default=None)
    p.add_argument("--enter-short", type=float, default=None)
    p.add_argument("--exit-short", type=float, default=None)
    # Local broker sim controls and ramp
    p.add_argument("--sim-slippage-bps", type=float, default=float(os.environ.get("SIM_SLIPPAGE_BPS", "0")))
    p.add_argument("--sim-fee-perc", type=float, default=float(os.environ.get("SIM_FEE_PERC", "0")))
    p.add_argument("--sim-fee-fixed", type=float, default=float(os.environ.get("SIM_FEE_FIXED", "0")))
    p.add_argument("--slip-ramp-start-bps", type=float, default=float(os.environ.get("SLIP_RAMP_START_BPS", "0")))
    p.add_argument("--slip-ramp-target-bps", type=float, default=float(os.environ.get("SLIP_RAMP_TARGET_BPS", "1")))
    p.add_argument("--slip-ramp-days", type=float, default=float(os.environ.get("SLIP_RAMP_DAYS", "5")))
    p.add_argument("--slip-ramp-epoch-ts", type=float, default=float(os.environ.get("SLIP_RAMP_EPOCH_TS", "0")))
    return p.parse_args()


def _base_from_grans(grans: List[str]) -> str:
    for g in ["M5","H1","D"]:
        if g in grans:
            return g
    return "M5"


def _next_boundary(base: str, with_grace_sec: int) -> datetime:
    now = datetime.now(timezone.utc)
    if base == "H1":
        last_close = now.replace(minute=0, second=0, microsecond=0)
        current_grace = last_close + timedelta(seconds=with_grace_sec)
        if now < current_grace:
            return current_grace
        next_close = last_close + timedelta(hours=1)
        return next_close + timedelta(seconds=with_grace_sec)
    if base == "D":
        last_close = now.replace(hour=0, minute=0, second=0, microsecond=0)
        current_grace = last_close + timedelta(seconds=with_grace_sec)
        if now < current_grace:
            return current_grace
        next_close = last_close + timedelta(days=1)
        return next_close + timedelta(seconds=with_grace_sec)
    # default M5
    minute = (now.minute // 5) * 5
    last_close = now.replace(minute=minute, second=0, microsecond=0)
    current_grace = last_close + timedelta(seconds=with_grace_sec)
    if now < current_grace:
        return current_grace
    next_close = last_close + timedelta(minutes=5)
    return next_close + timedelta(seconds=with_grace_sec)


def _is_blackout(ts_utc: datetime, window_min: int = 15) -> bool:
    try:
        ts_et = ts_utc.astimezone(ZoneInfo("America/New_York"))
        return (ts_et.hour == 17) and (0 <= ts_et.minute < max(1, int(window_min)))
    except Exception:
        return False


def _wait_until(ts: datetime, heartbeat_secs: float = 60.0, label: str = "M5") -> None:
    next_hb = datetime.now(timezone.utc)
    while True:
        now = datetime.now(timezone.utc)
        if now >= ts:
            return
        if heartbeat_secs and now >= next_hb:
            eta = (ts - now).total_seconds()
            print(f"[heartbeat] {now.isoformat()} waiting {eta:.1f}s for next {label}+grace")
            next_hb = now + timedelta(seconds=max(5.0, float(heartbeat_secs)))
        time.sleep(min(1.0, max(0.05, (ts - now).total_seconds())))


def _safe_inst(inst: str) -> str:
    return inst.replace("/", "_")


def _load_ct_features_module():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "continuous-trader", "features.py")
    if not os.path.exists(path):
        raise RuntimeError(f"missing features.py at {path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ct_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create module spec for continuous-trader/features.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = mod  # type: ignore[index]
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _build_live_feature_row(instruments: List[str], environment: str, access_token: Optional[str], include_grans: List[str]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    ct = _load_ct_features_module()
    periods = sorted(list({15, 45, 135, 405}))
    blocks: List[pd.DataFrame] = []
    base_ts_list: List[pd.Timestamp] = []
    # Determine base granularity (lowest of included)
    priority = ["M5", "H1", "D"]
    base = next((g for g in priority if g in include_grans), "M5")
    for inst in instruments:
        inst_blocks: List[pd.DataFrame] = []
        # Choose base dataframe for timestamp
        if base == "M5":
            base_df = ct._fetch_fx_recent(inst, "M5", max(450, max(periods)), environment, access_token)
        elif base == "H1":
            base_df = ct._fetch_fx_recent(inst, "H1", max(450, max(periods)), environment, access_token)
        else:
            base_df = ct._fetch_fx_recent(inst, "D", max(900, max(periods)), environment, access_token)
        if base_df.empty:
            continue
        base_ts = base_df.index.max()
        base_ts_list.append(base_ts)
        # Conditionally include M5/H1/D feature rows, aligned to base_ts
        if "M5" in include_grans:
            df_m5 = ct._fetch_fx_recent(inst, "M5", max(450, max(periods)), environment, access_token)
            if not df_m5.empty:
                feats_m5 = ct.compute_indicator_grid(df_m5, prefix=f"M5_{inst}_", periods=periods)
                sel_m5 = [f"M5_{inst}_{name}" for name in M5_FEATURES if f"M5_{inst}_{name}" in feats_m5.columns]
                row_m5 = feats_m5.tail(1)[sel_m5].copy()
                row_m5.index = pd.DatetimeIndex([base_ts])
                row_m5.columns = [f"M5::{c.split(f'M5_{inst}_',1)[1]}" for c in row_m5.columns]
                inst_blocks.append(row_m5)
        if "H1" in include_grans:
            df_h1 = ct._fetch_fx_recent(inst, "H1", max(450, max(periods)), environment, access_token)
            if not df_h1.empty:
                feats_h1 = ct.compute_indicator_grid(df_h1, prefix=f"H1_{inst}_", periods=periods)
                sel_h1 = [f"H1_{inst}_{name}" for name in H1_FEATURES if f"H1_{inst}_{name}" in feats_h1.columns]
                row_h1 = feats_h1.tail(1)[sel_h1].copy()
                row_h1.index = pd.DatetimeIndex([base_ts])
                row_h1.columns = [f"H1::{c.split(f'H1_{inst}_',1)[1]}" for c in row_h1.columns]
                inst_blocks.append(row_h1)
        if "D" in include_grans:
            df_d = ct._fetch_fx_recent(inst, "D", max(900, max(periods)), environment, access_token)
            if not df_d.empty:
                feats_d = ct.compute_indicator_grid(df_d, prefix=f"D_{inst}_", periods=periods)
                sel_d = [f"D_{inst}_{name}" for name in D_FEATURES if f"D_{inst}_{name}" in feats_d.columns]
                row_d = feats_d.tail(1)[sel_d].copy()
                row_d.index = pd.DatetimeIndex([base_ts])
                row_d.columns = [f"D::{c.split(f'D_{inst}_',1)[1]}" for c in row_d.columns]
                inst_blocks.append(row_d)
        if inst_blocks:
            inst_block = pd.concat(inst_blocks, axis=1)
            inst_block.columns = pd.MultiIndex.from_product([[inst], inst_block.columns])
            blocks.append(inst_block)
    if not blocks or not base_ts_list:
        return pd.DataFrame(), pd.Timestamp(0, tz=timezone.utc)
    X_panel = pd.concat(blocks, axis=1)
    return X_panel, max(base_ts_list)


def fetch_open_positions(api, account_id: str, *, ipc_client: Optional["broker_ipc.BrokerIPCClient"] = None, broker_account_id: Optional[int] = None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        if api is not None:
            import oandapyV20.endpoints.positions as positions  # type: ignore
            resp = api.request(positions.OpenPositions(accountID=account_id))
            for p in resp.get("positions", []):
                inst = str(p.get("instrument") or "")
                long_u = float((p.get("long") or {}).get("units") or 0.0)
                short_u = float((p.get("short") or {}).get("units") or 0.0)
                out[inst] = int(round(long_u + short_u))
        elif ipc_client is not None and broker_account_id is not None:
            pr = ipc_client.get_positions(int(broker_account_id))
            plist = (pr.data if pr and pr.ok else []) or []
            for pos in plist:
                sym = str(pos.get("symbol",""))
                qty = pos.get("net")
                if isinstance(qty, (int,float)):
                    out[sym] = int(round(float(qty)))
                else:
                    side = str(pos.get("side",""))
                    q = float(pos.get("quantity") or pos.get("qty") or 0.0)
                    out[sym] = int(round(q if side.lower()=="buy" else (-q if side.lower()=="sell" else 0.0)))
    except Exception:
        pass
    return out


def submit_order(api, account_id: str, instrument: str, delta_units: int, *, ipc_client: Optional["broker_ipc.BrokerIPCClient"] = None, broker_account_id: Optional[int] = None, cur_slip_bps: Optional[float] = None, sim_fee_perc: float = 0.0, sim_fee_fixed: float = 0.0) -> None:
    if delta_units == 0:
        return
    try:
        if api is not None:
            import oandapyV20.endpoints.orders as orders  # type: ignore
            order = {
                "order": {
                    "instrument": _safe_inst(instrument),
                    "units": str(int(delta_units)),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                }
            }
            api.request(orders.OrderCreate(accountID=account_id, data=order))
            print(f"order: {instrument} delta={delta_units} -> ok")
        elif ipc_client is not None and broker_account_id is not None:
            side = "buy" if int(delta_units) > 0 else "sell"
            qty = abs(int(delta_units))
            r = ipc_client.place_order(
                account_id=int(broker_account_id),
                symbol=_safe_inst(instrument),
                side=side,
                quantity=float(qty),
                order_type="market",
                limit_price=None,
                time_in_force="GTC",
                sim_slippage_bps=float(cur_slip_bps or 0.0),
                sim_fee_perc=float(sim_fee_perc),
                sim_fee_fixed=float(sim_fee_fixed),
            )
            print(f"order: {instrument} delta={delta_units} -> ok (ipc) {getattr(r,'data',None)}")
    except Exception as e:
        print(f"order: {instrument} delta={delta_units} -> fail: {e}")


def main() -> None:
    args = parse_args()

    # Load checkpoint
    ckpt = torch.load(args.model, map_location="cpu")
    meta = ckpt.get("meta", {})
    col_order: List[str] = ckpt.get("col_order", [])
    stats: Dict[str, Tuple[float, float]] = ckpt.get("feature_stats", {})
    hidden = int(meta.get("hidden", 256))

    # Instruments
    if args.instruments.strip():
        instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    else:
        instruments = list(meta.get("instruments", []))
    if len(instruments) == 0:
        raise RuntimeError("No instruments specified and checkpoint has no instruments meta")

    # API / Broker
    api = None
    ipc_client = None
    try:
        from oandapyV20 import API  # type: ignore
        token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
        if args.broker == "oanda":
            if not token:
                raise RuntimeError("Missing OANDA_DEMO_KEY or OANDA_ACCESS_TOKEN in env")
            api = API(access_token=token, environment=args.environment)
        else:
            if broker_ipc is None:
                raise RuntimeError("Local broker IPC client unavailable")
            ipc_client = broker_ipc.BrokerIPCClient(socket_path=args.ipc_socket)
    except Exception:
        print("Warning: oandapyV20 not available or credentials missing; DRYRUN mode")

    # Resolve account id with suffix override (default 4)
    if args.broker == "oanda":
        base = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
        if not base or len(base) < 3:
            raise RuntimeError("Provide --account-id or set OANDA_DEMO_ACCOUNT_ID (>=3 chars)")
        account_id = base[:-3] + f"{int(args.account_suffix):03d}"
    else:
        account_id = str(args.broker_account_id)

    # Model
    input_dim = int(meta.get("input_dim"))
    num_inst = int(meta.get("num_instruments", len(instruments)))
    meta_grans = meta.get("grans", ["M5","H1","D"])  # default if missing
    model = ActorCritic(input_dim=input_dim, num_instruments=num_inst, hidden=hidden)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    print(f"live-ac start env={args.environment} acct={account_id} n_inst={len(instruments)} max_units={args.max_units}")

    last_bar_ts: Optional[pd.Timestamp] = None
    while True:
        grans = [g.strip().upper() for g in (args.grans or ",".join(meta_grans)).split(',') if g.strip()]
        base = _base_from_grans(grans)
        step = timedelta(minutes=5) if base == "M5" else (timedelta(hours=1) if base == "H1" else timedelta(days=1))
        target_ts = _next_boundary(base, args.grace_seconds)
        while _is_blackout(target_ts):
            target_ts = target_ts + step
        label = ("M5" if base == "M5" else ("H1" if base == "H1" else "D1"))
        _wait_until(target_ts, heartbeat_secs=float(args.heartbeat_seconds), label=label)

        token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
        try:
            X_panel, now_idx = _build_live_feature_row(instruments, args.environment, token, grans)
        except Exception as e:
            import traceback
            print(f"feature build failed: {e!r}")
            traceback.print_exc()
            continue
        if X_panel.empty:
            print("no features built; skipping")
            continue

        # Flatten and align columns
        flat_cols: List[Tuple[str, str]] = []
        for inst in instruments:
            for col in X_panel[inst].columns:
                flat_cols.append((inst, col))
        X = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
        X.columns = [f"{i}::{c}" for i, c in X.columns]
        x_last = X.tail(1)
        # Add missing and reorder to training order
        if col_order:
            for c in col_order:
                if c not in x_last.columns:
                    x_last[c] = 0.0
            x_last = x_last[col_order]
        # Standardize
        if stats:
            for c in x_last.columns:
                m, s = stats.get(c, (0.0, 1.0))
                if s == 0.0:
                    s = 1.0
                x_last[c] = (x_last[c] - m) / s

        # Model inference
        xt = torch.tensor(x_last.values, dtype=torch.float32)
        with torch.no_grad():
            _, mu, _ = model(xt)
            a = torch.sigmoid(mu)[0].cpu().numpy()
        # Fetch current positions
        pos = fetch_open_positions(api, account_id, ipc_client=ipc_client, broker_account_id=args.broker_account_id) if (api or ipc_client) else {}
        print(f"{datetime.now(timezone.utc).isoformat()} close={now_idx}")
        send: List[Tuple[str, int]] = []
        th = meta.get("thresholds", {})
        el = float(args.enter_long if args.enter_long is not None else th.get("enter_long", 0.8))
        xl = float(args.exit_long if args.exit_long is not None else th.get("exit_long", 0.6))
        es = float(args.enter_short if args.enter_short is not None else th.get("enter_short", 0.2))
        xs = float(args.exit_short if args.exit_short is not None else th.get("exit_short", 0.4))
        pos_state: Dict[str, int] = {}
        for i, inst in enumerate(instruments):
            prev = int(pos.get(_safe_inst(inst), 0)) if api else 0
            prev_state = 0
            if prev > 0:
                prev_state = 1
            elif prev < 0:
                prev_state = -1
            v = float(a[i])
            new_state = prev_state
            if prev_state == 0:
                if v > el:
                    new_state = 1
                elif v < es:
                    new_state = -1
            elif prev_state == 1:
                if v < xl:
                    new_state = 0
            elif prev_state == -1:
                if v > xs:
                    new_state = 0
            pos_state[inst] = new_state
            target = int(round(float(new_state) * int(args.max_units)))
            delta = target - prev
            print(f"  {inst}: a={float(a[i]):.4f} prev_state={prev_state:+d} new_state={new_state:+d} prev={prev:+d} target={target:+d} delta={delta:+d}")
            if delta != 0:
                send.append((inst, delta))
        if last_bar_ts is not None and now_idx <= last_bar_ts:
            print("  stale bar -> no orders sent")
        else:
            for inst, delta in send:
                if api:
                    submit_order(api, account_id, inst, delta)
                elif ipc_client:
                    cur_slip_bps = linear_ramp_bps(
                        start_bps=float(args.slip_ramp_start_bps),
                        target_bps=float(args.slip_ramp_target_bps),
                        start_epoch_ts=(time.time() if float(args.slip_ramp_epoch_ts)==0.0 else float(args.slip_ramp_epoch_ts)),
                        ramp_days=float(args.slip_ramp_days),
                        now_ts=time.time(),
                    ) if linear_ramp_bps else float(args.sim_slippage_bps)
                    submit_order(None, account_id, inst, delta, ipc_client=ipc_client, broker_account_id=args.broker_account_id, cur_slip_bps=float(cur_slip_bps), sim_fee_perc=float(args.sim_fee_perc), sim_fee_fixed=float(args.sim_fee_fixed))
                else:
                    print(f"DRYRUN order: {inst} delta={delta}")
            if not send:
                print("  no orders needed")
            last_bar_ts = now_idx


if __name__ == "__main__":
    main()
