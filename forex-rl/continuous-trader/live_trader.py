#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from oandapyV20 import API  # type: ignore
import oandapyV20.endpoints.accounts as accounts  # type: ignore
import oandapyV20.endpoints.positions as positions  # type: ignore
import oandapyV20.endpoints.pricing as pricing  # type: ignore

# Repo paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from streamer.orders import place_market_order  # type: ignore
try:
    import broker_ipc  # type: ignore
except Exception:
    broker_ipc = None  # type: ignore
try:
    from slippage import linear_ramp_bps  # type: ignore
except Exception:
    linear_ramp_bps = None  # type: ignore
CT_ROOT = os.path.join(FX_ROOT, "continuous-trader")
if CT_ROOT not in sys.path:
    sys.path.append(CT_ROOT)
import features as ct_features  # type: ignore
import instruments as ct_instruments  # type: ignore
import model as ct_model  # type: ignore


@dataclass
class Thresholds:
    enter_long: float = 0.7
    exit_long: float = 0.6
    enter_short: float = 0.3
    exit_short: float = 0.4


def fetch_nav(api: API, account_id: str) -> Optional[float]:
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        return float(resp.get("account", {}).get("NAV"))
    except Exception:
        return None


def get_net_units_oanda(api: API, account_id: str, instrument: str) -> int:
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for pos in resp.get("positions", []):
            if pos.get("instrument") != instrument:
                continue
            long_units = float(pos.get("long", {}).get("units") or 0.0)
            short_units = float(pos.get("short", {}).get("units") or 0.0)
            return int(round(long_units + short_units))
    except Exception:
        pass
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Live continuous trader for EUR_USD using single-output actor-critic")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--environment", choices=["practice","live"], default="practice")
    p.add_argument("--broker", choices=["oanda","ipc"], default="oanda")
    p.add_argument("--broker-account-id", type=int, default=int(os.environ.get("BROKER_ACCOUNT_ID", "1")))
    p.add_argument("--ipc-socket", default=os.environ.get("PRAGMAGEN_IPC_SOCKET", "/run/pragmagen/pragmagen.sock"))
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--instruments-csv", default=None, help="CSV listing the 68 instruments for features")
    p.add_argument("--poll-seconds", type=float, default=60.0, help="Polling interval for new M5 candle")
    p.add_argument("--max-units", type=int, default=1000)
    p.add_argument("--min-units", type=int, default=10)
    p.add_argument("--order-cooldown", type=float, default=5.0)
    # No TP/SL; exits purely via thresholds
    p.add_argument("--model-path", default="forex-rl/continuous-trader/checkpoints/offline_eurusd.pt")
    p.add_argument("--feature-config", default="m5,h1,d")
    p.add_argument("--thresholds", default="0.7,0.6,0.3,0.4", help="enter_long,exit_long,enter_short,exit_short")
    # Local broker sim costs and ramp
    p.add_argument("--sim-slippage-bps", type=float, default=float(os.environ.get("SIM_SLIPPAGE_BPS", "0")))
    p.add_argument("--sim-fee-perc", type=float, default=float(os.environ.get("SIM_FEE_PERC", "0")))
    p.add_argument("--sim-fee-fixed", type=float, default=float(os.environ.get("SIM_FEE_FIXED", "0")))
    p.add_argument("--slip-ramp-start-bps", type=float, default=float(os.environ.get("SLIP_RAMP_START_BPS", "0")))
    p.add_argument("--slip-ramp-target-bps", type=float, default=float(os.environ.get("SLIP_RAMP_TARGET_BPS", "1")))
    p.add_argument("--slip-ramp-days", type=float, default=float(os.environ.get("SLIP_RAMP_DAYS", "5")))
    p.add_argument("--slip-ramp-epoch-ts", type=float, default=float(os.environ.get("SLIP_RAMP_EPOCH_TS", "0")))
    args = p.parse_args()

    api: Optional[API] = None
    ipc_client: Optional["broker_ipc.BrokerIPCClient"] = None
    if args.broker == "oanda":
        if not args.account_id or not args.access_token:
            raise RuntimeError("Missing OANDA credentials in env or flags")
        api = API(access_token=args.access_token, environment=args.environment)
    else:
        if broker_ipc is None:
            raise RuntimeError("Local broker IPC client unavailable")
        ipc_client = broker_ipc.BrokerIPCClient(socket_path=args.ipc_socket)
    ramp_epoch = time.time() if float(args.slip_ramp_epoch_ts) == 0.0 else float(args.slip_ramp_epoch_ts)

    # Instruments universe for features
    instruments = ct_instruments.load_68(args.instruments_csv)
    if args.instrument not in instruments:
        instruments = [args.instrument] + instruments

    # Load model checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    input_dim = int(meta.get("input_dim"))
    latent_dim = int(meta.get("latent_dim", 64))

    ae = ct_model.AutoEncoder(input_dim=input_dim, hidden_dims=(2048, 512, 128), latent_dim=latent_dim)
    model = ct_model.ActorCriticSingle(encoder=ae.encoder, latent_dim=latent_dim, policy_hidden=256, value_hidden=256)
    try:
        ae.encoder.load_state_dict(ckpt["ae_encoder_state"])  # type: ignore
        model.policy.load_state_dict(ckpt["policy_state"])   # type: ignore
        model.value.load_state_dict(ckpt["value_state"])     # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}")
    model.eval()

    # Feature selection
    fc = ct_features.FeatureConfig(include_m5=True, include_h1=True, include_d=True)

    # Parse thresholds
    tl, xl, ts, xs = (float(x) for x in args.thresholds.split(','))
    th = Thresholds(enter_long=tl, exit_long=xl, enter_short=ts, exit_short=xs)

    last_order_time = 0.0

    while True:
        try:
            # Build latest feature grid (single row) across instruments and granularities
            Xrow = ct_features.build_feature_grid(instruments, args.environment, args.access_token, fc)
            if Xrow.empty:
                time.sleep(max(5.0, args.poll_seconds))
                continue
            # Standardize using saved stats
            stats = ckpt.get("feature_stats", {})
            Xn = Xrow.copy()
            for c in Xn.columns:
                m, s = stats.get(c, (0.0, 1.0)) if isinstance(stats, dict) else (0.0, 1.0)
                if s == 0:
                    s = 1.0
                Xn[c] = (Xn[c] - m) / s
            x = torch.tensor(Xn.values, dtype=torch.float32)

            with torch.no_grad():
                a, _, _ = model(x)
            out_val = float(a.item())  # in [0,1)

            # Trading rules
            if api is not None:
                net_units = get_net_units_oanda(api, args.account_id, args.instrument)
            else:
                # IPC positions
                net_units = 0
                try:
                    pr = ipc_client.get_positions(args.broker_account_id) if ipc_client else None
                    plist = (pr.data if pr and pr.ok else []) or []
                    for pos in plist:
                        sym = str(pos.get("symbol",""))
                        if sym.upper() == args.instrument.upper():
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
            now_ts = time.time()
            place: Optional[int] = None
            if net_units == 0:
                if out_val > th.enter_long:
                    place = +args.max_units
                elif out_val < th.enter_short:
                    place = -args.max_units
            elif net_units > 0:  # long open
                if out_val < th.exit_long:
                    place = -net_units  # close
            else:  # short open
                if out_val > th.exit_short:
                    place = -net_units  # close

            if place is not None and (now_ts - last_order_time) >= args.order_cooldown:
                try:
                    if api is not None:
                        order = place_market_order(
                            api=api,
                            account_id=args.account_id,
                            instrument=args.instrument,
                            units=place,
                            tp_pips=None,
                            sl_pips=None,
                            anchor=None,
                            client_tag="ct",
                            client_comment="continuous trader",
                            fifo_safe=False,
                            fifo_adjust=False,
                        )
                        print(json.dumps({"type": "ORDER", "out": out_val, "units": place, "order": order}), flush=True)
                    else:
                        side = "buy" if int(place) > 0 else "sell"
                        qty = abs(int(place))
                        cur_slip_bps = linear_ramp_bps(
                            start_bps=float(args.slip_ramp_start_bps),
                            target_bps=float(args.slip_ramp_target_bps),
                            start_epoch_ts=ramp_epoch,
                            ramp_days=float(args.slip_ramp_days),
                            now_ts=time.time(),
                        ) if linear_ramp_bps else float(args.sim_slippage_bps)
                        r = ipc_client.place_order(
                            account_id=int(args.broker_account_id),
                            symbol=str(args.instrument),
                            side=side,
                            quantity=float(qty),
                            order_type="market",
                            limit_price=None,
                            time_in_force="GTC",
                            sim_slippage_bps=float(cur_slip_bps),
                            sim_fee_perc=float(args.sim_fee_perc),
                            sim_fee_fixed=float(args.sim_fee_fixed),
                        ) if ipc_client else None
                        print(json.dumps({"type": "ORDER", "out": out_val, "units": place, "order": getattr(r, 'data', None)}), flush=True)
                except Exception as exc:
                    print(json.dumps({"type": "ORDER_ERROR", "error": str(exc)}), flush=True)
                last_order_time = now_ts
            else:
                print(json.dumps({"type": "HB", "out": out_val, "units": net_units}), flush=True)

            time.sleep(max(5.0, args.poll_seconds))
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
            time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    main()
