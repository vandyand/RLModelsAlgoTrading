#!/usr/bin/env python3
"""
Align OANDA positions to target units per instrument.

- Reads target units mapping (instrument -> target units) from JSON file or stdin
- Fetches current net units per instrument from OANDA account
- Places market orders for the difference (target - current) per instrument
- Prints a single JSON array of results per instrument

Usage examples:
  # Pipe from inference
  python forex-rl/actor-critic/multi20_offline_infer.py \
    | python forex-rl/actor-critic/align_positions_to_targets.py --targets -

  # Or from saved JSON file
  python forex-rl/actor-critic/align_positions_to_targets.py --targets targets.json

Requires env vars or CLI flags for OANDA credentials:
  OANDA_DEMO_ACCOUNT_ID, OANDA_DEMO_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

# Import order helper
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from streamer.orders import place_market_order  # type: ignore


def read_targets(path_or_json: str) -> Dict[str, int]:
    """Read targets from stdin ('-'), a JSON string, or a file path.

    Accepts three forms:
      - '-' → read JSON from stdin
      - '{...}' → treat argument as inline JSON object
      - any other → treat as filesystem path and read file
    """
    s = (path_or_json or "").strip()
    if s == "-":
        data = sys.stdin.read()
    elif s.startswith("{"):
        data = s
    else:
        with open(s, "r", encoding="utf-8") as f:
            data = f.read()
    obj = json.loads(data)
    if isinstance(obj, dict):
        # instrument -> units
        return {str(k): int(v) for k, v in obj.items()}
    raise RuntimeError("Targets JSON must be an object mapping instrument->units")


essential_keys = ["instrument", "order_id", "fill_id", "units", "price", "time", "reason"]

def summarize_order(order_obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = (order_obj or {}).get("response", {})
        create = resp.get("orderCreateTransaction", {})
        fill = resp.get("orderFillTransaction", {})
        instrument = create.get("instrument") or fill.get("instrument")
        out = {
            "instrument": instrument,
            "order_id": create.get("id"),
            "fill_id": fill.get("id"),
            "units": fill.get("units") or create.get("units"),
            "price": fill.get("price"),
            "time": fill.get("time") or create.get("time"),
            "reason": fill.get("reason") or create.get("reason"),
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception:
        try:
            return {"raw": str(order_obj)[:400]}
        except Exception:
            return {"raw": "<unprintable>"}


def fetch_current_units_map(api: API, account_id: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for p in resp.get("positions", []):
            inst = p.get("instrument")
            try:
                long_u = float((p.get("long") or {}).get("units") or 0.0)
            except Exception:
                long_u = 0.0
            try:
                short_u = float((p.get("short") or {}).get("units") or 0.0)
            except Exception:
                short_u = 0.0
            out[str(inst)] = int(round(long_u + short_u))
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Align OANDA positions to target units per instrument")
    parser.add_argument("--targets", required=True, help="Path to targets JSON or '-' for stdin")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--account-suffix", type=int, choices=list(range(1, 10)), help="Override last 3 digits of account id with this 3-digit suffix (e.g., 2 -> '002')")
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--deadband-units", type=int, default=1, help="Skip orders with |delta| < deadband")
    parser.add_argument("--min-units", type=int, default=1, help="Minimum absolute units to submit an order")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--client-tag", default="align-positions")
    # --invert: optional boolean flag; supports forms: --invert, --invert true/false
    parser.add_argument(
        "--invert",
        nargs="?",
        const=True,
        default=False,
        type=lambda v: True if v is None else str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        help="Invert target units before aligning (negate all values)",
    )
    args = parser.parse_args()

    # Compute account id override with suffix if provided
    if args.account_suffix is not None:
        base = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
        if not base or len(base) < 3:
            raise RuntimeError("Base account id missing or malformed; provide --account-id or set OANDA_DEMO_ACCOUNT_ID")
        new_id = base[:-3] + f"{int(args.account_suffix):03d}"
        args.account_id = new_id

    if not args.account_id or not args.access_token:
        raise RuntimeError("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY or pass flags.")

    targets = read_targets(args.targets)
    # Optionally invert target units
    if args.invert:
        targets = {inst: -int(units) for inst, units in targets.items()}
    api = API(access_token=args.access_token, environment=args.environment)

    current_map = fetch_current_units_map(api, args.account_id)

    results: List[Dict[str, Any]] = []
    for inst, target in targets.items():
        current = int(current_map.get(inst, 0))
        delta = int(target) - current
        action = "SKIP"
        ord_sum: Optional[Dict[str, Any]] = None

        if abs(delta) >= max(args.deadband_units, args.min_units) and delta != 0:
            action = "ORDER"
            if args.dry_run:
                ord_sum = {"dry_run": True}
            else:
                try:
                    order = place_market_order(
                        api=api,
                        account_id=args.account_id,
                        instrument=inst,
                        units=delta,
                        tp_pips=None,
                        sl_pips=None,
                        anchor=None,
                        client_tag=args.client_tag,
                        client_comment="align to target",
                        fifo_safe=False,
                        fifo_adjust=False,
                    )
                    ord_sum = summarize_order(order)
                except Exception as exc:
                    action = "ERROR"
                    ord_sum = {"error": str(exc)}

        results.append({
            "instrument": inst,
            "current": current,
            "target": int(target),
            "delta": delta,
            "action": action,
            "order": ord_sum,
        })

    print(json.dumps(results))


if __name__ == "__main__":
    main()
