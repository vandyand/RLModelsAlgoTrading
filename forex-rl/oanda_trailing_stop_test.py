#!/usr/bin/env python3
"""Quick helper to test OANDA trailing-stop order syntax/quantization.

Usage examples:
  python forex-rl/oanda_trailing_stop_test.py --instrument USD_JPY --distance-pips 8.5
  python forex-rl/oanda_trailing_stop_test.py --instrument GBP_JPY --units 10 --distance 0.08 --send

Defaults assume practice env, account suffix 4, and credentials provided via
OANDA_DEMO_ACCOUNT_ID / OANDA_DEMO_KEY env vars.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts_ep
import oandapyV20.endpoints.orders as orders_ep


def quantize(value: float, precision: int) -> float:
    if precision < 0:
        return value
    step = 10 ** (-precision)
    rounded = round(round(value / step) * step, precision)
    return max(step, rounded)


def fetch_instrument_spec(api: API, account_id: str, instrument: str) -> Dict[str, Any]:
    req = accounts_ep.AccountInstruments(
        accountID=account_id,
        params={"instruments": instrument},
    )
    resp = api.request(req)
    instruments = resp.get("instruments") or []
    if not instruments:
        raise RuntimeError(f"Instrument not tradeable for account: {instrument}")
    return instruments[0]


def build_payload(
    instrument: str,
    units: int,
    distance: float,
    display_precision: int,
    client_tag: Optional[str],
    client_comment: Optional[str],
) -> Dict[str, Any]:
    precision = max(0, int(display_precision))
    distance_str = f"{distance:.{precision}f}"
    order: Dict[str, Any] = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "trailingStopLossOnFill": {
                "distance": distance_str,
                "timeInForce": "GTC",
            },
        }
    }
    if client_tag or client_comment:
        order["order"]["clientExtensions"] = {}
        if client_tag:
            order["order"]["clientExtensions"]["tag"] = client_tag
        if client_comment:
            order["order"]["clientExtensions"]["comment"] = client_comment
    return order


def resolve_account(base: Optional[str], suffix: Optional[int]) -> str:
    if suffix is None:
        if not base:
            raise RuntimeError("Provide --account-id or set OANDA_DEMO_ACCOUNT_ID")
        return str(base)
    base_id = base or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    if not base_id or len(base_id) < 3:
        raise RuntimeError("Account id/suffix combo invalid; need >=3 chars base")
    return base_id[:-3] + f"{int(suffix):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OANDA trailing stop order precision")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--units", type=int, default=1)
    parser.add_argument("--distance", type=float, default=None, help="Trailing distance in price units")
    parser.add_argument("--distance-pips", type=float, default=None, help="Trailing distance in pips")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--account-suffix", type=int, default=4)
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--client-tag", default="trailing-test")
    parser.add_argument("--client-comment", default=None)
    parser.add_argument("--send", action="store_true", help="Actually submit the order (default: dry run)")
    args = parser.parse_args()

    if not args.access_token:
        raise RuntimeError("Provide --access-token or set OANDA_DEMO_KEY")
    account_id = resolve_account(args.account_id, args.account_suffix)
    api = API(access_token=args.access_token, environment=args.environment)

    spec = fetch_instrument_spec(api, account_id, args.instrument)
    pip_location = int(spec.get("pipLocation", -4))
    pip_size = 10 ** pip_location
    display_precision = int(spec.get("displayPrecision", max(1, -pip_location)))
    min_trailing = float(spec.get("minimumTrailingStopDistance") or 0.0)

    if args.distance is None and args.distance_pips is None:
        raise RuntimeError("Provide --distance or --distance-pips")
    base_distance = args.distance if args.distance is not None else (float(args.distance_pips) * pip_size)
    if base_distance <= 0:
        raise RuntimeError("Distance must be positive")
    quantized = quantize(base_distance, display_precision)
    if quantized < min_trailing:
        quantized = max(min_trailing, quantized)

    payload = build_payload(
        instrument=args.instrument,
        units=args.units,
        distance=quantized,
        display_precision=display_precision,
        client_tag=args.client_tag,
        client_comment=args.client_comment,
    )

    info = {
        "account_id": account_id,
        "instrument": args.instrument,
        "units": args.units,
        "pip_size": pip_size,
        "pip_location": pip_location,
        "display_precision": display_precision,
        "min_trailing_distance": min_trailing,
        "raw_distance": base_distance,
        "quantized_distance": quantized,
    }
    print(json.dumps({"type": "PREVIEW", "info": info, "payload": payload}, indent=2))

    if args.send:
        resp = api.request(orders_ep.OrderCreate(accountID=account_id, data=payload))
        print(json.dumps({"type": "RESPONSE", "data": resp}, indent=2))
    else:
        print("(dry run - pass --send to actually place the test order)")


if __name__ == "__main__":
    main()
