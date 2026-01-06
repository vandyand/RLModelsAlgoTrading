#!/usr/bin/env python3
"""
Cancel all open Alpaca orders (paper or live).

This is a small utility to wipe the current order book for an Alpaca account.
It calls the Trading API's `cancel_orders` endpoint, which attempts to cancel
all outstanding orders.

Usage (from forex-rl root):

  # Cancel all open orders on the paper account
  python -m tools.cancel_all_alpaca_orders --paper

  # Or explicitly:
  python -m tools.cancel_all_alpaca_orders --paper true

Env:
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set for the target account.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def main() -> None:
    ap = argparse.ArgumentParser(description="Cancel all open Alpaca orders (paper or live).")
    ap.add_argument(
        "--paper",
        type=lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        nargs="?",
        const=True,
        default=True,
        help="Use Alpaca paper trading account (default: True). Pass 'false' to use live.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which account would be used but do not send the cancel request.",
    )
    args = ap.parse_args()

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    account = trading.get_account()
    info: Dict[str, Any] = {
        "account_id": str(getattr(account, "id", "")),
        "account_number": str(getattr(account, "account_number", "")),
        "equity": float(getattr(account, "equity", 0.0)),
        "paper": bool(args.paper),
    }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "account": info,
                    "message": "No cancel_orders call made (dry-run).",
                }
            )
        )
        return

    try:
        resp = trading.cancel_orders()
        # resp is a list of CancelOrderResponse models or raw JSON;
        # normalize to something JSON-serializable.
        out: List[Dict[str, Any]] = []
        for r in resp or []:
            try:
                # Try to access attributes like .id / .status; fallback to dict(r).
                entry = {
                    "id": str(getattr(r, "id", "")),
                    "status": str(getattr(r, "status", "")),
                }
            except Exception:
                try:
                    entry = dict(r)  # type: ignore[arg-type]
                except Exception:
                    entry = {"raw": str(r)}
            out.append(entry)
        print(
            json.dumps(
                {
                    "status": "cancel_orders_called",
                    "account": info,
                    "results": out,
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "account": info,
                    "error": str(exc),
                }
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()

