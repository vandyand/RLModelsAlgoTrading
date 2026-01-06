#!/usr/bin/env python3
"""
Submit a single Alpaca market order (equity or crypto).

This is a low-level utility to quickly test order placement, including
short-selling an equity on the paper account.

Usage examples (from forex-rl root, with ALPACA_API_* set):

  # Short approx $100 notional of SPY on paper
  python -m forex-rl.tools.alpaca_market_order --symbol SPY --side sell --notional 100

  # Go long 5 shares of AAPL
  python -m forex-rl.tools.alpaca_market_order --symbol AAPL --side buy --qty 5

Env:
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a single Alpaca market order (equity or crypto).")
    ap.add_argument(
        "--symbol",
        required=True,
        help="Symbol to trade (e.g., SPY, AAPL, BTCUSD or BTC/USD for crypto).",
    )
    ap.add_argument(
        "--side",
        required=True,
        choices=["buy", "sell", "BUY", "SELL"],
        help="Order side: buy or sell (sell from flat creates a short for equities).",
    )
    ap.add_argument(
        "--qty",
        type=float,
        default=None,
        help="Share/coin quantity for the order. Mutually exclusive with --notional.",
    )
    ap.add_argument(
        "--notional",
        type=float,
        default=None,
        help="Dollar notional for the order. Mutually exclusive with --qty.",
    )
    ap.add_argument(
        "--paper",
        type=lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        nargs="?",
        const=True,
        default=True,
        help="Use Alpaca paper trading account (default: True). Pass 'false' to use live.",
    )
    ap.add_argument(
        "--crypto",
        action="store_true",
        help=(
            "Treat symbol as crypto; uses GTC time-in-force. "
            "If not provided, the script will auto-detect crypto via Alpaca's asset_class."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request that would be sent, but do not submit.",
    )
    args = ap.parse_args()

    if (args.qty is None and args.notional is None) or (
        args.qty is not None and args.notional is not None
    ):
        raise SystemExit("You must specify exactly one of --qty or --notional.")

    side = OrderSide.BUY if str(args.side).lower() == "buy" else OrderSide.SELL

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    # Determine whether this is a crypto order.
    # Priority:
    # 1) Explicit --crypto flag
    # 2) Asset lookup from Alpaca (asset_class contains 'CRYPTO')
    # 3) Simple heuristic on symbol format as a fallback.
    is_crypto = bool(args.crypto)
    asset_class_str: Optional[str] = None
    if not is_crypto:
        try:
            asset = trading.get_asset(args.symbol)
            asset_class_str = str(getattr(asset, "asset_class", "")).upper()
            if "CRYPTO" in asset_class_str:
                is_crypto = True
        except Exception:
            # Ignore and fall back to simple heuristics.
            pass

    if not is_crypto:
        sym_upper = str(args.symbol).upper()
        if "/" in sym_upper or sym_upper.endswith("USD") or sym_upper.endswith("USDT"):
            is_crypto = True

    # For crypto, Alpaca does not support DAY time-in-force; use GTC by default.
    tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

    order_kwargs: Dict[str, Any] = {
        "symbol": args.symbol,
        "side": side,
        "time_in_force": tif,
    }
    if args.qty is not None:
        order_kwargs["qty"] = args.qty
    else:
        order_kwargs["notional"] = round(float(args.notional), 2)

    req = MarketOrderRequest(**order_kwargs)  # type: ignore[arg-type]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "paper": bool(args.paper),
                    "request": {
                        "symbol": args.symbol,
                        "side": str(side),
                        "time_in_force": str(tif),
                        "qty": args.qty,
                        "notional": float(order_kwargs.get("notional") or 0.0)
                        if "notional" in order_kwargs
                        else None,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    try:
        resp = trading.submit_order(order_data=req)
        # qty / notional may be absent or string-typed on the response; handle defensively.
        raw_qty = getattr(resp, "qty", None)
        if raw_qty is None:
            qty_val: Optional[float] = float(args.qty) if args.qty is not None else None
        else:
            try:
                qty_val = float(raw_qty)
            except Exception:
                qty_val = None

        raw_notional = getattr(resp, "notional", None)
        if raw_notional is None and "notional" in order_kwargs:
            notional_val: Optional[float] = float(order_kwargs["notional"])
        elif raw_notional is None:
            notional_val = None
        else:
            try:
                notional_val = float(raw_notional)
            except Exception:
                notional_val = None

        out: Dict[str, Optional[Any]] = {
            "symbol": str(getattr(resp, "symbol", args.symbol)),
            "side": str(getattr(resp, "side", side)),
            "qty": qty_val,
            "notional": notional_val,
            "id": str(getattr(resp, "id", "")),
            "status": str(getattr(resp, "status", "")),
            "client_order_id": str(getattr(resp, "client_order_id", "")),
        }
        print(json.dumps({"status": "ok", "paper": bool(args.paper), "order": out}, indent=2, sort_keys=True))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "paper": bool(args.paper),
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()

