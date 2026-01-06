#!/usr/bin/env python3
"""
Fetch Alpaca account equity and per-position PnL snapshot.

Outputs a single JSON object with:
- account-level fields (equity, cash, portfolio_value, equity_change_today, etc.)
- per-position PnL for all open positions (equity, crypto, etc.)

Usage (from forex-rl root):

  # Paper account (default)
  python -m forex-rl.tools.alpaca_account_pnl

  # Explicitly specify paper/live
  python -m forex-rl.tools.alpaca_account_pnl --paper true
  python -m forex-rl.tools.alpaca_account_pnl --paper false

Env:
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch Alpaca account equity and per-position PnL snapshot."
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
            "If set, only include crypto positions (asset_class contains 'CRYPTO') "
            "in the positions list and totals."
        ),
    )
    ap.add_argument(
        "--equities",
        action="store_true",
        help=(
            "If set, only include equity positions (asset_class contains 'EQUITY') "
            "in the positions list and totals."
        ),
    )
    args = ap.parse_args()

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    # --- Account-level snapshot ---
    account = trading.get_account()

    def _f(obj: Any, name: str, default: float = 0.0) -> float:
        try:
            val = getattr(obj, name, default)
            return float(val) if val is not None else float(default)
        except Exception:
            return float(default)

    equity = _f(account, "equity", 0.0)
    last_equity = _f(account, "last_equity", equity)
    cash = _f(account, "cash", 0.0)
    portfolio_value = _f(account, "portfolio_value", equity)
    long_market_value = _f(account, "long_market_value", 0.0)
    short_market_value = _f(account, "short_market_value", 0.0)

    equity_change_today = equity - last_equity
    equity_change_today_pct = (equity_change_today / last_equity) if last_equity not in (0.0, None) else 0.0

    acct_summary: Dict[str, Any] = {
        "account_id": str(getattr(account, "id", "")),
        "account_number": str(getattr(account, "account_number", "")),
        "status": str(getattr(account, "status", "")),
        "currency": str(getattr(account, "currency", "USD")),
        "paper": bool(args.paper),
        "equity": equity,
        "last_equity": last_equity,
        "equity_change_today": equity_change_today,
        "equity_change_today_pct": equity_change_today_pct,
        "cash": cash,
        "portfolio_value": portfolio_value,
        "long_market_value": long_market_value,
        "short_market_value": short_market_value,
        "buying_power": _f(account, "buying_power", 0.0),
        "daytrade_buying_power": _f(account, "daytrade_buying_power", 0.0),
        "created_at": str(getattr(account, "created_at", "")),
    }

    # --- Per-position PnL ---
    positions = trading.get_all_positions()
    pos_list: List[Dict[str, Any]] = []

    total_cost_basis = 0.0
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    total_unrealized_intraday_pl = 0.0

    for p in positions:
        sym = str(getattr(p, "symbol", ""))
        if not sym:
            continue

        # Optional filtering by asset class.
        asset_class_raw = str(getattr(p, "asset_class", ""))
        asset_class_upper = asset_class_raw.upper()
        if args.crypto or args.equities:
            is_crypto = "CRYPTO" in asset_class_upper
            is_equity = "EQUITY" in asset_class_upper
            # If only --crypto is set, require crypto.
            if args.crypto and not args.equities and not is_crypto:
                continue
            # If only --equities is set, require equity.
            if args.equities and not args.crypto and not is_equity:
                continue
            # If both flags are set, include union of crypto and equities; skip other asset classes.
            if args.crypto and args.equities and not (is_crypto or is_equity):
                continue

        try:
            qty = float(getattr(p, "qty", 0.0) or 0.0)
        except Exception:
            qty = 0.0

        market_value = _f(p, "market_value", 0.0)
        cost_basis = _f(p, "cost_basis", 0.0)
        unrealized_pl = _f(p, "unrealized_pl", 0.0)
        unrealized_plpc = _f(p, "unrealized_plpc", 0.0)
        unrealized_intraday_pl = _f(p, "unrealized_intraday_pl", 0.0)
        unrealized_intraday_plpc = _f(p, "unrealized_intraday_plpc", 0.0)

        total_cost_basis += cost_basis
        total_market_value += market_value
        total_unrealized_pl += unrealized_pl
        total_unrealized_intraday_pl += unrealized_intraday_pl

        pos_entry: Dict[str, Any] = {
            "symbol": sym,
            "asset_class": asset_class_raw,
            "exchange": str(getattr(p, "exchange", "")),
            "qty": qty,
            "side": str(getattr(p, "side", "")),
            "market_value": market_value,
            "cost_basis": cost_basis,
            "avg_entry_price": _f(p, "avg_entry_price", 0.0),
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc,
            "unrealized_intraday_pl": unrealized_intraday_pl,
            "unrealized_intraday_plpc": unrealized_intraday_plpc,
        }
        pos_list.append(pos_entry)

    totals = {
        "positions_count": len(pos_list),
        "positions_cost_basis": total_cost_basis,
        "positions_market_value": total_market_value,
        "positions_unrealized_pl": total_unrealized_pl,
        "positions_unrealized_intraday_pl": total_unrealized_intraday_pl,
    }

    snapshot = {
        "generated_at": _now_utc_iso(),
        "account": acct_summary,
        "positions": pos_list,
        "totals": totals,
    }

    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

