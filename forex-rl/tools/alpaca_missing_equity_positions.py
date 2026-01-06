#!/usr/bin/env python3
"""
List equities in the target universe for which there is currently no open position.

By default this:
- Loads the 100-name equity universe from `equities_universe_top100_five_years.json`.
- Connects to Alpaca (paper account by default).
- Fetches all open positions and filters to US_EQUITY.
- Prints the subset of universe symbols that have *no* open equity position.

Usage (from forex-rl root):

  # Paper account, default universe
  python -m forex-rl.tools.alpaca_missing_equity_positions

  # Custom universe JSON and live account
  python -m forex-rl.tools.alpaca_missing_equity_positions \\
      --universe-json forex-rl/equities_universe_top100.json \\
      --paper false

Env:
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set

from alpaca.trading.client import TradingClient


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def _load_universe_symbols(path: str) -> List[str]:
    if not os.path.exists(path):
        raise SystemExit(f"Universe JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    syms = data.get("symbols") or []
    if not isinstance(syms, list) or not syms:
        raise SystemExit(f"No 'symbols' list in universe JSON: {path}")
    return [str(s).strip().upper() for s in syms if str(s).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List universe equities that currently have no open Alpaca position."
    )
    ap.add_argument(
        "--universe-json",
        type=str,
        default="forex-rl/equities_universe_top100_five_years.json",
        help="Path to universe JSON with {symbols:[...]} (default: equities_universe_top100_five_years.json).",
    )
    ap.add_argument(
        "--paper",
        type=lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        nargs="?",
        const=True,
        default=True,
        help="Use Alpaca paper trading account (default: True). Pass 'false' to use live.",
    )
    args = ap.parse_args()

    # Resolve universe path relative to forex-rl root if needed.
    uni_path = args.universe_json
    if not os.path.isabs(uni_path):
        here = os.path.abspath(os.path.dirname(__file__))
        fx_root = os.path.abspath(os.path.join(here, os.pardir))
        uni_path = os.path.join(fx_root, os.path.basename(uni_path)) if "forex-rl" in uni_path else os.path.join(
            fx_root, os.path.basename(uni_path)
        )

    universe_syms = _load_universe_symbols(uni_path)

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    positions = trading.get_all_positions()
    open_equity_syms: Set[str] = set()

    for p in positions:
        sym = str(getattr(p, "symbol", "")).strip().upper()
        if not sym:
            continue
        asset_class = str(getattr(p, "asset_class", ""))
        # Alpaca-py typically formats this as 'AssetClass.US_EQUITY'; we just
        # check for 'EQUITY' substring to be robust.
        if "EQUITY" not in asset_class.upper():
            continue
        open_equity_syms.add(sym)

    missing = [s for s in universe_syms if s not in open_equity_syms]

    payload: Dict[str, Any] = {
        "status": "ok",
        "paper": bool(args.paper),
        "universe_json": uni_path,
        "universe_size": len(universe_syms),
        "open_equity_positions_count": len(open_equity_syms),
        "missing_count": len(missing),
        "missing_symbols": missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

