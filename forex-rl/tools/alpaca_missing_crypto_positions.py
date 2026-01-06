#!/usr/bin/env python3
"""
List crypto symbols in the target Alpaca universe for which there is currently no open position.

By default this:
- Loads the crypto target universe from `alpaca_crypto_target_symbols.json`.
- Connects to Alpaca (paper account by default).
- Fetches all open positions and filters to CRYPTO.
- Prints the subset of universe symbols that have *no* open crypto position.

Usage (from forex-rl root):

  # Paper account, default crypto universe
  python -m forex-rl.tools.alpaca_missing_crypto_positions

  # Custom universe JSON and live account
  python -m forex-rl.tools.alpaca_missing_crypto_positions \\
      --universe-json forex-rl/alpaca_crypto_target_symbols.json \\
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


def _canonical_crypto_symbol(sym: str) -> str:
    """
    Normalise a crypto symbol so that data/universe symbols like 'BTC/USD'
    match trading symbols like 'BTCUSD'.
    """
    return str(sym).replace("/", "").strip().upper()


def _load_universe_symbols(path: str) -> List[str]:
    if not os.path.exists(path):
        raise SystemExit(f"Universe JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    syms = data.get("symbols") or []
    if not isinstance(syms, list) or not syms:
        raise SystemExit(f"No 'symbols' list in universe JSON: {path}")
    # Keep the universe in its original slash format; we canonicalise only for matching.
    return [str(s).strip().upper() for s in syms if str(s).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List universe crypto symbols that currently have no open Alpaca position."
    )
    ap.add_argument(
        "--universe-json",
        type=str,
        default="forex-rl/alpaca_crypto_target_symbols.json",
        help="Path to universe JSON with {symbols:[...]} (default: alpaca_crypto_target_symbols.json).",
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
        # Keep behaviour consistent with the equity version: treat provided path
        # as a filename under the forex-rl root, regardless of intermediate dirs.
        uni_path = os.path.join(fx_root, os.path.basename(uni_path))

    # Universe as provided (e.g. 'BTC/USD', 'ETH/USD', ...).
    universe_syms = _load_universe_symbols(uni_path)
    # Canonical map: BTCUSD -> 'BTC/USD', etc.
    universe_canon_to_display = {
        _canonical_crypto_symbol(s): s for s in universe_syms
    }

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    positions = trading.get_all_positions()
    # Raw position symbols from Alpaca, e.g. 'BTCUSD'.
    open_crypto_syms: Set[str] = set()
    # Canonical forms for matching against the universe, e.g. 'BTCUSD'.
    open_crypto_syms_canon: Set[str] = set()

    for p in positions:
        sym_raw = str(getattr(p, "symbol", "")).strip()
        sym = sym_raw.upper()
        if not sym:
            continue
        asset_class = str(getattr(p, "asset_class", ""))
        # Alpaca-py typically formats this as 'AssetClass.CRYPTO'; we just
        # check for 'CRYPTO' substring to be robust.
        if "CRYPTO" not in asset_class.upper():
            continue
        open_crypto_syms.add(sym)
        open_crypto_syms_canon.add(_canonical_crypto_symbol(sym))

    # A universe symbol is "missing" if its canonical form does not appear
    # among the canonicalised open position symbols.
    missing: List[str] = [
        display_sym
        for canon_sym, display_sym in universe_canon_to_display.items()
        if canon_sym not in open_crypto_syms_canon
    ]

    payload: Dict[str, Any] = {
        "status": "ok",
        "paper": bool(args.paper),
        "universe_json": uni_path,
        "universe_size": len(universe_syms),
        "open_crypto_positions_count": len(open_crypto_syms),
        "missing_count": len(missing),
        "missing_symbols": missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

