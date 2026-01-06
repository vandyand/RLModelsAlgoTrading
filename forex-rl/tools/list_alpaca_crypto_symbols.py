#!/usr/bin/env python3
"""
List all Alpaca crypto symbols (asset_class=CRYPTO) via the trading API.

Usage (from forex-rl root, with ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY set):

  python -m tools.list_alpaca_crypto_symbols
"""

from __future__ import annotations

import json
import os


def main() -> None:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass
        from alpaca.trading.requests import GetAssetsRequest
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"alpaca-py trading components not available: {exc}")

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment")

    client = TradingClient(key, secret)
    assets = client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
    symbols = sorted(a.symbol for a in assets)

    payload = {
        "count": len(symbols),
        "symbols": symbols,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

