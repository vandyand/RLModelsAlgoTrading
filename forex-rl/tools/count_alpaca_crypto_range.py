#!/usr/bin/env python3
"""
Count Alpaca crypto bars in a given date range for specific symbols.

Usage (from forex-rl root, with ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY set):

  python -m tools.count_alpaca_crypto_range \\
    --symbols "SOL/USDC,SOL/USDT" \\
    --start 2023-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import pandas as pd


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment")
    return key, secret


def main() -> None:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.common.enums import Sort

    ap = argparse.ArgumentParser(description="Count Alpaca crypto bars in a given date range.")
    ap.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of Alpaca crypto symbols (e.g. SOL/USDC,SOL/USDT).",
    )
    ap.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD, inclusive).",
    )
    ap.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD, inclusive).",
    )
    args = ap.parse_args()

    symbols: List[str] = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    start_dt = pd.to_datetime(args.start, utc=True).to_pydatetime()
    # Make end inclusive by going to end-of-day
    end_dt = pd.to_datetime(args.end + " 23:59:59", utc=True).to_pydatetime()

    key, secret = _get_alpaca_keys()
    client = CryptoHistoricalDataClient(key, secret)

    results = {}
    for sym in symbols:
        req = CryptoBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            limit=None,
            sort=Sort.ASC,
        )
        try:
            barset = client.get_crypto_bars(request_params=req)
        except Exception as exc:
            results[sym] = {
                "status": "error",
                "reason": str(exc),
                "count": 0,
                "first_ts": None,
                "last_ts": None,
            }
            continue

        try:
            bars = barset[sym]
        except Exception:
            bars = getattr(barset, "data", {}).get(sym, []) or []

        if not bars:
            results[sym] = {
                "status": "no_data",
                "reason": None,
                "count": 0,
                "first_ts": None,
                "last_ts": None,
            }
            continue

        first_ts = bars[0].timestamp
        last_ts = bars[-1].timestamp
        results[sym] = {
            "status": "ok",
            "count": len(bars),
            "first_ts": first_ts.astimezone().isoformat().replace("+00:00", "Z"),
            "last_ts": last_ts.astimezone().isoformat().replace("+00:00", "Z"),
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

