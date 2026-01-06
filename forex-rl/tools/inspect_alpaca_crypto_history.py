#!/usr/bin/env python3
"""
Inspect Alpaca crypto/USD daily history for a set of symbols.

For each symbol, queries Alpaca's historical crypto data API for daily bars
and reports:

- first available bar timestamp
- last available bar timestamp
- total number of daily bars
- span in days between first and last bar (inclusive)

Results are written to a JSON file for downstream analysis.

Environment:
  ALPACA_API_KEY_ID       Alpaca API key (paper/live)
  ALPACA_API_SECRET_KEY   Alpaca API secret

Usage example (from forex-rl root):

  python -m forex-rl.tools.inspect_alpaca_crypto_history \\
    --symbols "BTC/USD,ETH/USD,SOL/USD" \\
    --output-json forex-rl/alpaca_crypto_history.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment")
    return key, secret


DEFAULT_SYMBOLS: List[str] = [
    "AAVE/USD",
    "AVAX/USD",
    "BAT/USD",
    "BCH/USD",
    "BTC/USD",
    "CRV/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "GRT/USD",
    "LINK/USD",
    "LTC/USD",
    "PEPE/USD",
    "SHIB/USD",
    "SKY/USD",
    "SOL/USD",
    "SUSHI/USD",
    "TRUMP/USD",
    "UNI/USD",
    "XRP/USD",
    "XTZ/USD",
    "YFI/USD",
]


@dataclass
class SymbolHistory:
    symbol: str
    first_bar_ts: Optional[str]
    last_bar_ts: Optional[str]
    first_bar_date: Optional[str]
    last_bar_date: Optional[str]
    bar_count: int
    span_days: Optional[int]


def inspect_history(symbols: List[str], output_json: str, start: str) -> None:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.common.enums import Sort

    key, secret = _get_alpaca_keys()
    client = CryptoHistoricalDataClient(key, secret)

    # Parse start date once; Alpaca will clamp to first available bar per symbol.
    start_dt = pd.to_datetime(start, utc=True).to_pydatetime()

    results: Dict[str, SymbolHistory] = {}

    for sym in symbols:
        req = CryptoBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=None,
            limit=None,
            sort=Sort.ASC,
        )
        try:
            barset = client.get_crypto_bars(request_params=req)
        except Exception as exc:
            results[sym] = SymbolHistory(
                symbol=sym,
                first_bar_ts=None,
                last_bar_ts=None,
                first_bar_date=None,
                last_bar_date=None,
                bar_count=0,
                span_days=None,
            )
            print(json.dumps({"symbol": sym, "status": "error", "reason": str(exc)}), flush=True)
            continue

        try:
            bars = barset[sym]
        except Exception:
            bars = getattr(barset, "data", {}).get(sym, []) or []

        if not bars:
            results[sym] = SymbolHistory(
                symbol=sym,
                first_bar_ts=None,
                last_bar_ts=None,
                first_bar_date=None,
                last_bar_date=None,
                bar_count=0,
                span_days=None,
            )
            print(json.dumps({"symbol": sym, "status": "no_data"}), flush=True)
            continue

        first_ts = bars[0].timestamp
        last_ts = bars[-1].timestamp
        first_iso = first_ts.astimezone().isoformat().replace("+00:00", "Z")
        last_iso = last_ts.astimezone().isoformat().replace("+00:00", "Z")
        first_date = first_ts.date().isoformat()
        last_date = last_ts.date().isoformat()
        count = len(bars)
        span_days = (last_ts.date() - first_ts.date()).days + 1

        results[sym] = SymbolHistory(
            symbol=sym,
            first_bar_ts=first_iso,
            last_bar_ts=last_iso,
            first_bar_date=first_date,
            last_bar_date=last_date,
            bar_count=count,
            span_days=span_days,
        )
        print(
            json.dumps(
                {
                    "symbol": sym,
                    "status": "ok",
                    "first_bar": first_iso,
                    "last_bar": last_iso,
                    "bar_count": count,
                    "span_days": span_days,
                }
            ),
            flush=True,
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "timeframe": "D",
        "start_requested": start,
        "symbols": symbols,
        "results": {sym: asdict(hist) for sym, hist in results.items()},
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    tmp = output_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, output_json)
    print(json.dumps({"status": "written", "output_json": output_json}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect Alpaca crypto/USD daily history per symbol.")
    ap.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated list of Alpaca crypto symbols (e.g. BTC/USD,ETH/USD,...)",
    )
    ap.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Earliest date to request from Alpaca (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default="forex-rl/alpaca_crypto_history.json",
        help="Path to JSON file to write results to.",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    inspect_history(symbols, args.output_json, args.start)


if __name__ == "__main__":  # pragma: no cover
    main()

