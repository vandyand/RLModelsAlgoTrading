#!/usr/bin/env python3
"""
Import Alpaca stock/crypto bars into the candle cache SQLite DB.

This mirrors the NQ CSV importer but uses the official `alpaca-py` SDK to
pull historical OHLCV bars and writes them into the existing
`candle_cache_service` schema so that:

- The existing HTTP cache service can serve equities/crypto via the same
  OANDA-compatible interface:

    GET /v3/instruments/{SYMBOL}/candles?granularity=D&price=M

- Downstream components (WFO, actor-critic, etc.) can query daily bars for
  stocks alongside FX/futures using the same path shape.

Environment:
  ALPACA_API_KEY_ID       Alpaca API key (paper/live)
  ALPACA_API_SECRET_KEY   Alpaca API secret

Notes:
- This script only touches historical *market data* (no trading), so using
  your paper-trading keys for both is safe. Alpaca's market data endpoint is
  shared between paper/live accounts.
- Bars are imported as OANDA-style "mid" candles:
    mid.o/mid.h/mid.l/mid.c = open/high/low/close
    volume                  = volume
  Bid/ask legs are left NULL.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pandas as pd

import candle_cache_service as ccs  # type: ignore

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment")
    return key, secret


def _parse_date(s: str) -> datetime:
    # Accept YYYY-MM-DD or full ISO-8601; always normalize to naive UTC for alpaca-py
    try:
        dt = pd.to_datetime(s, utc=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid date '{s}': {exc}")
    return dt.to_pydatetime()


def _bars_to_oanda_candles(symbol: str, bars: Sequence[Any]) -> List[Dict[str, Any]]:
    """
    Convert a sequence of Alpaca bar objects to the minimal
    OANDA candle JSON shape expected by candle_cache_service._upsert_candles.

    Supports both:
      - alpaca.data.models.Bar instances, and
      - plain dicts with keys like {"t","o","h","l","c","v"}.
    """
    candles: List[Dict[str, Any]] = []
    for b in bars:
        # Extract fields depending on whether `b` is a model instance or dict.
        if hasattr(b, "timestamp"):
            ts_dt = b.timestamp
            o = float(b.open)
            h = float(b.high)
            l = float(b.low)
            c = float(b.close)
            v = int(getattr(b, "volume", 0) or 0)
        else:
            # Fallback for dict-like entries from `.dict()` or raw JSON.
            ts_raw = b.get("t") or b.get("timestamp")
            ts_dt = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
            o = float(b.get("o", 0.0))
            h = float(b.get("h", 0.0))
            l = float(b.get("l", 0.0))
            c = float(b.get("c", 0.0))
            v = int(b.get("v") or b.get("volume") or 0)

        # Ensure RFC-3339 Z string
        ts = ts_dt.astimezone().isoformat().replace("+00:00", "Z")
        candles.append(
            {
                "time": ts,
                "complete": True,
                "mid": {
                    "o": o,
                    "h": h,
                    "l": l,
                    "c": c,
                },
                "volume": v,
            }
        )
    return candles


def _import_stock_bars(
    env: str,
    symbols: List[str],
    granularity: str,
    start: datetime,
    end: datetime | None,
) -> None:
    key, secret = _get_alpaca_keys()
    client = StockHistoricalDataClient(key, secret)

    if granularity.upper() != "D":
        raise SystemExit("Currently only daily (D) bars are supported for Alpaca imports")

    tf = TimeFrame.Day

    from alpaca.common.enums import Sort  # lazy import to avoid circulars

    # Alpaca returns up to 10k rows per call; we request full history once per symbol.
    # We let BarSet handle pagination internally.
    conn = ccs._connect_db()
    try:
        for symbol in symbols:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=None,
                sort=Sort.ASC,
            )
            barset = client.get_stock_bars(request_params=req)
            # BarSet behaves like a mapping: symbol -> List[Bar]
            try:
                bars = barset[symbol]
            except Exception:
                # Fallback to internal data mapping, if present.
                bars = getattr(barset, "data", {}).get(symbol, [])
            if not bars:
                print(f"[WARN] No bars returned for {symbol}; skipping")
                continue
            candles = _bars_to_oanda_candles(symbol, bars)
            print(f"{symbol}: importing {len(candles)} daily bars into env={env}")
            ccs._upsert_candles(conn, env, symbol, granularity.upper(), candles)
    finally:
        conn.close()


def _import_crypto_bars(
    env: str,
    symbols: List[str],
    granularity: str,
    start: datetime,
    end: datetime | None,
) -> None:
    """
    Optional helper to import Alpaca crypto bars into the same cache.
    Symbols must use Alpaca's crypto format, e.g. 'BTC/USD'.
    """
    key, secret = _get_alpaca_keys()
    client = CryptoHistoricalDataClient(key, secret)

    if granularity.upper() != "D":
        raise SystemExit("Currently only daily (D) bars are supported for Alpaca crypto imports")

    tf = TimeFrame.Day

    from alpaca.common.enums import Sort

    # The CryptoBarsRequest shares the same interface as StockBarsRequest.
    conn = ccs._connect_db()
    try:
        for symbol in symbols:
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=None,
                sort=Sort.ASC,
            )
            barset = client.get_crypto_bars(request_params=req)
            try:
                bars = barset[symbol]
            except Exception:
                bars = getattr(barset, "data", {}).get(symbol, [])
            if not bars:
                print(f"[WARN] No crypto bars returned for {symbol}; skipping")
                continue
            candles = _bars_to_oanda_candles(symbol, bars)
            print(f"{symbol}: importing {len(candles)} daily crypto bars into env={env}")
            ccs._upsert_candles(conn, env, symbol, granularity.upper(), candles)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Alpaca bars into candle_cache_service DB")
    ap.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g. SPY,QQQ,TLT) or Alpaca crypto symbols (e.g. BTC/USD)",
    )
    ap.add_argument(
        "--asset-class",
        type=str,
        default="stock",
        choices=["stock", "crypto"],
        help="Alpaca asset class to import (default: stock)",
    )
    ap.add_argument(
        "--granularity",
        type=str,
        default="D",
        help="Target granularity in OANDA-style code (currently only D is supported)",
    )
    ap.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Cache environment tag (default: candle_cache_service.OANDA_ENV, e.g. 'practice')",
    )
    ap.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD or ISO-8601)",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (inclusive, YYYY-MM-DD or ISO-8601). If omitted, uses Alpaca default (up to latest).",
    )
    args = ap.parse_args()

    env = str(args.environment or ccs.OANDA_ENV)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    start_dt = _parse_date(args.start)
    end_dt = _parse_date(args.end) if args.end else None

    print(
        f"Importing Alpaca {args.asset_class} bars for {len(symbols)} symbols into "
        f"DB={ccs.DB_PATH} env={env} granularity={args.granularity.upper()} "
        f"start={start_dt} end={end_dt}"
    )

    if args.asset_class == "stock":
        _import_stock_bars(env, symbols, args.granularity, start_dt, end_dt)
    else:
        _import_crypto_bars(env, symbols, args.granularity, start_dt, end_dt)

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()

