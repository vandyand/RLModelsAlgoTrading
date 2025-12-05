#!/usr/bin/env python3
from __future__ import annotations

import os
import urllib.parse
from typing import List

import requests


def _http_base() -> str:
    host = os.environ.get("QUESTDB_HOST", "127.0.0.1")
    port = int(os.environ.get("QUESTDB_HTTP_PORT", "9000"))
    return f"http://{host}:{port}"


def run_sql(query: str, timeout: float = 5.0) -> dict:
    base = _http_base()
    url = f"{base}/exec"
    params = {"query": query}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"status": r.status_code, "text": r.text[:2000]}


def ensure_fx_schema() -> None:
    ddls: List[str] = [
        # Top-of-book ticks from OANDA PricingStream
        (
            "CREATE TABLE IF NOT EXISTS fx_ticks ("
            "ts TIMESTAMP,"
            "instrument SYMBOL,"
            "bid DOUBLE, ask DOUBLE,"
            "bid_liquidity LONG, ask_liquidity LONG,"
            "closeout_bid DOUBLE, closeout_ask DOUBLE,"
            "tradeable BOOLEAN,"
            "status SYMBOL"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
        # DOM levels (all book entries per tick)
        (
            "CREATE TABLE IF NOT EXISTS fx_dom ("
            "ts TIMESTAMP,"
            "instrument SYMBOL,"
            "side SYMBOL,"
            "level INT,"
            "price DOUBLE,"
            "liquidity LONG"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
    ]
    for sql in ddls:
        run_sql(sql)


def ensure_crypto_schema() -> None:
    ddls: List[str] = [
        (
            "CREATE TABLE IF NOT EXISTS crypto_ticker ("
            "ts TIMESTAMP,"
            "symbol SYMBOL,"
            "last DOUBLE, open DOUBLE, high DOUBLE, low DOUBLE,"
            "base_volume DOUBLE, quote_volume DOUBLE, change_rate DOUBLE"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
        (
            "CREATE TABLE IF NOT EXISTS crypto_price ("
            "ts TIMESTAMP,"
            "symbol SYMBOL,"
            "mark_price DOUBLE, index_price DOUBLE, funding_rate DOUBLE,"
            "funding_time LONG, next_funding_time LONG"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
        (
            "CREATE TABLE IF NOT EXISTS crypto_depth ("
            "ts TIMESTAMP,"
            "symbol SYMBOL,"
            "side SYMBOL,"
            "level INT,"
            "price DOUBLE,"
            "qty DOUBLE"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
        (
            "CREATE TABLE IF NOT EXISTS crypto_trade ("
            "ts TIMESTAMP,"
            "symbol SYMBOL,"
            "price DOUBLE,"
            "qty DOUBLE"
            ") TIMESTAMP(ts) PARTITION BY DAY WAL"
        ),
    ]
    for sql in ddls:
        run_sql(sql)


__all__ = [
    "run_sql",
    "ensure_fx_schema",
    "ensure_crypto_schema",
]
