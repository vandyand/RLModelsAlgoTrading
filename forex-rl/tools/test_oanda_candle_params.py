#!/usr/bin/env python3
"""Quick smoke-test of OANDA candle parameter combinations.

This script talks directly to OANDA's /v3/instruments/{instrument}/candles endpoint
(using your OANDA_DEMO_KEY / OANDA_ACCESS_TOKEN) and prints summaries for a
few representative parameter combinations (price, granularity, count, from/to).

It does NOT use the local cache service.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import requests


def _env() -> str:
    return os.environ.get("OANDA_ENV", "practice")


def _base_url() -> str:
    return "https://api-fxpractice.oanda.com" if _env() == "practice" else "https://api-fxtrade.oanda.com"


def _token() -> str:
    tok = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
    if not tok:
        raise SystemExit("Missing OANDA_DEMO_KEY / OANDA_ACCESS_TOKEN")
    return tok


def _get(instrument: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_base_url()}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {_token()}"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def main() -> None:
    instrument = os.environ.get("TEST_INSTRUMENT", "EUR_USD")
    print(f"Environment={_env()} instrument={instrument}", file=sys.stderr)

    # 1) Simple mid, latest 10 M1
    resp1 = _get(instrument, {"price": "M", "granularity": "M1", "count": 10})
    cs1 = resp1.get("candles", [])
    print("CASE1 M1 M count=10:", json.dumps({
        "num": len(cs1),
        "first_complete": cs1[0].get("complete") if cs1 else None,
        "last_complete": cs1[-1].get("complete") if cs1 else None,
        "sample_keys": sorted(list(cs1[0].keys())) if cs1 else None,
    }, default=str))

    # 2) Bid+Ask, latest 10 M1
    resp2 = _get(instrument, {"price": "BA", "granularity": "M1", "count": 10})
    cs2 = resp2.get("candles", [])
    print("CASE2 M1 BA count=10:", json.dumps({
        "num": len(cs2),
        "first_keys": sorted(list(cs2[0].keys())) if cs2 else None,
        "has_bid": bool(cs2 and cs2[0].get("bid")),
        "has_ask": bool(cs2 and cs2[0].get("ask")),
    }, default=str))

    # 3) Mid+Bid+Ask, latest 10 M5
    resp3 = _get(instrument, {"price": "MBA", "granularity": "M5", "count": 10})
    cs3 = resp3.get("candles", [])
    print("CASE3 M5 MBA count=10:", json.dumps({
        "num": len(cs3),
        "first_keys": sorted(list(cs3[0].keys())) if cs3 else None,
    }, default=str))

    # 4) Range with from/to and includeFirst=true (S5, short window)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
    end = now.isoformat().replace("+00:00", "Z")
    resp4 = _get(instrument, {"price": "M", "granularity": "S5", "from": start, "to": end, "includeFirst": "true"})
    cs4 = resp4.get("candles", [])
    first_incomplete = next((c for c in cs4 if not c.get("complete", True)), None)
    print("CASE4 S5 range+includeFirst:", json.dumps({
        "num": len(cs4),
        "first_time": cs4[0].get("time") if cs4 else None,
        "last_time": cs4[-1].get("time") if cs4 else None,
        "any_incomplete": bool(first_incomplete),
    }, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
