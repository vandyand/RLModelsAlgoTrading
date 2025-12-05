#!/usr/bin/env python3
"""Compare responses between OANDA and the local candle cache service.

Requirements:
- OANDA_DEMO_KEY / OANDA_ACCESS_TOKEN set
- OANDA_ENV set (practice|live, default practice)
- Candle cache service running and reachable (CANDLE_CACHE_BASE_URL or --cache-base)

This script issues an identical GET /v3/instruments/{instrument}/candles
request to OANDA and to the cache, then deep-compares the JSON responses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _get_oanda(instrument: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_base_url()}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {_token()}"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _get_cache(cache_base: str, instrument: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{cache_base.rstrip('/')}/v3/instruments/{instrument}/candles"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def deep_equal(a: Any, b: Any, *, ignore_top_time: bool = True, _depth: int = 0) -> bool:
    """Deep structural/value comparison with an option to ignore top-level 'time'.

    OANDA's top-level 'time' field is effectively 'server now' and will differ
    slightly between OANDA and the cache (and between runs). For realistic
    comparisons we ignore that single field at the root by default.
    """
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if ignore_top_time and _depth == 0:
            # Compare all keys except 'time' at the root level.
            ka = {k for k in a.keys() if k != "time"}
            kb = {k for k in b.keys() if k != "time"}
            if ka != kb:
                return False
            return all(deep_equal(a[k], b[k], ignore_top_time=ignore_top_time, _depth=_depth + 1) for k in ka)
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k], ignore_top_time=ignore_top_time, _depth=_depth + 1) for k in a.keys())
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y, ignore_top_time=ignore_top_time, _depth=_depth + 1) for x, y in zip(a, b))
    return a == b


def summarize_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ca = a.get("candles", [])
    cb = b.get("candles", [])
    out: Dict[str, Any] = {
        "len_oanda": len(ca),
        "len_cache": len(cb),
    }
    if ca and cb:
        out["first_oanda"] = ca[0]
        out["first_cache"] = cb[0]
        out["last_oanda"] = ca[-1]
        out["last_cache"] = cb[-1]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare OANDA vs candle cache responses")
    p.add_argument("--instrument", default=os.environ.get("TEST_INSTRUMENT", "EUR_USD"))
    p.add_argument("--granularity", default="M1")
    p.add_argument("--price", default="M")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--from-time", dest="from_time", default=None)
    p.add_argument("--to-time", dest="to_time", default=None)
    p.add_argument("--include-first", dest="include_first", action="store_true")
    p.add_argument("--cache-base", default=os.environ.get("CANDLE_CACHE_BASE_URL", "http://127.0.0.1:9100"))
    p.add_argument("--no-ignore-top-time", dest="ignore_top_time", action="store_false",
                   help="Do not ignore top-level 'time' when comparing (may cause spurious mismatches)")
    p.set_defaults(ignore_top_time=True)
    args = p.parse_args()

    params: Dict[str, Any] = {"price": args.price, "granularity": args.granularity}
    if args.from_time:
        params["from"] = args.from_time
    if args.to_time:
        params["to"] = args.to_time
    if not (args.from_time and args.to_time):
        params["count"] = args.count
    if args.from_time:
        params["includeFirst"] = "true" if args.include_first else "false"

    print("Request params:", json.dumps(params, indent=2), file=sys.stderr)

    # Warm the cache once so the second call is more likely to hit the DB
    # path instead of proxying directly to OANDA.
    _ = _get_cache(args.cache_base, args.instrument, params)

    oanda = _get_oanda(args.instrument, params)
    cache = _get_cache(args.cache_base, args.instrument, params)

    eq = deep_equal(oanda, cache, ignore_top_time=args.ignore_top_time)
    print(json.dumps({"equal": eq}, default=str))
    if not eq:
        summary = summarize_diff(oanda, cache)
        print(json.dumps({"mismatch_summary": summary}, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
