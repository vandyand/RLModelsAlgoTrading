#!/usr/bin/env python3
"""
Basic OANDA connectivity validator for REST and Streaming.

- Verifies credentials from environment
- Resolves account id with optional 3-digit suffix override
- Tests Accounts Summary, Instruments Candles, Pricing REST
- Optional short PricingStream test to validate streaming channel

Examples:
  python forex-rl/tools/test_oanda_connectivity.py --instrument EUR_USD --environment practice --stream-seconds 5
  python forex-rl/tools/test_oanda_connectivity.py --instrument EUR_USD --account-id 001-011-123-000-001 --account-suffix 4

Environment:
  OANDA_DEMO_KEY (or OANDA_ACCESS_TOKEN) and OANDA_DEMO_ACCOUNT_ID typically set for practice
"""
import argparse
import json
import os
import socket
import sys
import time
from typing import Any, Dict, Optional


def mask_token(tok: Optional[str]) -> str:
    if not tok:
        return "(missing)"
    if len(tok) <= 6:
        return "*" * len(tok)
    return tok[:3] + "*" * (len(tok) - 6) + tok[-3:]


def resolve_account(base: Optional[str], suffix: Optional[int]) -> str:
    if not base:
        raise RuntimeError("Missing base account id. Provide --account-id or set OANDA_DEMO_ACCOUNT_ID")
    if suffix is None:
        return str(base)
    b = str(base)
    if len(b) < 3:
        raise RuntimeError("Account id must be at least 3 chars to use --account-suffix")
    return b[:-3] + f"{int(suffix):03d}"


def default_hosts(environment: str) -> Dict[str, str]:
    if environment == "practice":
        return {
            "rest": "api-fxpractice.oanda.com",
            "stream": "stream-fxpractice.oanda.com",
        }
    return {
        "rest": "api-fxtrade.oanda.com",
        "stream": "stream-fxtrade.oanda.com",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="OANDA connectivity validator")
    p.add_argument("--environment", choices=["practice","live"], default="practice")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    p.add_argument("--account-suffix", type=int, default=None)
    p.add_argument("--stream-seconds", type=float, default=5.0, help="Seconds to listen to PricingStream (0 to skip)")
    args = p.parse_args()

    token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
    account_id = resolve_account(args.account_id, args.account_suffix)

    print(json.dumps({
        "stage": "config",
        "environment": args.environment,
        "instrument": args.instrument,
        "account_id": account_id,
        "token_present": bool(token),
        "token_masked": mask_token(token),
        "proxies": {
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            "NO_PROXY": os.environ.get("NO_PROXY"),
        }
    }), flush=True)

    # DNS sanity for known hosts
    hosts = default_hosts(args.environment)
    for label, host in hosts.items():
        try:
            ip = socket.gethostbyname(host)
            print(json.dumps({"stage": "dns", "label": label, "host": host, "ip": ip}), flush=True)
        except Exception as e:
            print(json.dumps({"stage": "dns", "label": label, "host": host, "error": str(e)}), flush=True)

    # REST tests
    try:
        from oandapyV20 import API  # type: ignore
        import oandapyV20.endpoints.accounts as accounts  # type: ignore
        import oandapyV20.endpoints.instruments as instruments_ep  # type: ignore
        import oandapyV20.endpoints.pricing as pricing  # type: ignore
    except Exception as e:
        print(json.dumps({"stage": "import", "error": f"oandapyV20 not available: {e}"}), flush=True)
        sys.exit(1)

    if not token:
        print(json.dumps({"stage": "auth", "error": "Missing OANDA_DEMO_KEY / OANDA_ACCESS_TOKEN"}), flush=True)
        sys.exit(2)

    api = API(access_token=token, environment=args.environment)

    # Account summary
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        acct = resp.get("account", {})
        print(json.dumps({
            "stage": "account_summary",
            "NAV": acct.get("NAV"),
            "balance": acct.get("balance"),
            "currency": acct.get("currency"),
        }), flush=True)
    except Exception as e:
        print(json.dumps({"stage": "account_summary", "error": str(e)}), flush=True)

    # One candle (S5) to REST host
    try:
        req = instruments_ep.InstrumentsCandles(instrument=args.instrument, params={"granularity": "S5", "count": 5, "price": "M"})
        resp = api.request(req)
        cs = resp.get("candles", [])
        print(json.dumps({
            "stage": "candles",
            "count": len(cs),
            "first_time": (cs[0].get("time") if cs else None),
            "last_time": (cs[-1].get("time") if cs else None),
        }), flush=True)
    except Exception as e:
        print(json.dumps({"stage": "candles", "error": str(e)}), flush=True)

    # Pricing REST snapshot
    try:
        req = pricing.PricingInfo(accountID=account_id, params={"instruments": args.instrument})
        resp = api.request(req)
        prices = resp.get("prices", [])
        best = prices[0] if prices else {}
        print(json.dumps({
            "stage": "pricing_info",
            "instrument": best.get("instrument"),
            "bids": best.get("bids"),
            "asks": best.get("asks"),
        }, default=str), flush=True)
    except Exception as e:
        print(json.dumps({"stage": "pricing_info", "error": str(e)}), flush=True)

    # Streaming (optional)
    if args.stream_seconds and args.stream_seconds > 0:
        try:
            hb = 0
            px = 0
            t_end = time.time() + float(args.stream_seconds)
            stream = pricing.PricingStream(accountID=account_id, params={"instruments": args.instrument})
            for msg in api.request(stream):
                tnow = time.time()
                if msg.get("type") == "HEARTBEAT":
                    hb += 1
                elif msg.get("type") == "PRICE":
                    px += 1
                if tnow >= t_end:
                    break
            print(json.dumps({"stage": "stream", "hb": hb, "price_msgs": px, "seconds": args.stream_seconds}), flush=True)
        except Exception as e:
            print(json.dumps({"stage": "stream", "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
