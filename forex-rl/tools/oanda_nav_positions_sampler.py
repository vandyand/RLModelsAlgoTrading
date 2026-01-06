#!/usr/bin/env python3
"""
Periodic sampler for OANDA account NAV and open positions.

Intended use:
- Run as a long-lived process (e.g., systemd service)
- Every N seconds (default 300), fetch:
  - Account summary (NAV, balance, margin, etc.)
  - Open positions snapshot
- Append one JSON record per poll to a log file for downstream analysis.

Environment:
- OANDA_ENV                 practice|live (default: practice)
- For practice (demo) accounts:
  - OANDA_DEMO_KEY          REST API token
  - OANDA_DEMO_ACCOUNT_ID   account id
- For live accounts (optional):
  - OANDA_LIVE_KEY
  - OANDA_LIVE_ACCOUNT_ID
- OANDA_ACCESS_TOKEN        alternative token name (fallback)

CLI:
  python tools/oanda_nav_positions_sampler.py \
      --interval-secs 300 \
      --environment practice \
      --out logs/oanda_nav_positions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_token_and_account(
    environment: str,
    account_id_arg: Optional[str],
) -> Tuple[str, str]:
    """Resolve token and account id from args + environment variables."""
    # Base token / account from demo vars
    token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
    account_id = account_id_arg or os.environ.get("OANDA_DEMO_ACCOUNT_ID")

    if environment == "live":
        # Allow overrides for live
        token = os.environ.get("OANDA_LIVE_KEY", token)
        account_id = account_id_arg or os.environ.get("OANDA_LIVE_ACCOUNT_ID", account_id)

    if not token:
        raise RuntimeError(
            "Missing OANDA token. "
            "Set OANDA_DEMO_KEY (or OANDA_ACCESS_TOKEN), "
            "or OANDA_LIVE_KEY for live environment."
        )
    if not account_id:
        raise RuntimeError(
            "Missing OANDA account id. "
            "Set OANDA_DEMO_ACCOUNT_ID (or OANDA_LIVE_ACCOUNT_ID for live), "
            "or pass --account-id."
        )
    return str(token), str(account_id)


def _build_api(environment: str, token: str):
    try:
        from oandapyV20 import API  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(f"oandapyV20 not available: {exc}") from exc
    return API(access_token=token, environment=environment)


def fetch_account_and_positions(api, account_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch AccountSummary + OpenPositions from OANDA.

    Returns:
        (account_dict, positions_list)
        On errors, embeds simple {"error": "..."} markers instead of raising.
    """
    try:
        import oandapyV20.endpoints.accounts as accounts  # type: ignore
        import oandapyV20.endpoints.positions as positions_ep  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        return {"error": f"import_error: {exc}"}, []

    account: Dict[str, Any] = {}
    positions: List[Dict[str, Any]] = []

    # Account summary
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        account = resp.get("account", {}) or {}
    except Exception as exc:
        account = {"error": str(exc)}

    # Open positions
    try:
        req = positions_ep.OpenPositions(accountID=account_id)
        resp = api.request(req)
        positions = list(resp.get("positions", []))
    except Exception as exc:
        positions = [{"error": str(exc)}]

    return account, positions


def main() -> None:
    p = argparse.ArgumentParser(description="Periodic OANDA NAV + positions sampler")
    p.add_argument(
        "--environment",
        choices=["practice", "live"],
        default=os.environ.get("OANDA_ENV", "practice"),
        help="OANDA environment (practice|live). Default from OANDA_ENV or 'practice'.",
    )
    p.add_argument(
        "--account-id",
        default=None,
        help="Explicit account id (overrides env). If omitted, uses OANDA_DEMO_ACCOUNT_ID / OANDA_LIVE_ACCOUNT_ID.",
    )
    p.add_argument(
        "--interval-secs",
        type=float,
        default=300.0,
        help="Polling interval in seconds (default 300 = 5 minutes).",
    )
    p.add_argument(
        "--duration-secs",
        type=float,
        default=0.0,
        help="Optional total runtime limit in seconds (0 = run indefinitely).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output file path (JSONL). Default: logs/oanda_nav_positions_<timestamp>.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="logs",
        help="Base directory for default output path (default: logs).",
    )
    args = p.parse_args()

    try:
        token, account_id = _get_token_and_account(args.environment, args.account_id)
    except Exception as exc:
        # Fail fast with a clear message; systemd will surface this in logs.
        print(json.dumps({"stage": "config", "error": str(exc)}), file=sys.stderr, flush=True)
        raise SystemExit(1)

    try:
        api = _build_api(args.environment, token)
    except Exception as exc:
        print(json.dumps({"stage": "api_init", "error": str(exc)}), file=sys.stderr, flush=True)
        raise SystemExit(2)

    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
    else:
        ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = base_dir / f"oanda_nav_positions_{ts_label}.jsonl"

    # Ensure parent dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_event = {
        "event": "oanda_nav_positions_sampler_start",
        "environment": args.environment,
        "account_id": account_id,
        "interval_secs": float(args.interval_secs),
        "duration_secs": float(args.duration_secs),
        "out_path": str(out_path),
    }
    print(json.dumps(cfg_event), flush=True)

    start_ts = time.time()
    interval = max(1.0, float(args.interval_secs))

    # Open file once and append records
    with out_path.open("a", encoding="utf-8") as f:
        try:
            while True:
                now_iso = iso_now()
                epoch = time.time()

                account, positions = fetch_account_and_positions(api, account_id)
                record = {
                    "event": "oanda_nav_positions_snapshot",
                    "time_iso": now_iso,
                    "epoch": epoch,
                    "environment": args.environment,
                    "account_id": account_id,
                    "account": account,
                    "positions": positions,
                }
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()

                if args.duration_secs and (time.time() - start_ts) >= float(args.duration_secs):
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            # Graceful shutdown when run interactively
            pass


if __name__ == "__main__":
    main()

