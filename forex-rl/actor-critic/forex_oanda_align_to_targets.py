#!/usr/bin/env python3
"""
Align OANDA FX positions to target notionals derived from raw actor outputs.

Pipeline:
- Run forex_offline_infer.py to produce JSON mapping {instrument: weight},
  where weight ∈ [-1, 1] is the raw actor-critic output per FX pair.
- This script:
  - Scales each weight by --scale-notional to obtain a target *USD* notional.
  - Uses current FX prices to convert that USD notional into a target integer
    units value per OANDA instrument (long/short allowed).
  - Fetches current units per instrument from the OANDA account.
  - Submits market orders for the integer delta in units, subject to a
    --deadband-usd threshold.

Usage (from forex-rl root):

  python -m forex-rl.actor-critic.forex_offline_infer \\
    --checkpoint forex-rl/actor-critic/checkpoints/forex_offline_ac.pt \\
    --use-tcn \\
    | python -m forex-rl.actor-critic.forex_oanda_align_to_targets \\
        --targets - --scale-notional 10000 --deadband-usd 25

Env:
  OANDA_DEMO_ACCOUNT_ID, OANDA_DEMO_KEY (or live equivalents) must be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oandapyV20 import API
import oandapyV20.endpoints.positions as positions

# For FX prices we reuse the REST candles adapter. When running as
# `python -m forex-rl.actor-critic.forex_oanda_align_to_targets` the project
# root is on sys.path but the `forex-rl` directory itself is not, so we add it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore
from streamer.orders import place_market_order  # type: ignore


def read_targets(path_or_json: str) -> Dict[str, float]:
    """
    Read targets from stdin ('-'), a JSON string, or a file path.

    Expected format: { "EUR_USD": 0.12, "USD_JPY": -0.08, ... } with raw weights in [-1,1].
    """
    s = (path_or_json or "").strip()
    if s == "-":
        data = sys.stdin.read()
    elif s.startswith("{"):
        data = s
    else:
        with open(s, "r", encoding="utf-8") as f:
            data = f.read()

    # First try simple JSON object
    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            out: Dict[str, float] = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out
    except json.JSONDecodeError:
        # Fallback: handle multiple JSON objects (e.g. status + targets) separated by newlines.
        last_obj: Dict[str, Any] | None = None
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except Exception:
                continue
            if isinstance(cand, dict):
                last_obj = cand
        if last_obj is not None:
            out2: Dict[str, float] = {}
            for k, v in last_obj.items():
                try:
                    out2[str(k)] = float(v)
                except Exception:
                    continue
            if out2:
                return out2

    raise RuntimeError("Targets JSON must be an object mapping instrument->weight (stdin may contain multiple JSON lines).")


def fetch_current_units_map(api: API, account_id: str) -> Dict[str, int]:
    """Return current net units per OANDA instrument in the account."""
    out: Dict[str, int] = {}
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for p in resp.get("positions", []):
            inst = p.get("instrument")
            try:
                long_u = float((p.get("long") or {}).get("units") or 0.0)
            except Exception:
                long_u = 0.0
            try:
                short_u = float((p.get("short") or {}).get("units") or 0.0)
            except Exception:
                short_u = 0.0
            out[str(inst)] = int(round(long_u + short_u))
    except Exception:
        pass
    return out


def _split_instrument(inst: str) -> Tuple[str, str]:
    """Split OANDA instrument like 'EUR_USD' into (base, quote)."""
    parts = inst.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unrecognized FX instrument format: {inst}")
    return parts[0], parts[1]


def _fetch_last_close(
    instrument: str,
    environment: str,
    access_token: str,
) -> float:
    """Fetch the last available daily close price for an FX pair."""
    adapter = OandaRestCandlesAdapter(
        instrument=instrument,
        granularity="D",
        environment=environment,
        access_token=access_token,
    )
    # Fetch a modest recent window; we only need the last candle.
    from datetime import datetime, timezone, timedelta

    end = (datetime.now(timezone.utc)).date().isoformat()
    start = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    rows = list(
        adapter.fetch_range(
            from_time=f"{start}T00:00:00Z",
            to_time=f"{end}T23:59:59Z",
            batch=1000,
        )
    )
    if not rows:
        raise RuntimeError(f"No candles fetched for {instrument} D between {start} and {end}")
    last = rows[-1]
    try:
        close = float(last.get("mid", {}).get("c") or last.get("close"))
    except Exception:
        close = float(last.get("close") or 0.0)
    return float(close)


def _build_price_maps(
    instruments: List[str],
    environment: str,
    access_token: str,
    account_currency: str = "USD",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Build:
      - price_pair[inst] = price(BASE/QUOTE) (QUOTE per BASE)
      - fx_to_usd[currency] = USD per unit of `currency`
    for currencies appearing in the given instruments.
    """
    account_ccy = account_currency.upper()
    price_pair: Dict[str, float] = {}
    fx_to_usd: Dict[str, float] = {account_ccy: 1.0}

    # First fetch base/quote prices for all requested instruments.
    for inst in instruments:
        try:
            px = _fetch_last_close(inst, environment, access_token)
            price_pair[inst] = px
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "price_fetch_error",
                        "instrument": inst,
                        "reason": str(exc),
                    }
                ),
                flush=True,
            )

    # Collect unique currencies in the universe.
    currencies: set[str] = set()
    for inst in instruments:
        try:
            base, quote = _split_instrument(inst)
        except ValueError:
            continue
        currencies.add(base)
        currencies.add(quote)
    currencies.discard(account_ccy)

    # For each non-account currency c, we want USD per c.
    # We try c_USD first, then USD_c and invert if needed.
    for c in currencies:
        if c in fx_to_usd:
            continue
        pair1 = f"{c}_{account_ccy}"
        pair2 = f"{account_ccy}_{c}"
        px: Optional[float] = None
        try:
            px = price_pair.get(pair1) or _fetch_last_close(pair1, environment, access_token)
            fx_to_usd[c] = float(px)
            price_pair.setdefault(pair1, float(px))
            continue
        except Exception:
            pass
        try:
            px2 = price_pair.get(pair2) or _fetch_last_close(pair2, environment, access_token)
            price_pair.setdefault(pair2, float(px2))
            fx_to_usd[c] = 1.0 / float(px2)
        except Exception:
            print(
                json.dumps(
                    {
                        "status": "fx_to_usd_missing",
                        "currency": c,
                        "account_ccy": account_ccy,
                    }
                ),
                flush=True,
            )
            # Leave missing; pairs depending on this c may be skipped.

    return price_pair, fx_to_usd


def _usd_notional_per_unit(
    inst: str,
    price_pair: Dict[str, float],
    fx_to_usd: Dict[str, float],
    account_currency: str = "USD",
) -> Optional[float]:
    """
    Compute USD notional per 1 unit of an FX instrument inst = BASE_QUOTE.

    - If QUOTE == USD: 1 unit is price(BASE/QUOTE) USD.
    - If BASE == USD:  1 unit is 1 USD.
    - Else (cross):    1 unit is price(BASE/QUOTE) * (USD per QUOTE).
    """
    account_ccy = account_currency.upper()
    base, quote = _split_instrument(inst)
    px = price_pair.get(inst)
    if px is None:
        return None

    # QUOTE is account currency
    if quote == account_ccy:
        return float(px)
    # BASE is account currency
    if base == account_ccy:
        return 1.0

    # Cross: need USD per QUOTE
    usd_per_quote = fx_to_usd.get(quote)
    if usd_per_quote is None:
        return None
    return float(px) * float(usd_per_quote)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align OANDA FX positions to target USD notionals derived from raw actor outputs."
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Path to JSON mapping instrument->weight, inline JSON, or '-' for stdin.",
    )
    parser.add_argument(
        "--environment",
        choices=["practice", "live"],
        default="practice",
        help="Environment tag for OANDA (practice strongly recommended).",
    )
    parser.add_argument(
        "--account-id",
        default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"),
        help="OANDA account id (required for position fetch and orders).",
    )
    parser.add_argument(
        "--access-token",
        default=os.environ.get("OANDA_DEMO_KEY"),
        help="OANDA API access token (if omitted, falls back to OANDA_DEMO_KEY / OANDA_LIVE_KEY env).",
    )
    parser.add_argument(
        "--scale-notional",
        type=float,
        default=10000.0,
        help=(
            "If > 0, interpret raw weights directly and set target USD notional = "
            "weight * scale_notional (e.g. 10000 means weight=0.5 -> $5k)."
        ),
    )
    parser.add_argument(
        "--deadband-usd",
        type=float,
        default=25.0,
        help="Skip orders with |delta_notional| below this threshold (default: 25 USD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders; just print the planned deltas.",
    )
    parser.add_argument(
        "--account-currency",
        type=str,
        default="USD",
        help="Account base currency (currently only USD is fully supported).",
    )
    args = parser.parse_args()

    if not args.account_id or not args.access_token:
        raise SystemExit("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY or pass flags.")

    targets_raw = read_targets(args.targets)
    # Drop exact zeros
    weights: Dict[str, float] = {inst: w for inst, w in targets_raw.items() if abs(w) > 1e-6}
    if not weights:
        print(json.dumps({"status": "no_nonzero_targets"}))
        return

    instruments = list(weights.keys())

    # Build price maps for notional calculation
    price_pair, fx_to_usd = _build_price_maps(
        instruments=instruments,
        environment=args.environment,
        access_token=args.access_token,
        account_currency=args.account_currency,
    )

    # Fetch current positions
    api = API(access_token=args.access_token, environment=args.environment)
    current_units = fetch_current_units_map(api, args.account_id)

    results: List[Dict[str, Any]] = []

    for inst, w in weights.items():
        usd_per_unit = _usd_notional_per_unit(
            inst, price_pair=price_pair, fx_to_usd=fx_to_usd, account_currency=args.account_currency
        )
        if usd_per_unit is None or usd_per_unit <= 0.0:
            results.append(
                {
                    "instrument": inst,
                    "weight": w,
                    "status": "skip_missing_price",
                }
            )
            continue

        target_notional = float(w) * float(args.scale_notional)
        target_units_float = target_notional / usd_per_unit
        target_units = int(round(target_units_float))
        cur_units = int(current_units.get(inst, 0))
        delta_units = target_units - cur_units

        # Approximate USD delta notional for deadband
        approx_delta_notional = abs(delta_units) * usd_per_unit
        action = "SKIP"
        order_info: Optional[Dict[str, Any]] = None

        if approx_delta_notional >= float(args.deadband_usd) and delta_units != 0:
            action = "ORDER"
            if args.dry_run:
                order_info = {
                    "dry_run": True,
                    "delta_units": delta_units,
                    "approx_delta_notional": approx_delta_notional,
                }
            else:
                try:
                    order = place_market_order(
                        api=api,
                        account_id=args.account_id,
                        instrument=inst,
                        units=delta_units,
                        tp_pips=None,
                        sl_pips=None,
                        anchor=None,
                        client_tag="fx-align-to-target",
                        client_comment="align FX to target USD notional",
                        fifo_safe=False,
                        fifo_adjust=False,
                    )
                    # Reuse summarize style from align_positions_to_targets
                    resp = (order or {}).get("response", {})
                    create = resp.get("orderCreateTransaction", {})
                    fill = resp.get("orderFillTransaction", {})
                    ord_inst = create.get("instrument") or fill.get("instrument")
                    order_info = {
                        "instrument": ord_inst,
                        "order_id": create.get("id"),
                        "fill_id": fill.get("id"),
                        "units": fill.get("units") or create.get("units"),
                        "price": fill.get("price"),
                        "time": fill.get("time") or create.get("time"),
                        "reason": fill.get("reason") or create.get("reason"),
                    }
                    # Drop Nones
                    order_info = {k: v for k, v in order_info.items() if v is not None}
                except Exception as exc:
                    action = "ERROR"
                    order_info = {"error": str(exc)}

        results.append(
            {
                "instrument": inst,
                "weight": w,
                "usd_per_unit": usd_per_unit,
                "target_notional": target_notional,
                "target_units": target_units,
                "current_units": cur_units,
                "delta_units": delta_units,
                "approx_delta_notional": approx_delta_notional,
                "action": action,
                "order": order_info,
            }
        )

    print(json.dumps(results))


if __name__ == "__main__":  # pragma: no cover
    main()

