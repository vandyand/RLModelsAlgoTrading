#!/usr/bin/env python3
"""
Align Alpaca crypto positions to target weights per symbol.

Pipeline:
- Run crypto_offline_infer.py to produce JSON mapping {symbol: weight},
  where weight ∈ [-1, 1] is the raw actor-critic output.
- This script:
  - Interprets any negative weight as a zero desired position (no shorting).
  - Uses either:
      * direct notional scaling: target_notional = weight * scale_notional (if scale-notional > 0), or
      * equity-based sizing: normalize weights and allocate gross_target * equity (if scale-notional <= 0).
  - Fetches current crypto positions from Alpaca account.
  - Submits crypto market notional orders for the difference (target - current),
    using a crypto-compatible time_in_force (GTC).

Env:
  ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def _canonical_crypto_symbol(sym: str) -> str:
    """
    Normalise crypto symbols so that universe / target symbols like 'BTC/USD'
    match Alpaca trading/position symbols like 'BTCUSD'.

    We keep all external reporting in the original (targets) format and only use
    the canonicalised form for internal matching.
    """
    return str(sym).replace("/", "").strip().upper()


# Empirically observed minimum notional for crypto market orders on Alpaca.
_MIN_CRYPTO_ORDER_NOTIONAL_USD: float = 10.0


def read_targets(path_or_json: str) -> Dict[str, float]:
    """
    Read targets from stdin ('-'), a JSON string, or a file path.

    Expected format: { "BTC/USD": 0.12, "ETH/USD": -0.08, ... } with raw weights in [-1,1].
    Negative values are allowed on input but will be interpreted as zero desired
    exposure by the aligner (no shorting).
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

    raise RuntimeError("Targets JSON must be an object mapping symbol->weight (stdin may contain multiple JSON lines).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Align Alpaca crypto positions to target weights per symbol.")
    ap.add_argument(
        "--targets",
        required=True,
        help="Path to JSON mapping symbol->weight, inline JSON, or '-' for stdin.",
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use Alpaca paper trading account (default: True).",
    )
    ap.add_argument(
        "--gross-target",
        type=float,
        default=1.0,
        help=(
            "Target gross exposure as a multiple of account equity when using "
            "equity-based sizing (scale-notional <= 0). Default: 1.0."
        ),
    )
    ap.add_argument(
        "--scale-notional",
        type=float,
        default=100.0,
        help=(
            "If > 0, interpret raw weights directly and set target_notional = "
            "weight * scale_notional (e.g. 100 means BTC/USD=0.0409 -> $4.09). "
            "If <= 0, fall back to equity * gross-target sizing."
        ),
    )
    ap.add_argument(
        "--deadband-usd",
        type=float,
        default=5.0,
        help="Skip orders with |delta_notional| below this threshold (default: 5 USD).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders; just print the planned deltas.",
    )
    args = ap.parse_args()

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))

    targets_raw = read_targets(args.targets)
    # Clamp negative weights to zero (no short positions). For direct notional
    # scaling we STILL keep zero-weight symbols so that existing long positions
    # are flattened towards zero. For equity-based sizing we only allocate
    # budget across strictly positive weights.
    clamped: Dict[str, float] = {s: (w if w > 0.0 else 0.0) for s, w in targets_raw.items()}
    weights: Dict[str, float] = dict(clamped)  # include zeros for flattening

    # Track which symbols have strictly positive desired exposure.
    positive_symbols = [s for s, w in weights.items() if w > 1e-6]
    if not positive_symbols:
        # All targets are <= 0: this means "flatten everything".
        # We still proceed so that current positions are traded towards zero.
        positive_symbols = []

    # Two sizing modes:
    #  1) scale-notional > 0: direct scaling, target_notional = weight * scale_notional
    #  2) scale-notional <= 0: equity-based sizing, normalize weights and use
    #     gross_target * equity as the gross notional budget.
    use_direct_scaling = float(args.scale_notional) > 0.0

    if not use_direct_scaling:
        # Equity-based sizing: allocate only across positive weights.
        if not positive_symbols:
            print(json.dumps({"status": "no_positive_weights_for_equity_sizing"}))
            return

        total_abs = sum(abs(weights[s]) for s in positive_symbols)
        if total_abs <= 0:
            print(json.dumps({"status": "degenerate_weights"}))
            return
        norm_weights: Dict[str, float] = {s: (weights[s] / total_abs) for s in positive_symbols}

        # Fetch account equity
        account = trading.get_account()
        try:
            equity = float(account.equity)
        except Exception:
            equity = float(account.buying_power or 0.0)

        gross_budget = float(args.gross_target) * equity

    # Fetch current positions.
    #
    # For Alpaca crypto, position symbols are like 'BTCUSD', 'DOGEUSD', while
    # our targets / universe use 'BTC/USD', 'DOGE/USD'. We therefore build a
    # map keyed by a canonical symbol (slashes removed, uppercased) and keep
    # notional as the signed market_value.
    current_notional_by_canon: Dict[str, float] = {}
    try:
        positions = trading.get_all_positions()
        for p in positions:
            # Restrict to crypto positions only.
            asset_class = str(getattr(p, "asset_class", "")).upper()
            if "CRYPTO" not in asset_class:
                continue

            sym_raw = str(getattr(p, "symbol", "")).strip()
            if not sym_raw:
                continue
            canon = _canonical_crypto_symbol(sym_raw)

            try:
                mv = float(getattr(p, "market_value", 0.0))
            except Exception:
                mv = 0.0
            # Long positions have positive market_value, shorts negative.
            current_notional_by_canon[canon] = mv
    except Exception:
        positions = []

    results: List[Dict[str, Any]] = []

    for sym, w in weights.items():
        if use_direct_scaling:
            target_notional = float(w) * float(args.scale_notional)
        else:
            # Symbols without positive weight simply target zero notional.
            if sym in norm_weights:
                target_notional = float(norm_weights[sym]) * float(gross_budget)
            else:
                target_notional = 0.0

        canon_sym = _canonical_crypto_symbol(sym)
        cur = float(current_notional_by_canon.get(canon_sym, 0.0))
        delta = target_notional - cur
        action = "SKIP"
        order_info: Dict[str, Any] | None = None

        # --- Case 1: flattening (target at or very near zero) with an open long position.
        if target_notional <= _MIN_CRYPTO_ORDER_NOTIONAL_USD * 0.1 and cur > float(args.deadband_usd):
            # For closing, we prefer Alpaca's close_position API to avoid "insufficient balance"
            # issues from notional rounding.
            if args.dry_run:
                action = "CLOSE"
                order_info = {"dry_run": True, "close_position": True}
            else:
                try:
                    close_resp = trading.close_position(canon_sym)
                    action = "CLOSE"
                    order_info = {
                        "symbol": str(getattr(close_resp, "symbol", sym)),
                        "side": str(getattr(close_resp, "side", OrderSide.SELL)),
                        "id": str(getattr(close_resp, "id", "")),
                        "status": str(getattr(close_resp, "status", "")),
                    }
                    # After a close, we conceptually moved to zero.
                    delta = -cur
                    target_notional = 0.0
                except Exception as exc:
                    action = "ERROR"
                    order_info = {"error": str(exc)}
        else:
            # --- Case 2: regular adjustment towards a positive target.
            # Respect both the user deadband and Alpaca's minimum notional.
            eff_deadband = max(float(args.deadband_usd), _MIN_CRYPTO_ORDER_NOTIONAL_USD)
            if abs(delta) >= eff_deadband:
                # Alpaca requires notional values to be limited to 2 decimal places.
                notional = round(abs(delta), 2)
                if notional < eff_deadband:
                    # After rounding, if we fell below effective deadband, skip.
                    action = "SKIP"
                    order_info = None
                else:
                    action = "ORDER"
                    if args.dry_run:
                        order_info = {"dry_run": True, "notional": notional}
                    else:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        try:
                            # For crypto, Alpaca rejects DAY time_in_force; use GTC.
                            req = MarketOrderRequest(
                                symbol=sym,
                                notional=notional,
                                side=side,
                                time_in_force=TimeInForce.GTC,
                            )
                            resp = trading.submit_order(order_data=req)
                            order_info = {
                                "symbol": str(getattr(resp, "symbol", sym)),
                                "side": str(getattr(resp, "side", side)),
                                "notional": float(notional),
                                "id": str(getattr(resp, "id", "")),
                                "status": str(getattr(resp, "status", "")),
                            }
                        except Exception as exc:
                            action = "ERROR"
                            order_info = {"error": str(exc)}

        results.append(
            {
                "symbol": sym,
                "weight": w,
                "target_notional": target_notional,
                "current_notional": cur,
                "delta_notional": delta,
                "action": action,
                "order": order_info,
            }
        )

    print(json.dumps(results))


if __name__ == "__main__":  # pragma: no cover
    main()

