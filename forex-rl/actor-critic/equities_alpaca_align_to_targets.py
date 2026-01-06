#!/usr/bin/env python3
"""
Align Alpaca paper positions to target weights per symbol.

Pipeline:
- Run equities_offline_infer.py (or crypto_offline_infer.py) to produce JSON mapping {symbol: weight},
  where weight ∈ [-1, 1] is the raw actor-critic output.
- This script:
  - Interprets any negative weight as a zero desired position (no shorting).
  - Normalizes the remaining non-negative weights into portfolio weights when using equity-based sizing.
  - Uses account equity and a gross exposure budget (or a fixed scale_notional) to compute target dollar
    notionals per symbol.
  - Fetches current positions from Alpaca paper account.
  - Submits market notional orders for the difference (target - current).

Usage (from forex-rl root):

  python -m actor-critic.equities_offline_infer \\
    --checkpoint forex-rl/actor-critic/checkpoints/equities_offline_ac.pt \\
    --universe-json equities_universe_top100_five_years.json \\
    | python -m actor-critic.equities_alpaca_align_to_targets --targets - \\
        --gross-target 1.0

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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest


def _get_alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment.")
    return key, secret


def read_targets(path_or_json: str) -> Dict[str, float]:
    """
    Read targets from stdin ('-'), a JSON string, or a file path.

    Expected format: { "SPY": 0.12, "QQQ": -0.08, ... } with raw weights in [-1,1].
    Negative values are allowed on input. By default the aligner treats them as
    zero desired exposure (long-only). When --long-short is enabled, negative
    weights are interpreted as short targets which are executed in integer
    share quantities.
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
    ap = argparse.ArgumentParser(description="Align Alpaca paper positions to target weights per symbol.")
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
            "weight * scale_notional (e.g. 100 means SPY=0.0409 -> $4.09). "
            "If <= 0, fall back to equity * gross-target sizing."
        ),
    )
    ap.add_argument(
        "--deadband-usd",
        type=float,
        default=25.0,
        help="Skip orders with |delta_notional| below this threshold (default: 25 USD).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders; just print the planned deltas.",
    )
    ap.add_argument(
        "--long-short",
        action="store_true",
        help=(
            "Allow short positions for negative weights. Shorts are sized using integer "
            "shares derived from target notionals and latest prices. By default the "
            "aligner is long-only and clamps negative weights to zero."
        ),
    )
    args = ap.parse_args()

    key, secret = _get_alpaca_keys()
    trading = TradingClient(key, secret, paper=bool(args.paper))
    data_client = StockHistoricalDataClient(key, secret)

    targets_raw = read_targets(args.targets)
    # Process weights:
    # - Long-only (default): clamp negatives to zero and drop zeros.
    # - Long-short (--long-short): keep both positive and negative weights, drop zeros only.
    if args.long_short:
        weights: Dict[str, float] = {s: w for s, w in targets_raw.items() if abs(w) > 1e-6}
    else:
        clamped: Dict[str, float] = {s: (w if w > 0.0 else 0.0) for s, w in targets_raw.items()}
        weights = {s: w for s, w in clamped.items() if w > 1e-6}
    if not weights:
        print(json.dumps({"status": "no_nonzero_targets"}))
        return

    # Two sizing modes:
    #  1) scale-notional > 0: direct scaling, target_notional = weight * scale_notional
    #  2) scale-notional <= 0: equity-based sizing, normalize weights and use
    #     gross_target * equity as the gross notional budget.
    use_direct_scaling = float(args.scale_notional) > 0.0

    if not use_direct_scaling:
        total_abs = sum(abs(w) for w in weights.values())
        if total_abs <= 0:
            print(json.dumps({"status": "degenerate_weights"}))
            return
        norm_weights = {s: (w / total_abs) for s, w in weights.items()}

        # Fetch account equity
        account = trading.get_account()
        try:
            equity = float(account.equity)
        except Exception:
            equity = float(account.buying_power or 0.0)

        gross_budget = float(args.gross_target) * equity

    # Fetch current positions (notional and share quantity per symbol)
    current_notional: Dict[str, float] = {}
    current_qty: Dict[str, float] = {}
    try:
        positions = trading.get_all_positions()
        for p in positions:
            sym = str(p.symbol)
            try:
                mv = float(p.market_value)
            except Exception:
                mv = 0.0
            # Long positions have positive market_value, shorts negative.
            current_notional[sym] = mv
            try:
                q = float(p.qty)
            except Exception:
                q = 0.0
            current_qty[sym] = q
    except Exception:
        positions = []

    # Pre-fetch latest trade prices for all symbols we may touch, to convert
    # target notionals into integer share quantities when needed.
    price_by_sym: Dict[str, float] = {}
    try:
        trade_req = StockLatestTradeRequest(symbol_or_symbols=list(weights.keys()))
        latest = data_client.get_stock_latest_trade(trade_req)
        for sym_k, trade in latest.items():
            try:
                price_by_sym[str(sym_k)] = float(getattr(trade, "price", 0.0) or 0.0)
            except Exception:
                continue
    except Exception:
        # If latest trade fetch fails, we will fall back to inferring price from
        # existing positions where possible.
        pass

    results: List[Dict[str, Any]] = []

    for sym, w in weights.items():
        if use_direct_scaling:
            target_notional = float(w) * float(args.scale_notional)
        else:
            target_notional = float(norm_weights[sym]) * float(gross_budget)
        cur_notional = float(current_notional.get(sym, 0.0))
        delta_notional = target_notional - cur_notional
        action = "SKIP"
        order_info: Dict[str, Any] | None = None

        # Infer a best-effort latest price per share.
        px = price_by_sym.get(sym)
        if (px is None or px <= 0.0) and sym in current_qty and abs(current_qty.get(sym, 0.0)) > 0:
            try:
                px = abs(cur_notional) / max(1e-8, abs(current_qty[sym]))
            except Exception:
                px = None

        # When long-short is enabled and we have a price, convert target notionals
        # into integer share quantities (required for short sales). For long-only,
        # or when price is unavailable, preserve notional-based sizing.
        if args.long_short and px is not None and px > 0.0:
            # Target and current quantities (can be long or short).
            target_qty_float = target_notional / px
            target_qty = int(round(target_qty_float))
            cur_qty = float(current_qty.get(sym, 0.0))
            delta_qty = target_qty - cur_qty

            # Skip very small share changes.
            if delta_qty == 0:
                action = "SKIP"
                order_info = None
            else:
                # Approximate notional for deadband and logging.
                approx_delta_notional = abs(delta_qty) * px
                if approx_delta_notional < float(args.deadband_usd):
                    action = "SKIP"
                    order_info = None
                else:
                    action = "ORDER"
                    side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
                    qty = abs(int(delta_qty))
                    if args.dry_run:
                        order_info = {
                            "dry_run": True,
                            "qty": qty,
                            "approx_notional": approx_delta_notional,
                            "side": str(side),
                        }
                    else:
                        try:
                            req = MarketOrderRequest(
                                symbol=sym,
                                qty=qty,
                                side=side,
                                time_in_force=TimeInForce.DAY,
                            )
                            resp = trading.submit_order(order_data=req)
                            order_info = {
                                "symbol": str(getattr(resp, "symbol", sym)),
                                "side": str(getattr(resp, "side", side)),
                                "qty": qty,
                                "approx_notional": approx_delta_notional,
                                "id": str(getattr(resp, "id", "")),
                                "status": str(getattr(resp, "status", "")),
                            }
                        except Exception as exc:
                            action = "ERROR"
                            order_info = {"error": str(exc)}
        else:
            # Long-only or no price available: stick to notional-based sizing,
            # which remains long-only due to prior clamping of negative weights.
            if abs(delta_notional) >= float(args.deadband_usd):
                notional = round(abs(delta_notional), 2)
                if notional < float(args.deadband_usd):
                    # After rounding away tiny exposure, skip if below deadband.
                    action = "SKIP"
                    order_info = None
                else:
                    action = "ORDER"
                    side = OrderSide.BUY if delta_notional > 0 else OrderSide.SELL
                    if args.dry_run:
                        order_info = {"dry_run": True, "notional": notional, "side": str(side)}
                    else:
                        try:
                            req = MarketOrderRequest(
                                symbol=sym,
                                notional=notional,
                                side=side,
                                time_in_force=TimeInForce.DAY,
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
                "current_notional": cur_notional,
                "delta_notional": delta_notional,
                "action": action,
                "order": order_info,
            }
        )

    print(json.dumps(results))


if __name__ == "__main__":  # pragma: no cover
    main()

