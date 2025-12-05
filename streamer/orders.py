import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades


def get_api(environment: str, access_token: str) -> API:
    return API(access_token=access_token, environment=environment)


def fetch_instrument_spec(api: API, account_id: str, instrument: str) -> Dict[str, Any]:
    req = accounts.AccountInstruments(accountID=account_id, params={"instruments": instrument})
    resp = api.request(req)
    instruments = resp.get("instruments") or []
    if not instruments:
        raise RuntimeError(f"Instrument not found or not tradeable: {instrument}")
    return instruments[0]


def fetch_quote(api: API, account_id: str, instrument: str) -> Tuple[Optional[float], Optional[float]]:
    req = pricing.PricingInfo(accountID=account_id, params={"instruments": instrument})
    resp = api.request(req)
    prices = resp.get("prices") or []
    if not prices:
        return None, None
    p = prices[0]
    try:
        bid = float(p.get("bids")[0]["price"]) if p.get("bids") else None
    except Exception:
        bid = None
    try:
        ask = float(p.get("asks")[0]["price"]) if p.get("asks") else None
    except Exception:
        ask = None
    return bid, ask


def calc_pip_size(pip_location: int) -> float:
    # OANDA pipLocation is a negative exponent indicating pip position
    # pip size = 10 ** pipLocation (e.g., -4 -> 0.0001; -2 -> 0.01)
    return 10 ** pip_location


def round_price(value: float, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    return fmt.format(round(value, precision))


def build_order_payload(
    instrument: str,
    units: int,
    tp_price: Optional[float],
    sl_price: Optional[float],
    client_tag: Optional[str] = None,
    client_comment: Optional[str] = None,
    price_precision: int = 5,
) -> Dict[str, Any]:
    order: Dict[str, Any] = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
        }
    }

    if client_tag or client_comment:
        order["order"]["clientExtensions"] = {}
        if client_tag:
            order["order"]["clientExtensions"]["tag"] = client_tag
        if client_comment:
            order["order"]["clientExtensions"]["comment"] = client_comment

    if tp_price is not None:
        order["order"]["takeProfitOnFill"] = {"price": round_price(tp_price, price_precision)}
    if sl_price is not None:
        order["order"]["stopLossOnFill"] = {"price": round_price(sl_price, price_precision)}

    return order


def place_market_order(
    api: API,
    account_id: str,
    instrument: str,
    units: int,
    tp_pips: Optional[float],
    sl_pips: Optional[float],
    anchor: Optional[str],
    client_tag: Optional[str],
    client_comment: Optional[str],
    fifo_safe: bool,
    fifo_adjust: bool,
) -> Dict[str, Any]:
    spec = fetch_instrument_spec(api, account_id, instrument)
    pip_location = int(spec.get("pipLocation"))
    display_precision = int(spec.get("displayPrecision"))
    pip_size = calc_pip_size(pip_location)

    bid, ask = fetch_quote(api, account_id, instrument)
    anchor_price: Optional[float] = None
    if anchor:
        # Respect explicit anchor choice
        if anchor == "bid":
            anchor_price = bid
        elif anchor == "ask":
            anchor_price = ask
        elif anchor == "mid":
            if bid is not None and ask is not None:
                anchor_price = (bid + ask) / 2.0
        else:
            try:
                anchor_price = float(anchor)
            except Exception:
                anchor_price = None
    else:
        # Default: use ask for buys, bid for sells
        anchor_price = ask if units > 0 else bid

    # Compute base TP/SL prices from pips and anchor
    def base_tp_sl(anchor: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if anchor is None:
            return None, None
        is_buy = units > 0
        tp_price = None
        sl_price = None
        if tp_pips and tp_pips > 0:
            tp_price = anchor + (tp_pips * pip_size if is_buy else -tp_pips * pip_size)
        if sl_pips and sl_pips > 0:
            sl_price = anchor - (sl_pips * pip_size if is_buy else -sl_pips * pip_size)
        return tp_price, sl_price

    tp_price, sl_price = base_tp_sl(anchor_price)

    # Optionally adjust for FIFO safeguards using existing open trades
    if (fifo_safe or fifo_adjust) and (tp_price is not None or sl_price is not None):
        try:
            open_trades_resp = api.request(trades.OpenTrades(accountID=account_id))
            open_trades_list = [t for t in open_trades_resp.get("trades", []) if t.get("instrument") == instrument]
        except Exception:
            open_trades_list = []

        # Extract existing TP/SL for same side
        same_side = [t for t in open_trades_list if (float(t.get("currentUnits", 0)) > 0) == (units > 0)]
        existing_tps = []
        existing_sls = []
        for t in same_side:
            tp = (t.get("takeProfitOrder") or {}).get("price")
            sl = (t.get("stopLossOrder") or {}).get("price")
            try:
                if tp is not None:
                    existing_tps.append(float(tp))
            except Exception:
                pass
            try:
                if sl is not None:
                    existing_sls.append(float(sl))
            except Exception:
                pass

        epsilon = pip_size / 10.0
        if units > 0:
            # Long: TP must be >= max(existing TP), SL must be <= min(existing SL)
            if tp_price is not None and existing_tps:
                mx = max(existing_tps)
                if tp_price < mx:
                    tp_price = mx + epsilon if fifo_adjust else tp_price
            if sl_price is not None and existing_sls:
                mn = min(existing_sls)
                if sl_price > mn:
                    sl_price = mn - epsilon if fifo_adjust else sl_price
        else:
            # Short: TP must be <= min(existing TP), SL must be >= max(existing SL)
            if tp_price is not None and existing_tps:
                mn = min(existing_tps)
                if tp_price > mn:
                    tp_price = mn - epsilon if fifo_adjust else tp_price
            if sl_price is not None and existing_sls:
                mx = max(existing_sls)
                if sl_price < mx:
                    sl_price = mx + epsilon if fifo_adjust else sl_price

        # If only fifo_safe (not adjust), also stagger by tiny epsilon to avoid exact duplicates
        if fifo_safe and not fifo_adjust:
            count_same = len(same_side)
            if tp_price is not None:
                tp_price = tp_price + (epsilon * count_same if units > 0 else -epsilon * count_same)
            if sl_price is not None:
                sl_price = sl_price - (epsilon * count_same if units > 0 else -epsilon * count_same)

    payload = build_order_payload(
        instrument=instrument,
        units=units,
        tp_price=tp_price,
        sl_price=sl_price,
        client_tag=client_tag,
        client_comment=client_comment,
        price_precision=display_precision,
    )

    req = orders.OrderCreate(accountID=account_id, data=payload)
    resp = api.request(req)
    return {
        "request": {
            "instrument": instrument,
            "units": units,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "anchor": anchor,
            "anchor_price": anchor_price,
            "pip_size": pip_size,
            "price_precision": display_precision,
            "payload": payload,
        },
        "response": resp,
    }


def build_parser() -> argparse.ArgumentParser:
    examples = (
        "\n"
        "Examples:\n"
        "  # Buy 1k EUR/USD with 10 pip TP and 8 pip SL (default: FIFO-adjust on)\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units 1000 --tp-pips 10 --sl-pips 8\n\n"
        "  # Sell 2k EUR/USD with 12 pip TP and 6 pip SL\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units -2000 --tp-pips 12 --sl-pips 6\n\n"
        "  # Use mid price as anchor for TP/SL computation\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units 500 --tp-pips 5 --sl-pips 5 --anchor mid\n\n"
        "  # Also stagger prices slightly in addition to FIFO adjust\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units 12 --tp-pips 6 --sl-pips 6 --fifo-safe\n\n"
        "  # Provide explicit credentials if env vars are not set\n"
        "  streamer/orders.py --account-id $OANDA_DEMO_ACCOUNT_ID --access-token $OANDA_DEMO_KEY \\\n"
        "    place-market --instrument EUR_USD --units 100\n"
    )
    parser = argparse.ArgumentParser(
        description="OANDA orders CLI (market orders with TP/SL in pips)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_examples = (
        "Place a market order and attach TP/SL on fill.\n\n"
        "Examples:\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units 12 --tp-pips 6 --sl-pips 6\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units -25 --tp-pips 10 --sl-pips 5\n"
        "  streamer/orders.py place-market --instrument EUR_USD --units 100 --tp-pips 8 --sl-pips 8 --anchor bid\n"
    )
    p = sub.add_parser(
        "place-market",
        help="Place a market order with optional TP/SL in pips",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=p_examples,
    )
    p.add_argument("--instrument", required=True)
    p.add_argument("--units", type=int, required=True, help= "Positive for buy (long), negative for sell (short)")
    p.add_argument("--tp-pips", type=float, help="Take profit distance in pips")
    p.add_argument("--sl-pips", type=float, help="Stop loss distance in pips")
    p.add_argument(
        "--anchor",
        help="Anchor price: bid|ask|mid or explicit price (default: ask for buy, bid for sell)",
    )
    p.add_argument("--client-tag")
    p.add_argument("--client-comment")
    p.add_argument("--fifo-safe", action="store_true", help="Stagger TP/SL slightly to avoid FIFO safeguard on identical prices")
    p.add_argument("--fifo-adjust", action="store_true", help="Adjust TP/SL outward to satisfy FIFO constraints vs existing trades (default: on)")
    p.set_defaults(command="place-market", fifo_adjust=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.account_id or not args.access_token:
        raise RuntimeError(
            "Missing OANDA credentials. Ensure OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY are set or pass flags."
        )

    api = get_api(args.environment, args.access_token)

    if args.cmd == "place-market":
        result = place_market_order(
            api=api,
            account_id=args.account_id,
            instrument=args.instrument,
            units=args.units,
            tp_pips=args.tp_pips,
            sl_pips=args.sl_pips,
            anchor=args.anchor,
            client_tag=args.client_tag,
            client_comment=args.client_comment,
            fifo_safe=args.fifo_safe,
            fifo_adjust=args.fifo_adjust,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
