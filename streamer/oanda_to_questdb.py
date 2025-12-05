#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing

# Local helpers
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)
from questdb_ingest import QuestDBILPClient, iso_to_nanos  # type: ignore
from questdb_admin import ensure_fx_schema  # type: ignore


def load_fx_instruments() -> List[str]:
    p = os.path.join(HERE, "data", "fx_pairs.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        insts: List[str] = data.get("instruments") or []
        if not insts and data.get("instruments_csv"):
            insts = [s.strip() for s in data["instruments_csv"].split(",") if s.strip()]
        if insts:
            return insts
    except Exception:
        pass
    # Fallback 20
    return [
        "EUR_USD","USD_JPY","GBP_USD","AUD_USD","USD_CHF",
        "USD_CAD","NZD_USD","EUR_JPY","GBP_JPY","EUR_GBP",
        "EUR_CHF","EUR_AUD","EUR_CAD","AUD_JPY","CHF_JPY",
        "CAD_JPY","NZD_JPY","GBP_CHF","GBP_CAD","AUD_CAD",
    ]


def main() -> None:
    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID") or os.environ.get("OANDA_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials: set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY")

    # Ensure tables exist
    ensure_fx_schema()

    host = os.environ.get("QUESTDB_HOST", "127.0.0.1")
    ilp_port = int(os.environ.get("QUESTDB_ILP_PORT", "9009"))
    table_ticks = os.environ.get("QDB_FX_TICKS_TABLE", "fx_ticks")
    table_dom = os.environ.get("QDB_FX_DOM_TABLE", "fx_dom")

    qdb = QuestDBILPClient(host=host, port=ilp_port)

    instruments = load_fx_instruments()
    params = {"instruments": ",".join(instruments)}
    api = API(access_token=access_token, environment="practice")
    req = pricing.PricingStream(accountID=account_id, params=params)

    # Backpressure-safe batching
    batch_lines: List[str] = []
    batch_flush = int(os.environ.get("QDB_BATCH", "0"))
    flush_every = max(0, batch_flush)

    sent = 0
    last_flush_ns = time.time_ns()
    try:
        for tick in api.request(req):
            if tick.get("type") != "PRICE":
                continue
            inst = tick.get("instrument")
            iso = tick.get("time")
            ts_ns = iso_to_nanos(iso) if isinstance(iso, str) else time.time_ns()
            # Top of book and summary
            bids = tick.get("bids") or []
            asks = tick.get("asks") or []
            def p_float(x: Optional[str]) -> Optional[float]:
                try:
                    return float(x) if x is not None else None
                except Exception:
                    return None
            bid = p_float(bids[0].get("price")) if bids else None
            ask = p_float(asks[0].get("price")) if asks else None
            bid_liq = int(bids[0].get("liquidity")) if bids and bids[0].get("liquidity") is not None else None
            ask_liq = int(asks[0].get("liquidity")) if asks and asks[0].get("liquidity") is not None else None
            closeout_bid = p_float(tick.get("closeoutBid"))
            closeout_ask = p_float(tick.get("closeoutAsk"))
            tradeable = tick.get("tradeable")
            status = tick.get("status")

            line = qdb.build_line(
                table_ticks,
                tags={"instrument": inst},
                fields={
                    "bid": bid, "ask": ask,
                    "bid_liquidity": bid_liq, "ask_liquidity": ask_liq,
                    "closeout_bid": closeout_bid, "closeout_ask": closeout_ask,
                    "tradeable": bool(tradeable) if isinstance(tradeable, bool) else None,
                    "status": status if isinstance(status, str) else None,
                },
                ts_ns=ts_ns,
            )
            if flush_every > 0:
                batch_lines.append(line)
            else:
                qdb.write_line(line)
            sent += 1

            # DOM levels
            for idx, lvl in enumerate(bids):
                price = p_float(lvl.get("price"))
                liq = int(lvl.get("liquidity")) if lvl.get("liquidity") is not None else None
                line = qdb.build_line(
                    table_dom,
                    tags={"instrument": inst, "side": "bid"},
                    fields={"level": idx, "price": price, "liquidity": liq},
                    ts_ns=ts_ns,
                )
                if flush_every > 0:
                    batch_lines.append(line)
                else:
                    qdb.write_line(line)
            for idx, lvl in enumerate(asks):
                price = p_float(lvl.get("price"))
                liq = int(lvl.get("liquidity")) if lvl.get("liquidity") is not None else None
                line = qdb.build_line(
                    table_dom,
                    tags={"instrument": inst, "side": "ask"},
                    fields={"level": idx, "price": price, "liquidity": liq},
                    ts_ns=ts_ns,
                )
                if flush_every > 0:
                    batch_lines.append(line)
                else:
                    qdb.write_line(line)

            # Flush periodically if batching
            if flush_every > 0 and len(batch_lines) >= flush_every:
                qdb.write_line("\n".join(batch_lines))
                batch_lines.clear()
                last_flush_ns = time.time_ns()

            # Light log
            if sent % 100 == 0:
                print(json.dumps({"event": "fx_ingest", "count": sent, "last_instrument": inst}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        raise
    finally:
        if flush_every > 0 and batch_lines:
            try:
                qdb.write_line("\n".join(batch_lines))
            except Exception:
                pass


if __name__ == "__main__":
    main()
