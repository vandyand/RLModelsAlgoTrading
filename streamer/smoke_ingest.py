#!/usr/bin/env python3
from __future__ import annotations

import os
import time

from questdb_ingest import QuestDBILPClient
from questdb_admin import run_sql, ensure_fx_schema, ensure_crypto_schema


def main() -> None:
    ensure_fx_schema()
    ensure_crypto_schema()

    host = os.environ.get("QUESTDB_HOST", "127.0.0.1")
    ilp_port = int(os.environ.get("QUESTDB_ILP_PORT", "9009"))
    q = QuestDBILPClient(host=host, port=ilp_port)

    now_ns = time.time_ns()
    q.send_row("fx_ticks", tags={"instrument": "TEST_FX"}, fields={"bid": 1.2345, "ask": 1.2347, "bid_liquidity": 1000000, "ask_liquidity": 900000, "tradeable": True, "status": "tradeable"}, ts_ns=now_ns)
    q.send_row("fx_dom", tags={"instrument": "TEST_FX", "side": "bid"}, fields={"level": 0, "price": 1.2345, "liquidity": 1000000}, ts_ns=now_ns)
    q.send_row("crypto_ticker", tags={"symbol": "TEST"}, fields={"last": 100.0, "open": 99.0, "high": 101.0, "low": 98.0, "base_volume": 123.0, "quote_volume": 12345.0, "change_rate": 0.01}, ts_ns=now_ns)
    q.send_row("crypto_depth", tags={"symbol": "TEST", "side": "bid"}, fields={"level": 0, "price": 100.0, "qty": 1.23}, ts_ns=now_ns)

    # Query back counts
    fx = run_sql("SELECT count() FROM fx_ticks WHERE instrument='TEST_FX' AND ts > dateadd('h', -1, now())")
    cr = run_sql("SELECT count() FROM crypto_ticker WHERE symbol='TEST' AND ts > dateadd('h', -1, now())")
    print({"fx_ticks_recent": fx.get("dataset"), "crypto_ticker_recent": cr.get("dataset")})


if __name__ == "__main__":
    main()
