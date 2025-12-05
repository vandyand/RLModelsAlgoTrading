#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
import time

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(HERE)
CRYPTO_DIR = os.path.join(REPO_ROOT, "crypto-rl")
REF_DEMO_PY = os.path.join(REPO_ROOT, "ref", "bitunix-demo", "Demo", "Python")
for p in (REPO_ROOT, CRYPTO_DIR, HERE, REF_DEMO_PY):
    if p not in sys.path:
        sys.path.append(p)

# Import local/repo modules (avoid hyphen in package name by adding dir to sys.path)
from bitunix_ws import BitunixPublicWS  # type: ignore
from questdb_ingest import QuestDBILPClient  # type: ignore
from questdb_ingest import iso_to_nanos  # type: ignore
from questdb_admin import ensure_crypto_schema  # type: ignore

DEFAULT_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BCHUSDT", "LTCUSDT",
    "BNBUSDT", "ADAUSDT", "BATUSDT", "ETCUSDT", "XLMUSDT",
    "ZRXUSDT", "DOGEUSDT", "ATOMUSDT", "DOTUSDT", "LINKUSDT",
    "UNIUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT", "FILUSDT",
]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bitunix public WS -> QuestDB ILP")
    p.add_argument("--symbols", default=os.environ.get("SYMBOLS", ",".join(DEFAULT_SYMBOLS)), help="Comma-separated Bitunix symbols")
    p.add_argument("--ws-uri", default=os.environ.get("BITUNIX_WS_URI", "wss://fapi.bitunix.com/public/"))
    p.add_argument("--depth-level", type=int, choices=[1,5,15], default=5)
    # Enable useful channels by default for ease of use
    p.add_argument("--enable-trade", action="store_true", default=False)
    p.add_argument("--enable-ticker", action="store_true", default=True)
    p.add_argument("--enable-price", action="store_true", default=False)
    p.add_argument("--enable-depth", action="store_true", default=True)
    p.add_argument("--batch", type=int, default=int(os.environ.get("QDB_BATCH", "0")))
    return p.parse_args()


async def main_async() -> None:
    ensure_crypto_schema()

    args = build_args()
    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]

    host = os.environ.get("QUESTDB_HOST", "127.0.0.1")
    ilp_port = int(os.environ.get("QUESTDB_ILP_PORT", "9009"))
    qdb = QuestDBILPClient(host=host, port=ilp_port)

    t_trade = os.environ.get("QDB_CRYPTO_TRADE", "crypto_trade")
    t_ticker = os.environ.get("QDB_CRYPTO_TICKER", "crypto_ticker")
    t_price = os.environ.get("QDB_CRYPTO_PRICE", "crypto_price")
    t_depth = os.environ.get("QDB_CRYPTO_DEPTH", "crypto_depth")

    batch_lines: List[str] = []
    flush_every = max(0, int(args.batch))

    ws = BitunixPublicWS(public_ws_uri=args.ws_uri)

    # Stats and periodic logging
    stats = {"trade": 0, "ticker": 0, "price": 0, "depth": 0, "lines": 0}
    last_log = time.monotonic()

    def flush_batch() -> None:
        nonlocal batch_lines
        if batch_lines:
            qdb.write_line("\n".join(batch_lines))
            batch_lines = []

    def send_line(line: str) -> None:
        if flush_every > 0:
            batch_lines.append(line)
            if len(batch_lines) >= flush_every:
                flush_batch()
        else:
            qdb.write_line(line)

    def on_msg(m: Dict[str, Any]) -> None:
        ch = m.get("ch")
        if ch is None:
            return
        data = m.get("data") or {}
        symbol = (m.get("symbol") or data.get("symbol") or "").upper()
        ts = m.get("ts") or m.get("timestamp")
        if isinstance(ts, str):
            ts_ns = iso_to_nanos(ts)
        elif isinstance(ts, (int, float)):
            # Assume ms if too small for ns; convert to ns
            t = int(ts)
            ts_ns = t if t > 1_000_000_000_000_000_000 else (t * 1_000_000 if t > 1_000_000_000_000 else t * 1_000_000)
        else:
            ts_ns = None

        if ch == "trade" and args.enable_trade:
            price = _to_float(data.get("price"))
            qty = _to_float(data.get("qty"))
            line = qdb.build_line(t_trade, tags={"symbol": symbol}, fields={"price": price, "qty": qty}, ts_ns=ts_ns)
            send_line(line)
            stats["trade"] += 1
            stats["lines"] += 1
        elif ch == "ticker" and args.enable_ticker:
            last = _to_float(data.get("la"))
            open_ = _to_float(data.get("o"))
            high = _to_float(data.get("h"))
            low = _to_float(data.get("l"))
            base_vol = _to_float(data.get("b"))
            quote_vol = _to_float(data.get("q"))
            change_rate = _to_float(data.get("r"))
            line = qdb.build_line(
                t_ticker,
                tags={"symbol": symbol},
                fields={
                    "last": last, "open": open_, "high": high, "low": low,
                    "base_volume": base_vol, "quote_volume": quote_vol, "change_rate": change_rate,
                },
                ts_ns=ts_ns,
            )
            send_line(line)
            stats["ticker"] += 1
            stats["lines"] += 1
        elif ch == "price" and args.enable_price:
            mp = _to_float(data.get("mp"))
            ip = _to_float(data.get("ip"))
            fr = _to_float(data.get("fr"))
            ft = data.get("ft")
            nft = data.get("nft")
            line = qdb.build_line(
                t_price,
                tags={"symbol": symbol},
                fields={"mark_price": mp, "index_price": ip, "funding_rate": fr, "funding_time": ft, "next_funding_time": nft},
                ts_ns=ts_ns,
            )
            send_line(line)
            stats["price"] += 1
            stats["lines"] += 1
        elif isinstance(ch, str) and ch.startswith("depth_book") and args.enable_depth:
            bids = data.get("b") or []
            asks = data.get("a") or []
            # Emit best levels first, keep level index
            for idx, ent in enumerate(bids):
                try:
                    price = _to_float(ent[0])
                    qty = _to_float(ent[1])
                except Exception:
                    continue
                line = qdb.build_line(t_depth, tags={"symbol": symbol, "side": "bid"}, fields={"level": idx, "price": price, "qty": qty}, ts_ns=ts_ns)
                send_line(line)
                stats["depth"] += 1
                stats["lines"] += 1
            for idx, ent in enumerate(asks):
                try:
                    price = _to_float(ent[0])
                    qty = _to_float(ent[1])
                except Exception:
                    continue
                line = qdb.build_line(t_depth, tags={"symbol": symbol, "side": "ask"}, fields={"level": idx, "price": price, "qty": qty}, ts_ns=ts_ns)
                send_line(line)
                stats["depth"] += 1
                stats["lines"] += 1

        # Periodic log every ~10 seconds or 1000 lines
        now = time.monotonic()
        if stats["lines"] % 1000 == 0 or (now - last_log) > 10.0:
            print(json.dumps({"event": "crypto_ingest", **stats}), flush=True)
            last_log = now

    ws.add_listener(on_msg)

    channels: List[Dict[str, str]] = []
    for s in symbols:
        if args.enable_trade:
            channels.append({"symbol": s, "ch": "trade"})
        if args.enable_ticker:
            channels.append({"symbol": s, "ch": "ticker"})
        if args.enable_price:
            channels.append({"symbol": s, "ch": "price"})
        if args.enable_depth:
            channels.append({"symbol": s, "ch": f"depth_book{args.depth_level}"})

    # Connect and subscribe
    print(json.dumps({"event": "crypto_ws_connect", "uri": args.ws_uri, "symbols": symbols, "channels": len(channels)}), flush=True)
    connect_task = asyncio.create_task(ws.connect())
    await asyncio.sleep(1.0)
    try:
        await ws.subscribe(channels)
        print(json.dumps({"event": "crypto_ws_subscribed", "channels": channels[:5], "total": len(channels)}), flush=True)
    except Exception:
        pass
    await connect_task

    # Final flush if any
    flush_batch()


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
