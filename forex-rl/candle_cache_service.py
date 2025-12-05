#!/usr/bin/env python3
"""OANDA-compatible candlestick cache service.

- Exposes:  GET /v3/instruments/{instrument}/candles
- Supports: price=M, granularity, count, from, to, includeFirst (subset of OANDA v20)
- Backed by a local SQLite cache keyed by (environment, instrument, granularity, time).
- On cache miss:
  * For range (from/to) requests: proxies the full request to OANDA once, stores candles, and returns the upstream body.
  * For count-only requests: periodically refreshes from OANDA but serves repeated calls from cache.

Environment:
  OANDA_ENV              practice|live (default: practice)
  OANDA_DEMO_KEY         OANDA token (or OANDA_ACCESS_TOKEN)
  OANDA_ACCESS_TOKEN     alternative token name

Usage example:
  OANDA_DEMO_KEY=... OANDA_ENV=practice \
    python -m forex-rl.candle_cache_service --host 0.0.0.0 --port 9100

Then point clients at:
  http://127.0.0.1:9100/v3/instruments/EUR_USD/candles?granularity=M1&count=300&price=M

The JSON shape of responses matches OANDA's /v3/instruments/{instrument}/candles for price=M.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests


LOG = logging.getLogger("candle_cache_service")

DB_PATH: str = "oanda_candles_cache.db"
OANDA_ENV: str = os.environ.get("OANDA_ENV", "practice")
UPSTREAM_BASE: str = (
    "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com"
)
OANDA_TOKEN: Optional[str] = os.environ.get("OANDA_DEMO_KEY") or os.environ.get("OANDA_ACCESS_TOKEN")

# Approximate bar durations in seconds for a subset of granularities we care about.
GRANULARITY_SECONDS: Dict[str, int] = {
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
    "W": 7 * 86400,
}


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Main candle store: one row per (env, instrument, granularity, time) with optional mid/bid/ask legs.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            environment TEXT NOT NULL,
            instrument  TEXT NOT NULL,
            granularity TEXT NOT NULL,
            time        TEXT NOT NULL,
            complete    INTEGER NOT NULL,
            mid_o       TEXT,
            mid_h       TEXT,
            mid_l       TEXT,
            mid_c       TEXT,
            bid_o       TEXT,
            bid_h       TEXT,
            bid_l       TEXT,
            bid_c       TEXT,
            ask_o       TEXT,
            ask_h       TEXT,
            ask_l       TEXT,
            ask_c       TEXT,
            volume      INTEGER NOT NULL,
            PRIMARY KEY (environment, instrument, granularity, time)
        )
        """
    )
    # Range coverage table: records upstream ranges we know are fully cached for a specific price mask.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candle_ranges (
            environment TEXT NOT NULL,
            instrument  TEXT NOT NULL,
            granularity TEXT NOT NULL,
            price       TEXT NOT NULL,
            from_time   TEXT NOT NULL,
            to_time     TEXT NOT NULL,
            PRIMARY KEY (environment, instrument, granularity, price, from_time, to_time)
        )
        """
    )
    conn.commit()


def _upsert_candles(
    conn: sqlite3.Connection,
    environment: str,
    instrument: str,
    granularity: str,
    candles: List[Dict[str, Any]],
) -> None:
    if not candles:
        return
    rows: List[Tuple[Any, ...]] = []
    for c in candles:
        try:
            ts = str(c.get("time"))
            if not ts:
                continue
            complete = 1 if c.get("complete", False) else 0
            mid = c.get("mid") or {}
            bid = c.get("bid") or {}
            ask = c.get("ask") or {}
            mid_o = str(mid.get("o")) if mid.get("o") is not None else None
            mid_h = str(mid.get("h")) if mid.get("h") is not None else None
            mid_l = str(mid.get("l")) if mid.get("l") is not None else None
            mid_c = str(mid.get("c")) if mid.get("c") is not None else None
            bid_o = str(bid.get("o")) if bid.get("o") is not None else None
            bid_h = str(bid.get("h")) if bid.get("h") is not None else None
            bid_l = str(bid.get("l")) if bid.get("l") is not None else None
            bid_c = str(bid.get("c")) if bid.get("c") is not None else None
            ask_o = str(ask.get("o")) if ask.get("o") is not None else None
            ask_h = str(ask.get("h")) if ask.get("h") is not None else None
            ask_l = str(ask.get("l")) if ask.get("l") is not None else None
            ask_c = str(ask.get("c")) if ask.get("c") is not None else None
            vol_raw = c.get("volume", 0)
            try:
                vol = int(vol_raw)
            except Exception:
                try:
                    vol = int(float(vol_raw))
                except Exception:
                    vol = 0
            rows.append(
                (
                    environment,
                    instrument,
                    granularity,
                    ts,
                    complete,
                    mid_o,
                    mid_h,
                    mid_l,
                    mid_c,
                    bid_o,
                    bid_h,
                    bid_l,
                    bid_c,
                    ask_o,
                    ask_h,
                    ask_l,
                    ask_c,
                    vol,
                )
            )
        except Exception:
            continue
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO candles (
            environment, instrument, granularity, time, complete,
            mid_o, mid_h, mid_l, mid_c,
            bid_o, bid_h, bid_l, bid_c,
            ask_o, ask_h, ask_l, ask_c,
            volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(environment, instrument, granularity, time) DO UPDATE SET
            complete=excluded.complete,
            mid_o=COALESCE(excluded.mid_o, candles.mid_o),
            mid_h=COALESCE(excluded.mid_h, candles.mid_h),
            mid_l=COALESCE(excluded.mid_l, candles.mid_l),
            mid_c=COALESCE(excluded.mid_c, candles.mid_c),
            bid_o=COALESCE(excluded.bid_o, candles.bid_o),
            bid_h=COALESCE(excluded.bid_h, candles.bid_h),
            bid_l=COALESCE(excluded.bid_l, candles.bid_l),
            bid_c=COALESCE(excluded.bid_c, candles.bid_c),
            ask_o=COALESCE(excluded.ask_o, candles.ask_o),
            ask_h=COALESCE(excluded.ask_h, candles.ask_h),
            ask_l=COALESCE(excluded.ask_l, candles.ask_l),
            ask_c=COALESCE(excluded.ask_c, candles.ask_c),
            volume=excluded.volume
        """,
        rows,
    )
    conn.commit()


def _get_max_time(
    conn: sqlite3.Connection,
    environment: str,
    instrument: str,
    granularity: str,
) -> Optional[str]:
    cur = conn.execute(
        """
        SELECT time FROM candles
        WHERE environment=? AND instrument=? AND granularity=?
        ORDER BY time DESC
        LIMIT 1
        """,
        (environment, instrument, granularity),
    )
    row = cur.fetchone()
    return str(row["time"]) if row is not None else None


def _get_last_n(
    conn: sqlite3.Connection,
    environment: str,
    instrument: str,
    granularity: str,
    count: int,
) -> List[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT * FROM candles
        WHERE environment=? AND instrument=? AND granularity=?
        ORDER BY time DESC
        LIMIT ?
        """,
        (environment, instrument, granularity, int(count)),
    )
    rows = cur.fetchall()
    # We selected DESC, but OANDA returns ASC time order
    return list(reversed(rows))


def _get_range(
    conn: sqlite3.Connection,
    environment: str,
    instrument: str,
    granularity: str,
    from_time: Optional[str],
    to_time: Optional[str],
) -> List[sqlite3.Row]:
    sql = [
        "SELECT * FROM candles WHERE environment=? AND instrument=? AND granularity=?",
    ]
    params: List[Any] = [environment, instrument, granularity]
    if from_time:
        sql.append("AND time >= ?")
        params.append(from_time)
    if to_time:
        sql.append("AND time <= ?")
        params.append(to_time)
    sql.append("ORDER BY time ASC")
    cur = conn.execute(" ".join(sql), tuple(params))
    return cur.fetchall()


def _is_fresh(max_time: Optional[str], granularity: str) -> bool:
    if not max_time:
        return False
    seconds = GRANULARITY_SECONDS.get(granularity)
    if not seconds:
        return False
    try:
        dt = datetime.fromisoformat(max_time.replace("Z", "+00:00"))
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    # Slightly less than one full bar length to avoid constant remote refresh
    return age < seconds * 0.75


def _rows_have_prices(rows: List[sqlite3.Row], price: str) -> bool:
    """Return True if all requested price legs are present for each row."""
    price = (price or "M").upper()
    need_m = "M" in price
    need_b = "B" in price
    need_a = "A" in price
    for r in rows:
        if need_m and not (r["mid_o"] and r["mid_h"] and r["mid_l"] and r["mid_c"]):
            return False
        if need_b and not (r["bid_o"] and r["bid_h"] and r["bid_l"] and r["bid_c"]):
            return False
        if need_a and not (r["ask_o"] and r["ask_h"] and r["ask_l"] and r["ask_c"]):
            return False
    return True


def _build_response_from_rows(
    rows: List[sqlite3.Row], instrument: str, granularity: str, price: str
) -> Dict[str, Any]:
    price = (price or "M").upper()
    need_m = "M" in price
    need_b = "B" in price
    need_a = "A" in price
    candles: List[Dict[str, Any]] = []
    for r in rows:
        item: Dict[str, Any] = {
            "complete": bool(r["complete"]),
            "volume": int(r["volume"]),
            "time": str(r["time"]),
        }
        if need_m:
            item["mid"] = {
                "o": str(r["mid_o"]),
                "h": str(r["mid_h"]),
                "l": str(r["mid_l"]),
                "c": str(r["mid_c"]),
            }
        if need_b:
            item["bid"] = {
                "o": str(r["bid_o"]),
                "h": str(r["bid_h"]),
                "l": str(r["bid_l"]),
                "c": str(r["bid_c"]),
            }
        if need_a:
            item["ask"] = {
                "o": str(r["ask_o"]),
                "h": str(r["ask_h"]),
                "l": str(r["ask_l"]),
                "c": str(r["ask_c"]),
            }
        candles.append(item)
    return {
        "instrument": instrument,
        "granularity": granularity,
        "candles": candles,
        # OANDA responses also include a top-level time; approximate with now.
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _proxy_to_oanda(path: str, query: str) -> Dict[str, Any]:
    if not OANDA_TOKEN:
        raise RuntimeError("Missing OANDA token (OANDA_DEMO_KEY / OANDA_ACCESS_TOKEN) for upstream candle fetch")
    url = f"{UPSTREAM_BASE}{path}"
    if query:
        url = f"{url}?{query}"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


class CandleCacheHandler(BaseHTTPRequestHandler):
    server_version = "CandleCacheHTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - stdlib style logging
        LOG.info("%s - - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/v3/instruments/") or not parsed.path.endswith("/candles"):
                self.send_error(404, "Not Found")
                return
            # Path: /v3/instruments/{instrument}/candles
            parts = parsed.path.split("/")
            # ["", "v3", "instruments", "{instrument}", "candles"]
            if len(parts) != 5:
                self.send_error(404, "Not Found")
                return
            instrument = parts[3]
            query = parse_qs(parsed.query)
            self._handle_candles(instrument, parsed.path, parsed.query, query)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("request error: %s", exc)
            self.send_error(500, "Internal Server Error")

    def _handle_candles(
        self,
        instrument: str,
        path: str,
        raw_query: str,
        query: Dict[str, List[str]],
    ) -> None:
        price = (query.get("price", ["M"]) or ["M"])[0] or "M"
        granularity = (query.get("granularity", [None]) or [None])[0]
        if not granularity:
            self.send_error(400, "Missing granularity parameter")
            return

        from_time = (query.get("from", [None]) or [None])[0]
        to_time = (query.get("to", [None]) or [None])[0]
        count_raw = (query.get("count", [None]) or [None])[0]
        try:
            count = int(count_raw) if count_raw is not None else 0
        except Exception:
            count = 0
        if count <= 0:
            # OANDA default is 500 when count omitted; mimic that.
            count = 500

        try:
            with _connect_db() as conn:
                # Range requests with both from/to: serve from cache when a fully-covered range is known.
                if from_time and to_time:
                    cur = conn.execute(
                        """
                        SELECT 1 FROM candle_ranges
                        WHERE environment=? AND instrument=? AND granularity=? AND price=?
                          AND from_time <= ? AND to_time >= ?
                        LIMIT 1
                        """,
                        (OANDA_ENV, instrument, granularity, price, from_time, to_time),
                    )
                    if cur.fetchone() is not None:
                        rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
                        if rows and _rows_have_prices(rows, price):
                            payload = _build_response_from_rows(rows, instrument, granularity, price)
                            self._send_json(200, payload)
                            return

                # Range-based requests: default to proxy + cache when coverage is unknown or incomplete.
                if from_time or to_time:
                    try:
                        upstream_payload = _proxy_to_oanda(path, raw_query)
                    except Exception as exc:
                        LOG.error("upstream error (range): %s", exc)
                        self.send_error(502, "Upstream error")
                        return
                    try:
                        candles = upstream_payload.get("candles", [])
                        _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
                        # Record coverage only when both from/to are provided and we received candles.
                        if from_time and to_time and candles:
                            first_ts = str(candles[0].get("time") or from_time)
                            last_ts = str(candles[-1].get("time") or to_time)
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO candle_ranges
                                (environment, instrument, granularity, price, from_time, to_time)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (OANDA_ENV, instrument, granularity, price, first_ts, last_ts),
                            )
                            conn.commit()
                    except Exception:
                        LOG.exception("failed to cache range candles")
                    self._send_json(200, upstream_payload)
                    return

                # Count-only: prefer cache when recent and required price legs are present.
                max_time = _get_max_time(conn, OANDA_ENV, instrument, granularity)
                if _is_fresh(max_time, granularity):
                    rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
                    if rows and _rows_have_prices(rows, price):
                        payload = _build_response_from_rows(rows, instrument, granularity, price)
                        self._send_json(200, payload)
                        return

                # Either not fresh, missing prices, or empty: fetch from upstream once, store, then serve from cache.
                try:
                    upstream_payload = _proxy_to_oanda(path, raw_query)
                except Exception as exc:
                    LOG.error("upstream error (refresh): %s", exc)
                    # No stale fallback: bubble upstream error to caller.
                    self.send_error(502, "Upstream error")
                    return

                try:
                    candles = upstream_payload.get("candles", [])
                    _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
                except Exception:
                    LOG.exception("failed to upsert refreshed candles into cache")

                rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
                if rows and _rows_have_prices(rows, price):
                    payload = _build_response_from_rows(rows, instrument, granularity, price)
                    self._send_json(200, payload)
                else:
                    # If we still cannot serve from cache, return upstream payload directly.
                    self._send_json(200, upstream_payload)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("cache handling error: %s", exc)
            self.send_error(500, "Internal Server Error")


def main() -> None:
    parser = argparse.ArgumentParser(description="OANDA-compatible candlestick cache service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--db", default="oanda_candles_cache.db", help="SQLite DB path for candle cache")
    args = parser.parse_args()

    global DB_PATH
    DB_PATH = args.db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOG.info(
        "starting candle cache on %s:%d (env=%s, upstream=%s, db=%s)",
        args.host,
        args.port,
        OANDA_ENV,
        UPSTREAM_BASE,
        DB_PATH,
    )

    server = ThreadingHTTPServer((args.host, args.port), CandleCacheHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutdown requested; stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
