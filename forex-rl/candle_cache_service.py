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
from urllib.parse import urlparse, parse_qs, urlencode

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


def _is_fx_instrument(instrument: str) -> bool:
    """Heuristic to detect standard FX pairs (OANDA-backed).

    Instruments of the form ABC_XYZ (3-letter alphabetic, uppercase) are
    treated as FX and proxied to OANDA when needed.
    """
    parts = instrument.split("_")
    if len(parts) != 2:
        return False
    a, b = parts
    return (
        len(a) == 3
        and len(b) == 3
        and a.isalpha()
        and b.isalpha()
        and a.isupper()
        and b.isupper()
    )


# Known instruments that are backed purely by local imports (no upstream).
LOCAL_ONLY_INSTRUMENTS = {"NQ", "NQ_USD"}


def _is_alpaca_stock_symbol(instrument: str) -> bool:
    """Heuristic for Alpaca stock symbols.

    We treat plain uppercase tickers without underscores (e.g. SPY, QQQ, AAPL,
    MSFT) as Alpaca-backed equities. This keeps naming disjoint from OANDA FX
    (which always use ABC_XYZ).
    """
    if "_" in instrument or "/" in instrument:
        return False
    if not instrument:
        return False
    # Allow dots in tickers (e.g. BRK.B) while keeping the core alphanumeric.
    core = instrument.replace(".", "")
    return core.isupper() and core.isalnum() and 1 <= len(core) <= 10


def _is_alpaca_crypto_symbol(instrument: str) -> bool:
    """Heuristic for Alpaca crypto symbols (e.g. BTC/USD, ETH/USD).

    We treat symbols with a single '/' separator and no underscores as
    Alpaca-backed crypto instruments. Both legs must be short-ish, uppercase,
    and alphanumeric.
    """
    if "_" in instrument:
        return False
    if "/" not in instrument:
        return False
    base, quote = instrument.split("/", 1)
    if not base or not quote:
        return False
    if len(base) > 10 or len(quote) > 10:
        return False
    if not (base.replace(".", "").isalnum() and quote.replace(".", "").isalnum()):
        return False
    return base.isupper() and quote.isupper()


def _instrument_provider(instrument: str) -> str:
    """Return which upstream should be considered for this instrument.

    Values:
      - "oanda_fx"       → standard OANDA FX pair
      - "alpaca_stock"   → Alpaca-backed equity (via market data API)
      - "alpaca_crypto"  → Alpaca-backed crypto (e.g. BTC/USD)
      - "local"          → local-only (e.g. NQ futures import) with no upstream
    """
    if instrument in LOCAL_ONLY_INSTRUMENTS:
        return "local"
    if _is_fx_instrument(instrument):
        return "oanda_fx"
    if _is_alpaca_stock_symbol(instrument):
        return "alpaca_stock"
    if _is_alpaca_crypto_symbol(instrument):
        return "alpaca_crypto"
    return "local"

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


def _range_rows_fresh(
    rows: List[sqlite3.Row],
    granularity: str,
    to_time: Optional[str],
) -> bool:
    """Heuristic freshness check for range responses.

    For from/to requests that reach close to "now", we want to avoid serving a
    stale, truncated range from the cache when new candles may exist upstream.
    If a `to` bound is provided and the last cached candle lags that bound by
    more than ~0.75 of a bar, treat the cached range as stale and force an
    upstream refresh.
    """
    if not rows:
        return False
    if not to_time:
        # Historical backfills without an explicit upper bound can rely on the
        # cached rows as-is.
        return True

    seconds = GRANULARITY_SECONDS.get(granularity.upper())
    if not seconds:
        # Unknown granularity: fall back to trusting the cached rows.
        return True

    try:
        last_ts_raw = str(rows[-1]["time"])
        last_dt = datetime.fromisoformat(last_ts_raw.replace("Z", "+00:00"))
        to_dt = datetime.fromisoformat(str(to_time).replace("Z", "+00:00"))
    except Exception:
        # If we cannot parse timestamps reliably, err on the side of serving
        # from cache instead of causing extra upstream load.
        return True

    gap = (to_dt - last_dt).total_seconds()

    # If the last cached candle is within ~0.75 of one bar of the requested
    # upper bound, treat the range as "fresh enough". Otherwise, consider it
    # stale so callers (particularly live trading) can see newly-formed bars.
    return gap <= seconds * 0.75


def _range_rows_cover_from(
    rows: List[sqlite3.Row],
    granularity: str,
    from_time: Optional[str],
) -> bool:
    """Return True if cached rows cover the requested lower bound reasonably.

    For Alpaca-backed instruments we sometimes see partial coverage in the DB
    (e.g. only the last N days of a much longer requested window). When the
    earliest cached candle is far *after* the requested `from` time, we should
    treat the range as incomplete and force a full-range upstream fetch.
    """
    if not rows or not from_time:
        return True

    seconds = GRANULARITY_SECONDS.get(granularity.upper())
    if not seconds:
        # Unknown granularity: fall back to trusting the cached rows.
        return True

    try:
        first_ts_raw = str(rows[0]["time"])
        first_dt = datetime.fromisoformat(first_ts_raw.replace("Z", "+00:00"))
        from_dt = datetime.fromisoformat(str(from_time).replace("Z", "+00:00"))
    except Exception:
        # If we cannot parse timestamps reliably, err on the side of trusting
        # the cached rows instead of hammering Alpaca/OANDA.
        return True

    gap = (first_dt - from_dt).total_seconds()
    # If the first cached candle is within ~0.75 of one bar of the requested
    # lower bound, treat coverage as sufficient. Otherwise, consider the head
    # of the range incomplete and force a full-range fetch.
    return gap <= seconds * 0.75


def _maybe_top_up_tail(
    conn: sqlite3.Connection,
    instrument: str,
    granularity: str,
    price: str,
    rows: List[sqlite3.Row],
    to_time: Optional[str],
) -> bool:
    """Incrementally fetch only missing tail candles for a range, when cheap.

    When the cache has most of the requested [from,to] window but is missing a
    small number of candles at the upper bound, we avoid re-fetching the entire
    historical span from OANDA. Instead, we:
      * Estimate how many bars are missing at the tail.
      * If the gap is small, issue a narrow upstream request starting at the
        last cached timestamp up to the requested `to`, upsert those candles,
        and let callers re-read the full range from SQLite.

    Returns True if the operation succeeded or if no top-up is required (e.g.
    weekend, malformed timestamps). Returns False when the caller should fall
    back to a full-range upstream proxy.
    """
    if not rows or not to_time:
        return False

    seconds = GRANULARITY_SECONDS.get(granularity.upper())
    if not seconds:
        return False

    try:
        last_ts_raw = str(rows[-1]["time"])
        last_dt = datetime.fromisoformat(last_ts_raw.replace("Z", "+00:00"))
        to_dt = datetime.fromisoformat(str(to_time).replace("Z", "+00:00"))
    except Exception:
        # If parsing fails, let the caller decide on a full refresh.
        return False

    gap = (to_dt - last_dt).total_seconds()
    if gap <= 0:
        # Already at or beyond requested upper bound.
        return True

    # Weekend: if `to` is on Saturday/Sunday and the last cached candle is
    # Friday, there are no new bars to fetch; treat as effectively topped up.
    if to_dt.weekday() in (5, 6) and last_dt.weekday() == 4:
        return True

    # Estimated number of missing bars; if very large, we prefer falling back
    # to a full upstream proxy instead of issuing a huge incremental request.
    missing_bars = gap / float(seconds)
    MAX_TAIL_BARS = 1000
    if missing_bars > MAX_TAIL_BARS:
        return False

    # Issue a narrow upstream request for the missing tail only.
    if not OANDA_TOKEN:
        # Without a token we cannot top-up; let caller decide.
        return False

    tail_from = last_dt.isoformat().replace("+00:00", "Z")
    params = {
        "granularity": granularity,
        "price": price,
        "from": tail_from,
        "to": str(to_time),
    }
    try:
        query = urlencode(params)
        upstream_payload = _proxy_to_oanda(f"/v3/instruments/{instrument}/candles", query)
    except Exception as exc:
        LOG.error("upstream error (tail top-up): %s", exc)
        return False

    try:
        candles = upstream_payload.get("candles", [])
        _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
    except Exception:
        LOG.exception("failed to upsert tail candles into cache")
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


_ALPACA_STOCK_CLIENT: Optional[Any] = None
_ALPACA_CRYPTO_CLIENT: Optional[Any] = None


def _get_alpaca_stock_client() -> Any:
    """Lazy-initialized Alpaca stock historical data client."""
    global _ALPACA_STOCK_CLIENT
    if _ALPACA_STOCK_CLIENT is not None:
        return _ALPACA_STOCK_CLIENT

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY for Alpaca upstream")

    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"alpaca-py not available in environment: {exc}") from exc

    _ALPACA_STOCK_CLIENT = StockHistoricalDataClient(key, secret)
    return _ALPACA_STOCK_CLIENT


def _get_alpaca_crypto_client() -> Any:
    """Lazy-initialized Alpaca crypto historical data client."""
    global _ALPACA_CRYPTO_CLIENT
    if _ALPACA_CRYPTO_CLIENT is not None:
        return _ALPACA_CRYPTO_CLIENT

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY for Alpaca upstream")

    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"alpaca-py not available in environment: {exc}") from exc

    _ALPACA_CRYPTO_CLIENT = CryptoHistoricalDataClient(key, secret)
    return _ALPACA_CRYPTO_CLIENT


def _fetch_alpaca_stock_candles(
    instrument: str,
    granularity: str,
    from_time: Optional[str],
    to_time: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch stock bars from Alpaca and return OANDA-style candle dicts."""
    if granularity.upper() != "D":
        raise ValueError("Alpaca upstream currently supports only daily (D) granularity")

    client = _get_alpaca_stock_client()

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.common.enums import Sort
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"alpaca-py components missing: {exc}") from exc

    kwargs: Dict[str, Any] = {
        "symbol_or_symbols": instrument,
        "timeframe": TimeFrame.Day,
        "sort": Sort.ASC,
    }
    if from_time:
        try:
            kwargs["start"] = datetime.fromisoformat(str(from_time).replace("Z", "+00:00"))
        except Exception:
            pass
    if to_time:
        try:
            kwargs["end"] = datetime.fromisoformat(str(to_time).replace("Z", "+00:00"))
        except Exception:
            pass
    if limit > 0:
        kwargs["limit"] = int(limit)

    req = StockBarsRequest(**kwargs)
    barset = client.get_stock_bars(request_params=req)

    try:
        bars = barset[instrument]
    except Exception:
        bars = getattr(barset, "data", {}).get(instrument, []) or []

    candles: List[Dict[str, Any]] = []
    for b in bars:
        try:
            ts_dt = b.timestamp.astimezone(timezone.utc)
            ts = ts_dt.isoformat().replace("+00:00", "Z")
            candles.append(
                {
                    "time": ts,
                    "complete": True,
                    "mid": {
                        "o": float(b.open),
                        "h": float(b.high),
                        "l": float(b.low),
                        "c": float(b.close),
                    },
                    "volume": int(getattr(b, "volume", 0) or 0),
                }
            )
        except Exception:
            continue
    return candles


def _fetch_alpaca_crypto_candles(
    instrument: str,
    granularity: str,
    from_time: Optional[str],
    to_time: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch crypto bars from Alpaca and return OANDA-style candle dicts."""
    if granularity.upper() != "D":
        raise ValueError("Alpaca crypto upstream currently supports only daily (D) granularity")

    client = _get_alpaca_crypto_client()

    try:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.common.enums import Sort
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"alpaca-py components missing: {exc}") from exc

    kwargs: Dict[str, Any] = {
        "symbol_or_symbols": instrument,
        "timeframe": TimeFrame.Day,
        "sort": Sort.ASC,
    }
    if from_time:
        try:
            kwargs["start"] = datetime.fromisoformat(str(from_time).replace("Z", "+00:00"))
        except Exception:
            pass
    if to_time:
        try:
            kwargs["end"] = datetime.fromisoformat(str(to_time).replace("Z", "+00:00"))
        except Exception:
            pass
    if limit > 0:
        kwargs["limit"] = int(limit)

    req = CryptoBarsRequest(**kwargs)
    barset = client.get_crypto_bars(request_params=req)

    try:
        bars = barset[instrument]
    except Exception:
        bars = getattr(barset, "data", {}).get(instrument, []) or []

    candles: List[Dict[str, Any]] = []
    for b in bars:
        try:
            ts_dt = b.timestamp.astimezone(timezone.utc)
            ts = ts_dt.isoformat().replace("+00:00", "Z")
            candles.append(
                {
                    "time": ts,
                    "complete": True,
                    "mid": {
                        "o": float(b.open),
                        "h": float(b.high),
                        "l": float(b.low),
                        "c": float(b.close),
                    },
                    "volume": int(getattr(b, "volume", 0) or 0),
                }
            )
        except Exception:
            continue
    return candles


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
            prefix = "/v3/instruments/"
            suffix = "/candles"
            if not parsed.path.startswith(prefix) or not parsed.path.endswith(suffix):
                self.send_error(404, "Not Found")
                return
            # Path: /v3/instruments/{instrument}/candles
            # `{instrument}` may itself contain '/' (e.g. BTC/USD), so we slice
            # between the known prefix and suffix instead of splitting on '/'.
            instrument_path = parsed.path[len(prefix) : len(parsed.path) - len(suffix)]
            instrument = instrument_path.strip("/")
            if not instrument:
                self.send_error(404, "Not Found")
                return
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

        provider = _instrument_provider(instrument)

        try:
            with _connect_db() as conn:
                # Alpaca-backed equities/crypto: handle via Alpaca market data API.
                if provider == "alpaca_stock":
                    self._handle_alpaca_stock(
                        conn,
                        instrument=instrument,
                        granularity=granularity,
                        price=price,
                        from_time=from_time,
                        to_time=to_time,
                        count=count,
                    )
                    return
                if provider == "alpaca_crypto":
                    self._handle_alpaca_crypto(
                        conn,
                        instrument=instrument,
                        granularity=granularity,
                        price=price,
                        from_time=from_time,
                        to_time=to_time,
                        count=count,
                    )
                    return

                allow_upstream = bool(OANDA_TOKEN) and provider == "oanda_fx"
                # Range requests with both from/to: prefer cache, with optional
                # incremental tail top-up when only a few candles are missing.
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
                            if _range_rows_fresh(rows, granularity, to_time) or _maybe_top_up_tail(
                                conn, instrument, granularity, price, rows, to_time
                            ):
                                rows = _get_range(
                                    conn, OANDA_ENV, instrument, granularity, from_time, to_time
                                )
                                payload = _build_response_from_rows(rows, instrument, granularity, price)
                                self._send_json(200, payload)
                                return

                    # No explicit coverage record, but we might still have cached data.
                    rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
                    if rows and _rows_have_prices(rows, price):
                        if _range_rows_fresh(rows, granularity, to_time) or _maybe_top_up_tail(
                            conn, instrument, granularity, price, rows, to_time
                        ):
                            rows = _get_range(
                                conn, OANDA_ENV, instrument, granularity, from_time, to_time
                            )
                            payload = _build_response_from_rows(rows, instrument, granularity, price)
                            self._send_json(200, payload)
                            return

                # Range-based requests: default to proxy + cache for FX instruments when
                # coverage is unknown or incomplete, and no rows are present locally.
                if from_time or to_time:
                    # Best-effort local fallback before hitting upstream. When
                    # we already have most of the requested range in cache, we
                    # optionally top up just the missing tail candles instead
                    # of re-fetching the full [from,to] span from OANDA.
                    rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
                    if rows and _rows_have_prices(rows, price):
                        if (to_time and _range_rows_fresh(rows, granularity, to_time)) or _maybe_top_up_tail(
                            conn, instrument, granularity, price, rows, to_time
                        ):
                            rows = _get_range(
                                conn, OANDA_ENV, instrument, granularity, from_time, to_time
                            )
                            payload = _build_response_from_rows(rows, instrument, granularity, price)
                            self._send_json(200, payload)
                            return

                    # For non-FX instruments (e.g., Alpaca equities, imported futures),
                    # we treat the cache as the sole source of truth and never call OANDA.
                    if not allow_upstream:
                        payload = _build_response_from_rows(rows, instrument, granularity, price)
                        self._send_json(200, payload)
                        return

                    try:
                        upstream_payload = _proxy_to_oanda(path, raw_query)
                    except Exception as exc:
                        LOG.error("upstream error (range): %s", exc)
                        # Best-effort fallback: serve any cached rows we have for this range.
                        try:
                            fallback_rows = _get_range(
                                conn, OANDA_ENV, instrument, granularity, from_time, to_time
                            )
                        except Exception:
                            fallback_rows = []
                        if fallback_rows and _rows_have_prices(fallback_rows, price):
                            LOG.info(
                                "serving cached range for %s %s despite upstream error",
                                instrument,
                                granularity,
                            )
                            payload = _build_response_from_rows(
                                fallback_rows, instrument, granularity, price
                            )
                            self._send_json(200, payload)
                            return
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

                # Either not fresh, missing prices, or empty: for FX instruments we
                # fetch from upstream once, store, then serve from cache. For non-FX,
                # we fall back to whatever is cached (possibly stale) and never hit OANDA.
                if not allow_upstream:
                    rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
                    payload = _build_response_from_rows(rows, instrument, granularity, price)
                    self._send_json(200, payload)
                    return

                try:
                    upstream_payload = _proxy_to_oanda(path, raw_query)
                except Exception as exc:
                    LOG.error("upstream error (refresh): %s", exc)
                    # Best-effort stale fallback: if we have any cached candles at all for this
                    # instrument/granularity, serve them even if considered "stale".
                    try:
                        fallback_rows = _get_last_n(
                            conn, OANDA_ENV, instrument, granularity, count
                        )
                    except Exception:
                        fallback_rows = []
                    if fallback_rows and _rows_have_prices(fallback_rows, price):
                        LOG.info(
                            "serving cached stale candles for %s %s despite upstream error",
                            instrument,
                            granularity,
                        )
                            # NOTE: build and return payload from fallback rows
                        payload = _build_response_from_rows(
                            fallback_rows, instrument, granularity, price
                        )
                        self._send_json(200, payload)
                        return
                    # If we truly have nothing cached, bubble the upstream error.
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


    def _handle_alpaca_stock(
        self,
        conn: sqlite3.Connection,
        instrument: str,
        granularity: str,
        price: str,
        from_time: Optional[str],
        to_time: Optional[str],
        count: int,
    ) -> None:
        """Handle candles for Alpaca-backed stock symbols.

        We only support daily (D) granularity. Data is fetched from Alpaca's
        market data API and cached into the shared SQLite DB using the same
        OANDA-style candle shape as FX.
        """
        if granularity.upper() != "D":
            self.send_error(400, "Alpaca-backed instruments currently support only D granularity")
            return

        # Range requests with both from/to: prefer cache when available, but only
        # when coverage is reasonably complete at both the head and the tail.
        if from_time or to_time:
            rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
            if (
                rows
                and _rows_have_prices(rows, price)
                and _range_rows_cover_from(rows, granularity, from_time)
                and _range_rows_fresh(rows, granularity, to_time)
            ):
                payload = _build_response_from_rows(rows, instrument, granularity, price)
                self._send_json(200, payload)
                return

            # Fetch from Alpaca, upsert into cache, then serve from DB.
            try:
                candles = _fetch_alpaca_stock_candles(
                    instrument, granularity, from_time=from_time, to_time=to_time, limit=0
                )
            except Exception as exc:
                LOG.error("alpaca upstream error (range) for %s: %s", instrument, exc)
                # Fallback to any cached rows we might have.
                if rows and _rows_have_prices(rows, price):
                    payload = _build_response_from_rows(rows, instrument, granularity, price)
                    self._send_json(200, payload)
                    return
                self.send_error(502, "Alpaca upstream error")
                return

            try:
                _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
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
                LOG.exception("failed to cache Alpaca range candles for %s", instrument)

            rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
            payload = _build_response_from_rows(rows, instrument, granularity, price)
            self._send_json(200, payload)
            return

        # Count-only: prefer cache when recent and we have at least `count`
        # candles with required price legs; otherwise, fetch latest from Alpaca.
        max_time = _get_max_time(conn, OANDA_ENV, instrument, granularity)
        rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
        if (
            rows
            and len(rows) >= count
            and _is_fresh(max_time, granularity)
            and _rows_have_prices(rows, price)
        ):
            payload = _build_response_from_rows(rows, instrument, granularity, price)
            self._send_json(200, payload)
            return

        try:
            candles = _fetch_alpaca_stock_candles(
                instrument, granularity, from_time=None, to_time=None, limit=count
            )
        except Exception as exc:
            LOG.error("alpaca upstream error (refresh) for %s: %s", instrument, exc)
            # Best-effort fallback: serve any cached rows (possibly stale).
            rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
            if rows and _rows_have_prices(rows, price):
                payload = _build_response_from_rows(rows, instrument, granularity, price)
                self._send_json(200, payload)
                return
            self.send_error(502, "Alpaca upstream error")
            return

        try:
            _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
        except Exception:
            LOG.exception("failed to upsert Alpaca candles into cache for %s", instrument)

        rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
        payload = _build_response_from_rows(rows, instrument, granularity, price)
        self._send_json(200, payload)


    def _handle_alpaca_crypto(
        self,
        conn: sqlite3.Connection,
        instrument: str,
        granularity: str,
        price: str,
        from_time: Optional[str],
        to_time: Optional[str],
        count: int,
    ) -> None:
        """Handle candles for Alpaca-backed crypto symbols.

        We only support daily (D) granularity. Data is fetched from Alpaca's
        crypto market data API and cached into the shared SQLite DB using the
        same OANDA-style candle shape as FX.
        """
        if granularity.upper() != "D":
            self.send_error(400, "Alpaca-backed crypto currently supports only D granularity")
            return

        # Range requests with both from/to: prefer cache when available, but only
        # when coverage is reasonably complete at both the head and the tail.
        if from_time or to_time:
            rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
            if (
                rows
                and _rows_have_prices(rows, price)
                and _range_rows_cover_from(rows, granularity, from_time)
                and _range_rows_fresh(rows, granularity, to_time)
            ):
                payload = _build_response_from_rows(rows, instrument, granularity, price)
                self._send_json(200, payload)
                return

            # Fetch from Alpaca crypto, upsert into cache, then serve from DB.
            try:
                candles = _fetch_alpaca_crypto_candles(
                    instrument, granularity, from_time=from_time, to_time=to_time, limit=0
                )
            except Exception as exc:
                LOG.error("alpaca crypto upstream error (range) for %s: %s", instrument, exc)
                # Fallback to any cached rows we might have.
                if rows and _rows_have_prices(rows, price):
                    payload = _build_response_from_rows(rows, instrument, granularity, price)
                    self._send_json(200, payload)
                    return
                self.send_error(502, "Alpaca crypto upstream error")
                return

            try:
                _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
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
                LOG.exception("failed to cache Alpaca crypto range candles for %s", instrument)

            rows = _get_range(conn, OANDA_ENV, instrument, granularity, from_time, to_time)
            payload = _build_response_from_rows(rows, instrument, granularity, price)
            self._send_json(200, payload)
            return

        # Count-only: prefer cache when recent and we have at least `count`
        # candles with required price legs; otherwise, fetch latest from Alpaca crypto.
        max_time = _get_max_time(conn, OANDA_ENV, instrument, granularity)
        rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
        if (
            rows
            and len(rows) >= count
            and _is_fresh(max_time, granularity)
            and _rows_have_prices(rows, price)
        ):
            payload = _build_response_from_rows(rows, instrument, granularity, price)
            self._send_json(200, payload)
            return

        try:
            candles = _fetch_alpaca_crypto_candles(
                instrument, granularity, from_time=None, to_time=None, limit=count
            )
        except Exception as exc:
            LOG.error("alpaca crypto upstream error (refresh) for %s: %s", instrument, exc)
            # Best-effort fallback: serve any cached rows (possibly stale).
            rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
            if rows and _rows_have_prices(rows, price):
                payload = _build_response_from_rows(rows, instrument, granularity, price)
                self._send_json(200, payload)
                return
            self.send_error(502, "Alpaca crypto upstream error")
            return

        try:
            _upsert_candles(conn, OANDA_ENV, instrument, granularity, candles)
        except Exception:
            LOG.exception("failed to upsert Alpaca crypto candles into cache for %s", instrument)

        rows = _get_last_n(conn, OANDA_ENV, instrument, granularity, count)
        payload = _build_response_from_rows(rows, instrument, granularity, price)
        self._send_json(200, payload)


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
