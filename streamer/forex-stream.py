import json
import os
import sqlite3
import zlib
from typing import List, Optional, Tuple

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing


def load_fx_instruments_from_file() -> List[str]:
    """Load instrument list from streamer/data/fx_pairs.json with fallback."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, "fx_pairs.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instruments: List[str] = data.get("instruments") or []
        if not instruments and data.get("instruments_csv"):
            instruments = [s.strip() for s in data["instruments_csv"].split(",") if s.strip()]
        if instruments:
            return instruments
    except Exception:
        pass
    # Fallback hardcoded list
    return [
        "EUR_USD",
        "USD_JPY",
        "GBP_USD",
        "USD_CHF",
        "AUD_USD",
        "USD_CAD",
        "NZD_USD",
        "EUR_JPY",
        "GBP_JPY",
        "EUR_GBP",
        "EUR_CHF",
        "EUR_AUD",
        "EUR_CAD",
        "AUD_JPY",
        "CHF_JPY",
        "CAD_JPY",
        "NZD_JPY",
        "GBP_CHF",
        "GBP_CAD",
        "AUD_CAD",
        "AUD_NZD",
    ]


def ensure_sqlite_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    # Improve durability and write throughput for streaming inserts
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL,
            instrument TEXT NOT NULL,
            bid REAL,
            ask REAL,
            bid_liquidity INTEGER,
            ask_liquidity INTEGER,
            closeout_bid REAL,
            closeout_ask REAL,
            tradeable INTEGER,
            status TEXT,
            raw TEXT,
            raw_compressed BLOB
        );
        """
    )
    # Table for full book levels at each tick (all bid/ask entries with level index)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tick_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_id INTEGER NOT NULL,
            side TEXT NOT NULL,              -- 'bid' or 'ask'
            level INTEGER NOT NULL,          -- 0-based order in the stream payload
            price REAL,
            liquidity INTEGER,
            FOREIGN KEY (tick_id) REFERENCES ticks(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ticks_instrument_time
        ON ticks(instrument, time);
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tick_levels_tick_id
        ON tick_levels(tick_id);
        """
    )

    # In-place migration for older DBs missing new columns
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN closeout_bid REAL")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN closeout_ask REAL")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN tradeable INTEGER")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN status TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN raw TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE ticks ADD COLUMN raw_compressed BLOB")
    except Exception:
        pass
    connection.commit()
    return connection


def extract_bid_ask(tick: dict) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    bids = tick.get("bids") or []
    asks = tick.get("asks") or []
    bid_price = None
    ask_price = None
    bid_liquidity = None
    ask_liquidity = None
    if bids:
        try:
            bid_price = float(bids[0].get("price"))
        except Exception:
            bid_price = None
        try:
            bid_liquidity = int(bids[0].get("liquidity")) if bids[0].get("liquidity") is not None else None
        except Exception:
            bid_liquidity = None
    if asks:
        try:
            ask_price = float(asks[0].get("price"))
        except Exception:
            ask_price = None
        try:
            ask_liquidity = int(asks[0].get("liquidity")) if asks[0].get("liquidity") is not None else None
        except Exception:
            ask_liquidity = None
    return bid_price, ask_price, bid_liquidity, ask_liquidity


def main() -> None:
    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError(
            "Missing OANDA credentials. Ensure OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY are set."
        )

    api = API(access_token=access_token, environment="practice")  # Or "live"

    # Database path can be overridden via FOREX_DB_PATH env var
    default_db_path = os.path.join(os.path.dirname(__file__), "forex_ticks.sqlite")
    db_path = os.environ.get("FOREX_DB_PATH", default_db_path)
    conn = ensure_sqlite_db(db_path)
    cur = conn.cursor()

    instruments_list = load_fx_instruments_from_file()
    params = {"instruments": ",".join(instruments_list)}
    stream_request = pricing.PricingStream(accountID=account_id, params=params)

    try:
        for tick in api.request(stream_request):
            # Skip non-price messages (e.g., HEARTBEAT)
            if tick.get("type") != "PRICE":
                continue

            instrument = tick.get("instrument")
            time_iso = tick.get("time")
            bid, ask, bid_liq, ask_liq = extract_bid_ask(tick)
            closeout_bid = float(tick.get("closeoutBid")) if tick.get("closeoutBid") is not None else None
            closeout_ask = float(tick.get("closeoutAsk")) if tick.get("closeoutAsk") is not None else None
            tradeable = 1 if tick.get("tradeable") is True else (0 if tick.get("tradeable") is False else None)
            status = tick.get("status")

            # Prepare raw payload storage (compressed by default)
            raw_json = json.dumps(tick, separators=(",", ":"))
            compress_raw = os.environ.get("FOREX_COMPRESS_RAW", "1") != "0"
            raw_blob = zlib.compress(raw_json.encode("utf-8"), level=6) if compress_raw else None

            # Insert top-of-book and summary fields
            cur.execute(
                """
                INSERT INTO ticks (
                    time, instrument, bid, ask, bid_liquidity, ask_liquidity,
                    closeout_bid, closeout_ask, tradeable, status, raw, raw_compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time_iso,
                    instrument,
                    bid,
                    ask,
                    bid_liq,
                    ask_liq,
                    closeout_bid,
                    closeout_ask,
                    tradeable,
                    status,
                    None if compress_raw else raw_json,
                    raw_blob,
                ),
            )
            tick_id = cur.lastrowid

            # Insert full depth levels for bids and asks with level index
            for level_index, level in enumerate(tick.get("bids") or []):
                level_price = float(level.get("price")) if level.get("price") is not None else None
                level_liq = int(level.get("liquidity")) if level.get("liquidity") is not None else None
                cur.execute(
                    """
                    INSERT INTO tick_levels (tick_id, side, level, price, liquidity)
                    VALUES (?, 'bid', ?, ?, ?)
                    """,
                    (tick_id, level_index, level_price, level_liq),
                )
            for level_index, level in enumerate(tick.get("asks") or []):
                level_price = float(level.get("price")) if level.get("price") is not None else None
                level_liq = int(level.get("liquidity")) if level.get("liquidity") is not None else None
                cur.execute(
                    """
                    INSERT INTO tick_levels (tick_id, side, level, price, liquidity)
                    VALUES (?, 'ask', ?, ?, ?)
                    """,
                    (tick_id, level_index, level_price, level_liq),
                )

            conn.commit()

            # Light console output to indicate progress
            print(json.dumps({
                "instrument": instrument,
                "time": time_iso,
                "bid": bid,
                "ask": ask,
                "bids": len(tick.get("bids") or []),
                "asks": len(tick.get("asks") or []),
            }))
    except Exception as e:
        print("Error:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

