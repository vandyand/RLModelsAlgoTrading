import argparse
import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions


def ensure_account_tables(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL,
            account_id TEXT NOT NULL,
            currency TEXT,
            balance REAL,
            NAV REAL,
            unrealizedPL REAL,
            realizedPL REAL,
            marginAvailable REAL,
            marginUsed REAL,
            marginRate REAL,
            openTradeCount INTEGER,
            openPositionCount INTEGER,
            pendingOrderCount INTEGER,
            raw TEXT,
            raw_compressed BLOB
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_account_snapshots_time
        ON account_snapshots(account_id, time);
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL,
            account_id TEXT NOT NULL,
            instrument TEXT NOT NULL,
            long_units REAL,
            long_avg_price REAL,
            long_unrealized_pl REAL,
            short_units REAL,
            short_avg_price REAL,
            short_unrealized_pl REAL,
            net_units REAL,
            raw TEXT,
            raw_compressed BLOB
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_position_snapshots_time
        ON position_snapshots(account_id, time, instrument);
        """
    )

    # Best-effort migrations (ignore failures if columns exist)
    for ddl in [
        "ALTER TABLE account_snapshots ADD COLUMN raw TEXT",
        "ALTER TABLE account_snapshots ADD COLUMN raw_compressed BLOB",
        "ALTER TABLE position_snapshots ADD COLUMN raw TEXT",
        "ALTER TABLE position_snapshots ADD COLUMN raw_compressed BLOB",
    ]:
        try:
            cursor.execute(ddl)
        except Exception:
            pass

    connection.commit()


def ensure_sqlite_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    ensure_account_tables(connection)
    return connection


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 with 'Z' suffix and microseconds.

    Example: 2025-08-12T09:07:52.478062Z
    """
    now = datetime.now(timezone.utc)
    return now.replace(tzinfo=None).isoformat(timespec="microseconds") + "Z"


def fetch_account_summary(api: API, account_id: str) -> Dict[str, Any]:
    req = accounts.AccountSummary(accountID=account_id)
    response = api.request(req)
    return response.get("account", {})


def fetch_open_positions(api: API, account_id: str) -> List[Dict[str, Any]]:
    req = positions.OpenPositions(accountID=account_id)
    response = api.request(req)
    return response.get("positions", [])


def parse_float(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def store_account_snapshot(cursor: sqlite3.Cursor, account_id: str, summary: Dict[str, Any], snapshot_time: str, compress_raw: bool) -> None:
    raw_json = json.dumps(summary, separators=(",", ":"))
    raw_blob = zlib.compress(raw_json.encode("utf-8"), level=6) if compress_raw else None

    cursor.execute(
        """
        INSERT INTO account_snapshots (
            time, account_id, currency, balance, NAV, unrealizedPL, realizedPL,
            marginAvailable, marginUsed, marginRate,
            openTradeCount, openPositionCount, pendingOrderCount,
            raw, raw_compressed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_time,
            account_id,
            summary.get("currency"),
            parse_float(summary.get("balance")),
            parse_float(summary.get("NAV")),
            parse_float(summary.get("unrealizedPL")),
            parse_float(summary.get("pl")) if summary.get("pl") is not None else parse_float(summary.get("realizedPL")),
            parse_float(summary.get("marginAvailable")),
            parse_float(summary.get("marginUsed")),
            parse_float(summary.get("marginRate")),
            summary.get("openTradeCount"),
            summary.get("openPositionCount"),
            summary.get("pendingOrderCount"),
            None if compress_raw else raw_json,
            raw_blob,
        ),
    )


def store_positions_snapshot(cursor: sqlite3.Cursor, account_id: str, positions_list: List[Dict[str, Any]], snapshot_time: str, compress_raw: bool) -> None:
    for pos in positions_list:
        instrument = pos.get("instrument")
        long_side = pos.get("long", {}) or {}
        short_side = pos.get("short", {}) or {}

        long_units = parse_float(long_side.get("units"))
        long_avg_price = parse_float(long_side.get("averagePrice"))
        long_unrealized_pl = parse_float(long_side.get("unrealizedPL"))

        short_units = parse_float(short_side.get("units"))
        short_avg_price = parse_float(short_side.get("averagePrice"))
        short_unrealized_pl = parse_float(short_side.get("unrealizedPL"))

        net_units = (
            (long_units or 0.0) + (short_units or 0.0)
            if (long_units is not None or short_units is not None)
            else None
        )

        raw_json = json.dumps(pos, separators=(",", ":"))
        raw_blob = zlib.compress(raw_json.encode("utf-8"), level=6) if compress_raw else None

        cursor.execute(
            """
            INSERT INTO position_snapshots (
                time, account_id, instrument,
                long_units, long_avg_price, long_unrealized_pl,
                short_units, short_avg_price, short_unrealized_pl,
                net_units,
                raw, raw_compressed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_time,
                account_id,
                instrument,
                long_units,
                long_avg_price,
                long_unrealized_pl,
                short_units,
                short_avg_price,
                short_unrealized_pl,
                net_units,
                None if compress_raw else raw_json,
                raw_blob,
            ),
        )


def run_once(api: API, account_id: str, connection: sqlite3.Connection, compress_raw: bool) -> Tuple[int, int]:
    snapshot_time = _now_iso()
    summary = fetch_account_summary(api, account_id)
    open_positions = fetch_open_positions(api, account_id)

    cursor = connection.cursor()
    store_account_snapshot(cursor, account_id, summary, snapshot_time, compress_raw)
    store_positions_snapshot(cursor, account_id, open_positions, snapshot_time, compress_raw)
    connection.commit()

    return 1, len(open_positions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OANDA account summary and open positions into SQLite")
    parser.add_argument("--once", action="store_true", help="Fetch once and exit (default: continuous)")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval seconds for continuous mode")
    args = parser.parse_args()

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError(
            "Missing OANDA credentials. Ensure OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY are set."
        )

    api = API(access_token=access_token, environment="practice")

    default_db_path = os.path.join(os.path.dirname(__file__), "forex_ticks.sqlite")
    db_path = os.environ.get("FOREX_DB_PATH", default_db_path)
    connection = ensure_sqlite_db(db_path)

    compress_raw = os.environ.get("FOREX_COMPRESS_RAW", "1") != "0"

    if args.once:
        acc_count, pos_count = run_once(api, account_id, connection, compress_raw)
        print(json.dumps({
            "snapshots_written": acc_count,
            "positions_written": pos_count,
            "db": db_path,
        }))
        return

    # Continuous mode
    import time
    while True:
        try:
            acc_count, pos_count = run_once(api, account_id, connection, compress_raw)
            print(json.dumps({
                "time": _now_iso(),
                "snapshots_written": acc_count,
                "positions_written": pos_count,
            }))
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
        finally:
            time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
