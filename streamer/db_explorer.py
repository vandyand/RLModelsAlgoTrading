import argparse
import csv
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def get_default_db_path() -> str:
    return os.environ.get(
        "FOREX_DB_PATH",
        os.path.join(os.path.dirname(__file__), "forex_ticks.sqlite"),
    )


def open_connection(db_path: str, read_only: bool = True) -> sqlite3.Connection:
    if read_only:
        # Use read-only URI mode to avoid creating a new DB accidentally
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(db_path)


def rows_to_dicts(cursor: sqlite3.Cursor, rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def list_tables(connection: sqlite3.Connection) -> List[str]:
    with closing(connection.cursor()) as cursor:
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        return [name for (name,) in cursor.fetchall()]


def get_schema(connection: sqlite3.Connection, table: str) -> str:
    with closing(connection.cursor()) as cursor:
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else ""


def get_count(connection: sqlite3.Connection, table: str) -> int:
    with closing(connection.cursor()) as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        (count,) = cursor.fetchone()
        return int(count)


def get_latest_time(connection: sqlite3.Connection, table: str, time_column: str = "time") -> Optional[str]:
    with closing(connection.cursor()) as cursor:
        try:
            cursor.execute(f"SELECT MAX({time_column}) FROM {table}")
            (max_time,) = cursor.fetchone()
            return max_time
        except sqlite3.OperationalError:
            return None


def select_head_or_tail(
    connection: sqlite3.Connection,
    table: str,
    limit: int,
    order_column: Optional[str] = None,
    tail: bool = False,
) -> List[Dict[str, Any]]:
    with closing(connection.cursor()) as cursor:
        # Determine sensible order column
        candidate_columns = [c for c in [order_column, "time", "id"] if c]
        if order_column is None:
            candidate_columns = ["time", "id"]
        chosen = None
        for col in candidate_columns:
            try:
                cursor.execute(f"SELECT {col} FROM {table} LIMIT 0")
                chosen = col
                break
            except sqlite3.OperationalError:
                continue
        order_clause = f" ORDER BY {chosen} {'DESC' if tail else 'ASC'}" if chosen else ""
        cursor.execute(f"SELECT * FROM {table}{order_clause} LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return rows_to_dicts(cursor, rows)


def format_output(data: Any, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)
    if fmt == "csv":
        if isinstance(data, list) and data:
            columns = sorted({k for row in data for k in row.keys()})
            output_lines: List[str] = []
            writer = csv.DictWriter(
                output_lines := [], fieldnames=columns, extrasaction="ignore"
            )
            # csv module needs a file-like; emulate into list
            class _ListWriter:
                def __init__(self, sink: List[str]):
                    self.sink = sink

                def write(self, s: str) -> None:
                    self.sink.append(s)

            writer = csv.DictWriter(_ListWriter(output_lines), fieldnames=columns)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k) for k in columns})
            return "".join(output_lines)
        # Otherwise, just dump as JSON
        return json.dumps(data, default=str)
    return str(data)


def cmd_tables(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    tables = list_tables(conn)
    print(format_output(tables, args.format))


def cmd_schema(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    ddl = get_schema(conn, args.table)
    print(ddl)


def cmd_counts(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    if args.table:
        result = {args.table: get_count(conn, args.table)}
    else:
        result = {t: get_count(conn, t) for t in list_tables(conn)}
    print(format_output(result, args.format))


def cmd_head(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    data = select_head_or_tail(conn, args.table, args.limit, args.order, tail=False)
    print(format_output(data, args.format))


def cmd_tail(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    data = select_head_or_tail(conn, args.table, args.limit, args.order, tail=True)
    print(format_output(data, args.format))


def cmd_latest_ticks(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        sql = "SELECT * FROM ticks"
        params: List[Any] = []
        where: List[str] = []
        if args.instrument:
            where.append("instrument = ?")
            params.append(args.instrument)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY time DESC LIMIT ?"
        params.append(args.limit)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        print(format_output(rows_to_dicts(cursor, rows), args.format))


def cmd_query_ticks(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        sql = "SELECT time, instrument, bid, ask, bid_liquidity, ask_liquidity FROM ticks"
        params: List[Any] = []
        where: List[str] = []
        if args.instrument:
            where.append("instrument = ?")
            params.append(args.instrument)
        if args.start:
            where.append("time >= ?")
            params.append(args.start)
        if args.end:
            where.append("time <= ?")
            params.append(args.end)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY time"
        if args.limit:
            sql += " LIMIT ?"
            params.append(args.limit)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        print(format_output(rows_to_dicts(cursor, rows), args.format))


def cmd_tick_levels(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        tick_id = args.tick_id
        if tick_id is None:
            # Resolve by instrument + time exact match
            cursor.execute(
                "SELECT id FROM ticks WHERE instrument = ? AND time = ? LIMIT 1",
                (args.instrument, args.time),
            )
            row = cursor.fetchone()
            if not row:
                print(json.dumps({"error": "Tick not found for instrument+time"}))
                return
            tick_id = row[0]
        cursor.execute(
            "SELECT * FROM tick_levels WHERE tick_id = ? ORDER BY side, level",
            (tick_id,),
        )
        rows = cursor.fetchall()
        print(format_output(rows_to_dicts(cursor, rows), args.format))


def cmd_latest_positions(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        # Find latest snapshot time
        cursor.execute(
            "SELECT MAX(time) FROM position_snapshots WHERE account_id = ?",
            (args.account_id,),
        )
        (latest_time,) = cursor.fetchone() or (None,)
        if latest_time is None:
            print(format_output([], args.format))
            return
        sql = "SELECT * FROM position_snapshots WHERE account_id = ? AND time = ?"
        params: List[Any] = [args.account_id, latest_time]
        if args.instrument:
            sql += " AND instrument = ?"
            params.append(args.instrument)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        print(format_output(rows_to_dicts(cursor, rows), args.format))


def cmd_latest_account(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            "SELECT * FROM account_snapshots WHERE account_id = ? ORDER BY time DESC LIMIT ?",
            (args.account_id, args.limit),
        )
        rows = cursor.fetchall()
        print(format_output(rows_to_dicts(cursor, rows), args.format))


def cmd_export(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    with closing(conn.cursor()) as cursor:
        sql = f"SELECT * FROM {args.table}"
        params: List[Any] = []
        if args.where:
            sql += f" WHERE {args.where}"
        if args.limit:
            sql += " LIMIT ?"
            params.append(args.limit)
        cursor.execute(sql, tuple(params))
        rows = rows_to_dicts(cursor, cursor.fetchall())
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            if args.format == "csv" and rows:
                columns = sorted({k for r in rows for k in r.keys()})
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k) for k in columns})
            else:
                f.write(format_output(rows, args.format))
        print(json.dumps({"exported": len(rows), "file": args.out}))
    else:
        print(format_output(rows, args.format))


def cmd_info(args: argparse.Namespace) -> None:
    conn = open_connection(args.db, read_only=not args.rw)
    tables = [
        "ticks",
        "tick_levels",
        "account_snapshots",
        "position_snapshots",
    ]
    info: Dict[str, Any] = {}
    for t in tables:
        try:
            info[t] = {
                "count": get_count(conn, t),
                "latest_time": get_latest_time(conn, t) if t != "tick_levels" else None,
            }
        except Exception as exc:
            info[t] = {"error": str(exc)}
    print(format_output(info, args.format))


def cmd_help(args: argparse.Namespace) -> None:
    parser: Optional[argparse.ArgumentParser] = getattr(args, "_parser", None)
    subparsers = getattr(args, "_subparsers", None)
    if parser is None:
        print("Help unavailable: parser not attached")
        return
    if getattr(args, "command", None) and subparsers and hasattr(subparsers, "choices"):
        choices = subparsers.choices  # type: ignore[attr-defined]
        if args.command in choices:
            choices[args.command].print_help()
            return
        else:
            print(f"Unknown command '{args.command}'. Showing global help.\n")
    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    examples = (
        "\n"
        "Examples:\n"
        "  # Overview of dataset\n"
        "  streamer/db_explorer.py info\n\n"
        "  # List tables and show schema\n"
        "  streamer/db_explorer.py list-tables\n"
        "  streamer/db_explorer.py schema --table ticks\n\n"
        "  # Show latest 20 EUR/USD ticks\n"
        "  streamer/db_explorer.py latest-ticks --instrument EUR_USD --limit 20\n\n"
        "  # Query ticks over a time range\n"
        "  streamer/db_explorer.py query-ticks --instrument EUR_USD --start 2025-08-12T09:00:00Z --end 2025-08-12T09:05:00Z --limit 100\n\n"
        "  # Show order book levels for a specific tick\n"
        "  streamer/db_explorer.py tick-levels --tick-id 12345\n\n"
        "  # Latest positions/account snapshots\n"
        "  streamer/db_explorer.py latest-positions --account-id $OANDA_DEMO_ACCOUNT_ID\n"
        "  streamer/db_explorer.py latest-account --account-id $OANDA_DEMO_ACCOUNT_ID --limit 3\n\n"
        "  # Export rows to a file\n"
        "  streamer/db_explorer.py export --table ticks --where \"instrument='EUR_USD'\" --limit 1000 --out eurusd.json\n"
    )
    parser = argparse.ArgumentParser(
        description="SQLite database explorer for OANDA FX streaming data",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )
    parser.add_argument(
        "--db",
        default=get_default_db_path(),
        help="Path to SQLite DB (default: FOREX_DB_PATH or streamer/forex_ticks.sqlite)",
    )
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--rw", action="store_true", help="Open DB read-write (default read-only)")

    sub = parser.add_subparsers(dest="cmd", required=True)
    # Attach parser/subparsers so the help command can access them later
    parser.set_defaults(_parser=parser, _subparsers=sub)

    p = sub.add_parser("list-tables", help="List tables/views", formatter_class=argparse.RawTextHelpFormatter)
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("schema", help="Show CREATE TABLE for a table", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--table", required=True)
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("counts", help="Show row counts for all or one table", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--table")
    p.set_defaults(func=cmd_counts)

    p = sub.add_parser("head", help="Show first N rows of a table", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--table", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--order", help="Order by column (default: time or id)")
    p.set_defaults(func=cmd_head)

    p = sub.add_parser("tail", help="Show last N rows of a table", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--table", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--order", help="Order by column (default: time or id)")
    p.set_defaults(func=cmd_tail)

    p_latest_ticks_examples = (
        "Show newest ticks first; filter by instrument.\n\n"
        "Examples:\n"
        "  streamer/db_explorer.py latest-ticks\n"
        "  streamer/db_explorer.py latest-ticks --instrument EUR_USD --limit 50\n"
    )
    p = sub.add_parser("latest-ticks", help="Show latest ticks, optionally filtered by instrument", formatter_class=argparse.RawTextHelpFormatter, epilog=p_latest_ticks_examples)
    p.add_argument("--instrument")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_latest_ticks)

    p_query_ticks_examples = (
        "Query ticks by instrument and time range.\n\n"
        "Examples:\n"
        "  streamer/db_explorer.py query-ticks --instrument EUR_USD --start 2025-08-12T09:00:00Z --end 2025-08-12T10:00:00Z\n"
        "  streamer/db_explorer.py query-ticks --instrument GBP_USD --limit 1000\n"
    )
    p = sub.add_parser("query-ticks", help="Query ticks by instrument and time range", formatter_class=argparse.RawTextHelpFormatter, epilog=p_query_ticks_examples)
    p.add_argument("--instrument")
    p.add_argument("--start", help="ISO time start (inclusive)")
    p.add_argument("--end", help="ISO time end (inclusive)")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_query_ticks)

    p = sub.add_parser("tick-levels", help="Show order book levels for a tick", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tick-id", type=int)
    p.add_argument("--instrument")
    p.add_argument("--time", help="Exact ISO time to resolve tick when tick-id not provided")
    p.set_defaults(func=cmd_tick_levels)

    p_latest_positions_examples = (
        "Show most recent position snapshot for an account.\n\n"
        "Examples:\n"
        "  streamer/db_explorer.py latest-positions --account-id $OANDA_DEMO_ACCOUNT_ID\n"
        "  streamer/db_explorer.py latest-positions --account-id $OANDA_DEMO_ACCOUNT_ID --instrument EUR_USD\n"
    )
    p = sub.add_parser("latest-positions", help="Show the latest positions snapshot for an account", formatter_class=argparse.RawTextHelpFormatter, epilog=p_latest_positions_examples)
    p.add_argument("--account-id", required=True)
    p.add_argument("--instrument")
    p.set_defaults(func=cmd_latest_positions)

    p = sub.add_parser("latest-account", help="Show recent account snapshots for an account", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--account-id", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_latest_account)

    p_export_examples = (
        "Export rows from a table to a file or stdout.\n\n"
        "Examples:\n"
        "  streamer/db_explorer.py export --table ticks --where \"instrument='EUR_USD'\" --limit 100 > eurusd.json\n"
        "  streamer/db_explorer.py export --table position_snapshots --out positions.csv --format csv\n"
    )
    p = sub.add_parser("export", help="Export a table to stdout or a file", formatter_class=argparse.RawTextHelpFormatter, epilog=p_export_examples)
    p.add_argument("--table", required=True)
    p.add_argument("--where", help="Optional WHERE clause (without 'WHERE')")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", help="Output file path (stdout if omitted)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("info", help="Quick dataset overview (counts, latest times)", formatter_class=argparse.RawTextHelpFormatter)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("help", help="Show help for a command or global help")
    p.add_argument("command", nargs="?", help="Command to show help for")
    p.set_defaults(func=cmd_help)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
