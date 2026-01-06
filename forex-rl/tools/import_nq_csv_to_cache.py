#!/usr/bin/env python3
"""
Import NQ futures candles from local CSV/ZIP into the candle cache SQLite DB.

This lets the existing candle_cache_service serve NQ data via the same
OANDA-compatible HTTP interface the TCN WFO stack already uses.

Assumptions about the source data:
- Files live under forex-rl/nq-historical
- Filenames: nq-<gran>_bk.zip, e.g.:
    nq-1m_bk.zip, nq-5m_bk.zip, nq-15m_bk.zip, nq-30m_bk.zip,
    nq-1h_bk.zip, nq-4h_bk.zip, nq-1d_bk.zip, nq-1w_bk.zip, nq-1mo_bk.zip
- Inside each ZIP, a single CSV named nq-<gran>_bk.csv with rows:
    DATE;TIME;OPEN;HIGH;LOW;CLOSE;VOLUME
  Example:
    11/12/2008;01:35;1377.49334;1377.49334;1377.49334;1377.49334;1

Usage (from forex-rl root):

  # Dry-run (print what would be imported)
  python3 tools/import_nq_csv_to_cache.py --instrument NQ_USD --dry-run

  # Actual import for all supported granularities into the existing DB
  python3 tools/import_nq_csv_to_cache.py --instrument NQ_USD

After running this once, the candle_cache_service can serve NQ candles:
- Query URL: http://127.0.0.1:9100/v3/instruments/NQ_USD/candles?...,
  and TCNActionAdapter can use NQ via --instrument NQ_USD.
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import zipfile

import candle_cache_service as ccs  # type: ignore


# Map filename granularity tokens -> OANDA-style granularity strings
_GRAN_MAP: Dict[str, str] = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
    "1w": "W",
    # "1mo" has no direct OANDA analog; we currently skip it.
}


def _parse_row(date_str: str, time_str: str, o: str, h: str, l: str, c: str, v: str) -> Dict[str, Any]:
    """Parse one CSV row into an OANDA-style candle dict."""
    # Source format: DD/MM/YYYY;HH:MM in CME local time (America/Chicago).
    # Convert to UTC before storing so downstream adapters see UTC like OANDA.
    dt_local = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    try:
        # Python 3.9+ zoneinfo; avoid extra dependencies.
        from zoneinfo import ZoneInfo

        tz_chi = ZoneInfo("America/Chicago")
        dt_local = dt_local.replace(tzinfo=tz_chi)
        dt_utc = dt_local.astimezone(timezone.utc)
    except Exception:
        # Fallback: assume already UTC if zoneinfo isn't available for some reason.
        dt_utc = dt_local.replace(tzinfo=timezone.utc)

    ts = dt_utc.isoformat().replace("+00:00", "Z")

    def _to_str(x: str) -> str:
        return x.strip()

    vol: int
    try:
        vol = int(float(v.strip())) if v.strip() else 0
    except Exception:
        vol = 0

    return {
        "time": ts,
        "complete": True,
        "mid": {
            "o": _to_str(o),
            "h": _to_str(h),
            "l": _to_str(l),
            "c": _to_str(c),
        },
        "volume": vol,
    }


def _iter_zip_csv(zip_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Yield (granularity, candles) from one nq-*_bk.zip file.

    Returns:
        (granularity, candles)
      where granularity is an OANDA-style code (e.g. "M5", "H1").
    """
    name = zip_path.stem  # e.g. "nq-5m_bk"
    # Expect pattern nq-<token>_bk
    parts = name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Unrecognized NQ filename format: {zip_path.name}")
    token_with_suffix = parts[1]  # e.g. "5m_bk"
    token = token_with_suffix.split("_")[0]  # e.g. "5m"

    gran = _GRAN_MAP.get(token)
    if not gran:
        raise ValueError(f"Unsupported NQ granularity token '{token}' in {zip_path.name}")

    # Find the inner CSV
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV files found inside {zip_path}")
        if len(csv_names) > 1:
            # Pick the first deterministically
            csv_names.sort()
        inner_name = csv_names[0]

        candles: List[Dict[str, Any]] = []
        with zf.open(inner_name, "r") as f:
            wrapper = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.reader(wrapper, delimiter=";")
            for row in reader:
                # Expect 7 columns: date;time;o;h;l;c;v
                if len(row) < 7:
                    continue
                date_str, time_str, o, h, l, c, v = row[:7]
                try:
                    candle = _parse_row(date_str, time_str, o, h, l, c, v)
                except Exception:
                    continue
                candles.append(candle)

    return gran, candles


def main() -> None:
    ap = argparse.ArgumentParser(description="Import NQ CSV candles into candle_cache_service DB")
    ap.add_argument(
        "--root",
        type=str,
        default="nq-historical",
        help="Root directory with nq-*_bk.zip files (relative to forex-rl).",
    )
    ap.add_argument(
        "--instrument",
        type=str,
        default="NQ",
        help="Instrument name to store in cache (e.g. NQ).",
    )
    ap.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Cache environment tag (default: candle_cache_service.OANDA_ENV).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarize but do not write to the DB.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        # Assume relative to forex-rl root when run as: python3 tools/import_nq_csv_to_cache.py
        root = Path(__file__).resolve().parent.parent / root

    if not root.exists():
        raise SystemExit(f"Root directory '{root}' does not exist")

    env = str(args.environment or ccs.OANDA_ENV)
    instrument = str(args.instrument).upper()

    zip_files = sorted(root.glob("nq-*_bk.zip"))
    if not zip_files:
        raise SystemExit(f"No nq-*_bk.zip files found under {root}")

    print(f"Importing NQ candles into cache DB={ccs.DB_PATH} env={env} instrument={instrument}")
    print(f"Found {len(zip_files)} ZIP files:")
    for z in zip_files:
        print(f"  - {z.name}")

    conn = ccs._connect_db()

    for zip_path in zip_files:
        try:
            gran, candles = _iter_zip_csv(zip_path)
        except Exception as e:
            print(f"[WARN] Skipping {zip_path.name}: {e}")
            continue

        print(f"{zip_path.name}: gran={gran}, candles={len(candles)}")
        if args.dry_run:
            continue

        # Chunk writes to avoid huge transactions
        batch_size = 5000
        for i in range(0, len(candles), batch_size):
            batch = candles[i : i + batch_size]
            ccs._upsert_candles(conn, env, instrument, gran, batch)
        print(f"  -> wrote {len(candles)} candles to DB")

    conn.close()
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()

