#!/usr/bin/env python3
"""Quick utility to inspect min/max timestamps for local OHLC CSVs.

Helps choose sensible train/val ranges that actually overlap on-disk data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect OHLC CSV coverage for instruments/granularities")
    p.add_argument("--raw-dir", default="continuous-trader/data", help="Directory with raw OHLC CSVs")
    p.add_argument("--instruments", default="USD_PLN", help="Comma-separated instruments")
    p.add_argument("--granularities", default="M5,D", help="Comma-separated granularities, e.g. 'M1,M5,H1,D'")
    return p.parse_args()


def inspect_coverage(raw_dir: Path, instrument: str, gran: str) -> Dict[str, str | None]:
    safe = instrument.replace("/", "_")
    fname = f"{safe}_{gran.upper()}.csv"
    path = raw_dir / fname
    if not path.exists():
        return {"file": str(path), "exists": False, "start": None, "end": None, "rows": None}
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df.empty:
                return {"file": str(path), "exists": True, "start": None, "end": None, "rows": 0}
            start = df["timestamp"].iloc[0]
            end = df["timestamp"].iloc[-1]
            return {
                "file": str(path),
                "exists": True,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "rows": int(df.shape[0]),
            }
        else:
            # No timestamp column; just report row count
            return {"file": str(path), "exists": True, "start": None, "end": None, "rows": int(df.shape[0])}
    except Exception as exc:
        return {"file": str(path), "exists": False, "start": None, "end": None, "rows": None, "error": str(exc)}


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    grans = [tok.strip().upper() for tok in args.granularities.split(",") if tok.strip()]

    rows: List[Dict[str, str | None]] = []
    for inst in instruments:
        for gran in grans:
            rows.append(inspect_coverage(raw_dir, inst, gran))

    for row in rows:
        print(row)


if __name__ == "__main__":  # pragma: no cover
    main()
