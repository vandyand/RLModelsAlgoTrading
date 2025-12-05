#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any

import pandas as pd

# Make repo modules importable (oanda_rest_adapter)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore


@dataclass
class Args:
    instrument: str
    granularity: str
    environment: str
    access_token: Optional[str]
    start: Optional[str]
    end: Optional[str]
    out_csv: str
    resume: bool
    batch: int
    max_candles: Optional[int]


def parse_cli() -> Args:
    p = argparse.ArgumentParser(description="Download OANDA REST candles to CSV (with resume)")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--granularity", default="M5",
                   help="OANDA granularity (e.g., S5,S10,S30,M1,M5,M15,M30,H1,H4,D)")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))

    p.add_argument("--start", help="ISO timestamp or YYYY-MM-DD (UTC). If omitted and --resume with existing file, continues after last row.")
    p.add_argument("--end", help="ISO timestamp or YYYY-MM-DD (UTC). Optional.")

    p.add_argument("--out-csv",
                   help="Output CSV path (default: forex-rl/ga-ness/data/{instrument}_{granularity}.csv)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing CSV last timestamp if present")
    p.add_argument("--batch", type=int, default=5000, help="Page size per request (<= 5000)")
    p.add_argument("--max-candles", type=int, help="Stop after N candles (debug)")

    a = p.parse_args()

    out_csv = a.out_csv or os.path.join(
        REPO_ROOT, "forex-rl", "ga-ness", "data", f"{a.instrument}_{a.granularity}.csv"
    )

    return Args(
        instrument=a.instrument,
        granularity=a.granularity,
        environment=a.environment,
        access_token=a.access_token,
        start=a.start,
        end=a.end,
        out_csv=out_csv,
        resume=bool(a.resume),
        batch=max(1, min(int(a.batch), 5000)),
        max_candles=a.max_candles if a.max_candles is not None and a.max_candles > 0 else None,
    )


def _to_iso_utc(s: str) -> str:
    """Coerce strings like YYYY-MM-DD or arbitrary parseable time to RFC3339 Z.
    Accepts already-ISO strings and returns a Z-suffixed ISO8601 string.
    """
    if s is None or s == "":
        raise ValueError("start/end time string is empty")
    try:
        # Fast-path: if already ends with 'Z', trust it
        if s.endswith("Z"):
            return s
    except Exception:
        pass
    ts = pd.to_datetime(s, utc=True, errors="raise")
    # Normalize naive date to start-of-day if only a date was provided
    if len(str(s)) == 10 and str(s)[4] == '-' and str(s)[7] == '-':
        ts = ts.normalize()
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_existing(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # Ensure canonical columns
        want = ["open", "high", "low", "close", "volume"]
        for w in want:
            if w not in df.columns:
                df[w] = float("nan")
        return df[want].astype(float)
    except Exception:
        return None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def download_and_save(args: Args) -> None:
    if args.access_token is None or len(args.access_token.strip()) == 0:
        raise RuntimeError("Access token required for OANDA REST (set OANDA_DEMO_KEY or pass --access-token)")

    existing = _read_existing(args.out_csv) if args.resume else None
    start_iso: Optional[str] = _to_iso_utc(args.start) if args.start else None
    end_iso: Optional[str] = _to_iso_utc(args.end) if args.end else None

    if existing is not None and args.resume:
        if start_iso is None:
            # Continue from the last timestamp + 1 second
            last_ts = existing.index.max()
            start_iso = (pd.to_datetime(last_ts, utc=True) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # else: user-specified start overrides resume point

    if start_iso is None:
        # Default to yesterday 00:00Z
        utc_now = pd.Timestamp.utcnow()
        # Normalize tz handling across pandas versions
        if getattr(utc_now, "tz", None) is None:
            utc_now = utc_now.tz_localize("UTC")
        else:
            utc_now = utc_now.tz_convert("UTC")
        start_iso = (utc_now.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(json.dumps({
        "event": "begin",
        "instrument": args.instrument,
        "granularity": args.granularity,
        "start": start_iso,
        "end": end_iso,
        "out_csv": args.out_csv,
        "resume": bool(existing is not None),
    }), flush=True)

    adapter = OandaRestCandlesAdapter(
        instrument=args.instrument,
        granularity=args.granularity,
        environment=args.environment,
        access_token=args.access_token,
    )

    rows: List[Dict[str, Any]] = []
    yielded = 0
    for c in adapter.fetch_range(from_time=start_iso, to_time=end_iso, batch=args.batch, max_candles=args.max_candles):
        rows.append(c)
        yielded += 1
        if yielded % 1000 == 0:
            print(json.dumps({"event": "progress", "candles": yielded, "last": c.get("timestamp")}), flush=True)

    if len(rows) == 0:
        print(json.dumps({"event": "no_data", "start": start_iso, "end": end_iso or None}), flush=True)
        return

    df_new = pd.DataFrame(rows)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)
    df_new = df_new.set_index("timestamp").sort_index()
    df_new = df_new[["open", "high", "low", "close", "volume"]].astype(float)

    if existing is not None:
        df_all = pd.concat([existing, df_new])
        df_all = df_all[~df_all.index.duplicated(keep="last")]  # keep latest
    else:
        df_all = df_new

    _ensure_parent_dir(args.out_csv)
    tmp = args.out_csv + ".tmp"
    df_all.to_csv(tmp, index=True)
    os.replace(tmp, args.out_csv)

    print(json.dumps({
        "event": "saved",
        "rows_total": int(df_all.shape[0]),
        "rows_new": int(df_new.shape[0]),
        "first": df_new.index.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last": df_new.index.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_csv": args.out_csv,
    }), flush=True)


if __name__ == "__main__":
    args = parse_cli()
    download_and_save(args)
