#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

# Repo paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore
import instruments as ct_instruments  # type: ignore

DEFAULT_OUT_DIR = os.path.join(FX_ROOT, "continuous-trader", "data")


def fetch_save(instrument: str, granularity: str, start: str, end: Optional[str], access_token: Optional[str], environment: str, out_dir: str) -> str:
    adapter = OandaRestCandlesAdapter(instrument=instrument, granularity=granularity, environment=environment, access_token=access_token)
    rows = list(adapter.fetch_range(from_time=f"{start}T00:00:00Z", to_time=(f"{end}T23:59:59Z" if end else None), batch=5000))
    if not rows:
        raise RuntimeError(f"No {granularity} candles fetched for {instrument}")
    df = pd.DataFrame(rows)
    safe = instrument.replace('/', '_')
    path = os.path.join(out_dir, f"{safe}_{granularity}.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest OANDA candles to CSV in continuous-trader/data")
    p.add_argument("--instruments", default="", help="Comma-separated list or leave empty to use 68-instrument CSV")
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--environment", choices=["practice","live"], default="practice")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs if present")
    p.add_argument("--granularities", default="M5", help="Comma-separated granularities to fetch (e.g., M5,H1,D)")
    args = p.parse_args()

    # Determine instruments list
    if args.instruments.strip():
        instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]
    else:
        try:
            instruments = ct_instruments.load_68()
        except Exception:
            instruments = ["EUR_USD"]

    grans = [g.strip().upper() for g in (args.granularities or "M5").split(',') if g.strip()]
    print({"status": "start", "instruments": instruments, "count": len(instruments), "granularities": grans, "out_dir": args.out_dir})
    out_paths: List[str] = []
    for inst in instruments:
        for gran in grans:
            try:
                safe = inst.replace('/', '_')
                target_path = os.path.join(args.out_dir, f"{safe}_{gran}.csv")
                if (not args.overwrite) and os.path.exists(target_path):
                    print({"status": "skip_exists", "instrument": inst, "granularity": gran, "path": target_path})
                    out_paths.append(target_path)
                    continue
                print({"status": "fetch", "instrument": inst, "granularity": gran})
                out = fetch_save(inst, gran, args.start, args.end, args.access_token, args.environment, args.out_dir)
                # Quick row count
                try:
                    nrows = sum(1 for _ in open(out, 'r', encoding='utf-8')) - 1
                except Exception:
                    nrows = None
                print({"status": "saved", "instrument": inst, "granularity": gran, "path": out, "rows": nrows})
                out_paths.append(out)
            except Exception as exc:
                print({"status": "error", "instrument": inst, "granularity": gran, "error": str(exc)})
    print({"done": True, "count": len(out_paths)})


if __name__ == "__main__":
    main()
