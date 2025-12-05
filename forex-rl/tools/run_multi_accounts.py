#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple trader instances for many broker accounts")
    p.add_argument("--script", required=True, help="Path to Python trader script (e.g., forex-rl/online_trader.py)")
    p.add_argument("--accounts", required=True, help="Comma-separated broker account IDs (e.g., '1,2,3,4')")
    p.add_argument("--base-args", default="", help="Additional CLI args passed to each process (quoted)")
    p.add_argument("--logs-dir", default="logs/multi", help="Directory to write per-account stdout logs")
    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use")
    return p.parse_args()


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    script = Path(args.script).resolve()
    if not script.exists():
        raise SystemExit(f"script not found: {script}")
    accounts = [int(x) for x in args.accounts.split(',') if x.strip()]
    ensure_dir(args.logs_dir)

    procs: List[subprocess.Popen] = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for acct in accounts:
        log_path = Path(args.logs_dir) / f"acct{acct}_{timestamp}.log"
        cmd = [args.python_bin, str(script), "--broker", "ipc", "--broker-account-id", str(acct)]
        if args.base_args:
            cmd.extend(shlex.split(args.base_args))
        stdout_f = open(log_path, "w", buffering=1)
        env = os.environ.copy()
        p = subprocess.Popen(cmd, stdout=stdout_f, stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"started acct={acct} pid={p.pid} log={log_path}")

    # Wait for all
    rc = 0
    for p in procs:
        try:
            rc |= p.wait()
        except KeyboardInterrupt:
            pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
