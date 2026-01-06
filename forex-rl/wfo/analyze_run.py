from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_windows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_windows(rows: List[Dict[str, Any]]) -> None:
    metrics = []
    for r in rows:
        # Newer WFO writes metrics at top level; fall back to nested "metrics"
        m = r.get("metrics") or r
        win = r.get("win")
        sharpe = float(m.get("sharpe") or 0.0)
        cum_return = float(m.get("cum_return") or 0.0)
        max_dd = float(m.get("max_dd") or 0.0)
        trades = float(m.get("trades") or 0.0)
        metrics.append((win, sharpe, cum_return, max_dd, trades))

    if not metrics:
        print("No metrics found in windows.jsonl")
        return

    print("win\tsharpe\tcum_ret\tmax_dd\ttrades")
    for win, sharpe, cum_ret, max_dd, trades in metrics:
        print(f"{win}\t{sharpe:.3f}\t{cum_ret:.4f}\t{max_dd:.4f}\t{trades:.0f}")

    arr = np.array([[s, c, d, t] for _, s, c, d, t in metrics], dtype=float)
    sharpe_mean = float(np.mean(arr[:, 0]))
    sharpe_med = float(np.median(arr[:, 0]))
    cum_mean = float(np.mean(arr[:, 1]))
    dd_mean = float(np.mean(arr[:, 2]))

    print("\nSummary across windows:")
    print(f"  avg_sharpe\t= {sharpe_mean:.3f}")
    print(f"  med_sharpe\t= {sharpe_med:.3f}")
    print(f"  avg_cum_ret\t= {cum_mean:.4f}")
    print(f"  avg_max_dd\t= {dd_mean:.4f}")



def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize WFO run windows.jsonl")
    ap.add_argument("run_dir", help="Path to a wfo run directory")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    windows_path = run_dir / "windows.jsonl"
    if not windows_path.exists():
        raise SystemExit(f"windows.jsonl not found under {run_dir}")

    rows = load_windows(windows_path)
    summarize_windows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
