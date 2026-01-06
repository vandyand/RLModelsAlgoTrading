from __future__ import annotations

"""
Compact reporter/visualizer for `tcn_policy_run` JSONL outputs.

This reads a per-window JSONL file produced by:

  python3 -m wfo.tcn_policy_run \
    --base-config wfo/your_policy.json \
    --out-jsonl wfo/runs/your_policy_windows.jsonl

Each line is expected to be a JSON object of the form:

  {
    "win": 1,
    "train": ["...","..."],
    "val": ["...","..."],
    "metrics_train": {...},
    "metrics_val": {...}
  }

The reporter then:
  - Computes equity over validation windows from cum_return.
  - Prints an ASCII equity chart (using the same helper as WFO).
  - Prints small ASCII charts for per-window metrics such as
    cum_return, sharpe, and time_in_mkt.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import plotext as plt  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _load_windows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            rows.append(rec)
    return rows


def _extract_metric_series(
    rows: List[Dict[str, Any]],
    key: str,
) -> List[float]:
    out: List[float] = []
    for r in rows:
        mv = r.get("metrics_val") or {}
        try:
            out.append(float(mv.get(key) or 0.0))
        except Exception:
            out.append(0.0)
    return out


def _print_series_chart(
    title: str,
    values: List[float],
) -> None:
    print(f"\n== {title} ==")
    if not values:
        print("(no data)")
        return
    if plt is None:
        # Fallback: simple textual summary if plotext is not installed.
        print("(plotext not installed; values:) ", values)
        return

    xs = list(range(1, len(values) + 1))
    plt.clear_figure()
    plt.clear_data()
    plt.title(title)
    plt.plot(xs, values, marker="dot")
    plt.xlabel("window")
    plt.show()


def summarize_and_plot(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"JSONL file not found: {path}")

    rows = _load_windows(path)
    if not rows:
        print("No windows found in JSONL file.")
        return

    n = len(rows)
    print(f"Loaded {n} windows from {path}")

    cumrets = _extract_metric_series(rows, "cum_return")
    sharpes = _extract_metric_series(rows, "sharpe")
    tims = _extract_metric_series(rows, "time_in_mkt")
    trades = _extract_metric_series(rows, "trades")

    # Equity over validation windows: product of (1 + cum_return_i)
    equity = []
    eq = 1.0
    for r in cumrets:
        eq *= (1.0 + r)
        equity.append(eq)

    # Summary stats
    arr_cum = np.asarray(cumrets, dtype=float)
    arr_sh = np.asarray(sharpes, dtype=float)
    arr_tim = np.asarray(tims, dtype=float)
    arr_tr = np.asarray(trades, dtype=float)

    print("\n=== Aggregate validation metrics over windows ===")
    print(f"Mean cum_return: {arr_cum.mean():.6f}")
    print(f"Total cum_return (stitched): {equity[-1] - 1.0:.6f}")
    print(f"Mean sharpe: {arr_sh.mean():.3f}")
    print(f"Mean time_in_mkt: {arr_tim.mean():.3f}")
    print(f"Total trades: {int(arr_tr.sum())}")

    # Charts
    _print_series_chart("Validation equity (stitched across windows)", equity)
    _print_series_chart("Per-window cum_return", cumrets)
    _print_series_chart("Per-window sharpe", sharpes)
    _print_series_chart("Per-window time_in_mkt", tims)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize and visualize per-window validation metrics from tcn_policy_run",
    )
    ap.add_argument(
        "--run-jsonl",
        required=True,
        help="Path to JSONL file produced by --out-jsonl in tcn_policy_run",
    )
    args = ap.parse_args()

    summarize_and_plot(Path(args.run_jsonl))


if __name__ == "__main__":  # pragma: no cover
    main()

