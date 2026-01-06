from __future__ import annotations

"""
Convenience wrapper to run a full TCN policy WFO evaluation *and* immediately
summarize it with in-terminal charts.

This is equivalent to:

  python3 -m wfo.tcn_policy_run \
      --base-config wfo/your_policy.json \
      --out-jsonl wfo/runs/<auto>.jsonl

  python3 -m wfo.tcn_policy_report \
      --run-jsonl wfo/runs/<auto>.jsonl

but wired together so you only need a single command.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .tcn_policy_run import _run_windows
from .tcn_policy_report import summarize_and_plot


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run TCN policy WFO and report validation metrics in one step",
    )
    ap.add_argument(
        "--base-config",
        required=True,
        help="Path to base JSON config for geometry/hyperparams/instrument",
    )
    ap.add_argument(
        "--runs-dir",
        default="wfo/runs",
        help="Directory to store the auto-generated JSONL run file",
    )
    args = ap.parse_args()

    base_path = Path(args.base_config)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Millisecond-resolution timestamp for uniqueness.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    jsonl_path = runs_dir / f"{base_path.stem}_windows_{ts}.jsonl"

    print(f"Running tcn_policy_run for {base_path} ...")
    _run_windows(str(base_path), out_jsonl=str(jsonl_path))
    print(f"\nSaved per-window metrics to {jsonl_path}")

    print("\n=== Policy report ===")
    summarize_and_plot(jsonl_path)


if __name__ == "__main__":  # pragma: no cover
    main()

