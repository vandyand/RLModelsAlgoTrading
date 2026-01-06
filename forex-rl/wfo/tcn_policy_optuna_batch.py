from __future__ import annotations

"""
Batch Optuna search over TCN+MLP policy configs for multiple FX instruments.

This script:
  - Reads the default 20 most liquid OANDA FX pairs from ac_multi20.instruments.
  - Takes the top ~15 majors/crosses.
  - For each instrument with a known base config JSON, runs the
    `tcn_policy_optuna_search` script and writes an optimized config into
    `wfo/optuna_search_results/` (or a user-specified output directory),
    including a timestamp in the filename.

Usage (from forex-rl root):

  python3 -m wfo.tcn_policy_optuna_batch \
    --trials 40 \
    --seed 0

You can also override the output directory:

  python3 -m wfo.tcn_policy_optuna_batch \
    --trials 50 \
    --optuna-dir wfo/optuna_search_results
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import __package__ as wfo_package  # noqa: F401
from .ac_multi20 import instruments as ac_instruments


# Top 20 from ac-multi20; we keep the first 15 as the most liquid majors/crosses.
DEFAULT_OANDA_20: List[str] = list(ac_instruments.DEFAULT_OANDA_20)
TOP_15: List[str] = DEFAULT_OANDA_20[:15]


# Map instruments we currently have dedicated base-configs for.
BASE_CONFIGS: Dict[str, str] = {
    # Core majors
    "EUR_USD": "wfo/tcn_eurusd_policy_debug_3w1w.json",
    "USD_JPY": "wfo/tcn_usdjpy_policy_debug.json",
    "GBP_USD": "wfo/tcn_gbpusd_policy_debug.json",
    "AUD_USD": "wfo/tcn_audusd_policy_debug.json",
    # Cross we already tuned by hand / HPSS-style config
    "EUR_JPY": "wfo/tcn_eurjpy_policy_hpss.json",
}


def _run_optuna_for_instrument(
    instrument: str,
    base_cfg_rel: str,
    optuna_dir: Path,
    trials: int,
    seed: int,
) -> None:
    """
    Invoke `tcn_policy_optuna_search` for a single instrument via subprocess,
    writing the best config into optuna_dir with a timestamped filename.
    """
    inst_tag = instrument.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = optuna_dir / f"optimal_{inst_tag.lower()}_{ts}.json"

    cmd = [
        sys.executable,
        "-m",
        "wfo.tcn_policy_optuna_search",
        "--base-config",
        base_cfg_rel,
        "--trials",
        str(int(trials)),
        "--seed",
        str(int(seed)),
        "--out-config",
        str(out_path),
    ]

    print(f"\n=== Optuna search for {instrument} ===")
    print(f"Base config: {base_cfg_rel}")
    print(f"Output config: {out_path}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Optuna search failed for {instrument}: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch Optuna search for multiple FX instruments using TCN+MLP policies",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Number of Optuna trials per instrument",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Optuna sampler (shared across instruments)",
    )
    ap.add_argument(
        "--optuna-dir",
        type=str,
        default="wfo/optuna_search_results",
        help="Directory for optimized policy configs",
    )
    args = ap.parse_args()

    optuna_dir = Path(args.optuna_dir)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    print("Top 15 OANDA FX instruments (from DEFAULT_OANDA_20):")
    print(", ".join(TOP_15))

    for inst in TOP_15:
        base_cfg_rel = BASE_CONFIGS.get(inst)
        if not base_cfg_rel:
            print(f"[INFO] No base config mapping for {inst}; skipping.")
            continue

        base_cfg_path = Path(base_cfg_rel)
        if not base_cfg_path.exists():
            # Allow both repo-root and wfo-relative invocations.
            fallback = Path("forex-rl") / base_cfg_rel
            if fallback.exists():
                base_cfg_rel = str(fallback)
            else:
                print(f"[WARN] Base config '{base_cfg_rel}' for {inst} not found; skipping.")
                continue

        _run_optuna_for_instrument(
            instrument=inst,
            base_cfg_rel=base_cfg_rel,
            optuna_dir=optuna_dir,
            trials=int(args.trials),
            seed=int(args.seed),
        )


if __name__ == "__main__":  # pragma: no cover
    main()

