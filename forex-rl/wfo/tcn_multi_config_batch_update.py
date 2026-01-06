from __future__ import annotations

"""
Batch helper to update the TCN multi-policy live config with all optimized
configs from an Optuna search results directory in one shot.

Usage (from forex-rl root):

  python3 -m wfo.tcn_multi_config_batch_update \
    --optuna-dir wfo/optuna_search_results \
    --multi-config wfo/tcn_multi_policy_live_config.json \
    --units-per-side 200

For each file in --optuna-dir matching "optimal_*.json", this script:
  - Infers the instrument name from the filename, e.g.
      optimal_eur_usd_20251229-163455.json -> EUR_USD
  - Upserts an entry in the multi config's `instruments` list with:
      {
        "name": "EUR_USD",
        "base_config": "wfo/optuna_search_results/optimal_eur_usd_20251229-163455.json",
        "units_per_side": 200
      }
  - If an instrument with the same name already exists, its `base_config` and
    `units_per_side` fields are updated.
"""

import argparse
from pathlib import Path
from typing import List

from .tcn_multi_config_update import _ensure_instruments_list, _load_json


def _infer_instrument_from_filename(path: Path) -> str | None:
    """
    Extract instrument name from an optimal_*.json filename.

    Pattern we expect (lowercase instrument, timestamp suffix):
      optimal_eur_usd_YYYYMMDD-HHMMSS.json

    We:
      - Strip "optimal_" prefix
      - Strip final "_YYYYMMDD-HHMMSS" segment
      - Uppercase and return, e.g. "EUR_USD".
    """
    stem = path.stem  # e.g. "optimal_eur_usd_20251229-163455"
    prefix = "optimal_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]  # "eur_usd_20251229-163455"
    inst_part, sep, _ = rest.rpartition("_")
    if not sep:
        # No underscore found after "optimal_"; unable to infer.
        return None
    inst = inst_part.strip()
    if not inst:
        return None
    return inst.upper()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch update TCN multi-policy live config from Optuna search results",
    )
    ap.add_argument(
        "--optuna-dir",
        type=str,
        default="wfo/optuna_search_results",
        help="Directory containing optimal_*.json Optuna result configs",
    )
    ap.add_argument(
        "--multi-config",
        type=str,
        default="wfo/tcn_multi_policy_live_config.json",
        help="Path to the multi-instrument live config JSON (edited in place)",
    )
    ap.add_argument(
        "--units-per-side",
        type=int,
        default=200,
        help="Default logical max units per side for each instrument",
    )
    args = ap.parse_args()

    optuna_dir = Path(args.optuna_dir)
    if not optuna_dir.exists():
        raise SystemExit(f"Optuna directory not found: {optuna_dir}")

    # Collect all optimal_*.json files in the directory.
    files: List[Path] = sorted(optuna_dir.glob("optimal_*.json"))
    if not files:
        print(f"No optimal_*.json files found under {optuna_dir}")
        return

    # When there are multiple files per instrument (different timestamps),
    # keep only the *most recent* one for each instrument based on the
    # timestamp suffix in the filename.
    latest_by_inst: dict[str, Path] = {}
    for path in files:
        inst_name = _infer_instrument_from_filename(path)
        if inst_name is None:
            print(f"[INFO] Skipping {path.name}: unable to infer instrument name.")
            continue
        prev = latest_by_inst.get(inst_name)
        # Filenames are of the form optimal_<inst>_YYYYMMDD-HHMMSS.json, so
        # lexicographic order on the stem is equivalent to timestamp order.
        if prev is None or path.stem > prev.stem:
            latest_by_inst[inst_name] = path

    multi_path = Path(args.multi_config)
    cfg = _load_json(multi_path)
    insts = _ensure_instruments_list(cfg)

    updated_any = False

    # Deterministic order by instrument name
    for inst_name, path in sorted(latest_by_inst.items()):
        # Upsert this instrument entry using only the latest config path.
        updated = False
        for inst in insts:
            name = str(inst.get("name", "")).upper()
            if name == inst_name:
                inst["name"] = inst_name
                inst["base_config"] = str(path)
                inst["units_per_side"] = int(args.units_per_side)
                updated = True
                updated_any = True
                break

        if not updated:
            insts.append(
                {
                    "name": inst_name,
                    "base_config": str(path),
                    "units_per_side": int(args.units_per_side),
                }
            )
            updated_any = True

        print(
            f"Upserted instrument {inst_name} with config {path} "
            f"(units_per_side={int(args.units_per_side)})"
        )

    if updated_any:
        multi_path.parent.mkdir(parents=True, exist_ok=True)
        with multi_path.open("w", encoding="utf-8") as f:
            # Keep formatting readable; do not sort keys to preserve original layout
            import json

            json.dump(cfg, f, indent=2)
            f.write("\n")

        print(f"\nUpdated multi-instrument config: {multi_path}")
    else:
        print("No changes were made to the multi-instrument config.")


if __name__ == "__main__":  # pragma: no cover
    main()

