from __future__ import annotations

"""
Helper to add or update an instrument entry in the multi-instrument
TCN live trader config JSON.

Usage (from forex-rl root):

  python3 -m wfo.tcn_multi_config_update \
    --instrument EUR_JPY \
    --base-config wfo/tcn_eurjpy_policy_live_2025Q4.json \
    --units-per-side 200

By default this edits `wfo/tcn_multi_policy_live_config.json` in place,
upserting an entry in the `instruments` list with:

  {
    "name": "EUR_JPY",
    "base_config": "wfo/tcn_eurjpy_policy_live_2025Q4.json",
    "units_per_side": 200
  }

If an instrument with the same `name` already exists, its `base_config`
and `units_per_side` fields are updated; other keys remain untouched.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Multi config JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Multi config JSON must contain an object at the top level")
    return data


def _ensure_instruments_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    insts = cfg.get("instruments")
    if insts is None:
        insts = []
    if not isinstance(insts, list):
        raise SystemExit("`instruments` field in multi config must be a list")
    # Normalize to list[dict]
    out: List[Dict[str, Any]] = []
    for x in insts:
        if isinstance(x, dict):
            out.append(x)
    cfg["instruments"] = out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add or update an instrument entry in the TCN multi-policy live config",
    )
    ap.add_argument(
        "--multi-config",
        default="wfo/tcn_multi_policy_live_config.json",
        help="Path to the multi-instrument live config JSON (edited in place)",
    )
    ap.add_argument(
        "--instrument",
        required=True,
        help="Instrument name (e.g. EUR_USD, AUD_JPY) to upsert",
    )
    ap.add_argument(
        "--base-config",
        required=True,
        help="Path to the optimized per-instrument policy config JSON (as used by tcn_policy_optuna_search)",
    )
    ap.add_argument(
        "--units-per-side",
        type=int,
        required=True,
        help="Logical max units per side for this instrument in live trading",
    )
    args = ap.parse_args()

    multi_path = Path(args.multi_config)
    cfg = _load_json(multi_path)

    insts = _ensure_instruments_list(cfg)
    target_name = str(args.instrument).upper()

    updated = False
    for inst in insts:
        name = str(inst.get("name", "")).upper()
        if name == target_name:
            inst["name"] = target_name
            inst["base_config"] = str(args.base_config)
            inst["units_per_side"] = int(args.units_per_side)
            updated = True
            break

    if not updated:
        insts.append(
            {
                "name": target_name,
                "base_config": str(args.base_config),
                "units_per_side": int(args.units_per_side),
            }
        )

    multi_path.parent.mkdir(parents=True, exist_ok=True)
    with multi_path.open("w", encoding="utf-8") as f:
        # Keep formatting readable; do not sort keys to preserve original layout
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "event": "multi_config_updated",
                "multi_config": str(multi_path),
                "instrument": target_name,
                "base_config": str(args.base_config),
                "units_per_side": int(args.units_per_side),
                "mode": "updated" if updated else "added",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

