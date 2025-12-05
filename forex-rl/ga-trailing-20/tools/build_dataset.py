#!/usr/bin/env python3
"""CLI stub for building aligned datasets and scaler manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ga_trailing_20.data import DatasetLoader, LoaderConfig, SplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned dataset artifacts for ga-trailing-20")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw OHLC CSVs")
    parser.add_argument("--feature-dir", required=True, help="Directory containing feature exports")
    parser.add_argument("--instruments", required=True, help="Comma-separated instruments")
    parser.add_argument("--train", required=True, help="start:end date range for training split")
    parser.add_argument("--val", help="start:end date range for validation split")
    parser.add_argument("--test", help="start:end date range for test split")
    parser.add_argument("--manifest", required=True, help="Path to write dataset manifest JSON")
    parser.add_argument("--scaler-json", help="Optional path to export scaler state JSON")
    parser.add_argument("--cache-dir", help="Optional cache directory for aligned arrays")
    parser.add_argument("--feature-subset", help="Optional path to file listing selected features")
    parser.add_argument("--aux", default="M5,H1,D", help="Auxiliary granularities to include")
    return parser.parse_args()


def parse_range(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Date range must be start:end, got {value}")
    return parts[0].strip(), parts[1].strip()


def load_feature_subset(path: str | None) -> List[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]
    subset = load_feature_subset(args.feature_subset)
    cfg = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        aux_granularities=aux,
        feature_subset=subset,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    loader = DatasetLoader(cfg)
    split_cfg = SplitConfig(
        train=parse_range(args.train),
        val=parse_range(args.val) if args.val else None,
        test=parse_range(args.test) if args.test else None,
    )
    loader.split_by_dates(split_cfg)
    loader.export_manifest(Path(args.manifest))
    print(f"Manifest written to {args.manifest}")
    if args.scaler_json and loader.scaler_state is not None:
        payload = {
            "version": loader.scaler_state.version,
            "feature_types": loader.scaler_state.feature_types,
            "stats": loader.scaler_state.stats,
            "manifest_hash": loader.scaler_state.manifest_hash,
        }
        scaler_path = Path(args.scaler_json)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Scaler state written to {args.scaler_json}")


if __name__ == "__main__":
    main()
