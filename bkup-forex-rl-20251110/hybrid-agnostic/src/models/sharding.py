#!/usr/bin/env python3
"""
Feature sharding utilities.
- Deterministic, stable sharding by feature name using xxhash or hashlib fallback
- Persisted mapping for consistent train/infer
- Evenly distributes features across K shards while keeping related columns together if desired
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import xxhash  # type: ignore
except Exception:  # pragma: no cover
    xxhash = None

import hashlib


@dataclass(frozen=True)
class ShardPlan:
    num_shards: int
    feature_to_shard: Dict[str, int]
    shard_to_features: Dict[int, List[str]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "num_shards": int(self.num_shards),
            "feature_to_shard": self.feature_to_shard,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str) -> "ShardPlan":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        num = int(obj["num_shards"])  # type: ignore[index]
        f2s: Dict[str, int] = {str(k): int(v) for k, v in obj["feature_to_shard"].items()}  # type: ignore[index]
        s2f: Dict[int, List[str]] = {}
        for feat, sid in f2s.items():
            s2f.setdefault(sid, []).append(feat)
        # preserve order
        for sid in range(num):
            s2f.setdefault(sid, [])
            s2f[sid].sort()
        return ShardPlan(num_shards=num, feature_to_shard=f2s, shard_to_features=s2f)


def _stable_hash(text: str) -> int:
    if xxhash is not None:
        return int(xxhash.xxh64(text).intdigest())
    # Fallback
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def _coarse_group_key(feature_name: str) -> str:
    """Group semantically related features so they land in same shard if possible.
    Examples:
      - "FX_EUR_USD_rsi_14" and "FX_EUR_USD_macd_12_26" -> same group
      - "ETF_SPY_ema_45" grouped by instrument "ETF_SPY"
    """
    # Common pattern: PREFIX_{instrument}_suffix
    m = re.match(r"^(FX|ETF)_[^_]+_[^_]+", feature_name)
    if m:
        return m.group(0)
    # Fallback: prefix before first double-underscore separator
    m2 = re.match(r"^([A-Z]+_[A-Z]+)", feature_name)
    if m2:
        return m2.group(1)
    # Default to full feature name (no grouping)
    return feature_name


def plan_shards(feature_names: Sequence[str], num_shards: int, group_related: bool = True) -> ShardPlan:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    feature_to_shard: Dict[str, int] = {}
    if group_related:
        # Bucket features by coarse group, then assign groups to shards round-robin by hashed key
        groups: Dict[str, List[str]] = {}
        for f in feature_names:
            g = _coarse_group_key(f)
            groups.setdefault(g, []).append(f)
        # Stable ordering by group hash
        ordered_groups = sorted(groups.items(), key=lambda kv: _stable_hash(kv[0]))
        # Assign groups to shards in a greedy load-balanced fashion by number of features
        shard_loads = [0] * num_shards
        shard_lists: List[List[str]] = [[] for _ in range(num_shards)]
        for gname, feats in ordered_groups:
            # choose shard with minimum current load; tie-break by hash
            min_idx = min(range(num_shards), key=lambda i: (shard_loads[i], (hash(_stable_hash(gname) ^ i) & 0xffffffff)))
            shard_lists[min_idx].extend(feats)
            shard_loads[min_idx] += len(feats)
        # Build maps
        for sid, feats in enumerate(shard_lists):
            feats_sorted = sorted(feats)
            for f in feats_sorted:
                feature_to_shard[f] = sid
    else:
        # Direct hash partition
        for f in feature_names:
            sid = _stable_hash(f) % num_shards
            feature_to_shard[f] = int(sid)

    shard_to_features: Dict[int, List[str]] = {i: [] for i in range(num_shards)}
    for f, sid in feature_to_shard.items():
        shard_to_features[sid].append(f)
    for sid in range(num_shards):
        shard_to_features[sid].sort()

    return ShardPlan(num_shards=num_shards, feature_to_shard=feature_to_shard, shard_to_features=shard_to_features)


def shard_dataframe_columns(columns: Sequence[str], plan: ShardPlan) -> Dict[int, List[int]]:
    """Return mapping shard_id -> list of column indices for a pandas DataFrame with given columns.
    Missing features are ignored; extra columns not in the plan are ignored.
    """
    col_index: Dict[str, int] = {c: i for i, c in enumerate(columns)}
    out: Dict[int, List[int]] = {sid: [] for sid in range(plan.num_shards)}
    for feat, sid in plan.feature_to_shard.items():
        idx = col_index.get(feat)
        if idx is not None:
            out[sid].append(idx)
    # Ensure deterministic ordering inside shards
    for sid in range(plan.num_shards):
        out[sid].sort()
    return out
