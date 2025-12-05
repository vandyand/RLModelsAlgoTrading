"""Feature scaling utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class ScalerType(str, Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


@dataclass
class ScalerState:
    version: str
    feature_types: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    manifest_hash: Optional[str] = None


class ScalerManager:
    def __init__(self, default_type: ScalerType = ScalerType.STANDARD) -> None:
        self.default_type = default_type

    def fit(
        self,
        df: pd.DataFrame,
        *,
        feature_types: Optional[Dict[str, ScalerType]] = None,
        version: str = "2025-11-28",
        manifest_hash: Optional[str] = None,
    ) -> ScalerState:
        stats: Dict[str, Dict[str, float]] = {}
        types: Dict[str, str] = {}
        ft_map = feature_types or {}
        for col in df.columns:
            series = df[col].astype(float)
            stype = ft_map.get(col, self.default_type)
            types[col] = stype.value
            if stype == ScalerType.STANDARD:
                mean = float(series.mean())
                std = float(series.std())
                stats[col] = {
                    "mean": mean,
                    "std": max(std, 1e-8),
                }
            elif stype == ScalerType.ROBUST:
                median = float(series.median())
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = max(q3 - q1, 1e-8)
                stats[col] = {
                    "median": median,
                    "iqr": iqr,
                }
            elif stype == ScalerType.MINMAX:
                min_v = float(series.min())
                max_v = float(series.max())
                spread = max(max_v - min_v, 1e-8)
                stats[col] = {
                    "min": min_v,
                    "max": max_v,
                    "spread": spread,
                }
            else:
                stats[col] = {}
        return ScalerState(version=version, feature_types=types, stats=stats, manifest_hash=manifest_hash)

    def apply(self, df: pd.DataFrame, state: ScalerState) -> pd.DataFrame:
        out = df.copy()
        for col in df.columns:
            stype = ScalerType(state.feature_types.get(col, self.default_type.value))
            info = state.stats.get(col, {})
            if stype == ScalerType.STANDARD:
                mean = info.get("mean", 0.0)
                std = info.get("std", 1.0)
                out[col] = (df[col] - mean) / (std if std != 0 else 1.0)
            elif stype == ScalerType.ROBUST:
                median = info.get("median", 0.0)
                iqr = info.get("iqr", 1.0)
                out[col] = (df[col] - median) / (iqr if iqr != 0 else 1.0)
            elif stype == ScalerType.MINMAX:
                min_v = info.get("min", 0.0)
                spread = info.get("spread", 1.0)
                out[col] = (df[col] - min_v) / (spread if spread != 0 else 1.0)
            else:
                out[col] = df[col]
        return out
