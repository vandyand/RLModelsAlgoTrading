"""DatasetLoader aligning candles and multi-granularity feature grids."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .scaler import ScalerManager, ScalerState

@dataclass
class LoaderConfig:
    instruments: Sequence[str]
    raw_dir: Path
    feature_dir: Path
    base_granularity: str = "M1"
    aux_granularities: Sequence[str] = ("M5", "H1", "D")
    normalize: bool = True
    feature_subset: Optional[Sequence[str]] = None
    cache_dir: Optional[Path] = None
    # Optional hard cap on rows loaded per instrument per granularity
    max_rows: Optional[int] = None


@dataclass
class SplitConfig:
    train: tuple[str, str]
    val: Optional[tuple[str, str]] = None
    test: Optional[tuple[str, str]] = None


class DatasetSplit(Iterable[Dict[str, Any]]):
    """Iterable wrapper over aligned bar/feature records."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)


class DatasetLoader:
    """Coordinates alignment of raw candles + auxiliary feature grids."""

    def __init__(self, config: LoaderConfig) -> None:
        self.cfg = config
        self.raw_dir = Path(config.raw_dir)
        self.feature_dir = Path(config.feature_dir)
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        self.inventory: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.scaler_state: Optional[ScalerState] = None
        self.manifest: Dict[str, Any] = {}
        self.scaler_manager = ScalerManager()
        self._catalog_sources()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_by_dates(self, split_cfg: SplitConfig) -> tuple[DatasetSplit, Optional[DatasetSplit], Optional[DatasetSplit]]:
        panel = self._build_panel()
        records = self._records_from_panel(panel)

        def select(range_tuple: Optional[tuple[str, str]]) -> List[Dict[str, Any]]:
            if range_tuple is None:
                return []
            start_raw, end_raw = map(pd.Timestamp, range_tuple)
            # Normalize to UTC to match loader indices
            start = start_raw.tz_localize("UTC") if start_raw.tzinfo is None else start_raw.tz_convert("UTC")
            end = end_raw.tz_localize("UTC") if end_raw.tzinfo is None else end_raw.tz_convert("UTC")
            out: List[Dict[str, Any]] = []
            for rec in records:
                ts = pd.Timestamp(rec["timestamp"])
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if start <= ts <= end:
                    out.append(rec)
            return out

        return (
            DatasetSplit(select(split_cfg.train)),
            DatasetSplit(select(split_cfg.val)) if split_cfg.val else None,
            DatasetSplit(select(split_cfg.test)) if split_cfg.test else None,
        )

    def make_rolling_folds(self, n_splits: int, val_bars: int, gap_bars: int = 0) -> List[DatasetSplit]:
        panel = self._build_panel()
        records = self._records_from_panel(panel)
        total = len(records)
        folds: List[DatasetSplit] = []
        step = max(1, (total - val_bars) // max(1, n_splits))
        for start in range(0, total - val_bars + 1, step):
            fold_records = records[start + gap_bars : start + gap_bars + val_bars]
            folds.append(DatasetSplit(fold_records))
            if len(folds) == n_splits:
                break
        return folds

    def export_manifest(self, path: Path) -> None:
        payload = {
            "instruments": list(self.cfg.instruments),
            "base_granularity": self.cfg.base_granularity,
            "aux_granularities": list(self.cfg.aux_granularities),
            "inventory": self.inventory,
            "feature_names": self.feature_names,
        }
        if self.scaler_state is not None:
            payload["scaler_state"] = {
                "version": self.scaler_state.version,
                "feature_types": self.scaler_state.feature_types,
                "stats": self.scaler_state.stats,
                "manifest_hash": self.scaler_state.manifest_hash,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load_manifest(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.feature_names = list(self.manifest.get("feature_names", []))
        scaler_meta = self.manifest.get("scaler_state")
        if scaler_meta:
            self.scaler_state = ScalerState(
                version=scaler_meta.get("version", ""),
                feature_types=scaler_meta.get("feature_types", {}),
                stats=scaler_meta.get("stats", {}),
                manifest_hash=scaler_meta.get("manifest_hash"),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _catalog_sources(self) -> None:
        inventory: Dict[str, Dict[str, List[str]]] = {}
        for inst in self.cfg.instruments:
            inst_inv: Dict[str, List[str]] = {}
            for gran in self.cfg.aux_granularities:
                pattern = f"{inst.replace('/', '_')}_{gran}"
                matches = [str(path) for path in self.feature_dir.rglob(f"*{pattern}*")]
                if matches:
                    inst_inv[gran] = matches
            inventory[inst] = inst_inv
        self.inventory = inventory

    def _build_panel(self) -> Dict[str, Any]:
        raw_frames: Dict[str, pd.DataFrame] = {}
        for inst in self.cfg.instruments:
            raw_frames[inst] = self._load_ohlc(inst, self.cfg.base_granularity)
        base_index = None
        for df in raw_frames.values():
            base_index = df.index if base_index is None else base_index.intersection(df.index)
        if base_index is None or base_index.empty:
            raise ValueError("No overlapping timestamps found across instruments")

        feature_blocks: List[pd.DataFrame] = []
        for inst in self.cfg.instruments:
            inst_blocks = []
            for gran in self.cfg.aux_granularities:
                feats = self._load_feature_grid(inst, gran)
                if feats is None:
                    continue
                feats = feats.reindex(base_index).ffill().fillna(0.0)
                feats = feats.add_prefix(f"{gran}_{inst}::")
                inst_blocks.append(feats)
            if inst_blocks:
                feature_blocks.append(pd.concat(inst_blocks, axis=1))
        feature_matrix = pd.concat(feature_blocks, axis=1) if feature_blocks else pd.DataFrame(index=base_index)

        if self.cfg.feature_subset:
            subset = [col for col in self.cfg.feature_subset if col in feature_matrix.columns]
            feature_matrix = feature_matrix.reindex(columns=subset)

        self.feature_names = feature_matrix.columns.tolist()
        if self.cfg.normalize and not feature_matrix.empty:
            self.scaler_state = self.scaler_manager.fit(feature_matrix)
            feature_matrix = self.scaler_manager.apply(feature_matrix, self.scaler_state)

        bars_map = {inst: raw_frames[inst].reindex(base_index) for inst in self.cfg.instruments}
        return {"index": base_index, "bars": bars_map, "features": feature_matrix}

    def _records_from_panel(self, panel: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_index: pd.Index = panel["index"]
        bars_map: Dict[str, pd.DataFrame] = panel["bars"]
        feature_df: pd.DataFrame = panel["features"]
        records: List[Dict[str, Any]] = []
        for ts in base_index:
            bars = {inst: bars_map[inst].loc[ts].to_dict() for inst in bars_map}
            features = feature_df.loc[ts].values.astype(np.float32) if not feature_df.empty else np.array([], dtype=np.float32)
            records.append({"timestamp": ts, "bars": bars, "features": features})
        return records

    def _load_ohlc(self, instrument: str, granularity: str) -> pd.DataFrame:
        fname = f"{instrument.replace('/', '_')}_{granularity.upper()}.csv"
        path = self.raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing OHLC file: {path}")
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.date_range(start=0, periods=len(df), freq="T")
        cols = {c.lower(): c for c in df.columns}
        rename = {
            cols.get("open", "open"): "open",
            cols.get("high", "high"): "high",
            cols.get("low", "low"): "low",
            cols.get("close", "close"): "close",
            cols.get("volume", "volume"): "volume",
        }
        df = df.rename(columns=rename)[["open", "high", "low", "close", "volume"]]
        # Apply optional tail cap to limit memory
        if getattr(self.cfg, "max_rows", None) is not None and len(df) > int(self.cfg.max_rows):  # type: ignore[arg-type]
            df = df.tail(int(self.cfg.max_rows))  # type: ignore[arg-type]
        return df.sort_index()

    def _load_feature_grid(self, instrument: str, granularity: str) -> Optional[pd.DataFrame]:
        gran = granularity.upper()
        safe = instrument.replace('/', '_')
        path = self.feature_dir / gran / f"{safe}_{gran}_features.csv.gz"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        # Drop duplicate index labels to allow safe reindexing
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        # Apply same tail cap to features to roughly match OHLC window
        if getattr(self.cfg, "max_rows", None) is not None and len(df) > int(self.cfg.max_rows):  # type: ignore[arg-type]
            df = df.tail(int(self.cfg.max_rows))  # type: ignore[arg-type]
        return df
