"""Smoke test wiring DatasetLoader into TrailingStopSimulator.

This is intentionally lightweight and uses existing M5 data + no feature grids.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import numpy as np

# Add ga-trailing-20 as an importable root
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]


class DummyStrategy:
    """Always-flat strategy (no entries) for simulator wiring test."""

    def predict(self, features: np.ndarray) -> np.ndarray:  # type: ignore[override]
        # One instrument → single logit 0.0 → flat
        return np.zeros(1, dtype=float)


def build_loader() -> DatasetLoader:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "continuous-trader" / "data"
    feature_dir = raw_dir / "features"
    cfg = LoaderConfig(
        instruments=["USD_PLN"],  # has M5 data in continuous-trader/data
        raw_dir=raw_dir,
        feature_dir=feature_dir,
        base_granularity="M5",
        aux_granularities=(),  # keep features empty for now
        normalize=False,
    )
    return DatasetLoader(cfg)


def take_first_n(records_iter, n: int) -> List[dict]:
    out: List[dict] = []
    for idx, rec in enumerate(records_iter):
        if idx >= n:
            break
        out.append(rec)
    return out


def test_simulator_smoke() -> None:
    loader = build_loader()
    split_cfg = SplitConfig(train=("2000-01-01", "2100-01-01"))
    train_split, _, _ = loader.split_by_dates(split_cfg)
    # Use a small prefix to keep test fast
    records = take_first_n(train_split, 200)
    assert records, "Expected at least one record from loader"

    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))
    result = sim.evaluate(DummyStrategy(), records, record_equity=True, return_trades=True)

    assert result.metrics is not None
    assert isinstance(result.metrics.cum_return, float)
    # Flat strategy should not open trades
    assert len(result.trades) == 0
    assert result.equity_curve is None or result.equity_curve.size == len(records)


if __name__ == "__main__":  # pragma: no cover
    test_simulator_smoke()
    print("Simulator smoke test completed")
