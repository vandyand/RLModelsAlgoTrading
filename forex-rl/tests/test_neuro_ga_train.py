"""Smoke test for NeuroGAStrategy GA training + simulator integration."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Add ga-trailing-20 as import root
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from simulator import CostModel, CostModelConfig, TrailingConfig, TrailingStopSimulator  # type: ignore[import]
from strategies.neuro_ga import NeuroGAStrategy, NeuroGAConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def _build_small_split():
    raw_dir = REPO_ROOT / "continuous-trader" / "data"
    feature_dir = raw_dir / "features"
    cfg = LoaderConfig(
        instruments=["USD_PLN"],
        raw_dir=raw_dir,
        feature_dir=feature_dir,
        base_granularity="M5",
        aux_granularities=(),
        normalize=True,
    )
    loader = DatasetLoader(cfg)
    split_cfg = SplitConfig(train=("2000-01-01", "2100-01-01"))
    train_split, _, _ = loader.split_by_dates(split_cfg)
    records = []
    for idx, rec in enumerate(train_split):
        if idx >= 200:
            break
        records.append(rec)
    assert len(records) > 10

    class SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)

    return SmallSplit()


def test_neuro_ga_train_and_eval() -> None:
    train_split = _build_small_split()
    cfg = NeuroGAConfig(population=6, generations=2, hidden_layers=(16,))
    reg = RegularizationConfig(l2=1e-4, complexity_penalty=0.0)
    strat = NeuroGAStrategy(config=cfg, regularization=reg)
    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(train_split, None, sim)

    # After GA, genome should be present and predict should emit finite scalar
    assert strat._genome is not None  # type: ignore[attr-defined]
    first_rec = next(iter(train_split))
    feats = np.asarray(first_rec["features"], dtype=np.float32)
    out = strat.predict(feats)
    assert out.shape == (1,)
    assert np.isfinite(out[0])

    # Simulator evaluation should run without error
    result = sim.evaluate(strat, train_split, record_equity=False, return_trades=True)
    assert result.metrics is not None


if __name__ == "__main__":  # pragma: no cover
    test_neuro_ga_train_and_eval()
    print("NeuroGAStrategy smoke test passed")
