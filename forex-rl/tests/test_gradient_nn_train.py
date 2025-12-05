"""Smoke test for GradientNNStrategy training + simulator integration."""
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
from strategies.gradient_nn import GradientNNStrategy, GradientNNConfig  # type: ignore[import]
from strategies.base import RegularizationConfig  # type: ignore[import]


def _build_small_splits():
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
    # Take a prefix of records to keep training quick
    records = []
    for idx, rec in enumerate(train_split):
        if idx >= 300:
            break
        records.append(rec)
    assert len(records) > 10
    # Wrap back into an iterable DatasetSplit-like object
    class SmallSplit:
        def __iter__(self):  # pragma: no cover - trivial
            return iter(records)
    return SmallSplit()


def test_gradient_nn_train_and_eval() -> None:
    train_split = _build_small_splits()
    cfg = GradientNNConfig(hidden_dims=(64, 32), epochs=3, batch_size=64, patience=2)
    reg = RegularizationConfig(l2=1e-4)
    strat = GradientNNStrategy(config=cfg, regularization=reg, device="cpu")
    sim = TrailingStopSimulator(TrailingConfig(), CostModel(CostModelConfig()))

    strat.fit(train_split, None, sim)

    # After training, model should exist and produce a scalar logit
    assert strat._model is not None  # type: ignore[attr-defined]
    first_batch = next(iter(train_split))
    feats = np.asarray(first_batch["features"], dtype=np.float32)
    out = strat.predict(feats)
    assert out.shape == (1,)
    assert np.isfinite(out[0])

    # Simulator should be able to evaluate the trained strategy
    result = sim.evaluate(strat, train_split, record_equity=False, return_trades=True)
    assert result.metrics is not None


if __name__ == "__main__":  # pragma: no cover
    test_gradient_nn_train_and_eval()
    print("GradientNNStrategy smoke test passed")
