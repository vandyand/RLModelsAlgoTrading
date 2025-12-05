"""Tests for ScalerManager round-trip behavior.

This test uses synthetic data to avoid large I/O, focusing on the scaler
fit/apply logic itself.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add ga-trailing-20 as importable root
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data.scaler import ScalerManager, ScalerState, ScalerType  # type: ignore[import]


def test_scaler_round_trip_synthetic() -> None:
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "feat_std": rng.normal(0, 2, size=n),
        "feat_robust": rng.standard_t(df=3, size=n),
        "feat_minmax": rng.uniform(-5, 5, size=n),
    })
    feature_types = {
        "feat_std": ScalerType.STANDARD,
        "feat_robust": ScalerType.ROBUST,
        "feat_minmax": ScalerType.MINMAX,
    }
    mgr = ScalerManager()
    state = mgr.fit(df, feature_types=feature_types)
    df_scaled = mgr.apply(df, state)

    # Standard: mean ~0, std ~1
    assert abs(float(df_scaled["feat_std"].mean())) < 1e-1
    assert 0.5 < float(df_scaled["feat_std"].std()) < 1.5

    # Robust: median ~0
    assert abs(float(df_scaled["feat_robust"].median())) < 1e-1

    # Min-max: within [0,1]
    assert df_scaled["feat_minmax"].min() >= -1e-6
    assert df_scaled["feat_minmax"].max() <= 1.0 + 1e-6


if __name__ == "__main__":  # pragma: no cover
    test_scaler_round_trip_synthetic()
    print("ScalerManager synthetic round-trip test passed")
