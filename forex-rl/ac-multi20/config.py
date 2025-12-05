from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from instruments import DEFAULT_OANDA_20
from feature_select import M5_FEATURES, H1_FEATURES, D_FEATURES


@dataclass
class Config:
    # Universe and horizon
    instruments: List[str] = field(default_factory=lambda: list(DEFAULT_OANDA_20))
    lookback_days: int = 20
    # Optional explicit date range overrides lookback_days when provided
    start_date: Optional[str] = None  # format: YYYY-MM-DD (UTC)
    end_date: Optional[str] = None    # format: YYYY-MM-DD (UTC)

    # Cost per instrument position change applied to raw returns
    # Default = 0.0002 (2% of 1%)
    trade_cost: float = 0.0002

    # Feature selections (20 per instrument)
    m5_features: List[str] = field(default_factory=lambda: list(M5_FEATURES))
    h1_features: List[str] = field(default_factory=lambda: list(H1_FEATURES))
    d_features: List[str] = field(default_factory=lambda: list(D_FEATURES))

    # Data roots (relative to repo root)
    data_root: str = "continuous-trader/data"
    features_root: str = "continuous-trader/data/features"

    # Model/GA settings
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    population: int = 40
    generations: int = 20
    elite_frac: float = 0.2
    mutation_prob: float = 0.3
    weight_sigma: float = 0.05
    affine_sigma: float = 0.02
    crossover_frac: float = 0.3
    random_frac: float = 0.1

    # Normalization for inputs: 'rolling' or 'affine'
    input_norm: str = "affine"

    # Hysteresis thresholds for mapping outputs to positions
    threshold_mode: str = "band"  # 'band' or 'absolute'
    # Absolute mode thresholds
    enter_long: float = 0.60
    exit_long: float = 0.55
    enter_short: float = 0.40
    exit_short: float = 0.45
    # Band mode thresholds around 0.5
    band_enter: float = 0.05  # enter if > 0.5 + band_enter or < 0.5 - band_enter
    band_exit: float = 0.02   # exit if < 0.5 + band_exit (long) or > 0.5 - band_exit (short)

    # Segmentation for stats
    num_segments: int = 10

    # Randomness (None = unseeded RNG)
    seed: Optional[int] = None

    # Performance controls
    n_jobs: int = 0           # 0 => auto (cpu_count), 1 => serial
    downsample: int = 1       # evaluate every k-th bar (>=1)
