# Normalization & Scaler Export Pipeline

## Goals
- Provide consistent feature normalization across training, evaluation, and live inference.
- Support multiple scaler modes (z-score, robust/quantile, min-max) selectable per feature group.
- Version and persist scaler metadata so any strategy checkpoint can restore the exact preprocessing state.

## Key Components

### 1. Scaler Registry
- Implement `ScalerType` enum: `standard`, `robust`, `minmax`, `none`.
- Map feature groups → scaler type (e.g., price-derived indicators use `standard`, bounded oscillators use `minmax`).
- Allow overrides via loader config or CLI flags.

### 2. Fit / Apply API
- `ScalerManager.fit(df: pd.DataFrame) -> ScalerState`
  - Computes per-column stats based on selected scaler.
  - Stores metadata: mean/std, median/IQR, min/max, quantile cutoffs, epsilon safeguards.
- `ScalerManager.apply(df, state) -> pd.DataFrame`
  - Applies stored stats to new data.
  - Handles missing columns by filling with zeros or dropping based on policy.

### 3. Persistence Format
- Store scaler state as JSON (`scalers/*.json`) with structure:
  ```json
  {
    "version": "2025-11-28",
    "feature_groups": {
      "standard": ["EUR_USD::ret_1", ...],
      "robust": [...]
    },
    "stats": {
      "EUR_USD::ret_1": {"mean": 0.0, "std": 0.0123},
      "EUR_USD::rsi14": {"median": 0.5, "iqr": 0.15},
      ...
    }
  }
  ```
- Include hash of input dataset manifest to tie scaler to a specific data build.

### 4. Versioning & CLI
- CLI tool `ga-trailing-20/tools/export_scaler.py`:
  - Arguments: dataset manifest path, scaler config, output path.
  - Validates dataset hash match before exporting.
  - Optionally emits binary pickle for faster load (while keeping JSON canonical source).

### 5. Integration Points
- `DatasetLoader` will call `ScalerManager.fit` after assembling the training split; stores `scaler_state` attribute.
- Strategies receive scaler state via loader or manifest; training pipelines ensure normalized features before model consumption.
- Live trader/backtester loads scaler JSON and applies identical transforms.

### 6. Edge Case Handling
- Features with near-zero variance: default to leaving as-is (std=1) or drop them based on threshold.
- NaNs in stats (e.g., fully zero columns) → record flag `is_constant: true` and skip normalization.
- Backwards compatibility: maintain `schema_version` to handle future format changes.

### 7. Testing Plan
- Unit tests for `ScalerManager` covering each scaler type, persistence round-trips, and missing-feature behavior.
- Integration test where dataset manifest + scaler JSON + feature matrix reproduce the same normalized outputs as training run.

## Deliverables
1. `ga-trailing-20/data/scaler.py` implementing registry + manager.
2. CLI exporter script in `ga-trailing-20/tools/`.
3. Documentation snippet in the roadmap and README describing how to regenerate scalers.
