# Data Loader + Feature Alignment Plan

## Objectives
1. Produce synchronized per-bar samples at M1 resolution with optional auxiliary features (M5/H1/D feature grids + exogenous signals).
2. Support rolling train/val/test splits, time-series cross-validation folds, and reproducible normalization stats.
3. Stream efficiently from large CSV / gzipped feature tables without loading entire history into memory.
4. Provide feature metadata (names, dtype, lag alignment) for downstream strategies and scaler exports.

## Source Inputs
- **Primary candles (M1)**: pulled from OANDA or reconstructed by upsampling M5 data where M1 is missing. Required columns: `timestamp, open, high, low, close, volume`.
- **Auxiliary candles/features**:
  - M5 raw OHLC (for intraday derived metrics if desired).
  - Pre-computed features from `continuous-trader/data/features/{D,H1}` (wide indicator grids per instrument).
  - Optional ETF/eco features (future extension) keyed by date.

## Loader Architecture
1. **Catalog & Schema Registry**
   - Scan `continuous-trader/data` and `data/features` once to map instrument → available granularities & files.
   - Persist schema metadata (column order, dtype, row count) in `ga-trailing-20/cache/data_inventory.json`.

2. **Chunked Readers**
   - Use pandas chunked reading (or pyarrow) for large CSV/gz files.
   - Standardize column names and ensure UTC timestamps.
   - Apply forward/back-fill to handle missing bars per granularity.

3. **Alignment Pipeline**
   - Choose base index = intersection of M1 timestamps for selected instruments (with optional trimming to ensure full auxiliary coverage).
   - For each auxiliary granularity:
     - Reindex to base index using forward-fill within granularity validity window (e.g., D data holds for next 1440 M1 bars).
     - Optionally add lagged versions (e.g., D features shifted by 1 day to avoid lookahead).
   - Concatenate per-instrument blocks with MultiIndex columns (`instrument::feature`).

4. **Windowing & Dataset Splits**
   - Support sequential slicing into train/val/test given explicit date ranges or ratios.
   - Provide rolling-origin splitter for cross-validation (defined by Phase 3 CV config).
   - Each split yields iterables of `(features, targets)` where targets can be:
     - Next-bar returns per instrument (for GA fitness proxies).
     - Full candle window for simulator (close/open/high/low for trailing-stop replay).

5. **Caching Strategy**
   - Cache aligned NumPy memmaps per instrument/time slice to accelerate repeated runs.
   - Maintain hash of config (instruments, granularity set, feature selection, normalization choices) to reuse caches.

6. **Interface Sketch (pseudo-code)**
   ```python
   loader = DatasetLoader(
       instruments=["EUR_USD","USD_JPY"],
       base_granularity="M1",
       aux=["M5","H1","D"],
       feature_dir="continuous-trader/data/features",
       raw_dir="continuous-trader/data",
       normalize=True,
       feature_subset=None,
   )
   train, val, test = loader.split_by_dates(train=("2018-01-01","2022-12-31"), ...)
   cv_folds = loader.make_rolling_folds(n_splits=5, val_bars=5000)
   scaler_state = loader.scaler_state  # for export
   feature_names = loader.feature_names
   ```

## Handling Missing Data & Quality Checks
- Drop leading/trailing rows where any instrument lacks base-granularity coverage.
- Track fill ratios per feature; if coverage < threshold (e.g., 90%) raise warning or drop feature.
- Validate monotonic timestamps and absence of duplicates.

## Output Artifacts
- `feature_names.json`: ordered list of columns fed to strategies.
- `scaler_state.json`: mean/std (or quantile) per feature for reuse.
- `dataset_manifest.json`: metadata summarizing instrument list, date range, splits, number of samples.

## Next Steps
1. Implement `DatasetLoader` class under `ga-trailing-20/data/loader.py` using above architecture.
2. Add CLI utility (`ga-trailing-20/tools/build_dataset.py`) to materialize aligned datasets and export scaler stats.
3. Integrate with strategies so each `Strategy.fit()` can request splits or folds from the loader.
