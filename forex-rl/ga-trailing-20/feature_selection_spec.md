# Feature Selection & Dimensionality Reduction Plan

## Objectives
- Prioritize features that offer predictive signal for per-bar entrance decisions while minimizing redundancy.
- Provide a ranked feature list (with scores + metadata) that can be consumed by DatasetLoader and strategies.
- Operate on time-series data without leaking future information.

## Pipeline Overview
1. **Pre-filtering**
   - Remove constant/low-variance features (variance < threshold, e.g., 1e-6).
   - Drop features with excessive missingness (>10% NaNs after alignment).

2. **Correlation Pruning**
   - Compute rolling Pearson correlation (e.g., 90-day window) between every pair; drop one feature from pairs with |corr| > 0.97.
   - Optionally use distance correlation or HSIC for non-linear dependencies.

3. **Univariate Ranking**
   - For each feature, compute rolling mutual information / distance correlation with next-bar returns (per instrument).
   - Aggregate scores across folds (mean + std) to capture stability.

4. **Wrapper Stage (Greedy Selection)**
   - Use a fast proxy model (e.g., LightGBM or L1-regularized logistic regression) trained on rolling folds.
   - Calculate permutation importance / SHAP values; select top-k features per instrument.
   - Iteratively add features to a candidate set while monitoring validation metric (Sharpe/PF). Stop when marginal gain < epsilon.

5. **Backtester Validation**
   - Feed candidate feature subsets into a simplified strategy (e.g., hysteresis baseline) to ensure performance gains translate to simulator metrics.

6. **Output Artifacts**
   - `feature_ranking.json`: list of features with scores, selection stage info, coverage stats.
   - `selected_features.txt`: final ordered set to pass to DatasetLoader (can be instrument-specific or global).
   - Logs/plots summarizing MI trends, correlation heatmaps (optional).

## Script Layout
- File: `ga-trailing-20/tools/feature_select.py`
  - CLI arguments: instruments, date range, scoring metric, max_features, output dir.
  - Steps: load aligned dataset → run pipeline → emit artifacts.

## Integration
- DatasetLoader accepts `feature_subset` argument referencing `selected_features.txt`.
- Strategies log which feature ranking version they used via manifest metadata.

## Future Extensions
- Consider autoencoder-based embeddings as an optional dimensionality reduction path (trained offline, used as additional features).
- Support incremental updates (only recompute rankings for new data segments).
