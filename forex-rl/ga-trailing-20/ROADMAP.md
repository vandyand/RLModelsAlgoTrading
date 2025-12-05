# GA Trailing-20 Roadmap

## Context Snapshot
- **Goal:** Offline-optimized multi-instrument entrance policy that relies on per-bar trailing stops for exits.
- **Data:** Minute candles + cross-granularity features already staged under `continuous-trader/data` (raw OHLC + gzipped feature grids).
- **Reference systems:**
  - Actor-Critic stack in `actor-critic/multi20_offline_actor_critic*.py` (dataset builder + AE encoder + policy/value nets).
  - Live trailing trader in `actor-critic/multi20_trailing_trader.py` (vol-aware trailing stop, hysteresis, multi-head policy heads).
  - GA framework in `ga-multi20/*` (genome model, hysteresis mapping, backtester, ballast-style fitness functions).
- **New folder:** `ga-trailing-20/` now contains the roadmap plus a `strategies/` module housing RuleTree GA, Neuro-GA, and Gradient NN scaffolds behind a unified `Strategy` interface with CV + regularization hooks.
- **Guiding principles:** time-series cross-validation, explicit L1/L2/complexity penalties, modular strategies, and shareable trailing-stop simulator.

## Phase 1 — Data + Feature Plumbing
- [ ] Catalog available candle/feature files (compressed + csv) within `continuous-trader/data` and map cross-granularity joins.
- [ ] Implement loader that aligns M1 candles with auxiliary feature stacks (M5/H1/D + exogenous features) and produces train/validation/test splits.
- [ ] Add configurable scaling/normalization stats export for future inference.
- [ ] Build a dimensionality-reduction script (filter + wrapper pipeline) that ranks/outputs optimal feature subsets per instrument.
- [ ] Wire `DatasetLoader` + `tools/build_dataset.py` scaffolding into end-to-end flow once alignment logic is complete.
- [ ] Promote scaler JSON artifacts (`--scaler-json`) to first-class inputs for strategies and live traders.

## Phase 2 — Trailing Stop Simulator (Per-Bar Engine)
- [ ] Port/adapt `compute_trailing_distance`, ATR/pip/tick-vol logic, and hysteresis mapping to a bar-close world.
- [ ] Build deterministic backtest loop that only allows entrance/exit decisions at bar close while continuously walking trailing stops forward intrabar via high/low projections.
- [ ] Encode transaction cost, slippage, and exposure accounting so GA fitness reflects realistic fills.
- [ ] Expose simulator hooks required by `Strategy` protocols (batch predict, CV evaluation, per-fold metrics).
- [ ] Implement API + metrics from `trailing_simulator_spec.md` (SimulatorResult, aggregation helpers, context hooks).
- [ ] Add configurable cost/slippage module per `cost_model_spec.md` (spread tables, slippage modes, financing).

## Phase 3 — Entrance Policy Optimization
- [ ] Implement GA search that reuses `MultiHeadGenome` idea but targets entrance logits only (exits handled by trailing stop module).
- [ ] Implement rule-tree GA (AND/OR feature thresholds) with depth/complexity penalties.
- [ ] Support dual fitness heads: Sharpe-style ballast plus stability metrics (PF, win rate, drawdown, time-in-market).
- [ ] Allow warm-start from actor-critic checkpoints (latent encoder + lightweight head) for hybrid GA/NN sweeps.
- [ ] Add time-series cross-validation driver so GA selection honors multiple folds before finalizing champions.

## Phase 4 — Neural Baseline (Optional)
- [ ] Assemble offline actor-critic + supervised variants that train against the new simulator rewards (per-trade trailing PnL) instead of Sharpe-like per-day reward.
- [ ] Integrate L1/L2 regularization + early stopping based on CV folds.
- [ ] Compare GA vs NN outputs via shared validation backtests; persist top policies + metadata.

## Phase 5 — Deliverables & Ops
- [ ] CLI entrypoints (`backtest.py`, `train_ga.py`, `train_nn.py`) inside `ga-trailing-20/`.
- [ ] Model artifact schema (checkpoint + scaler stats + trailing config) for replay/live hand-off.
- [ ] Documentation: usage examples, config templates, feature-selection workflow, and evaluation report snapshots.

## Data Inventory Snapshot (Nov 28, 2025)
- **Raw OHLC CSVs (`continuous-trader/data`)**
  - Daily bars: AUD_SGD, AUD_USD, CAD_JPY, EUR_HKD, GBP_JPY, GBP_PLN, USD_NOK, USD_SGD, etc.
  - Hourly bars: AUD_USD_H1, EUR_PLN_H1, EUR_SEK_H1, NZD_SGD_H1, SGD_JPY_H1, USD_CNH_H1.
  - M5 bars: CAD_HKD_M5, CHF_HKD_M5, EUR_DKK_M5, EUR_NOK_M5, USD_PLN_M5, plus others as new downloads arrive.
- **Feature exports (`continuous-trader/data/features`)**
  - Daily feature grids (`D/*.csv.gz`) covering ~70 FX crosses with the indicator set from `continuous-trader/features.py`.
  - Hourly export currently limited to `H1/EUR_PLN_H1_features.csv.gz`; more H1/M5 feature dumps need to be generated if we want cross-granularity feature parity.
- **Gaps / follow-ups**
  - No native M1 history on disk—must roll from OANDA or aggregate from M5.
  - Need confirmation on feature column ordering + schema (will be handled when building the loader that reads and aligns these files).

## Decision Log
- _11/28/2025:_ Established roadmap structure and confirmed dependency surfaces (actor-critic + GA + trailing stop modules).
- _11/28/2025:_ Added strategy abstraction (`Strategy` base + rule-tree GA + Neuro-GA + Gradient NN scaffolds), committed to CV + regularization requirements, and captured dimensionality-reduction deliverable.
- _11/28/2025:_ Implemented `simulator/` package scaffolding (trailing config, cost model, engine evaluate loop) aligned with specs.
- _11/28/2025:_ Added `data/loader.py` + `tools/build_dataset.py` scaffolds covering inventory catalog + CLI manifest export.
- _11/28/2025:_ Implemented `GradientNNStrategy` with `tools/train_gradient_nn.py` plus smoke tests; defaults now focus on recent 3–4 years of data.
- _11/28/2025:_ Implemented `NeuroGAStrategy` (MLP genome GA) with `tools/train_neuro_ga.py` and smoke tests over downsampled recent data.
- _11/28/2025:_ Added `baseline_threshold` strategy + `tools/run_baseline.py` for simple end-to-end backtests.
