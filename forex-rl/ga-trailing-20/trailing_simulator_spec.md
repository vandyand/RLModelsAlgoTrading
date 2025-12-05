# Trailing Stop Simulator Specification

## Goals
- Replay historical candles at M1 granularity, making exactly one decision per bar close.
- Apply user-defined trailing stop logic (ATR/pip/tick-vol variants) to manage exits autonomously between bar closes.
- Provide deterministic, vectorized evaluation for strategies implementing the shared `Strategy` interface.

## Core Concepts
1. **State Timeline**
   - Each bar `t` provides OHLCV for every instrument plus aligned auxiliary features.
   - Strategy receives feature tensor at bar close and emits entrance logits/actions.
   - Positions are entered/updated only at bar close; trailing stops are advanced immediately after entry and after each new bar.

2. **Trailing Stop Engine**
   - Implements modes: `atr`, `pip`, `tick_vol` (reused from live trader but adapted to bar-level data).
   - For an open trade:
     - Maintain `trail_price` and `max_favorable_price` (for longs)/`min_favorable_price` (shorts).
     - Update using bar highs/lows to approximate intrabar movement; e.g., for long positions, use bar high to extend trailing stop if high exceeds previous max.
     - If trailing stop crosses bar low (long) or bar high (short), mark exit at stop price; exit is effective for next bar open (to avoid lookahead) or at stop price depending on simulation strictness.

3. **Decision Flow per Bar**
   1. Advance trailing stop using current bar high/low.
   2. Detect stop hits; close trades accordingly, logging exit reason `TRAIL_HIT`.
   3. Provide strategy with current features (including flags about open positions, trailing distance, etc.).
   4. Strategy outputs entrance signals (e.g., logits → discrete {-1,0,1}).
   5. Apply hysteresis mapping and risk constraints (max positions per instrument, net exposure limit).
   6. When entering, compute trailing distance via configured mode and set stop level (respecting min distance, quantization, etc.).

4. **Targets & Rewards**
   - Primary reward: trade-level normalized PnL (same formula as live trailing trader) or per-bar contributions for GA fitness.
   - Provide optional per-bar metrics (returns, drawdown, exposure) for gradient-based strategies.

## Implementation Outline
- Module: `ga-trailing-20/simulator/engine.py`
- Classes:
  - `TrailingConfig`: dataclass capturing ATS params (mode, ATR period/mult, pip distance, tick-vol mult, min distance/step, slippage assumptions).
  - `InstrumentState`: holds per-instrument position info (direction, units, entry price, trailing distance, trail price, open timestamp).
  - `SimulatorResult`: metrics, trade list, equity curve, diagnostics.
- Functions:
  - `simulate(strategy, dataset, config, cost_model) -> SimulatorResult`
  - `advance_trailing(state, bar, config)` -> updates trail & returns exit signal.

## API Details
```python
class TrailingStopSimulator:
    def __init__(self, trailing_cfg: TrailingConfig, cost_model: CostModel) -> None: ...

    def evaluate(
        self,
        strategy: Strategy,
        split: DatasetSplit,
        *,
        record_equity: bool = True,
        return_trades: bool = True,
    ) -> SimulatorResult:
        ...
```
- `DatasetSplit` supplies an iterator of `(features, bar_data)` where `bar_data` includes OHLCV + timestamp per instrument.
- `SimulatorResult` fields:
  - `metrics`: `PortfolioMetrics` dataclass (cum_return, sharpe, sortino, max_drawdown, profit_factor, win_rate, avg_trade_duration, time_in_market, exposure, avg_trailing_distance, etc.).
  - `trades`: list of trade dicts (`entry_time`, `exit_time`, `direction`, `entry_price`, `exit_price`, `trail_mode`, `stop_distance`, `reason`).
  - `equity_curve`: optional numpy array of cumulative equity per bar.
  - `per_bar_stats`: optional structure for debugging (returns, exposure, trailing adjustments).
- Simulator exposes `reset()` to clear state between runs and `set_context_hooks()` so strategies can consume custom contextual features.

### Metrics Aggregation
- Provide helper `aggregate_results(results: List[SimulatorResult]) -> Dict[str, float]` for CV folds.
- Support JSON-serializable summary with percentile stats across instruments and folds.

## Metrics Captured
- PnL stats: cumulative return, Sharpe, Sortino, max drawdown, profit factor, win rate, average trade duration.
- Exposure stats: time in market, gross leverage.
- Operational stats: number of entries, trailing adjustments, stop hits.
- Support per-fold aggregation for cross-validation.

## Hooks for Strategies
- Strategy can request `context` dictionary containing:
  - `current_trail_distance`
  - `time_since_entry`
  - `nav` proxy, etc.
- Simulator exposes `evaluate(strategy, split)` method (matching `TrailingStopSimulator` protocol) returning metrics + trade logs.

## Next Steps
1. Formalize simulator API + metrics (todo `todo_sim_arch`).
2. Design transaction cost & slippage model (todo `todo_cost_model`).
3. Implement `simulator/engine.py` with unit tests.
