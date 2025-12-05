# Transaction Cost & Slippage Model

## Goals
- Capture realistic fills for per-bar simulations without requiring tick data.
- Parameterize spreads, commissions, and slippage so strategies can be evaluated under different market conditions.

## Components

### 1. Spread Model
- Base spread per instrument (in pips) derived from historical OANDA data or config defaults.
- Optionally dynamic: use rolling average of (ask - bid) from available intraday data; fallback to static table.
- Spread is paid on both entry and exit: cost = spread * pip_value * units.

### 2. Commission / Financing
- Flat commission per million traded (configurable per instrument).
- Overnight financing/rollover modeled as daily rate applied when holding across day boundaries (optional).

### 3. Slippage Model
- **Deterministic:** slippage = `slippage_perc * price` applied on entry/exit.
- **Volatility-aware:** slippage = `k * ATR_m1` or proportional to bar range (high - low).
- **Stochastic option:** draw from normal or Laplace distribution with zero mean, std tied to volatility; clamp to ±N pips.

### 4. Stop Execution Rules
- When trailing stop is hit within bar:
  - Determine hit price using bar extremes (for long: min(trail, low); short: max(trail, high)).
  - Apply slippage (negative) to simulate adverse execution.
  - If bar gap occurs (open < trail), use bar open ± slippage to respect gap risk.

### 5. CostModel API
```python
@dataclass
class CostModelConfig:
    spread_mode: str = "static"
    spread_table: dict[str, float] = field(default_factory=dict)
    commission_per_million: float = 0.0
    slippage_mode: str = "deterministic"  # or "vol", "stochastic"
    slippage_params: dict[str, float] = field(default_factory=dict)
    financing_rate_bps: float = 0.0

class CostModel:
    def __init__(self, cfg: CostModelConfig): ...
    def entry_cost(self, instrument: str, price: float, units: float, bar_info) -> float: ...
    def exit_cost(self, instrument: str, price: float, units: float, bar_info) -> float: ...
    def apply_slippage(self, instrument: str, intended_price: float, direction: int, bar_info) -> float: ...
```
- `bar_info` includes ATR, high/low, timestamp, etc., enabling volatility-aware logic.

### 6. Integration with Simulator
- Simulator requests adjusted fill price = `intended_price + slippage`.
- Costs deducted from trade PnL immediately; financing applied at day roll.
- CostModel also reports per-trade cost components for diagnostics.

### 7. Configuration Files
- Store default spread/commission tables under `ga-trailing-20/config/cost_models/*.json`.
- Allow CLI overrides (e.g., `--spread-multiplier 1.5` for stress testing).

### 8. Testing
- Unit tests validating deterministic, volatility-based, and stochastic slippage behavior.
- Scenario tests: ensure cost model produces expected net PnL when spread/commission set to zero vs non-zero.
