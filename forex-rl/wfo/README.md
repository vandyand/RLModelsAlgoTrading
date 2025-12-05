## Generic Walk-Forward Optimization (WFO)

A lightweight framework to run walk-forward optimization across different trading systems using simple adapter interfaces.

### Key ideas
- **Adapters**: Small classes that bridge your strategy to the WFO core. Implement 3 methods and you’re done.
- **Windows**: The core slices time into [train → validate] windows and iterates forward.
- **Metrics**: Your adapter returns metrics; the core writes `windows.jsonl` per window (plus helpful basis-point fields).

---

## Quick start

- **AC Multi-20 (Actor-Critic) example**
```bash
python -m wfo.cli --adapter ac-multi20 \
  --start 2023-01-01 --end 2023-03-01 \
  --train-n 0.5 --val-n 0.5 --step-n 0.5 --unit months \
  --grans H1,D --epochs 2 --hidden 64 --reward-scale 10000
```

- **GA-Ness (simple GA) example**
```bash
python -m wfo.cli --adapter ga-ness \
  --start 2023-01-01 --end 2023-02-01 \
  --train-n 2 --val-n 1 --step-n 1 --unit weeks
```

Outputs appear in `wfo/runs/<adapter>-wfo-<timestamp>/`:
- `meta.json` — run metadata (window count, args)
- `windows.jsonl` — one JSON object per window with metrics

---

## CLI overview

- **Global windowing**
  - `--start`, `--end`: UTC timestamps (YYYY-MM-DD)
  - `--train-n`, `--val-n`, `--step-n`: window sizes as floats
  - `--unit`: one of `months|weeks|days|steps`
  - `--windows-limit`: cap windows for quick runs
  - `--base-gran`: used only when `--unit steps` (M5|H1|D)
  - `--out-dir`: base output directory (default `wfo/runs`)

- **Adapter selection**
  - `--adapter ac-multi20`
  - `--adapter ga-ness`

- **Adapter-specific options**
  - AC: accepts many passthroughs (e.g., `--grans`, `--epochs`, `--hidden`, thresholds, etc.)
  - GA-Ness: currently uses defaults baked into the adapter; customize by editing `wfo/cli.py` or instantiating the adapter from Python.

---

## Windowing behavior
- A window is defined as:
  - Train: `[train_start, train_end]`
  - Validation: `(train_end, val_end]` (starts one second after train end)
- The framework advances `train_start` by `step-n` in the chosen `--unit` until the validation end exceeds `--end`.
- Floating sizes are supported (e.g., `0.5 months`, `2.5 weeks`).
- `--unit steps` uses the core’s `--base-gran` to define a step as one bar of that granularity (M5/H1/D).

---

## Metrics and logging
- Each `validate(...)` returns a metrics dict; the core writes it verbatim into `windows.jsonl` and also augments with:
  - `cum_return_bp = cum_return * 1e4` (basis points)
  - `max_dd_bp = max_dd * 1e4` (basis points; typically negative)
- Typical fields (when available):
  - `cum_return`, `sharpe`, `sortino`, `max_dd`, `trades`, `time_in_mkt`, `win_rate`, `win_loss`, `profit_factor`, `equity_r2`, etc.
- Nulls (NaN/Inf) are serialized as `null` for JSON consumers.

Example `windows.jsonl` rows:
```json
{"event":"window","win":1,"cum_return":0.0123,"sharpe":1.1,"max_dd":-0.034,"trades":42,"cum_return_bp":123.0,"max_dd_bp":-340.0}
{"event":"window","win":2,"cum_return":-0.001,"sharpe":-0.3,"max_dd":-0.010,"trades":5,"cum_return_bp":-10.0,"max_dd_bp":-100.0}
```

---

## Current adapters

- **AC Multi-20 (`ac-multi20`)**
  - Reuses existing components in `ac-multi20/` (feature loader, model, backtester)
  - Adapter: `wfo/adapters/ac_multi20.py`
  - Useful options:
    - `--grans`: e.g., `M5,H1,D` (default) or `H1,D` for hourly base step
    - `--epochs`, `--hidden`, `--lr`, `--reward-scale`
    - Thresholds: `--enter-long`, `--exit-long`, `--enter-short`, `--exit-short`
    - `--carry-forward`, `--init-model`
  - Expects features under `continuous-trader/data` (same as your AC scripts)

- **GA-Ness (`ga-ness`)**
  - Wraps `ga-ness/backtest_simple_strategy.py` and `ga-ness/ga_optimize.py`
  - Adapter: `wfo/adapters/ga_ness.py`
  - Uses `ga-ness/data/EUR_USD_M5.csv` by default
  - GA hyperparameters currently set in the adapter; extend `wfo/cli.py` to expose them if needed

---

## Writing a new adapter (simple guide)
Create `wfo/adapters/my_strategy.py`:

```python
from __future__ import annotations
from typing import Tuple, Dict, Any
import pandas as pd

class MyStrategyAdapter:
    name = "my-strategy"

    def __init__(self, **kwargs) -> None:
        # parse/store any config
        pass

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 1) Load historical data/features for [start,end]
        # 2) Return (X_panel, closes)
        #    - X_panel: features per timestamp (DataFrame)
        #    - closes: prices aligned to X_panel.index (DataFrame or Series → DataFrame)
        ...

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        # Train and return a model object (any Python object you need later)
        ...

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        # Evaluate on validation data and return metrics dict
        # Include at least: cum_return, sharpe, max_dd, trades (when possible)
        ...
```

Then wire it in `wfo/cli.py`:
```python
elif args.adapter == "my-strategy":
    from .adapters.my_strategy import MyStrategyAdapter
    adapter = MyStrategyAdapter(param1=..., param2=...)
```

Run:
```bash
python -m wfo.cli --adapter my-strategy --start 2023-01-01 --end 2023-06-01 --train-n 1 --val-n 1 --step-n 1 --unit months
```

### What the three methods mean (plain language)
- **load_window(start, end)**: “Give me inputs and prices for this time range.”
- **fit(X, closes)**: “Learn the strategy from these inputs (training).”
- **validate(model, X, closes)**: “Test the learned strategy on new data and score it.”

Tips:
- Keep `X` and `closes` aligned by timestamp (same index).
- Return floats; if you do not compute certain metrics, set them to NaN.
- The core adds basis-point versions of `cum_return` and `max_dd` automatically.

---

## Consuming results
- Read `windows.jsonl` to analyze per-window behavior.
- Aggregate, plot, or compute stability metrics across windows.
- For AC runs, you’ll also find per-window checkpoints under `runs/.../checkpoints/` when enabled.

---

## Troubleshooting
- Empty windows: ensure your data covers the requested ranges and timezones are UTC.
- Zero trades or zeros in metrics: may indicate thresholds too strict or no signals in the window.
- Large or tiny values: use the `_bp` fields for readability.

---

## Extending
- Expose more adapter-specific CLI args in `wfo/cli.py` as needed.
- Add new standardized metrics to your adapter (e.g., `equity_r2`, `win_rate`).
- Implement a positions-returning path if you want richer trade-level statistics downstream.
