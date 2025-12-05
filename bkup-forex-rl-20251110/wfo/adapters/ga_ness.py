from __future__ import annotations

from typing import Any, Dict, Tuple
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure ga-ness is importable
_ROOT = Path(__file__).resolve().parents[2]
_GA_DIR = _ROOT / "ga-ness"
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))

from backtest_simple_strategy import load_candles, evaluate_strategy, StrategyParams  # type: ignore
from ga_optimize import GAConfig, run_ga, build_fitness_fn  # type: ignore


class GANessAdapter:
    name = "ga-ness"

    def __init__(
        self,
        *,
        csv: str = None,
        population: int = 30,
        generations: int = 20,
        elite_frac: float = 0.1,
        mutation_prob: float = 0.3,
        crossover_prob: float = 0.8,
        cost_bps: float = 1.0,
        w_sharpe: float = 0.4,
        w_return: float = 0.3,
        w_dd: float = 0.2,
        w_trades: float = 0.1,
        neg_sharpe_penalty: float = 10.0,
        neg_return_penalty: float = 5.0,
    ) -> None:
        self.csv = csv or str(_GA_DIR / "data" / "EUR_USD_M5.csv")
        self.population = int(population)
        self.generations = int(generations)
        self.elite_frac = float(elite_frac)
        self.mutation_prob = float(mutation_prob)
        self.crossover_prob = float(crossover_prob)
        self.cost_bps = float(cost_bps)
        self.w_sharpe = float(w_sharpe)
        self.w_return = float(w_return)
        self.w_dd = float(w_dd)
        self.w_trades = float(w_trades)
        self.neg_sharpe_penalty = float(neg_sharpe_penalty)
        self.neg_return_penalty = float(neg_return_penalty)

    # Adapter API
    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = load_candles(self.csv, start=str(start), end=str(end))
        # For GA simple strategy: use one instrument's close as closes frame
        closes = pd.DataFrame({"EUR_USD": df["close"].astype(float)})
        return df, closes

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame):
        # Fit GA on X; no separate close needed beyond X columns
        cfg = GAConfig(
            population=self.population,
            generations=self.generations,
            elite_frac=self.elite_frac,
            mutation_prob=self.mutation_prob,
            crossover_prob=self.crossover_prob,
            cost_bps=self.cost_bps,
        )
        fitness_fn = build_fitness_fn(self.w_sharpe, self.w_return, self.w_dd, self.w_trades, self.neg_sharpe_penalty, self.neg_return_penalty)
        result = run_ga(X, cfg, df_val=None, fitness_fn=fitness_fn)
        self.best_params = StrategyParams(**result["best_params"])  # type: ignore[index]
        return self.best_params

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        # model is StrategyParams for this adapter
        m = evaluate_strategy(X, model, cost_bps=self.cost_bps)
        # Standardize keys to generic core fields where possible
        return {
            "cum_return": float(m.get("cum_return", 0.0)),
            "sharpe": float(m.get("sharpe", 0.0)),
            "sortino": float("nan"),  # not provided by GA simple eval
            "max_dd": float(m.get("max_drawdown", 0.0)),
            "trades": float(m.get("trades", 0.0)),
            "time_in_mkt": float("nan"),
            "num_days": int((X.index[-1] - X.index[0]).days) if len(X) > 1 else 0,
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
            "win_loss": float("nan"),
            "n_win": int(0),
            "n_loss": int(0),
        }
