"""Direct GA-optimized multi-instrument policy.

This strategy defines a small feed-forward network mapping the current
feature vector into per-instrument:

    [gate_logit, dir_raw, pos_raw, trail_raw]

for each instrument. The raw outputs are then transformed into:

    - enter_flags in [0,1] via sigmoid(gate_logit)
    - direction scores in [-1,1] via tanh(dir_raw)
    - position size fraction in [0,1] via sigmoid(pos_raw)
    - trailing-distance fraction in [0,1] via sigmoid(trail_raw)

The simulator interprets these as:

    preds = [enter_flags, dir_scores, pos_fracs, trail_fracs]  (length 4N)

and applies its existing logic to:

    - skip entries when enter_flag < 0.5
    - map dir_scores to {-1,0,+1} via thresholds (0.5, -0.5)
    - scale units by pos_frac
    - map trail_frac into [min_pips, max_pips] for dynamic trailing stops

Network weights are optimized via a genetic algorithm directly on
backtest fitness using TrailingStopSimulator.evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import (
    CrossValidationConfig,
    DatasetSplit,
    RegularizationConfig,
    Strategy,
    StrategyArtifact,
    TrailingStopSimulator,
)


@dataclass
class GAPolicyConfig:
    """Hyper-parameters for GA-optimized multi-instrument policy."""

    population: int = 32
    generations: int = 20
    hidden_dims: Tuple[int, ...] = (64,)
    mutation_prob: float = 0.2
    weight_sigma: float = 0.05
    bias_sigma: float = 0.02
    crossover_frac: float = 0.3
    elite_frac: float = 0.1

    # Fitness shaping: legacy parameter no longer used directly in the current
    # asymmetric PnL-based score, but kept for potential future experiments.
    lambda_tim: float = 2.0


class GAPolicyStrategy(Strategy):
    """Direct GA search over a compact multi-instrument entrance policy."""

    def __init__(
        self,
        instruments: Sequence[str],
        config: Optional[GAPolicyConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or GAPolicyConfig()
        self.instruments: Tuple[str, ...] = tuple(instruments)
        self.num_instruments: int = len(self.instruments)
        if self.num_instruments <= 0:
            raise ValueError("GAPolicyStrategy requires at least one instrument")

        self._input_dim: Optional[int] = None
        self._genome: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit],
        simulator: TrailingStopSimulator,
    ) -> None:
        # Materialize records once for GA fitness evaluations.
        train_records = list(train_split)
        if len(train_records) < 2:
            return

        # Infer input dimension and basic sanity from the first record.
        first = train_records[0]
        feats0 = np.asarray(first.get("features"), dtype=np.float32)
        if feats0.ndim != 1 or feats0.size == 0:
            raise ValueError("Training records missing 1D 'features' vector")
        bars0 = first.get("bars")
        if not bars0:
            raise ValueError("Training records missing 'bars'")
        if len(bars0) != self.num_instruments:
            # We rely on a fixed instrument order; warn if mismatch.
            raise ValueError(f"GAPolicyStrategy expected {self.num_instruments} instruments, got {len(bars0)}")

        self._input_dim = int(feats0.shape[0])

        # Wrap records so simulator can iterate multiple times per genome.
        class FixedSplit:
            def __iter__(self_inner):  # pragma: no cover - trivial
                return iter(train_records)

        eval_split: DatasetSplit = FixedSplit()

        cfg = self.config
        pop_size = int(cfg.population)
        elite_k = max(1, int(round(cfg.elite_frac * pop_size)))
        cross_k = max(0, int(round(cfg.crossover_frac * pop_size)))

        population: List[Dict[str, Any]] = [self._init_genome(self._input_dim) for _ in range(pop_size)]
        best_genome: Optional[Dict[str, Any]] = None
        best_score = -1e30

        for _gen in range(cfg.generations):
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for g in population:
                self._genome = g
                result = simulator.evaluate(self, eval_split, record_equity=False, return_trades=True)

                # Fitness: use gross profit (sum of positive PnL across all
                # trades) as the score. This deliberately ignores losses so
                # that GA will strongly prefer policies that occasionally make
                # large gains, even if net PnL is not yet positive. The goal
                # here is purely to test whether GA can discover any policy
                # that accumulates significant positive-side PnL.
                pnls = [float(t.metadata.get("pnl", 0.0)) for t in result.trades]
                total_pos = sum(p for p in pnls if p > 0.0)
                score = total_pos

                # Optional regularization penalties.
                if self.regularization.l2 > 0.0:
                    l2 = self._genome_l2(g)
                    score -= float(self.regularization.l2) * l2
                if self.regularization.complexity_penalty > 0.0:
                    score -= float(self.regularization.complexity_penalty) * self._param_count(g)

                scored.append((score, g))

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > best_score:
                best_score = scored[0][0]
                best_genome = scored[0][1]

            elites = [g for _, g in scored[:elite_k]]
            new_pop: List[Dict[str, Any]] = elites.copy()

            # Crossovers
            while len(new_pop) < elite_k + cross_k:
                a, b = np.random.choice(len(elites), size=2, replace=True)
                child = self._crossover(elites[a], elites[b])
                new_pop.append(self._mutate(child))

            # Mutated elites / randoms
            while len(new_pop) < pop_size:
                base = elites[len(new_pop) % elite_k]
                new_pop.append(self._mutate(base))

            population = new_pop

        if best_genome is not None:
            self._genome = best_genome

    def cross_validate(
        self,
        data: DatasetSplit,
        simulator: TrailingStopSimulator,
    ) -> Dict[str, Any]:
        result = simulator.evaluate(self, data)
        return {"metrics": getattr(result, "metrics", None)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._genome is None or self._input_dim is None:
            return np.zeros(self.num_instruments * 4, dtype=np.float32)
        if features.ndim == 1:
            x = features.astype(np.float32)
        else:
            x = features[0].astype(np.float32)
        if x.size != self._input_dim:
            # Basic guard against feature-mismatch.
            x = np.resize(x, self._input_dim).astype(np.float32)

        raw = self._forward(self._genome, x)  # shape [4 * num_instruments]
        N = self.num_instruments
        if raw.size != 4 * N:
            raw = np.resize(raw, 4 * N)

        gate_logits = raw[0:N]
        dir_raw = raw[N : 2 * N]
        pos_raw = raw[2 * N : 3 * N]
        trail_raw = raw[3 * N : 4 * N]

        enter_flags = 1.0 / (1.0 + np.exp(-gate_logits))  # [0,1]
        dir_scores = np.tanh(dir_raw)  # [-1,1]
        pos_fracs = 1.0 / (1.0 + np.exp(-pos_raw))  # [0,1]
        trail_fracs = 1.0 / (1.0 + np.exp(-trail_raw))  # [0,1]

        return np.concatenate(
            [
                enter_flags.astype(np.float32),
                dir_scores.astype(np.float32),
                pos_fracs.astype(np.float32),
                trail_fracs.astype(np.float32),
            ],
            axis=0,
        )

    def save(self, path: str) -> StrategyArtifact:
        if self._genome is None or self._input_dim is None:
            raise RuntimeError("Genome not trained; cannot save")
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "input_dim": self._input_dim,
            "config": {
                "population": self.config.population,
                "generations": self.config.generations,
                "hidden_dims": list(self.config.hidden_dims),
                "mutation_prob": self.config.mutation_prob,
                "weight_sigma": self.config.weight_sigma,
                "bias_sigma": self.config.bias_sigma,
                "crossover_frac": self.config.crossover_frac,
                "elite_frac": self.config.elite_frac,
                "lambda_tim": self.config.lambda_tim,
            },
            "genome": {
                "weights": [w.tolist() for w in self._genome["weights"]],
                "biases": [b.tolist() for b in self._genome["biases"]],
            },
        }
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        art = StrategyArtifact(
            name="ga_policy",
            version="v0",
            feature_names=None,
            scaler_state=None,
            extra={"path": path},
        )
        self.set_artifact(art)
        return art

    @classmethod
    def load(cls, path: str) -> "GAPolicyStrategy":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cfg_dict = payload.get("config") or {}
        cfg = GAPolicyConfig(
            population=int(cfg_dict.get("population", 32)),
            generations=int(cfg_dict.get("generations", 20)),
            hidden_dims=tuple(cfg_dict.get("hidden_dims", (64,))),
            mutation_prob=float(cfg_dict.get("mutation_prob", 0.2)),
            weight_sigma=float(cfg_dict.get("weight_sigma", 0.05)),
            bias_sigma=float(cfg_dict.get("bias_sigma", 0.02)),
            crossover_frac=float(cfg_dict.get("crossover_frac", 0.3)),
            elite_frac=float(cfg_dict.get("elite_frac", 0.1)),
            lambda_tim=float(cfg_dict.get("lambda_tim", 2.0)),
        )
        # Instruments are not serialized here; caller must know them or we
        # could extend the payload to include them if needed.
        strat = cls(instruments=[], config=cfg)  # type: ignore[arg-type]
        strat._input_dim = int(payload.get("input_dim"))
        gw = payload.get("genome", {})
        weights = [np.array(w, dtype=np.float32) for w in gw.get("weights", [])]
        biases = [np.array(b, dtype=np.float32) for b in gw.get("biases", [])]
        strat._genome = {"weights": weights, "biases": biases}
        return strat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_genome(self, input_dim: int) -> Dict[str, Any]:
        # One small MLP: input_dim -> hidden_dims... -> 4 * num_instruments
        layers = [input_dim] + list(self.config.hidden_dims) + [4 * self.num_instruments]
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            scale = np.sqrt(2.0 / max(1, fan_in))
            W = np.random.randn(fan_in, fan_out).astype(np.float32) * float(scale)
            b = np.zeros((fan_out,), dtype=np.float32)
            weights.append(W)
            biases.append(b)
        return {"weights": weights, "biases": biases}

    def _mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        new_g = {
            "weights": [w.copy() for w in genome["weights"]],
            "biases": [b.copy() for b in genome["biases"]],
        }
        for i in range(len(new_g["weights"])):
            if np.random.rand() < cfg.mutation_prob:
                noise_w = np.random.randn(*new_g["weights"][i].shape).astype(np.float32) * float(cfg.weight_sigma)
                new_g["weights"][i] += noise_w
            if np.random.rand() < cfg.mutation_prob:
                noise_b = np.random.randn(*new_g["biases"][i].shape).astype(np.float32) * float(cfg.bias_sigma)
                new_g["biases"][i] += noise_b
        return new_g

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child_w: List[np.ndarray] = []
        child_b: List[np.ndarray] = []
        for wa, wb, ba, bb in zip(a["weights"], b["weights"], a["biases"], b["biases"]):
            mask = np.random.rand(*wa.shape) < 0.5
            cw = np.where(mask, wa, wb)
            mask_b = np.random.rand(*ba.shape) < 0.5
            cb = np.where(mask_b, ba, bb)
            child_w.append(cw.astype(np.float32))
            child_b.append(cb.astype(np.float32))
        return {"weights": child_w, "biases": child_b}

    def _forward(self, genome: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(zip(genome["weights"], genome["biases"])):
            h = h @ W + b
            if i < len(genome["weights"]) - 1:
                h = np.tanh(h)
        return h.astype(np.float32)

    def _param_count(self, genome: Dict[str, Any]) -> int:
        total = 0
        for W, b in zip(genome["weights"], genome["biases"]):
            total += int(W.size + b.size)
        return total

    def _genome_l2(self, genome: Dict[str, Any]) -> float:
        acc = 0.0
        for W, b in zip(genome["weights"], genome["biases"]):
            acc += float(np.sum(W * W) + np.sum(b * b))
        return acc

