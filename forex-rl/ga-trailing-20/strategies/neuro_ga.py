"""Genetic algorithm over dense-network weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
class NeuroGAConfig:
    """Hyper-parameters for weight-evolving GA."""

    population: int = 64
    generations: int = 40
    hidden_layers: tuple[int, ...] = (256, 128)
    mutation_prob: float = 0.15
    weight_sigma: float = 0.05
    bias_sigma: float = 0.02
    crossover_frac: float = 0.3
    elite_frac: float = 0.1


class NeuroGAStrategy(Strategy):
    """Optimizes neural weights via genetic search."""

    def __init__(
        self,
        config: Optional[NeuroGAConfig] = None,
        *,
        regularization: Optional[RegularizationConfig] = None,
        cv: Optional[CrossValidationConfig] = None,
    ) -> None:
        super().__init__(regularization=regularization, cv=cv)
        self.config = config or NeuroGAConfig()
        self._genome: Optional[Dict[str, Any]] = None  # simple MLP genome
        self._input_dim: Optional[int] = None

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
        X, _ = self._build_xy(train_records)
        self._input_dim = X.shape[1]

        # Wrap records so simulator can iterate multiple times.
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

        for gen in range(cfg.generations):
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for g in population:
                self._genome = g
                result = simulator.evaluate(self, eval_split, record_equity=False, return_trades=False)
                metrics = getattr(result, "metrics", None)
                sharpe = float(getattr(metrics, "sharpe", 0.0) if metrics is not None else 0.0)
                cum_ret = float(getattr(metrics, "cum_return", 0.0) if metrics is not None else 0.0)
                score = sharpe + 0.1 * cum_ret
                # Regularization penalties
                if self.regularization.l2 > 0.0:
                    l2 = self._genome_l2(g)
                    score -= float(self.regularization.l2) * l2
                if self.regularization.complexity_penalty > 0.0:
                    score -= float(self.regularization.complexity_penalty) * self._param_count(g)
                scored.append((score, g))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_score:
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
        if self._genome is None:
            return np.zeros(1, dtype=float)
        if features.ndim == 1:
            x = features.astype(np.float32)
        else:
            x = features[0].astype(np.float32)
        out = self._forward(self._genome, x)
        return np.array([float(out)], dtype=float)

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
                "hidden_layers": list(self.config.hidden_layers),
                "mutation_prob": self.config.mutation_prob,
                "weight_sigma": self.config.weight_sigma,
                "bias_sigma": self.config.bias_sigma,
                "crossover_frac": self.config.crossover_frac,
                "elite_frac": self.config.elite_frac,
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
            name="neuro_ga",
            version="v0",
            feature_names=None,
            scaler_state=None,
            extra={"path": path},
        )
        self.set_artifact(art)
        return art

    @classmethod
    def load(cls, path: str) -> "NeuroGAStrategy":
        import json
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cfg_dict = payload.get("config") or {}
        cfg = NeuroGAConfig(
            population=int(cfg_dict.get("population", 64)),
            generations=int(cfg_dict.get("generations", 40)),
            hidden_layers=tuple(cfg_dict.get("hidden_layers", (256, 128))),
            mutation_prob=float(cfg_dict.get("mutation_prob", 0.15)),
            weight_sigma=float(cfg_dict.get("weight_sigma", 0.05)),
            bias_sigma=float(cfg_dict.get("bias_sigma", 0.02)),
            crossover_frac=float(cfg_dict.get("crossover_frac", 0.3)),
            elite_frac=float(cfg_dict.get("elite_frac", 0.1)),
        )
        strat = cls(config=cfg)
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
        layers = [input_dim] + list(self.config.hidden_layers) + [1]
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

    def _forward(self, genome: Dict[str, Any], x: np.ndarray) -> float:
        h = x
        for i, (W, b) in enumerate(zip(genome["weights"], genome["biases"])):
            h = h @ W + b
            if i < len(genome["weights"]) - 1:
                h = np.tanh(h)
        return float(h.reshape(-1)[0])

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

    def _build_xy(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        for i in range(len(records) - 1):
            rec_t = records[i]
            rec_tp1 = records[i + 1]
            feats = np.asarray(rec_t["features"], dtype=np.float32)
            bars_t = rec_t["bars"]
            bars_tp1 = rec_tp1["bars"]
            if not bars_t:
                continue
            first_inst = sorted(bars_t.keys())[0]
            close_t = float(bars_t[first_inst]["close"])
            close_tp1 = float(bars_tp1[first_inst]["close"])
            if close_t <= 0.0:
                ret = 0.0
            else:
                ret = (close_tp1 / close_t) - 1.0
            X_list.append(feats)
            y_list.append(float(ret))
        if not X_list:
            raise ValueError("Not enough records to infer input dimension")
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.float32)
        return X, y
