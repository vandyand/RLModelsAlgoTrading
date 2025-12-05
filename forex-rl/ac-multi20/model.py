from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class MultiHeadGenome:
    # Hidden layer sizes shared across all heads
    arch: List[int]
    # Per-head weight stacks including final 1-unit output
    # Each head has weights: List[np.ndarray], biases: List[np.ndarray]
    weights: List[List[np.ndarray]]
    biases: List[List[np.ndarray]]
    # Optional learned affine on inputs
    input_scale: np.ndarray | None
    input_bias: np.ndarray | None

    @staticmethod
    def init(input_dim: int, arch: List[int], num_heads: int, use_affine: bool, rng: np.random.Generator | None = None) -> "MultiHeadGenome":
        rng = rng or np.random.default_rng()
        layers: List[int] = list(arch) + [1]
        dims = [input_dim] + layers
        all_weights: List[List[np.ndarray]] = []
        all_biases: List[List[np.ndarray]] = []
        for _ in range(num_heads):
            weights: List[np.ndarray] = []
            biases: List[np.ndarray] = []
            for i in range(len(dims) - 1):
                fan_in = int(dims[i]); fan_out = int(dims[i + 1])
                scale = np.sqrt(2.0 / max(1, fan_in)) if i < len(dims) - 2 else np.sqrt(1.0 / max(1, fan_in))
                W = (rng.standard_normal((fan_in, fan_out)).astype(np.float32)) * float(scale)
                b = np.zeros((fan_out,), dtype=np.float32)
                weights.append(W)
                biases.append(b)
            all_weights.append(weights)
            all_biases.append(biases)
        input_scale = np.ones((input_dim,), dtype=np.float32) if use_affine else None
        input_bias = np.zeros((input_dim,), dtype=np.float32) if use_affine else None
        return MultiHeadGenome(arch=list(arch), weights=all_weights, biases=all_biases, input_scale=input_scale, input_bias=input_bias)

    def clone(self) -> "MultiHeadGenome":
        return MultiHeadGenome(
            arch=list(self.arch),
            weights=[[w.copy() for w in head] for head in self.weights],
            biases=[[b.copy() for b in head] for head in self.biases],
            input_scale=(self.input_scale.copy() if self.input_scale is not None else None),
            input_bias=(self.input_bias.copy() if self.input_bias is not None else None),
        )

    def to_dict(self) -> dict:
        return {
            "arch": list(self.arch),
            "weights": [[w.tolist() for w in head] for head in self.weights],
            "biases": [[b.tolist() for b in head] for head in self.biases],
            "input_scale": (self.input_scale.tolist() if self.input_scale is not None else None),
            "input_bias": (self.input_bias.tolist() if self.input_bias is not None else None),
        }

    @staticmethod
    def from_dict(d: dict) -> "MultiHeadGenome":
        return MultiHeadGenome(
            arch=list(d.get("arch", [])),
            weights=[[np.array(w, dtype=np.float32) for w in head] for head in d.get("weights", [])],
            biases=[[np.array(b, dtype=np.float32) for b in head] for head in d.get("biases", [])],
            input_scale=(np.array(d.get("input_scale"), dtype=np.float32) if d.get("input_scale") is not None else None),
            input_bias=(np.array(d.get("input_bias"), dtype=np.float32) if d.get("input_bias") is not None else None),
        )

    def mutate(self, p_mut: float, w_sigma: float, aff_sigma: float, rng: np.random.Generator | None = None) -> "MultiHeadGenome":
        rng = rng or np.random.default_rng()
        out = self.clone()
        for h in range(len(out.weights)):
            for i in range(len(out.weights[h])):
                if rng.random() < p_mut:
                    noise = rng.standard_normal(out.weights[h][i].shape).astype(np.float32) * float(w_sigma)
                    out.weights[h][i] = out.weights[h][i] + noise
                if rng.random() < p_mut:
                    bnoise = rng.standard_normal(out.biases[h][i].shape).astype(np.float32) * float(w_sigma)
                    out.biases[h][i] = out.biases[h][i] + bnoise
                out.weights[h][i] = np.clip(out.weights[h][i], -5.0, 5.0)
                out.biases[h][i] = np.clip(out.biases[h][i], -5.0, 5.0)
        if out.input_scale is not None and rng.random() < p_mut:
            out.input_scale = out.input_scale + (rng.standard_normal(out.input_scale.shape).astype(np.float32) * float(aff_sigma))
            out.input_scale = np.clip(out.input_scale, 0.05, 20.0)
        if out.input_bias is not None and rng.random() < p_mut:
            out.input_bias = out.input_bias + (rng.standard_normal(out.input_bias.shape).astype(np.float32) * float(aff_sigma))
            out.input_bias = np.clip(out.input_bias, -5.0, 5.0)
        return out

    def crossover(self, other: "MultiHeadGenome", rng: np.random.Generator | None = None) -> "MultiHeadGenome":
        rng = rng or np.random.default_rng()
        child = self.clone()
        # Single-point crossover per layer per head
        for h in range(len(child.weights)):
            for i in range(len(child.weights[h])):
                if rng.random() < 0.5:
                    child.weights[h][i] = other.weights[h][i].copy()
                    child.biases[h][i] = other.biases[h][i].copy()
        if child.input_scale is not None and other.input_scale is not None and rng.random() < 0.5:
            child.input_scale = other.input_scale.copy()
        if child.input_bias is not None and other.input_bias is not None and rng.random() < 0.5:
            child.input_bias = other.input_bias.copy()
        return child


def forward(genome: MultiHeadGenome, X: pd.DataFrame) -> np.ndarray:
    # X: shape (T, F)
    Xt = X.to_numpy(dtype=np.float32, copy=False)
    if genome.input_scale is not None:
        Xt = Xt * genome.input_scale
    if genome.input_bias is not None:
        Xt = Xt + genome.input_bias
    T = Xt.shape[0]
    H = Xt
    outputs: List[np.ndarray] = []
    for h in range(len(genome.weights)):
        Hh = H
        for i in range(len(genome.weights[h])):
            W = genome.weights[h][i]; b = genome.biases[h][i]
            Z = Hh @ W + b
            if i < len(genome.weights[h]) - 1:
                Hh = np.minimum(20.0, np.maximum(0.0, Z))
            else:
                Hh = 0.5 * (1.0 + np.tanh(0.5 * Z))
        outputs.append(Hh.reshape(T))
    # shape (T, num_heads)
    return np.stack(outputs, axis=1)


def hysteresis_map(out_vals: np.ndarray, enter_long: float = 0.7, exit_long: float = 0.6, enter_short: float = 0.3, exit_short: float = 0.4, *, mode: str = "absolute", band_enter: float = 0.05, band_exit: float = 0.02) -> np.ndarray:
    # Vectorized hysteresis mapping across heads; still sequential across time (dependency on state)
    T, H = out_vals.shape
    pos = np.zeros((T, H), dtype=np.float32)
    state = np.zeros((H,), dtype=np.int8)
    if mode == "band":
        up_enter = 0.5 + band_enter
        dn_enter = 0.5 - band_enter
        up_exit = 0.5 + band_exit
        dn_exit = 0.5 - band_exit
        for t in range(T):
            v = out_vals[t, :]
            # flat -> long/short
            to_long = (state == 0) & (v > up_enter)
            to_short = (state == 0) & (v < dn_enter)
            # exits
            exit_long_mask = (state == 1) & (v < up_exit)
            exit_short_mask = (state == -1) & (v > dn_exit)
            state = np.where(to_long, 1, np.where(to_short, -1, state))
            state = np.where(exit_long_mask | exit_short_mask, 0, state)
            pos[t, :] = state
    else:
        for t in range(T):
            v = out_vals[t, :]
            to_long = (state == 0) & (v > enter_long)
            to_short = (state == 0) & (v < enter_short)
            exit_long_mask = (state == 1) & (v < exit_long)
            exit_short_mask = (state == -1) & (v > exit_short)
            state = np.where(to_long, 1, np.where(to_short, -1, state))
            state = np.where(exit_long_mask | exit_short_mask, 0, state)
            pos[t, :] = state
    # shift by 1 bar
    pos = np.vstack([np.zeros((1, H), dtype=np.float32), pos[:-1, :]])
    return pos
