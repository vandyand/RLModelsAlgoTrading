#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd

from multi_features import build_features, INSTRUMENTS


# ---- GP nodes ----
@dataclass
class Node:
    pass

@dataclass
class Feature(Node):
    name: str  # column in X

@dataclass
class Threshold(Node):
    value: float

@dataclass
class UnaryOp(Node):
    op: str  # 'NOT'
    child: Node

@dataclass
class BinaryOp(Node):
    op: str  # 'AND','OR','>','<'
    left: Node
    right: Node


@dataclass
class ConstBool(Node):
    value: bool


def node_size(n: Node) -> int:
    if isinstance(n, (Feature, Threshold)):
        return 1
    if isinstance(n, UnaryOp):
        return 1 + node_size(n.child)
    if isinstance(n, BinaryOp):
        return 1 + node_size(n.left) + node_size(n.right)
    return 1


def node_depth(n: Node) -> int:
    if isinstance(n, (Feature, Threshold)):
        return 1
    if isinstance(n, UnaryOp):
        return 1 + node_depth(n.child)
    if isinstance(n, BinaryOp):
        return 1 + max(node_depth(n.left), node_depth(n.right))
    return 1


def pretty(n: Node) -> str:
    if isinstance(n, Feature):
        return n.name
    if isinstance(n, Threshold):
        return f"{n.value:.4f}"
    if isinstance(n, UnaryOp):
        return f"({n.op} {pretty(n.child)})"
    if isinstance(n, BinaryOp):
        return f"({pretty(n.left)} {n.op} {pretty(n.right)})"
    if isinstance(n, ConstBool):
        return "TRUE" if n.value else "FALSE"
    return "?"


def clone_node(n: Optional[Node]) -> Optional[Node]:
    if n is None:
        return None
    if isinstance(n, Feature):
        return Feature(name=n.name)
    if isinstance(n, Threshold):
        return Threshold(value=float(n.value))
    if isinstance(n, UnaryOp):
        return UnaryOp(op=n.op, child=clone_node(n.child))
    if isinstance(n, BinaryOp):
        return BinaryOp(op=n.op, left=clone_node(n.left), right=clone_node(n.right))
    return None


def clone_individual(ind: Tuple[Node, Node, Optional[Node], Optional[Node]]) -> Tuple[Node, Node, Optional[Node], Optional[Node]]:
    return (
        clone_node(ind[0]),
        clone_node(ind[1]),
        clone_node(ind[2]),
        clone_node(ind[3]),
    )  # type: ignore[return-value]


# ---- Node (de)serialization ----
def node_to_dict(n: Optional[Node]) -> Optional[Dict[str, Any]]:
    if n is None:
        return None
    if isinstance(n, Feature):
        return {"type": "Feature", "name": n.name}
    if isinstance(n, Threshold):
        return {"type": "Threshold", "value": n.value}
    if isinstance(n, UnaryOp):
        return {"type": "UnaryOp", "op": n.op, "child": node_to_dict(n.child)}
    if isinstance(n, BinaryOp):
        return {"type": "BinaryOp", "op": n.op, "left": node_to_dict(n.left), "right": node_to_dict(n.right)}
    if isinstance(n, ConstBool):
        return {"type": "ConstBool", "value": bool(n.value)}
    return None


def node_from_dict(d: Optional[Dict[str, Any]]) -> Optional[Node]:
    if d is None:
        return None
    t = d.get("type")
    if t == "Feature":
        return Feature(name=d["name"])  # type: ignore[arg-type]
    if t == "Threshold":
        return Threshold(value=float(d["value"]))  # type: ignore[arg-type]
    if t == "UnaryOp":
        return UnaryOp(op=d["op"], child=node_from_dict(d.get("child")))  # type: ignore[arg-type]
    if t == "BinaryOp":
        return BinaryOp(op=d["op"], left=node_from_dict(d.get("left")), right=node_from_dict(d.get("right")))  # type: ignore[arg-type]
    if t == "ConstBool":
        return ConstBool(value=bool(d.get("value", False)))  # type: ignore[arg-type]
    return None


# ---- GP generation ----
OPS_BOOL = ["AND", "OR"]
OPS_CMP = [">", "<"]
OPS_UN = ["NOT"]


def random_feature(feature_names: List[str]) -> Feature:
    return Feature(name=random.choice(feature_names))


def random_threshold() -> Threshold:
    # Features are z-scored; thresholds in (-3, 3)
    return Threshold(value=random.uniform(-2.5, 2.5))


def random_tree(feature_names: List[str], max_depth: int) -> Node:
    if max_depth <= 1:
        # base case: comparison
        return BinaryOp(op=random.choice(OPS_CMP), left=random_feature(feature_names), right=random_threshold())
    # build boolean composition of sub-comparisons
    if random.random() < 0.2:
        # unary NOT
        return UnaryOp(op="NOT", child=random_tree(feature_names, max_depth - 1))
    return BinaryOp(
        op=random.choice(OPS_BOOL),
        left=random_tree(feature_names, max_depth - 1),
        right=random_tree(feature_names, max_depth - 1),
    )


def mutate(node: Node, feature_names: List[str], max_depth: int, p_mut: float = 0.2) -> Node:
    # Work on a deep copy to avoid mutating parents/elites by reference
    node = clone_node(node)  # type: ignore[assignment]
    if random.random() > p_mut:
        return node  # type: ignore[return-value]
    # random replacement at a node
    if isinstance(node, Feature) and random.random() < 0.5:
        return random_feature(feature_names)
    if isinstance(node, Threshold) and random.random() < 0.5:
        return random_threshold()
    if isinstance(node, BinaryOp):
        if random.random() < 0.3:
            node.op = random.choice(OPS_BOOL if node.op in OPS_BOOL else OPS_CMP)
        node.left = mutate(node.left, feature_names, max_depth - 1, p_mut)
        node.right = mutate(node.right, feature_names, max_depth - 1, p_mut)
        return node
    if isinstance(node, UnaryOp):
        if random.random() < 0.3:
            node.op = "NOT"
        node.child = mutate(node.child, feature_names, max_depth - 1, p_mut)
        return node
    # fallback: grow a new subtree
    return random_tree(feature_names, max_depth)


def simplify_const(n: Node) -> Node:
    # Constant folding and pruning of obvious TRUE/FALSE expressions
    if isinstance(n, UnaryOp):
        child = simplify_const(n.child)
        if isinstance(child, ConstBool):
            return ConstBool(value=(not child.value))
        return UnaryOp(op=n.op, child=child)
    if isinstance(n, BinaryOp):
        left = simplify_const(n.left)
        right = simplify_const(n.right)
        # If both sides are consts
        if isinstance(left, ConstBool) and isinstance(right, ConstBool):
            if n.op in OPS_BOOL:
                return ConstBool(value=(left.value and right.value) if n.op == 'AND' else (left.value or right.value))
            # Compare booleans: True/False converted to 1/0 semantics
            lnum = 1.0 if left.value else 0.0
            rnum = 1.0 if right.value else 0.0
            return ConstBool(value=(lnum > rnum) if n.op == '>' else (lnum < rnum))
        # If left is const and op is AND/OR
        if isinstance(left, ConstBool) and n.op in OPS_BOOL:
            if n.op == 'AND':
                return right if left.value else ConstBool(False)
            else:
                return ConstBool(True) if left.value else right
        if isinstance(right, ConstBool) and n.op in OPS_BOOL:
            if n.op == 'AND':
                return left if right.value else ConstBool(False)
            else:
                return ConstBool(True) if right.value else left
        return BinaryOp(op=n.op, left=left, right=right)
    return n


def individual_key(ind: Tuple[Node, Node, Optional[Node], Optional[Node]]) -> str:
    # Structural uniqueness key
    return json.dumps({
        'el': node_to_dict(ind[0]), 'es': node_to_dict(ind[1]),
        'xl': node_to_dict(ind[2]), 'xs': node_to_dict(ind[3])
    }, sort_keys=True)


def crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    # Simple: swap left/right subtrees if both BinaryOps; else return clones
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        if random.random() < 0.5:
            return BinaryOp(a.op, b.left, a.right), BinaryOp(b.op, a.left, b.right)
        else:
            return BinaryOp(a.op, a.left, b.right), BinaryOp(b.op, b.left, a.right)
    return a, b


# ---- Evaluation ----

def eval_node_vec(n: Node, X: pd.DataFrame) -> pd.Series:
    # Vectorized evaluation returning boolean Series
    if isinstance(n, BinaryOp):
        if n.op in OPS_BOOL:
            ls = eval_node_vec(n.left, X).astype(bool)
            rs = eval_node_vec(n.right, X).astype(bool)
            return (ls & rs) if n.op == "AND" else (ls | rs)
        # Comparison
        if isinstance(n.left, Feature) and isinstance(n.right, Threshold):
            return (X[n.left.name] > n.right.value) if n.op == ">" else (X[n.left.name] < n.right.value)
        if isinstance(n.left, Threshold) and isinstance(n.right, Feature):
            return (n.left.value > X[n.right.name]) if n.op == ">" else (n.left.value < X[n.right.name])
        # Fallback generic
        l = X[getattr(n.left, 'name', '')] if isinstance(n.left, Feature) else pd.Series(getattr(n.left, 'value', 0.0), index=X.index)
        r = X[getattr(n.right, 'name', '')] if isinstance(n.right, Feature) else pd.Series(getattr(n.right, 'value', 0.0), index=X.index)
        return (l > r) if n.op == ">" else (l < r)
    if isinstance(n, UnaryOp):
        return ~eval_node_vec(n.child, X)
    if isinstance(n, ConstBool):
        return pd.Series(bool(n.value), index=X.index)
    # Leaf nodes shouldn't be evaluated directly in vector mode; return False
    return pd.Series(False, index=X.index)


def positions_from_tree_three_state(
    entry_long: Node,
    entry_short: Node,
    exit_long: Optional[Node],
    exit_short: Optional[Node],
    X: pd.DataFrame,
    min_hold_candles: int = 3,
    cooldown_candles: int = 12,
) -> pd.Series:
    # Compute signals vectorized first, then run a simple O(N) FSM
    gl = eval_node_vec(entry_long, X).to_numpy(dtype=bool)
    gs = eval_node_vec(entry_short, X).to_numpy(dtype=bool)
    xl = eval_node_vec(exit_long, X).to_numpy(dtype=bool) if exit_long is not None else np.zeros(len(X), dtype=bool)
    xs = eval_node_vec(exit_short, X).to_numpy(dtype=bool) if exit_short is not None else np.zeros(len(X), dtype=bool)

    state = 0
    hold = 0
    cooldown = 0
    out = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        if cooldown > 0:
            cooldown -= 1
        if state != 0:
            hold += 1
        else:
            hold = 0

        if state == 0:
            if cooldown == 0:
                if gl[i]:
                    state = 1
                    hold = 0
                elif gs[i]:
                    state = -1
                    hold = 0
        elif state == 1:
            if hold >= min_hold_candles and (xl[i] or gs[i]):
                state = 0
                cooldown = cooldown_candles
        elif state == -1:
            if hold >= min_hold_candles and (xs[i] or gl[i]):
                state = 0
                cooldown = cooldown_candles
        out[i] = float(state)
    pos = pd.Series(out, index=X.index)
    return pos.shift(1).fillna(0.0)


# ---- Metrics ----
from backtest_simple_strategy import evaluate_strategy, StrategyParams  # reuse metrics impl


def metrics_from_positions(close: pd.Series, pos: pd.Series, cost_bps: float = 1.0) -> Dict[str, float]:
    ret = close.pct_change().fillna(0.0)
    strat_ret = pos * ret
    cost = float(cost_bps) * 1e-4
    if cost > 0:
        trade_flag = pos.ne(pos.shift(1)).fillna(False)
        strat_ret[trade_flag] = strat_ret[trade_flag] - cost
    equity = (1.0 + strat_ret).cumprod()
    cum_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0
    # Safe std computations (avoid ddof warnings on small samples)
    vals = strat_ret.values
    vol = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    mean_ret = float(np.mean(vals)) if len(vals) > 0 else 0.0
    sharpe = float((mean_ret / vol) * np.sqrt(252 * 288)) if vol > 1e-12 else 0.0
    neg_vals = vals[vals < 0.0]
    dstd = float(np.std(neg_vals, ddof=1)) if len(neg_vals) > 1 else 0.0
    sortino = float((strat_ret.mean() / dstd) * np.sqrt(252 * 288)) if dstd > 1e-12 else 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).fillna(0.0)
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    trades = int(pos.ne(pos.shift(1)).sum())
    return {"cum_return": cum_return, "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd, "trades": float(trades)}


# ---- Trade frequency normalization ----
def trade_frequency_score(trades: int, bars: int, low: float = 10.0, high: float = 25.0, max_ok: float = 50.0) -> float:
    """Score trade frequency in [0,1] normalized by sample length.
    Computes trades per 1000 bars, targets [low, high] as ideal band.
    """
    if bars <= 0:
        return 0.0
    per_k = (float(trades) * 4000.0) / float(bars)
    if per_k < low:
        return max(0.0, per_k / low)
    if per_k <= high:
        return 1.0
    # Above high, linearly decay to 0 at max_ok
    return max(0.0, (max_ok - per_k) / max(1e-6, (max_ok - high)))


# ---- GP Optimizer ----
@dataclass
class GPConfig:
    population: int = 20
    generations: int = 5
    max_depth: int = 3
    max_nodes: int = 12
    complexity_penalty: float = 0.5  # per node beyond 8
    cost_bps: float = 1.0
    min_hold_candles: int = 3
    cooldown_candles: int = 12
    elite_frac: float = 0.2
    elite_count: int = 0
    # Optional pluggable scoring function. If provided, this overrides the
    # built-in composite in fitness(). The callable receives:
    #  - metrics: Dict[str, float] as produced by metrics_from_positions
    #  - extras: Dict[str, float] (e.g., {"tf": trade_frequency_score})
    #  - size: total node count across entry/exit trees
    #  - depth: max depth across entry/exit trees
    # and must return a float score (higher is better).
    score_fn: Optional[Callable[[Dict[str, float], Dict[str, float], int, int], float]] = None


def fitness(entry_long: Node, entry_short: Node, exit_long: Optional[Node], exit_short: Optional[Node], X_train: pd.DataFrame, close_col: str, cfg: GPConfig) -> Tuple[float, Dict[str, float], int]:
    pos = positions_from_tree_three_state(
        entry_long, entry_short, exit_long, exit_short, X_train,
        min_hold_candles=cfg.min_hold_candles, cooldown_candles=cfg.cooldown_candles,
    )
    m = metrics_from_positions(X_train[close_col], pos, cost_bps=cfg.cost_bps)
    trades = int(m.get("trades", 0.0))

    # Trade frequency score normalized by sample length (per 1000 bars)
    tf_score = trade_frequency_score(trades, len(X_train))
    # Tree complexity stats for optional custom scorer
    size_total = (
        node_size(entry_long)
        + node_size(entry_short)
        + (node_size(exit_long) if exit_long else 0)
        + (node_size(exit_short) if exit_short else 0)
    )
    depth_total = max(
        node_depth(entry_long),
        node_depth(entry_short),
        (node_depth(exit_long) if exit_long else 0),
        (node_depth(exit_short) if exit_short else 0),
    )
    
    # Allow an injected scorer to define the fitness directly
    if cfg.score_fn is not None:
        try:
            score_override = cfg.score_fn(m, {"tf": float(tf_score)}, int(size_total), int(depth_total))
            return float(score_override), m, trades
        except Exception:
            # Fall back to the default composite if custom scorer fails
            pass

    sharpe = float(m.get("sharpe", 0.0))
    sortino = float(m.get("sortino", 0.0))
    cumret = float(m.get("cum_return", 0.0))
    max_dd = float(m.get("max_drawdown", 0.0))

    # Composite ratio with ballast
    numer = (sharpe + 100.0 if sharpe >= 0.0 else 100.0) *  (sortino + 100.0 if sortino >= 0.0 else 100.0) * ((cumret * 100.0) + 100.0 if cumret >= 0.0 else 100.0) * (100.0 + tf_score * 10)
    denom = (abs(sharpe) + 100.0 if sharpe < 0.0 else 100.0) * (abs(sortino) + 100.0 if sortino < 0.0 else 100.0) * (abs(cumret * 100.0) + 100.0 if cumret < 0.0 else 100.0) * (100.0 + abs(max_dd) * 500.0)
    score = numer / max(1e-12, denom)
    return float(score), m, trades


def run_gp(X: pd.DataFrame, close_col: str, cfg: GPConfig, X_val: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    feature_names = [c for c in X.columns if c != close_col and not c.endswith("_close")]
    # Individuals are tuples of (entry_long, entry_short, exit_long, exit_short)
    pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = [
        (
            random_tree(feature_names, cfg.max_depth),
            random_tree(feature_names, cfg.max_depth),
            random_tree(feature_names, cfg.max_depth - 1),
            random_tree(feature_names, cfg.max_depth - 1),
        )
        for _ in range(cfg.population)
    ]
    history: List[Dict[str, Any]] = []
    for gen in range(cfg.generations):
        scored: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]] = []
        for ind in pop:
            f, m, tr = fitness(ind[0], ind[1], ind[2], ind[3], X, close_col, cfg)
            scored.append((f, ind, m, tr))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_f, best_ind, best_m, best_trades = scored[0]
        val_m = None
        if X_val is not None:
            pos_val = positions_from_tree_three_state(
                best_ind[0], best_ind[1], best_ind[2], best_ind[3], X_val,
                min_hold_candles=cfg.min_hold_candles, cooldown_candles=cfg.cooldown_candles,
            )
            val_m = metrics_from_positions(X_val[close_col], pos_val, cost_bps=cfg.cost_bps)
        size = node_size(best_ind[0]) + node_size(best_ind[1]) + (node_size(best_ind[2]) if best_ind[2] else 0) + (node_size(best_ind[3]) if best_ind[3] else 0)
        depth = max(node_depth(best_ind[0]), node_depth(best_ind[1]), node_depth(best_ind[2]) if best_ind[2] else 0, node_depth(best_ind[3]) if best_ind[3] else 0)
        history.append({
            "gen": gen, "fitness": best_f, "metrics": best_m, "val_metrics": val_m, "trades": best_trades,
            "tree": {
                "entry_long": pretty(best_ind[0]),
                "entry_short": pretty(best_ind[1]),
                "exit_long": pretty(best_ind[2]) if best_ind[2] else None,
                "exit_short": pretty(best_ind[3]) if best_ind[3] else None,
            },
            "size": size, "depth": depth,
        })
        print(json.dumps(history[-1]), flush=True)

        # Selection: keep top 20%, fill rest via crossover/mutation
        elite_n = max(1, int(0.2 * cfg.population))
        next_pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = [scored[i][1] for i in range(elite_n)]
        while len(next_pop) < cfg.population:
            parents = random.sample(scored[:max(10, cfg.population)], 2)
            a = parents[0][1]
            b = parents[1][1]
            # crossover each subtree independently
            e1, e2 = crossover(a[0], b[0])
            s1, s2 = crossover(a[1], b[1])
            xl1, xl2 = crossover(a[2], b[2]) if (a[2] and b[2]) else (a[2] or b[2], None)
            xs1, xs2 = crossover(a[3], b[3]) if (a[3] and b[3]) else (a[3] or b[3], None)
            # mutate
            e1 = mutate(e1, feature_names, cfg.max_depth, 0.3)
            s1 = mutate(s1, feature_names, cfg.max_depth, 0.3)
            if xl1 is not None:
                xl1 = mutate(xl1, feature_names, cfg.max_depth - 1, 0.3)
            if xs1 is not None:
                xs1 = mutate(xs1, feature_names, cfg.max_depth - 1, 0.3)
            next_pop.append((e1, s1, xl1, xs1))
            if len(next_pop) < cfg.population:
                e2 = mutate(e2, feature_names, cfg.max_depth, 0.3)
                s2 = mutate(s2, feature_names, cfg.max_depth, 0.3)
                if xl2 is not None:
                    xl2 = mutate(xl2, feature_names, cfg.max_depth - 1, 0.3)
                if xs2 is not None:
                    xs2 = mutate(xs2, feature_names, cfg.max_depth - 1, 0.3)
                next_pop.append((e2, s2, xl2, xs2))
        pop = next_pop

    # Final best
    scored_final: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]] = []
    for ind in pop:
        f, m, tr = fitness(ind[0], ind[1], ind[2], ind[3], X, close_col, cfg)
        scored_final.append((f, ind, m, tr))
    scored_final.sort(key=lambda x: x[0], reverse=True)
    bf, bt, bm, tr = scored_final[0]
    # Sanity: compute additional stats for best individual
    pos_train = positions_from_tree_three_state(bt[0], bt[1], bt[2], bt[3], X, cfg.min_hold_candles, cfg.cooldown_candles)
    avg_hold = float((pos_train.ne(pos_train.shift(1))).cumsum().diff().groupby((pos_train != pos_train.shift()).cumsum()).transform('size').dropna().mean() or 0.0) if len(pos_train) > 0 else 0.0
    return {
        "best_fitness": bf,
        "best_tree": {
            "entry_long": pretty(bt[0]),
            "entry_short": pretty(bt[1]),
            "exit_long": pretty(bt[2]) if bt[2] else None,
            "exit_short": pretty(bt[3]) if bt[3] else None,
        },
        "best_metrics": bm,
        "best_trades": tr,
        "avg_hold_bars_train": avg_hold,
        "history": history,
    }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GP-style logic optimization across multiple instruments")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--val-start", default="2023-01-01")
    p.add_argument("--val-end", default="2023-12-31")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-nodes", type=int, default=12)
    p.add_argument("--complexity-penalty", type=float, default=0.5)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--min-hold", type=int, default=3, help="Minimum bars to hold a position (e.g., 3 for 15 minutes)")
    p.add_argument("--cooldown", type=int, default=12, help="Cooldown bars after exit before new entry (e.g., 12 for 1 hour)")
    p.add_argument("--subsample", type=int, default=40, help="Use every k-th bar for speed (1=no downsample)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(__file__), "runs", "gp"))
    p.add_argument("--top-k", type=int, default=3, help="Save top-K individuals per generation")
    p.add_argument("--final-top-k", type=int, default=5, help="Evaluate top-K elites on validation at end")
    # Conservative defaults for small hosts
    p.add_argument("--train-max-rows", type=int, default=20000, help="Limit training rows (tail)")
    p.add_argument("--val-max-rows", type=int, default=8000, help="Limit validation rows (tail)")
    # Config profiles
    p.add_argument("--profile", default="", help="Named arg set: small|medium|large or custom id (optional)")
    p.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    p.add_argument("--save-profile", help="Save current args to profiles.json under this id")
    # Elitism controls
    p.add_argument("--elite-frac", type=float, default=0.2, help="Fraction of parents carried over (ignored if elite-count>0)")
    p.add_argument("--elite-count", type=int, default=0, help="Exact number of parents carried over; overrides elite-frac if >0")
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    # Apply profiles
    profiles_path = os.path.join(os.path.dirname(__file__), "profiles.json")
    # Predefined profiles
    default_profiles = {
        "small": {"population": 8, "generations": 3, "subsample": 40, "train_max_rows": 20000, "val_max_rows": 8000, "elite_frac": 0.2},
        "medium": {"population": 50, "generations": 15, "subsample": 20, "train_max_rows": 80000, "val_max_rows": 30000, "elite_frac": 0.2},
        "large": {"population": 120, "generations": 30, "subsample": 10, "train_max_rows": 160000, "val_max_rows": 60000, "elite_frac": 0.2},
    }
    # Load user profiles if any
    user_profiles = {}
    if os.path.exists(profiles_path):
        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                user_profiles = json.load(f)
        except Exception:
            user_profiles = {}
    all_profiles = {**default_profiles, **user_profiles}
    if a.list_profiles:
        print(json.dumps({"event": "profiles", "profiles": all_profiles}, indent=2))
        return
    if a.profile and a.profile in all_profiles:
        prof = all_profiles[a.profile]
        for k, v in prof.items():
            setattr(a, k, v)
    # Optionally save current args as new profile
    if a.save_profile:
        user_profiles[a.save_profile] = {
            "population": int(a.population),
            "generations": int(a.generations),
            "subsample": int(a.subsample),
            "train_max_rows": int(a.train_max_rows),
            "val_max_rows": int(a.val_max_rows),
            "elite_frac": float(a.elite_frac),
            "elite_count": int(a.elite_count),
        }
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump({**user_profiles}, f, indent=2)
        print(json.dumps({"event": "profile_saved", "id": a.save_profile}), flush=True)
    random.seed(int(a.seed))
    np.random.seed(int(a.seed))
    start_ts = time.time()
    # Compute build-time limits (can be disabled via flags or <=0 values)
    build_max_rows: Optional[int]
    if hasattr(a, "no_max_rows") and a.no_max_rows:
        build_max_rows = None
    else:
        tm = int(a.train_max_rows) if a.train_max_rows is not None else 0
        vm = int(a.val_max_rows) if a.val_max_rows is not None else 0
        if tm <= 0 and vm <= 0:
            build_max_rows = None
        else:
            tm = max(0, tm)
            vm = max(0, vm)
            build_max_rows = (tm + vm) if (tm + vm) > 0 else None

    feats = build_features(
        a.data_dir,
        start=a.train_start,
        end=a.val_end,
        subsample=int(a.subsample) if a.subsample else 1,
        max_rows=build_max_rows,
    )
    X_all = feats["X"]
    print(json.dumps({
        "event": "features_loaded",
        "loaded_instruments": feats.get("loaded"),
        "missing_instruments": feats.get("missing"),
        "n_features": int(X_all.shape[1]),
    }), flush=True)
    # Choose primary close for PnL computation: EUR_USD close
    close_col = "EUR_USD_close"
    # Split
    X_train = X_all[(X_all.index >= pd.to_datetime(a.train_start, utc=True)) & (X_all.index <= pd.to_datetime(a.train_end, utc=True))]
    X_val = X_all[(X_all.index >= pd.to_datetime(a.val_start, utc=True)) & (X_all.index <= pd.to_datetime(a.val_end, utc=True))]

    # Downsample and limit rows to reduce memory/CPU (post split)
    if int(a.subsample) > 1:
        step = int(a.subsample)
        X_train = X_train.iloc[::step]
        X_val = X_val.iloc[::step] if len(X_val) > 0 else X_val
    if not (hasattr(a, "no_max_rows") and a.no_max_rows):
        if a.train_max_rows is not None and int(a.train_max_rows) > 0 and len(X_train) > int(a.train_max_rows):
            X_train = X_train.tail(int(a.train_max_rows))
        if a.val_max_rows is not None and int(a.val_max_rows) > 0 and len(X_val) > int(a.val_max_rows):
            X_val = X_val.tail(int(a.val_max_rows))

    run_dir = os.path.join(str(a.checkpoint_dir), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(a),
            "close_col": close_col,
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "num_features": int(X_train.shape[1]),
            "start_time": start_ts,
        }, f)
    print(json.dumps({
        "event": "start", "run_dir": run_dir,
        "train_rows": int(len(X_train)), "val_rows": int(len(X_val)), "features": int(X_train.shape[1]),
        "subsample": int(a.subsample), "build_max_rows": (build_max_rows if build_max_rows is not None else "none"),
    }), flush=True)

    cfg = GPConfig(
        population=int(a.population),
        generations=int(a.generations),
        max_depth=int(a.max_depth),
        max_nodes=int(a.max_nodes),
        complexity_penalty=float(a.complexity_penalty),
        cost_bps=float(a.cost_bps),
        min_hold_candles=int(a.min_hold),
        cooldown_candles=int(a.cooldown),
        elite_frac=float(a.elite_frac),
        elite_count=int(a.elite_count),
    )
    # Wrap run to stream checkpoints
    def save_generation(gen: int, scored: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]]) -> None:
        top_k = max(1, int(a.top_k))
        path = os.path.join(run_dir, f"gen_{gen:03d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(min(top_k, len(scored))):
                fit, ind, m, tr = scored[i]
                record = {
                    "rank": i + 1,
                    "fitness": float(fit),
                    "metrics": m,
                    "trades": int(tr),
                    "size": node_size(ind[0]) + node_size(ind[1]) + (node_size(ind[2]) if ind[2] else 0) + (node_size(ind[3]) if ind[3] else 0),
                    "depth": max(node_depth(ind[0]), node_depth(ind[1]), node_depth(ind[2]) if ind[2] else 0, node_depth(ind[3]) if ind[3] else 0),
                    "entry_long": node_to_dict(ind[0]),
                    "entry_short": node_to_dict(ind[1]),
                    "exit_long": node_to_dict(ind[2]),
                    "exit_short": node_to_dict(ind[3]),
                }
                f.write(json.dumps(record) + "\n")

    # Monkey-patch run_gp loop to save checkpoints per generation
    # We re-implement a slim runner here to keep code simple
    feature_names = [c for c in X_train.columns if c != close_col and not c.endswith("_close")]
    pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = [
        (
            random_tree(feature_names, cfg.max_depth),
            random_tree(feature_names, cfg.max_depth),
            random_tree(feature_names, cfg.max_depth - 1),
            random_tree(feature_names, cfg.max_depth - 1),
        ) for _ in range(cfg.population)
    ]

    history: List[Dict[str, Any]] = []
    ever_best_f: Optional[float] = None
    for gen in range(cfg.generations):
        t0 = time.time()
        scored: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]] = []
        for ind in pop:
            fval, met, tr = fitness(ind[0], ind[1], ind[2], ind[3], X_train, close_col, cfg)
            scored.append((fval, ind, met, tr))
        scored.sort(key=lambda x: x[0], reverse=True)
        save_generation(gen, scored)
        # logging summary
        best_f, best_ind, best_m, best_tr = scored[0]
        # Median fitness
        median_f = float(np.median([s[0] for s in scored])) if scored else float("nan")
        # Trade-frequency score consistent with fitness() and normalized by rows
        best_tf = trade_frequency_score(int(best_tr), len(X_train))
        print(json.dumps({
            "event": "gen_summary",
            "gen": gen,
            "med": float(median_f),
            "sec": round(time.time() - t0, 3),
            "tree": int(
                node_size(best_ind[0])
                + node_size(best_ind[1])
                + (node_size(best_ind[2]) if best_ind[2] else 0)
                + (node_size(best_ind[3]) if best_ind[3] else 0)
            ),
            "best": {
                "fit": float(best_f),
                "trd": int(best_tr),
                "shrp": float(best_m.get("sharpe", 0.0)),
                "srtn": float(best_m.get("sortino", 0.0)),
                "cum": float(best_m.get("cum_return", 0.0)),
                "dd": float(best_m.get("max_drawdown", 0.0)),
                "tf": float(best_tf),
            },
        }), flush=True)

        # evolve
        elite_n = int(cfg.elite_count) if int(cfg.elite_count) > 0 else max(1, int(cfg.elite_frac * cfg.population))
        elite_n = min(max(1, elite_n), cfg.population - 1 if cfg.population > 1 else 1)
        # Deep copy elites to preserve their structure unchanged and simplify constants
        next_pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = []
        seen: set[str] = set()
        for i in range(elite_n):
            raw = scored[i][1]
            elite = (
                simplify_const(clone_node(raw[0])),
                simplify_const(clone_node(raw[1])),
                simplify_const(clone_node(raw[2])) if raw[2] else None,
                simplify_const(clone_node(raw[3])) if raw[3] else None,
            )
            key = individual_key(elite)
            if key not in seen:
                next_pop.append(elite)  # type: ignore[arg-type]
                seen.add(key)
        while len(next_pop) < cfg.population:
            parents = random.sample(scored[:max(10, cfg.population)], 2)
            a_ind = parents[0][1]; b_ind = parents[1][1]
            # Work on cloned parents to avoid subtree aliasing
            pa = clone_individual(a_ind)
            pb = clone_individual(b_ind)
            e1, e2 = crossover(pa[0], pb[0])
            s1, s2 = crossover(pa[1], pb[1])
            xl1, xl2 = crossover(pa[2], pb[2]) if (pa[2] and pb[2]) else (pa[2] or pb[2], None)
            xs1, xs2 = crossover(pa[3], pb[3]) if (pa[3] and pb[3]) else (pa[3] or pb[3], None)
            e1 = mutate(e1, feature_names, cfg.max_depth, 0.3)
            s1 = mutate(s1, feature_names, cfg.max_depth, 0.3)
            if xl1 is not None:
                xl1 = mutate(xl1, feature_names, cfg.max_depth - 1, 0.3)
            if xs1 is not None:
                xs1 = mutate(xs1, feature_names, cfg.max_depth - 1, 0.3)
            child1 = (simplify_const(e1), simplify_const(s1), simplify_const(xl1) if xl1 else None, simplify_const(xs1) if xs1 else None)
            k1 = individual_key(child1)
            if k1 not in seen:
                next_pop.append(child1)  # type: ignore[arg-type]
                seen.add(k1)
            if len(next_pop) < cfg.population:
                e2m = mutate(e2, feature_names, cfg.max_depth, 0.3)
                s2m = mutate(s2, feature_names, cfg.max_depth, 0.3)
                if xl2 is not None:
                    xl2 = mutate(xl2, feature_names, cfg.max_depth - 1, 0.3)
                if xs2 is not None:
                    xs2 = mutate(xs2, feature_names, cfg.max_depth - 1, 0.3)
                child2 = (simplify_const(e2m), simplify_const(s2m), simplify_const(xl2) if xl2 else None, simplify_const(xs2) if xs2 else None)
                k2 = individual_key(child2)
                if k2 not in seen:
                    next_pop.append(child2)  # type: ignore[arg-type]
                    seen.add(k2)
        pop = next_pop

    # final evaluation of last population
    final_scores: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]] = []
    for ind in pop:
        fval, met, tr = fitness(ind[0], ind[1], ind[2], ind[3], X_train, close_col, cfg)
        final_scores.append((fval, ind, met, tr))
    final_scores.sort(key=lambda x: x[0], reverse=True)
    bf, bt, bm, tr = final_scores[0]
    # Save final
    with open(os.path.join(run_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fitness": float(bf), "metrics": bm, "trades": int(tr),
            "tree": {
                "entry_long": node_to_dict(bt[0]),
                "entry_short": node_to_dict(bt[1]),
                "exit_long": node_to_dict(bt[2]),
                "exit_short": node_to_dict(bt[3]),
            }
        }, f)
    print(json.dumps({"event": "result", "best": {"fitness": float(bf), "tree": {
        "entry_long": pretty(bt[0]), "entry_short": pretty(bt[1]),
        "exit_long": pretty(bt[2]) if bt[2] else None, "exit_short": pretty(bt[3]) if bt[3] else None,
    }, "metrics": bm, "trades": int(tr), "run_dir": run_dir}}), flush=True)

    # Evaluate top-K elites on validation (if available)
    if X_val is not None and len(X_val) > 0:
        k = max(1, int(a.final_top_k))
        k = min(k, len(final_scores))
        path = os.path.join(run_dir, "final_elites.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rank in range(k):
                fit, ind, train_m, train_tr = final_scores[rank]
                pos_val = positions_from_tree_three_state(ind[0], ind[1], ind[2], ind[3], X_val, cfg.min_hold_candles, cfg.cooldown_candles)
                val_m = metrics_from_positions(X_val[close_col], pos_val, cost_bps=cfg.cost_bps)
                rec = {
                    "rank": rank + 1,
                    "fitness_train": float(fit),
                    "metrics_train": train_m,
                    "metrics_val": val_m,
                    "trades_train": int(train_tr),
                    "trades_val": int(val_m.get("trades", 0.0)),
                    "tree": {
                        "entry_long": pretty(ind[0]),
                        "entry_short": pretty(ind[1]),
                        "exit_long": pretty(ind[2]) if ind[2] else None,
                        "exit_short": pretty(ind[3]) if ind[3] else None,
                    },
                }
                f.write(json.dumps(rec) + "\n")
        print(json.dumps({"event": "final_elites", "count": k, "file": path}), flush=True)


if __name__ == "__main__":
    main()
