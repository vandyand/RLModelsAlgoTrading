from __future__ import annotations

"""
Genetic algorithm hyperparameter search for TCNActionAdapter over 2023 WFO.

This reuses the same WFO geometry and global-trade objective as model_zoo_2023,
but uses a GA instead of plain random search to explore the TCNAction space.

Usage (from forex-rl root):

  python3 -m wfo.tcn_action_ga_search \\
      --base-config wfo/tcn_eurusd_2023_weeks.json \\
      --population 5 \\
      --generations 2 \\
      --seed 10 \\
      --plot-equity
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .core import WFOConfig, run_wfo, _instantiate_adapter, _render_chart
from .adapters.tcn_scalar_threshold import TCNActionAdapter
from .tcn_action_random_search import sample_hyperparams as sample_action_hyperparams
from .model_zoo_2023 import compute_global_objective, load_base_config, build_wfo_cfg


def _plot_global_equity(run_dir: Path) -> None:
    """Render ASCII equity curve from stitched trades for a WFO run."""
    trades_path = run_dir / "trades.jsonl"
    if not trades_path.exists():
        return

    all_profits: list[float] = []
    with trades_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for t in rec.get("trades", []):
                try:
                    all_profits.append(float(t.get("net_pnl", 0.0)))
                except Exception:
                    continue
    if not all_profits:
        return

    profits = np.asarray(all_profits, dtype=float)
    equity = np.cumprod(1.0 + profits)
    chart = _render_chart(equity.tolist(), width=60, height=8)
    if chart:
        print(chart, flush=True)


def _tcn_hp_to_adapter_kwargs(base_cfg: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Map TCNAction hyperparameters into adapter_kwargs."""
    candle_cache_base = str(base_cfg.get("candle_cache_base", "http://127.0.0.1:9100"))
    fixed_adapter: Dict[str, Any] = {
        "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
        "grans": str(base_cfg.get("grans", "M15,H1,D")),
        "base_gran": str(base_cfg.get("tcn_base_gran", base_cfg.get("base_gran", "M15"))),
        "epochs": int(base_cfg.get("epochs", 3)),
        "lr": float(base_cfg.get("lr", 1e-3)),
        "candle_cache_base": candle_cache_base,
        # Labeling mode comes from base config; quantile/threshold values may be GA-optimized.
        "label_mode": str(base_cfg.get("label_mode", "bps")),
        # NQ-only debug flag, ignored for normal FX configs.
        "debug_force_all_long": bool(base_cfg.get("debug_force_all_long", False)),
    }
    adapter_kwargs = dict(fixed_adapter)
    adapter_kwargs.update(
        lookback_bars=int(hp["tcn_lookback"]),
        target_horizon=int(hp["tcn_target_horizon"]),
        hidden_channels=int(hp["tcn_hidden"]),
        num_blocks=int(hp["tcn_blocks"]),
        kernel_size=int(hp["tcn_kernel"]),
        dropout=float(hp["tcn_dropout"]),
        label_long_thresh=float(hp["label_long_thresh"]),
        label_short_thresh=float(hp["label_short_thresh"]),
        # For quantile mode, these may be unused but are still passed.
        label_long_quantile=float(hp.get("label_long_quantile", base_cfg.get("label_long_quantile", 0.7))),
        label_short_quantile=float(hp.get("label_short_quantile", base_cfg.get("label_short_quantile", 0.3))),
        trade_cost_bps=float(hp["tcn_trade_cost_bps"]),
        # NQ-only causal rule hyperparameters; FX adapters simply ignore these.
        nq_ret_thresh=float(hp.get("nq_ret_thresh", base_cfg.get("nq_ret_thresh", 0.0005))),
        nq_vol_lookback=int(hp.get("nq_vol_lookback", base_cfg.get("nq_vol_lookback", 64))),
        nq_z_thresh=float(hp.get("nq_z_thresh", base_cfg.get("nq_z_thresh", 1.0))),
        # NQ-only policy hysteresis thresholds. For FX configs these are unused,
        # but for the NQ TCN+MLP adapter they directly control trade decisions.
        nq_enter_long=float(hp.get("nq_enter_long", base_cfg.get("nq_enter_long", 0.2))),
        nq_exit_long=float(hp.get("nq_exit_long", base_cfg.get("nq_exit_long", 0.05))),
        nq_enter_short=float(hp.get("nq_enter_short", base_cfg.get("nq_enter_short", -0.2))),
        nq_exit_short=float(hp.get("nq_exit_short", base_cfg.get("nq_exit_short", -0.05))),
        # Optional MLP layout for policy head; GA does not presently mutate this
        # but debug configs (e.g. EURUSD policy) can pass it through.
        mlp_layers=base_cfg.get("mlp_layers"),
    )
    return adapter_kwargs


def _eval_individual(
    base_cfg: Dict[str, Any],
    hp: Dict[str, Any],
    individual_id: int,
) -> Tuple[float, Dict[str, float], str, int]:
    """Worker: run WFO for a single TCNAction config and compute global objective."""
    # Allow base config to override the adapter implementation, so we can
    # use NQ-specific variants without touching the FX path.
    adapter_spec = str(
        base_cfg.get(
            "adapter_spec",
            "wfo.adapters.tcn_scalar_threshold:TCNActionAdapter",
        )
    )
    adapter_kwargs = _tcn_hp_to_adapter_kwargs(base_cfg, hp)
    adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)
    cfg = build_wfo_cfg(base_cfg, adapter_spec=adapter_spec, adapter_kwargs=adapter_kwargs)
    run_dir = Path(run_wfo(adapter, cfg))
    score, summary = compute_global_objective(run_dir)

    # Optional NQ-specific objective shaping:
    # - Encourage a roughly balanced mix of long and short trades.
    # - Encourage a reasonable number of trades per year without hard cut-offs.
    #
    # The goal is to steer search away from degenerate "short-only, few-trade"
    # solutions while still letting Sharpe/cum_return drive the core objective.
    inst = str(base_cfg.get("tcn_instrument", "")).upper()
    if inst == "NQ":
        total_trades = int(summary.get("total_trades", 0) or 0)
        total_long_trades = int(summary.get("total_long_trades", 0) or 0)
        total_short_trades = int(summary.get("total_short_trades", 0) or 0)

        # Trade-count term: 0 trades -> 0 pts; 250+ trades -> +10 pts, linear in between.
        if total_trades <= 0:
            trade_score = 0.0
        else:
            trade_term = min(float(total_trades), 250.0) / 250.0
            trade_score = 10.0 * trade_term

        # Long/short balance term:
        # - ratio = total_long / total_short
        # - ratio in [0.1, 1] or [1, 10] is mapped linearly:
        #     1.0  -> +10 pts (perfect balance)
        #     0.1  -> -10 pts (too few longs)
        #     10.0 -> -10 pts (too few shorts)
        # - Outside [0.1, 10] or if one side has zero trades: -10 pts.
        if total_long_trades == 0 and total_short_trades == 0:
            ratio_score = -10.0
        elif total_short_trades == 0 or total_long_trades == 0:
            ratio_score = -10.0
        else:
            ratio = float(total_long_trades) / float(total_short_trades)
            if ratio <= 0.1 or ratio >= 10.0:
                ratio_score = -10.0
            elif ratio == 1.0:
                ratio_score = 10.0
            elif ratio < 1.0:
                # Map 0.1 -> -10, 1.0 -> +10 linearly.
                ratio_score = -10.0 + (ratio - 0.1) * (20.0 / (1.0 - 0.1))
            else:
                # 1.0 < ratio < 10.0: map 1.0 -> +10, 10.0 -> -10 linearly.
                ratio_score = 10.0 - (ratio - 1.0) * (20.0 / (10.0 - 1.0))

        score += trade_score + ratio_score

    return score, summary, str(run_dir), individual_id


def _sample_from_ga_spec(rng: random.Random, spec: Any, base_val: Any) -> Any:
    """Sample a single hyperparameter value from a compact GA range spec.

    Spec formats:
      - [start, end]        -> sample uniformly in [start, end]
      - [start, step, end]  -> sample from {start, start+step, ..., end}

    Integer specs produce integer values; otherwise floats are used.
    On any error, we fall back to base_val.
    """
    try:
        if not isinstance(spec, (list, tuple)) or len(spec) == 0:
            return base_val

        # Single value: treat as constant
        if len(spec) == 1:
            return spec[0]

        if len(spec) == 2:
            start, end = spec
            step = None
        else:
            start, step, end = spec[0], spec[1], spec[2]

        # Integer grid
        if all(isinstance(x, int) for x in (start, end)) and (step is None or isinstance(step, int)):
            if step is None or step == 0:
                lo = min(start, end)
                hi = max(start, end)
                return rng.randint(lo, hi)
            step_int = int(step)
            if step_int == 0:
                lo = min(start, end)
                hi = max(start, end)
                return rng.randint(lo, hi)
            if step_int > 0 and end < start:
                start, end = end, start
            if step_int < 0 and end > start:
                start, end = end, start
            values = list(range(start, end + (1 if step_int > 0 else -1), step_int))
            if not values:
                return base_val
            return rng.choice(values)

        # Float range/grid
        start_f = float(start)
        end_f = float(end)
        if step is None or float(step) == 0.0:
            lo = min(start_f, end_f)
            hi = max(start_f, end_f)
            return rng.uniform(lo, hi)

        step_f = float(step)
        if step_f == 0.0:
            lo = min(start_f, end_f)
            hi = max(start_f, end_f)
            return rng.uniform(lo, hi)
        lo = min(start_f, end_f)
        hi = max(start_f, end_f)
        n_steps = int((hi - lo) / abs(step_f))
        if n_steps <= 0:
            return lo
        idx = rng.randint(0, n_steps)
        return lo + idx * abs(step_f)
    except Exception:
        return base_val


def _apply_ga_config_to_sample(
    rng: random.Random, base_hp: Dict[str, Any], ga_cfg: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Override a base hyperparameter sample using a compact GA config, if provided.

    ga_cfg is a dict like:

        {
          "tcn_lookback": [12, 8, 240],
          "tcn_target_horizon": [2, 4, 192],
          "tcn_hidden": [8, 4, 32],
          "tcn_blocks": [1, 1, 4],
          "tcn_kernel": [1, 1, 5],
          "tcn_dropout": [0.0, 0.01, 0.15],
          "label_long_thresh": [0.0005, 0.0001, 0.0080],
          "label_short_thresh": [-0.0080, 0.0001, -0.0005],
          "tcn_trade_cost_bps": [0.5, 0.5, 1.5],
          "nq_ret_thresh": [0.0002, 0.0001, 0.0020],
          "nq_vol_lookback": [32, 16, 128],
          "nq_z_thresh": [0.5, 0.1, 2.5]
        }

    For each key present in ga_cfg we resample that field according to its spec.
    Keys not present in ga_cfg keep the original base_hp value.
    """
    if not ga_cfg:
        return dict(base_hp)

    hp = dict(base_hp)
    for k, spec in ga_cfg.items():
        if k not in hp:
            continue
        hp[k] = _sample_from_ga_spec(rng, spec, hp[k])

    # Ensure quantile labeler hyperparameters are sane if present.
    s_q = hp.get("label_short_quantile", None)
    l_q = hp.get("label_long_quantile", None)
    if isinstance(s_q, (int, float)) and isinstance(l_q, (int, float)):
        s_q_f = float(s_q)
        l_q_f = float(l_q)
        if not (0.0 < s_q_f < l_q_f < 1.0):
            # Simple repair: clamp into [0.05, 0.45] and [0.55, 0.95]
            s_q_f = max(0.05, min(0.45, s_q_f))
            l_q_f = max(0.55, min(0.95, max(l_q_f, s_q_f + 0.1)))
            hp["label_short_quantile"] = s_q_f
            hp["label_long_quantile"] = l_q_f

    return hp


def _sample_hp_for_ga(rng: random.Random, ga_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Sample a GA hyperparameter dict, optionally constrained by ga_cfg."""
    base_hp = sample_action_hyperparams(rng)
    return _apply_ga_config_to_sample(rng, base_hp, ga_cfg)


def _crossover(rng: random.Random, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
    """Uniform crossover: choose each field from A or B with 0.5 probability."""
    child: Dict[str, Any] = {}
    for k in parent_a.keys():
        if k not in parent_b:
            child[k] = parent_a[k]
            continue
        child[k] = parent_a[k] if rng.random() < 0.5 else parent_b[k]
    return child


def _mutate(
    rng: random.Random,
    hp: Dict[str, Any],
    mutation_prob: float = 0.3,
    ga_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Mutation: with some probability, resample each hyperparameter from the sampler.

    If ga_config is provided, the sampler respects its compact range specs.
    """
    base_sample = _sample_hp_for_ga(rng, ga_config)
    child = dict(hp)
    for k, v in base_sample.items():
        if rng.random() < mutation_prob:
            child[k] = v
    return child


@dataclass
class Individual:
    hp: Dict[str, Any]
    score: float = float("-inf")
    summary: Dict[str, float] | None = None
    run_dir: str | None = None


def run_ga(
    base_cfg: Dict[str, Any],
    population_size: int,
    generations: int,
    rng: random.Random,
    plot_equity: bool = False,
    max_workers: int | None = None,
) -> List[Individual]:
    """Run a simple GA over TCNAction hyperparameters."""
    # Optional GA config to constrain hyperparameter ranges.
    ga_config: Dict[str, Any] | None = base_cfg.get("_ga_config")  # type: ignore[assignment]

    # Initialize population
    population: List[Individual] = [
        Individual(hp=_sample_hp_for_ga(rng, ga_config)) for _ in range(population_size)
    ]

    adapter_spec = "wfo.adapters.tcn_scalar_threshold:TCNActionAdapter"

    for gen in range(generations):
        print(f"=== GA GENERATION {gen} ===", flush=True)

        # Evaluate population in parallel.
        # We cap max_workers to avoid exhausting RAM when models are large.
        if max_workers is None:
            # Conservative default: don't flood all cores by default.
            max_workers = min(4, os.cpu_count() or 1)
        # Ensure sensible bounds.
        max_workers = max(1, int(max_workers))
        max_workers = min(max_workers, population_size, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, indiv in enumerate(population):
                indiv_id = idx
                # Log start
                print(
                    json.dumps(
                        {
                            "event": "ga_individual_start",
                            "generation": gen,
                            "idx": idx,
                            "hp": indiv.hp,
                        }
                    ),
                    flush=True,
                )
                fut = ex.submit(_eval_individual, base_cfg, indiv.hp, indiv_id)
                futures[fut] = indiv

            for fut in as_completed(futures):
                indiv = futures[fut]
                score, summary, run_dir, indiv_id = fut.result()
                indiv.score = float(score)
                indiv.summary = summary
                indiv.run_dir = run_dir
                print(
                    json.dumps(
                        {
                            "event": "ga_individual_result",
                            "generation": gen,
                            "idx": indiv_id,
                            "score": indiv.score,
                            "summary": indiv.summary,
                            "run_dir": indiv.run_dir,
                        }
                    ),
                    flush=True,
                )
                if plot_equity:
                    _plot_global_equity(Path(run_dir))

        # Sort by fitness (descending)
        population.sort(key=lambda ind: ind.score, reverse=True)

        # Log generation summary
        best = population[0]
        print(
            json.dumps(
                {
                    "event": "ga_generation_summary",
                    "generation": gen,
                    "best_score": best.score,
                    "best_hp": best.hp,
                    "best_summary": best.summary,
                    "best_run_dir": best.run_dir,
                }
            ),
            flush=True,
        )

        # Last generation: don't create next population
        if gen == generations - 1:
            break

        # Selection + reproduction
        next_pop: List[Individual] = []
        elite_count = max(1, population_size // 4)
        # Elitism: carry top elites unchanged
        next_pop.extend(population[:elite_count])

        # Fill the rest via crossover + mutation
        while len(next_pop) < population_size:
            # Tournament selection of parents
            def pick_parent() -> Individual:
                k = min(3, len(population))
                cand = rng.sample(population, k)
                cand.sort(key=lambda ind: ind.score, reverse=True)
                return cand[0]

            p1 = pick_parent()
            p2 = pick_parent()
            child_hp = _crossover(rng, p1.hp, p2.hp)
            child_hp = _mutate(rng, child_hp, mutation_prob=0.3, ga_config=ga_config)
            next_pop.append(Individual(hp=child_hp))

        population = next_pop

    return population


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GA hyperparameter search for TCNActionAdapter over 2023 (3m/1w/1w WFO)"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_2023_weeks.json",
        help="Path to base JSON config (geometry/instrument/etc.)",
    )
    ap.add_argument("--population", type=int, default=24, help="Population size")
    ap.add_argument("--generations", type=int, default=10, help="Number of generations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--plot-equity",
        action="store_true",
        help="Render ASCII equity curve for each individual's global trades",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override TCN training epochs in base config",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel WFO workers (cap to reduce RAM usage)",
    )
    ap.add_argument(
        "--ga-config",
        type=str,
        default=None,
        help=(
            "Optional JSON file with compact GA hyperparameter ranges. "
            "Each key is a hyperparameter name, value is [start, end] or [start, step, end]."
        ),
    )
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))
    # Optional CLI override for epochs so GA runs can use fewer epochs
    # without editing the JSON config.
    if args.epochs is not None:
        base_cfg["epochs"] = int(args.epochs)

    # Optional GA hyperparameter range config
    if args.ga_config:
        ga_path = Path(args.ga_config)
        with ga_path.open("r") as f:
            ga_data = json.load(f)
        if not isinstance(ga_data, dict):
            raise SystemExit("--ga-config JSON must contain an object at the top level")
        # Stash under a private key so run_ga can access it without changing its signature.
        base_cfg["_ga_config"] = ga_data

    rng = random.Random(int(args.seed))

    population = run_ga(
        base_cfg,
        population_size=int(args.population),
        generations=int(args.generations),
        rng=rng,
        plot_equity=bool(args.plot_equity),
        max_workers=args.max_workers,
    )

    # Final summary: best individual
    population.sort(key=lambda ind: ind.score, reverse=True)
    best = population[0]
    print("\n=== GA final best individual ===")
    print(
        json.dumps(
            {
                "best_score": best.score,
                "best_hp": best.hp,
                "best_summary": best.summary,
                "best_run_dir": best.run_dir,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()

