from __future__ import annotations
import argparse
import importlib.util
from typing import Optional
import os
import pandas as pd

from config import Config
from features_loader import load_feature_panel
from fitness import ballast_score, FitnessFn
from engine import run_ga
from backtester import evaluate_genome
from model import MultiHeadGenome
import json


def load_fitness(path_or_name: Optional[str]) -> FitnessFn:
    """Resolve a fitness function without hard-requiring optional symbols.

    Supports:
    - Empty/None -> ballast_score
    - Simple names: "ballast"; "daily_trades_consistency" if present, else fallback to ballast
    - Module path:attr, e.g. "ga-multi20.fitness:daily_trades_consistency_score"
    - Importable module exposing attr "fitness"
    - File path to module exposing attr "fitness"
    """
    from fitness import ballast_score as _ballast

    if not path_or_name:
        return _ballast
    name = path_or_name.strip()
    lname = name.lower()
    # Simple aliases
    if lname == "ballast":
        return _ballast
    if lname in {"daily_trades_consistency", "daily_trades", "dtc"}:
        try:
            from fitness import daily_trades_consistency_score as _dtc  # type: ignore
            return _dtc
        except Exception:
            print("[warn] daily_trades_consistency_score not found; falling back to ballast")
            return _ballast
    # module:attr form
    if ":" in name:
        mod_name, attr = name.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
        return fn  # type: ignore[return-value]
    # Try module exposing `fitness`
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "fitness", _ballast)
    except Exception:
        # Try loading from file path exposing `fitness`
        spec = importlib.util.spec_from_file_location("custom_fitness", name)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            return getattr(mod, "fitness", _ballast)
    return _ballast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GA multi-20 NN strategy and backtest")
    parser.add_argument("--instruments", default=",")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--start", default="", help="YYYY-MM-DD start date (UTC). If omitted, use full history.")
    parser.add_argument("--end", default="", help="YYYY-MM-DD end date (UTC). If omitted, use last available.")
    parser.add_argument("--trade-cost", type=float, default=0.0002, help="Cost per instrument flip (absolute return deduction)")
    parser.add_argument(
        "--fitness",
        default="",
        help=(
            "Fitness to use: empty=ballast | 'ballast' | "
            "'module:attr' | importable module exposing 'fitness'"
        ),
    )
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--hidden", default="512,256,128")
    parser.add_argument("--n-jobs", type=int, default=0, help="Parallel workers (0=auto cpu_count, 1=serial)")
    parser.add_argument("--downsample", type=int, default=1, help="Evaluate every k-th bar (>=1)")
    # segments removed; fitness gets full trades instead
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mutation", type=float, default=None, help="Override mutation probability (0..1)")
    parser.add_argument("--weight-sigma", type=float, default=None, help="Override weight mutation sigma")
    parser.add_argument("--affine-sigma", type=float, default=None, help="Override input affine mutation sigma")
    parser.add_argument("--enter-long", type=float, default=None)
    parser.add_argument("--exit-long", type=float, default=None)
    parser.add_argument("--enter-short", type=float, default=None)
    parser.add_argument("--exit-short", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["band","absolute"], default=None)
    parser.add_argument("--band-enter", type=float, default=None)
    parser.add_argument("--band-exit", type=float, default=None)
    parser.add_argument("--resume", default="", help="Path to best_genome-*.json to warm-start population")
    args = parser.parse_args()

    cfg = Config()
    # Use defaults unless instruments explicitly provided
    if args.instruments.strip():
        csv = args.instruments.strip()
        if csv != ",":
            cfg.instruments = [s.strip().upper() for s in csv.split(",") if s.strip()]
    cfg.lookback_days = int(args.lookback_days)
    cfg.start_date = args.start.strip() or None
    cfg.end_date = args.end.strip() or None
    cfg.trade_cost = float(args.trade_cost)
    cfg.population = int(args.population)
    cfg.generations = int(args.generations)
    cfg.hidden_layers = [int(s) for s in args.hidden.split(",") if s.strip()]
    cfg.seed = (int(args.seed) if args.seed is not None else None)
    cfg.n_jobs = int(args.n_jobs)
    cfg.downsample = max(1, int(args.downsample))
    # Optional overrides
    if args.mutation is not None:
        cfg.mutation_prob = float(args.mutation)
    if args.weight_sigma is not None:
        cfg.weight_sigma = float(args.weight_sigma)
    if args.affine_sigma is not None:
        cfg.affine_sigma = float(args.affine_sigma)
    if args.enter_long is not None:
        cfg.enter_long = float(args.enter_long)
    if args.exit_long is not None:
        cfg.exit_long = float(args.exit_long)
    if args.enter_short is not None:
        cfg.enter_short = float(args.enter_short)
    if args.exit_short is not None:
        cfg.exit_short = float(args.exit_short)
    if args.threshold_mode is not None:
        cfg.threshold_mode = str(args.threshold_mode)
    if args.band_enter is not None:
        cfg.band_enter = float(args.band_enter)
    if args.band_exit is not None:
        cfg.band_exit = float(args.band_exit)

    print(f"Loading features for {len(cfg.instruments)} instruments...")
    X_panel, closes = load_feature_panel(cfg)
    # Flatten X to (T, 20*20) with deterministic column order
    flat_cols = []
    for inst in cfg.instruments:
        for col in X_panel[inst].columns:
            flat_cols.append((inst, col))
    X = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
    X.columns = [f"{i}::{c}" for i, c in X.columns]

    fit_fn = load_fitness(args.fitness)

    print("Running GA...")
    # Optional warm-start from checkpoint
    seed_genomes = None
    if args.resume.strip():
        try:
            with open(args.resume.strip(), "r") as f:
                d = json.load(f)
            seed_genomes = [MultiHeadGenome.from_dict(d)]
            print(f"Loaded seed genome from {args.resume.strip()}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint: {e}")
            seed_genomes = None
    # Callback no longer prints; engine prints single one-liner per generation
    state = run_ga(cfg, X, closes, fitness_fn=fit_fn, seed_genomes=seed_genomes)

    print(f"Best score: {state.best_score:.6f}")
    score, result = evaluate_genome(
        state.best_genome, X, closes, cfg.trade_cost, fitness_fn=fit_fn,
        thresholds=(cfg.enter_long, cfg.exit_long, cfg.enter_short, cfg.exit_short) if cfg.threshold_mode=="absolute" else None,
        mode=("absolute" if cfg.threshold_mode=="absolute" else "band"),
        band_enter=cfg.band_enter, band_exit=cfg.band_exit,
    )
    print(f"Eval score: {score:.6f}")
    print("Portfolio metrics:")
    print({k: getattr(result.metrics, k) for k in ("cum_return","sharpe","sortino","max_drawdown","trades")})
    print(f"Trades emitted: {len(result.trades)}")

    # Save best genome and meta-params for later reuse
    from datetime import datetime, UTC
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    save_path = f"ga-multi20/checkpoints/best_genome-{ts}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(state.best_genome.to_dict(), f)
    print(f"Saved best genome to {save_path}")
    meta = {
        "timestamp": ts,
        "instruments": cfg.instruments,
        "threshold_mode": cfg.threshold_mode,
        "band_enter": cfg.band_enter,
        "band_exit": cfg.band_exit,
        "enter_long": cfg.enter_long,
        "exit_long": cfg.exit_long,
        "enter_short": cfg.enter_short,
        "exit_short": cfg.exit_short,
        "hidden_layers": cfg.hidden_layers,
        "population": cfg.population,
        "generations": cfg.generations,
        "mutation_prob": cfg.mutation_prob,
        "weight_sigma": cfg.weight_sigma,
        "affine_sigma": cfg.affine_sigma,
        "trade_cost": cfg.trade_cost,
        "lookback_days": cfg.lookback_days,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
    }
    meta_path = f"ga-multi20/checkpoints/best_genome-{ts}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Saved run meta to {meta_path}")


if __name__ == "__main__":
    main()
