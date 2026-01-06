#!/usr/bin/env python3
"""
Optuna-based hyperparameter search for the crypto offline actor-critic.

This script wraps `crypto_offline_actor_critic` and uses Optuna to search over:
- Policy/value MLP architectures
- Actor noise (sigma)
- Entropy and value loss coefficients
- Reward scale
- Max notional and discount factor (gamma)

Objective:
- Maximize the final validation Sharpe-like reward
  (`val_avg_daily_reward`) over a fixed train/val split in time.

Usage (from forex-rl root):

  python -m forex-rl.actor-critic.crypto_optuna_search \\
      --start 2021-01-01 --end 2025-12-31 \\
      --trials 40 --epochs 12

Notes:
- Data (X, R) is built once per run and reused across trials.
- All price data is fetched via `candle_cache_service` (CANDLE_CACHE_BASE_URL).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna  # type: ignore[import-untyped]
import torch
import pandas as pd

from .crypto_offline_actor_critic import (
    Config,
    DEFAULT_CRYPTO_SYMBOLS,
    build_crypto_dataset,
    standardize_fit,
    train_actor_critic,
    _build_tcn_features_from_checkpoint,
)


def _split_train_val(
    Xn,
    R,
    val_frac: float,
) -> Tuple[Any, Any, Any, Any]:
    n = len(Xn)
    split = int(n * (1.0 - val_frac))
    if split < 1 or split >= n:
        split = max(1, n - 1)
    X_train = Xn.iloc[:split]
    R_train = R.iloc[:split]
    X_val = Xn.iloc[split:]
    R_val = R.iloc[split:]
    return X_train, R_train, X_val, R_val


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna search for crypto offline actor-critic hyperparameters.")
    ap.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_CRYPTO_SYMBOLS),
        help="Comma-separated list of crypto/USD symbols (default: built-in 19-symbol universe).",
    )
    ap.add_argument("--environment", type=str, default="practice", choices=["practice", "live"])
    ap.add_argument(
        "--candle-cache-base-url",
        type=str,
        default=os.environ.get("CANDLE_CACHE_BASE_URL", "http://127.0.0.1:9100"),
    )
    ap.add_argument("--start", type=str, default="2021-01-01")
    ap.add_argument("--end", type=str, default="2025-12-31")
    ap.add_argument("--epochs", type=int, default=12, help="Training epochs per trial (default: 12).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (default: 0.2).",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of independent repeats per trial (different seeds) whose "
            "val_avg_daily_reward is averaged to form the objective. "
            "Increases robustness to noise at the cost of runtime (default: 1)."
        ),
    )
    ap.add_argument(
        "--study-name",
        type=str,
        default="crypto_ac_optuna",
        help="Optuna study name (for RDB storage, if used).",
    )
    ap.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///crypto_ac_optuna.db). Default: in-memory.",
    )
    ap.add_argument(
        "--direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction for val_avg_daily_reward (default: maximize).",
    )
    ap.add_argument(
        "--best-out",
        type=str,
        default="crypto_optuna_best.json",
        help=(
            "Path to JSON file where the best trial's params/attrs will be "
            "saved whenever a new best trial is found (default: crypto_optuna_best.json)."
        ),
    )
    ap.add_argument(
        "--use-tcn",
        action="store_true",
        help="When set, load a pre-trained TCN checkpoint and append its features to X "
        "before running Optuna; by default, use only raw OHLCV-derived + time features.",
    )
    args = ap.parse_args()

    # Seed base RNGs for reproducibility of data split
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve root for relative output paths
    here = os.path.abspath(os.path.dirname(__file__))
    fx_root = os.path.abspath(os.path.join(here, os.pardir))

    best_out_path = args.best_out
    if not os.path.isabs(best_out_path):
        best_out_path = os.path.join(fx_root, best_out_path)

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided for crypto Optuna search.")

    # Base config used for dataset construction; training hyperparams come from trials.
    base_cfg = Config(
        symbols=symbols,
        environment=args.environment,
        candle_cache_base_url=args.candle_cache_base_url,
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(
        json.dumps(
            {
                "status": "optuna_build_dataset",
                "symbols": symbols,
                "start": args.start,
                "end": args.end,
                "candle_cache_base_url": args.candle_cache_base_url,
            }
        ),
        flush=True,
    )

    # Build dataset once and reuse across all trials.
    X, R, dates = build_crypto_dataset(base_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally augment features with pre-trained TCN outputs.
    if args.use_tcn:
        tcn_ckpt_path = os.path.join(fx_root, "actor-critic", "checkpoints", "crypto_tcn.pt")
        tcn_feats = _build_tcn_features_from_checkpoint(
            X, base_cfg, symbols, ckpt_path=tcn_ckpt_path, device=device
        )
        X_aug = pd.concat([X, tcn_feats], axis=1)
    else:
        X_aug = X

    # Log dataset geometry for debugging / performance analysis.
    print(
        json.dumps(
            {
                "status": "optuna_dataset_ready",
                "use_tcn": bool(args.use_tcn),
                "rows": int(len(X_aug)),
                "cols": int(X_aug.shape[1]),
            }
        ),
        flush=True,
    )

    Xn, stats = standardize_fit(X_aug)
    X_train, R_train, X_val, R_val = _split_train_val(Xn, R, val_frac=args.val_frac)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        # 1) Policy MLP depth and widths
        pol_depth = trial.suggest_int("policy_depth", 1, 3)
        policy_hidden: List[int] = []
        for i in range(pol_depth):
            h = trial.suggest_int(f"policy_hidden_{i}", 64, 512, step=64)
            policy_hidden.append(h)

        # 2) Value MLP depth and widths
        val_depth = trial.suggest_int("value_depth", 1, 3)
        value_hidden: List[int] = []
        for i in range(val_depth):
            h = trial.suggest_int(f"value_hidden_{i}", 64, 512, step=64)
            value_hidden.append(h)

        # 3) RL hyperparameters
        actor_sigma = trial.suggest_float("actor_sigma", 0.05, 0.6)
        entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 5e-3, log=True)
        value_coef = trial.suggest_float("value_coef", 0.1, 1.0)
        reward_scale = trial.suggest_float("reward_scale", 0.5, 5.0)
        max_notional = trial.suggest_float("max_notional", 0.5, 5.0)
        gamma = trial.suggest_float("gamma", 0.95, 0.995)

        # Evaluate this hyperparameter setting multiple times with different seeds
        # and average the resulting validation rewards. This reduces sensitivity
        # to stochasticity in training at the cost of extra compute.
        rewards: List[float] = []
        num_repeats = max(1, int(args.repeats))

        for rep in range(num_repeats):
            cfg_seed = int(args.seed + trial.number * 1000 + rep)
            cfg = Config(
                symbols=symbols,
                environment=args.environment,
                candle_cache_base_url=args.candle_cache_base_url,
                start=args.start,
                end=args.end,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gamma=float(gamma),
                actor_sigma=float(actor_sigma),
                entropy_coef=float(entropy_coef),
                value_coef=float(value_coef),
                reward_scale=float(reward_scale),
                max_notional=float(max_notional),
                seed=cfg_seed,
                policy_hidden=policy_hidden,
                value_hidden=value_hidden,
            )

            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

            out = train_actor_critic(X_train, R_train, X_val, R_val, cfg, device)
            rewards.append(float(out.get("val_avg_daily_reward", 0.0)))

        val_reward = float(np.mean(rewards))

        # Report for monitoring
        trial.set_user_attr("val_avg_daily_reward", val_reward)
        trial.set_user_attr("val_avg_daily_reward_repeats", rewards)
        return val_reward

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction=args.direction,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction=args.direction)

    print(
        json.dumps(
            {
                "status": "optuna_start",
                "trials": args.trials,
                "study_name": args.study_name,
                "storage": args.storage or "in-memory",
                "best_out": best_out_path,
            }
        ),
        flush=True,
    )

    def _save_best_callback(study_obj: optuna.Study, trial: optuna.FrozenTrial) -> None:
        # Only act when this trial is the current best.
        if study_obj.best_trial.number != trial.number:
            return
        payload = {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "val_frac": float(args.val_frac),
            "repeats": int(args.repeats),
            "seed_base": int(args.seed),
            "study_name": args.study_name,
            "storage": args.storage or "in-memory",
            "best_value": float(trial.value) if trial.value is not None else None,
            "best_params": dict(trial.params),
            "best_attrs": dict(trial.user_attrs),
        }
        try:
            os.makedirs(os.path.dirname(best_out_path), exist_ok=True)
        except Exception:
            # Directory may already exist or be current dir; ignore errors here.
            pass
        with open(best_out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[_save_best_callback])

    best = study.best_trial
    print(
        json.dumps(
            {
                "status": "optuna_done",
                "best_value": best.value,
                "best_params": best.params,
                "best_attrs": best.user_attrs,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

