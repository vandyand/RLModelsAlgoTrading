from __future__ import annotations

"""
Bayesian (Optuna) hyperparameter search for TCNActionAdapter over a WFO period.

This is an alternative to the GA-based search in `tcn_action_ga_search.py`.
It reuses the same WFO wiring and objective, but uses Optuna to explore the
hyperparameter space more efficiently.

Usage (from forex-rl root):

  # Example NQ run
  python3 -m wfo.tcn_action_optuna_search \
      --base-config wfo/tcn_nq_2023_weeks.json \
      --trials 50 \
      --seed 51

Requirements:
  pip install optuna
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import optuna  # type: ignore[import-untyped]

from .model_zoo_2023 import load_base_config
from .tcn_action_ga_search import _eval_individual, _plot_global_equity


def _suggest_hyperparams(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest a TCNAction hyperparameter set for this trial.

    We largely mirror the ranges from `tcn_action_random_search.sample_hyperparams`,
    but fix num_blocks and kernel_size to 1 for speed (especially for NQ).
    """

    hp: Dict[str, Any] = {}

    # Core TCN structure
    # Upper bounds chosen so the range is divisible by `step` and Optuna
    # does not need to adjust them (avoids warnings).
    hp["tcn_lookback"] = trial.suggest_int("tcn_lookback", 12, 236, step=8)
    hp["tcn_target_horizon"] = trial.suggest_int("tcn_target_horizon", 2, 190, step=4)
    hp["tcn_hidden"] = trial.suggest_int("tcn_hidden", 8, 32, step=4)
    hp["tcn_blocks"] = 1  # fixed
    hp["tcn_kernel"] = 1  # fixed
    hp["tcn_dropout"] = trial.suggest_float("tcn_dropout", 0.0, 0.15)

    # Label thresholds in forward-return space.
    hp["label_long_thresh"] = trial.suggest_float("label_long_thresh", 0.0005, 0.0080)
    hp["label_short_thresh"] = -trial.suggest_float("label_short_thresh_abs", 0.0005, 0.0080)

    # Quantile-based labeler parameters (only used when label_mode == "quantile").
    short_q_base = trial.suggest_float("label_short_quantile_base", 0.05, 0.35)
    long_q_min = max(0.55, short_q_base + 0.10)
    long_q = trial.suggest_float("label_long_quantile", long_q_min, 0.95)
    hp["label_short_quantile"] = short_q_base
    hp["label_long_quantile"] = long_q

    # Trade cost is usually fixed at 1 bp.
    hp["tcn_trade_cost_bps"] = float(base_cfg.get("tcn_trade_cost_bps", 1.0))

    # NQ-only causal rule parameters are now largely unused by the NQ adapter
    # (which defers to the classifier's validate), but we keep them here so
    # configs remain compatible. They default to the base config values.
    hp["nq_ret_thresh"] = float(base_cfg.get("nq_ret_thresh", 0.0005))
    hp["nq_vol_lookback"] = int(base_cfg.get("nq_vol_lookback", 64))
    hp["nq_z_thresh"] = float(base_cfg.get("nq_z_thresh", 1.0))

    return hp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna Bayesian hyperparameter search for TCNActionAdapter over a WFO period"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_2023_weeks.json",
        help="Path to base JSON config (geometry/instrument/etc.)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Number of Optuna trials",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Optuna sampler",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override TCN training epochs in base config (for all trials).",
    )
    ap.add_argument(
        "--plot-equity",
        action="store_true",
        help="Render ASCII equity curve for each trial's global trades.",
    )
    args = ap.parse_args()

    base_cfg: Dict[str, Any] = load_base_config(Path(args.base_config))
    if args.epochs is not None:
        base_cfg["epochs"] = int(args.epochs)

    def objective(trial: optuna.Trial) -> float:
        # Optionally let Optuna choose the labeling mode per trial so it can
        # discover whether BPS or quantile labeling works better for the
        # current instrument/year.
        base_cfg_local = dict(base_cfg)
        base_cfg_local["label_mode"] = trial.suggest_categorical(
            "label_mode", ["bps", "quantile"]
        )

        hp = _suggest_hyperparams(trial, base_cfg_local)
        # Use trial.number as an identifier for logging/run_dir tagging.
        score, summary, run_dir, _ = _eval_individual(
            base_cfg_local, hp, individual_id=trial.number
        )
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("run_dir", run_dir)

        # Optionally show equity curve for this trial.
        if args.plot_equity:
            _plot_global_equity(Path(run_dir))

        return float(score)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(args.trials))

    best = study.best_trial
    best_hp = {
        "tcn_lookback": int(best.params.get("tcn_lookback", 0)),
        "tcn_target_horizon": int(best.params.get("tcn_target_horizon", 0)),
        "tcn_hidden": int(best.params.get("tcn_hidden", 0)),
        "tcn_blocks": 1,
        "tcn_kernel": 1,
        "tcn_dropout": float(best.params.get("tcn_dropout", 0.0)),
        "label_long_thresh": float(best.params.get("label_long_thresh", 0.0)),
        "label_short_thresh": -float(best.params.get("label_short_thresh_abs", 0.0)),
        "label_long_quantile": float(best.params.get("label_long_quantile", 0.7)),
        "label_short_quantile": float(best.params.get("label_short_quantile_base", 0.3)),
        "tcn_trade_cost_bps": float(base_cfg.get("tcn_trade_cost_bps", 1.0)),
        "label_mode": best.params.get("label_mode", base_cfg.get("label_mode", "bps")),
    }
    summary = best.user_attrs.get("summary", {})
    run_dir = best.user_attrs.get("run_dir", "")

    print("\n=== Optuna final best trial ===")
    payload = {
        "best_score": float(best.value),
        "best_hp": best_hp,
        "best_summary": summary,
        "best_run_dir": run_dir,
    }

    print(json.dumps(payload), flush=True)

    # Also print a compact human-readable summary for quick inspection.
    if summary:
        sharpe = summary.get("global_sharpe", 0.0)
        cumret = summary.get("global_cum_return", 0.0)
        max_dd = summary.get("global_max_dd", 0.0)
        trades = int(summary.get("total_trades", 0) or 0)
        long_tr = int(summary.get("total_long_trades", 0) or 0)
        short_tr = int(summary.get("total_short_trades", 0) or 0)
        print(
            f"\nBest trial metrics: sharpe={sharpe:.3f}, cum_ret={cumret:.3f}, "
            f"max_dd={max_dd:.3f}, trades={trades}, longs={long_tr}, shorts={short_tr}",
            flush=True,
        )


if __name__ == "__main__":  # pragma: no cover
    main()

