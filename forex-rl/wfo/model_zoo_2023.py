from __future__ import annotations

"""
Model zoo random search over multiple strategy families using WFO.

Geometry (fixed):
- Start/end: 2023-01-01 .. 2023-12-31
- Train: 12 weeks (~3 months)
- Validation: 1 week
- Step: 1 week

Families currently included:
- tcn_action : TCNActionAdapter (supervised {-1,0,+1} classifier)

NOTE: The old tcn_scalar family (scalar z + thresholds) is deprecated/disabled.

Usage (from forex-rl root):

  python3 -m wfo.model_zoo_2023 \\
      --base-config wfo/tcn_eurusd_2023_weeks.json \\
      --families tcn_action \\
      --trials-per-family 5 \\
      --seed 5

This will print highlighted JSON lines per trial and a final compact summary table.
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .core import WFOConfig, run_wfo, _instantiate_adapter, _render_chart
from .adapters.tcn_scalar_threshold import TCNActionAdapter
from .adapters.lean_models import MLPActionAdapter, RuleActionAdapter
from .tcn_action_random_search import (
    sample_hyperparams as sample_action_hyperparams,
)
import numpy as np


def load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Base config JSON must contain an object at the top level")
    return data


def build_wfo_cfg(
    base_cfg: Dict[str, Any],
    adapter_spec: str,
    adapter_kwargs: Dict[str, Any],
) -> WFOConfig:
    """Construct a WFOConfig using geometry from the base config.

    Defaults match the original 2023 setup (3m/1w/1w in weeks), but callers can
    override start/end and geometry fields in the JSON base config.
    """
    start = str(base_cfg.get("start", "2023-01-01"))
    end = str(base_cfg.get("end", "2023-12-31"))
    train_n = float(base_cfg.get("train_n", 12.0))
    val_n = float(base_cfg.get("val_n", 1.0))
    step_n = float(base_cfg.get("step_n", 1.0))
    unit = str(base_cfg.get("unit", "weeks"))

    base_gran = str(base_cfg.get("base_gran", "M5"))
    out_dir = str(base_cfg.get("out_dir", "wfo/runs"))
    rolling_from_yesterday = bool(base_cfg.get("rolling_from_yesterday", False))

    return WFOConfig(
        start=start,
        end=end,
        train_n=train_n,
        val_n=val_n,
        step_n=step_n,
        unit=unit,
        windows_limit=int(base_cfg.get("windows_limit", 0)),
        base_gran=base_gran,
        out_dir=out_dir,
        rolling_from_yesterday=rolling_from_yesterday,
        adapter_spec=adapter_spec,
        adapter_kwargs=adapter_kwargs,
        parallel=1,
        no_chart=True,
        chart_every=1,
        quiet=True,
        mode="fast",
    )


def compute_global_objective(run_dir: Path) -> Tuple[float, Dict[str, float]]:
    """Compute global objective from stitched trades.jsonl for a WFO run.

    - Sharpe: trade-based (mean(net_pnl) / std(net_pnl) over all trades)
    - Cum return: product(1 + net_pnl_i) - 1 over trades in time order
    - Max DD: from stitched equity curve over trades
    """
    trades_path = run_dir / "trades.jsonl"
    if not trades_path.exists():
        return 0.0, {}

    all_profits: list[float] = []
    all_sides: list[int] = []

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
                    all_sides.append(int(t.get("side", 0)))
                except Exception:
                    continue

    if not all_profits:
        return 0.0, {}

    profits = np.asarray(all_profits, dtype=float)
    mu = float(profits.mean())
    sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
    sharpe = float(mu / sigma) if sigma > 1e-12 else 0.0

    # Stitched equity over trades (1 + net_pnl_i)
    equity = np.cumprod(1.0 + profits)
    cum_return = float(equity[-1] - 1.0)
    if equity.size > 0:
        running_max = np.maximum.accumulate(equity)
        dd = (equity / np.clip(running_max, 1e-12, None)) - 1.0
        max_dd = float(np.min(dd))
    else:
        max_dd = 0.0

    gains = profits[profits > 0]
    losses = -profits[profits < 0]
    if losses.size > 0:
        pf = float(gains.sum() / max(1e-12, losses.sum()))
    elif gains.size > 0:
        pf = float("inf")
    else:
        pf = 0.0

    n_win = int((profits > 0).sum())
    n_loss = int((profits < 0).sum())
    win_rate = float(n_win / max(1, len(profits)))
    if n_loss > 0:
        win_loss = float(n_win / max(1, n_loss))
    elif n_win > 0:
        win_loss = float("inf")
    else:
        win_loss = 0.0

    # Mean PnL per winning / losing trade (global)
    if n_win > 0:
        mean_win = float(profits[profits > 0].mean())
    else:
        mean_win = 0.0
    if n_loss > 0:
        # report loss magnitude as positive number
        mean_loss = float((-profits[profits < 0]).mean())
    else:
        mean_loss = 0.0

    # Long/short breakdown
    sides = np.asarray(all_sides, dtype=int)
    long_mask = sides > 0
    short_mask = sides < 0
    long_trades = int(long_mask.sum())
    short_trades = int(short_mask.sum())
    long_win = int(((profits > 0) & long_mask).sum())
    long_loss = int(((profits < 0) & long_mask).sum())
    short_win = int(((profits > 0) & short_mask).sum())
    short_loss = int(((profits < 0) & short_mask).sum())

    summary = {
        "global_sharpe": sharpe,
        "global_cum_return": cum_return,
        "global_cum_return_bp": cum_return * 1e4,
        "global_max_dd": max_dd,
        "global_max_dd_bp": max_dd * 1e4,
        "global_profit_factor": pf,
        "global_win_rate": win_rate,
        "global_win_loss": win_loss,
        "mean_winning_trade_profit": mean_win,
        "mean_losing_trade_loss": mean_loss,
        "total_trades": int(len(profits)),
        "total_long_trades": long_trades,
        "total_short_trades": short_trades,
        "total_long_win_trades": long_win,
        "total_long_loss_trades": long_loss,
        "total_short_win_trades": short_win,
        "total_short_loss_trades": short_loss,
    }

    # Simple score: Sharpe primary, reward cum_return, penalize drawdown
    score = sharpe + 0.1 * cum_return - 0.1 * abs(max_dd)
    return score, summary


def _plot_global_equity(run_dir: Path) -> None:
    """Render an ASCII equity curve from stitched trades for a WFO run."""
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


def _run_trial_job(
    family: str,
    base_cfg: Dict[str, Any],
    adapter_spec: str,
    adapter_kwargs: Dict[str, Any],
    hp: Dict[str, Any],
    trial_index: int,
) -> Tuple[float, Dict[str, Any], Dict[str, float], str, int]:
    """Worker function to run a single trial in a separate process."""
    # Instantiate adapter inside the worker to avoid pickling issues
    adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)
    cfg = build_wfo_cfg(base_cfg, adapter_spec=adapter_spec, adapter_kwargs=adapter_kwargs)
    run_dir = Path(run_wfo(adapter, cfg))
    score, summary = compute_global_objective(run_dir)
    return score, hp, summary, str(run_dir), trial_index


def _sample_mlp_hyperparams(rng: random.Random) -> Dict[str, Any]:
    lookback_options = [16, 24, 32]
    horizon_options = [2, 4]
    hidden_options = [0, 16, 32]  # 0 => linear baseline

    label_long_thresh = rng.uniform(0.0003, 0.0010)
    label_short_thresh = -rng.uniform(0.0003, 0.0010)

    return {
        "lookback_bars": rng.choice(lookback_options),
        "target_horizon": rng.choice(horizon_options),
        "hidden_dim": rng.choice(hidden_options),
        "dropout": rng.uniform(0.0, 0.3),
        "label_long_thresh": label_long_thresh,
        "label_short_thresh": label_short_thresh,
        "trade_cost_bps": rng.choice([0.5, 1.0]),
    }


def _sample_rule_hyperparams(rng: random.Random) -> Dict[str, Any]:
    momentum_lookback = rng.choice([24, 48, 72])
    long_ret_bp = rng.uniform(3.0, 15.0)
    short_ret_bp = rng.uniform(3.0, 15.0)
    return {
        "momentum_lookback": int(momentum_lookback),
        "long_ret_bp": float(long_ret_bp),
        "short_ret_bp": float(short_ret_bp),
        "trade_cost_bps": rng.choice([0.5, 1.0]),
    }


def run_family_trials(
    family: str,
    base_cfg: Dict[str, Any],
    trials: int,
    rng: random.Random,
    plot_equity: bool = False,
) -> List[Tuple[float, Dict[str, Any], Dict[str, float], str]]:
    """Run random search for a single strategy family.

    Returns list of (score, hp, summary, run_dir).
    """
    results: List[Tuple[float, Dict[str, Any], Dict[str, float], str]] = []

    # Family-specific wiring
    candle_cache_base = str(base_cfg.get("candle_cache_base", "http://127.0.0.1:9100"))

    if family == "tcn_action":
        adapter_cls = TCNActionAdapter
        adapter_spec = "wfo.adapters.tcn_scalar_threshold:TCNActionAdapter"
        sample_hp = sample_action_hyperparams
        compute_obj = compute_global_objective
        fixed_adapter: Dict[str, Any] = {
            "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
            "grans": str(base_cfg.get("grans", "M5,H1,D")),
            "base_gran": str(base_cfg.get("tcn_base_gran", base_cfg.get("base_gran", "M5"))),
            "epochs": int(base_cfg.get("epochs", 5)),
            "lr": float(base_cfg.get("lr", 1e-3)),
            "candle_cache_base": candle_cache_base,
        }
    elif family == "mlp_action":
        adapter_cls = MLPActionAdapter
        adapter_spec = "wfo.adapters.lean_models:MLPActionAdapter"
        sample_hp = _sample_mlp_hyperparams
        compute_obj = compute_global_objective
        fixed_adapter = {
            "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
            "grans": str(base_cfg.get("grans", "M5,H1,D")),
            "base_gran": str(base_cfg.get("tcn_base_gran", base_cfg.get("base_gran", "M5"))),
            "epochs": 3,
            "lr": float(base_cfg.get("lr", 1e-3)),
            "candle_cache_base": candle_cache_base,
        }
    elif family == "rule_action":
        adapter_cls = RuleActionAdapter
        adapter_spec = "wfo.adapters.lean_models:RuleActionAdapter"
        sample_hp = _sample_rule_hyperparams
        compute_obj = compute_global_objective
        # Rule adapter is stateless; only needs instrument / grans / base_gran / candle_cache_base
        fixed_adapter = {
            "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
            "grans": str(base_cfg.get("grans", "M5,H1,D")),
            "base_gran": str(base_cfg.get("tcn_base_gran", base_cfg.get("base_gran", "M5"))),
            "candle_cache_base": candle_cache_base,
        }
    else:
        raise SystemExit(f"Unknown family: {family}")

    # Prepare all trial jobs (sampling HPs is done in the main process so seeding is deterministic)
    jobs: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    for i in range(int(trials)):
        hp = sample_hp(rng)
        adapter_kwargs = dict(fixed_adapter)

        if family == "tcn_action":
            adapter_kwargs.update(
                lookback_bars=int(hp["tcn_lookback"]),
                target_horizon=int(hp["tcn_target_horizon"]),
                hidden_channels=int(hp["tcn_hidden"]),
                num_blocks=int(hp["tcn_blocks"]),
                kernel_size=int(hp["tcn_kernel"]),
                dropout=float(hp["tcn_dropout"]),
                label_long_thresh=float(hp["label_long_thresh"]),
                label_short_thresh=float(hp["label_short_thresh"]),
                trade_cost_bps=float(hp["tcn_trade_cost_bps"]),
            )
        elif family == "mlp_action":
            adapter_kwargs.update(
                lookback_bars=int(hp["lookback_bars"]),
                target_horizon=int(hp["target_horizon"]),
                hidden_dim=int(hp["hidden_dim"]),
                dropout=float(hp["dropout"]),
                label_long_thresh=float(hp["label_long_thresh"]),
                label_short_thresh=float(hp["label_short_thresh"]),
                trade_cost_bps=float(hp["trade_cost_bps"]),
            )
        elif family == "rule_action":
            adapter_kwargs.update(
                momentum_lookback=int(hp["momentum_lookback"]),
                long_ret_bp=float(hp["long_ret_bp"]),
                short_ret_bp=float(hp["short_ret_bp"]),
                trade_cost_bps=float(hp["trade_cost_bps"]),
            )

        trial_idx = i + 1
        # Highlighted start marker + JSON for easier grep/visual scanning
        print(f"=== ZOO_TRIAL_START [{family}] #{trial_idx} ===", flush=True)
        print(
            json.dumps(
                {
                    "event": "trial_start",
                    "family": family,
                    "trial": trial_idx,
                    "hp": hp,
                }
            ),
            flush=True,
        )
        jobs.append((hp, adapter_kwargs, trial_idx))

    if not jobs:
        return results

    # Run trials in parallel across processes to exploit multiple CPU cores.
    max_workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_trial = {}
        for hp, adapter_kwargs, trial_idx in jobs:
            fut = ex.submit(
                _run_trial_job,
                family,
                base_cfg,
                adapter_spec,
                adapter_kwargs,
                hp,
                trial_idx,
            )
            future_to_trial[fut] = trial_idx

        for fut in as_completed(future_to_trial):
            score, hp, summary, run_dir, trial_idx = fut.result()
            # Highlighted result marker + JSON
            print(f"=== ZOO_TRIAL_RESULT [{family}] #{trial_idx} ===", flush=True)
            print(
                json.dumps(
                    {
                        "event": "trial_result",
                        "family": family,
                        "trial": trial_idx,
                        "score": score,
                        "summary": summary,
                        "run_dir": run_dir,
                    }
                ),
                flush=True,
            )
            if plot_equity:
                _plot_global_equity(Path(run_dir))
            results.append((score, hp, summary, run_dir))

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Model zoo random search over multiple strategy families for 2023 (3m/1w/1w WFO)"
    )
    ap.add_argument(
        "--base-config",
        default="wfo/tcn_eurusd_2023_weeks.json",
        help="Path to base JSON config (geometry/instrument/etc.)",
    )
    ap.add_argument(
        "--families",
        default="tcn_action,mlp_action,rule_action",
        help="Comma-separated list of families to run (e.g. tcn_action,mlp_action,rule_action)",
    )
    ap.add_argument(
        "--trials-per-family",
        type=int,
        default=3,
        help="Number of random trials per family",
    )
    ap.add_argument(
        "--plot-equity",
        action="store_true",
        help="Render ASCII equity curve for each trial's global trades",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))
    rng = random.Random(int(args.seed))

    families = [f.strip() for f in args.families.split(",") if f.strip()]

    all_results: Dict[str, List[Tuple[float, Dict[str, Any], Dict[str, float], str]]] = {}

    for fam in families:
        fam_results = run_family_trials(
            fam,
            base_cfg,
            int(args.trials_per_family),
            rng,
            plot_equity=bool(args.plot_equity),
        )
        all_results[fam] = fam_results

    # Compact summary table
    print("\n=== Model zoo 2023 summary ===")
    header = [
        "family",
        "best_score",
        "best_mean_sharpe",
        "best_mean_cum_return",
        "best_total_trades",
        "best_run_dir",
    ]
    print(",".join(header))
    for fam, fam_results in all_results.items():
        if not fam_results:
            continue
        best = max(fam_results, key=lambda t: t[0])
        best_score, _, best_summary, best_run = best
        # Use global trade-based metrics from compute_global_objective
        row = [
            fam,
            f"{best_score:.4f}",
            f"{best_summary.get('global_sharpe', 0.0):.4f}",
            f"{best_summary.get('global_cum_return', 0.0):.4f}",
            str(best_summary.get("total_trades", 0)),
            best_run,
        ]
        print(",".join(row))


if __name__ == "__main__":  # pragma: no cover
    main()

