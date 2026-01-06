from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .core import WFOConfig, run_wfo
from .adapters.tcn_scalar_threshold import TCNScalarThresholdAdapter


def load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Base config JSON must contain an object at the top level")
    return data


def sample_hyperparams(rng: random.Random) -> Dict[str, Any]:
    """Randomly sample a hyperparameter set for the TCN scalar adapter.

    Adjust ranges as needed; this is a reasonable starting point.
    """
    # Keep ranges modest so per-trial training stays fast.
    # Complexity is roughly O(lookback * hidden * blocks * kernel * epochs * windows),
    # so we intentionally cap these.
    lookback_options = [48, 64, 96, 128]
    horizon_options = [2, 4]
    hidden_options = [16, 32]
    blocks_options = [1, 2, 3]
    # Thresholds in z-scored space (~N(0,1)).
    # Long side uses positive bands; short side uses symmetric negative bands.
    enter_long = rng.uniform(0.6, 0.9)
    exit_long = rng.uniform(0.3, min(0.7, enter_long - 0.05))
    # Sample magnitudes, then map to negative thresholds with exit_short > enter_short.
    short_enter_mag = rng.uniform(0.6, 0.9)
    short_exit_mag = rng.uniform(0.3, min(0.7, short_enter_mag - 0.05))
    enter_short = -short_enter_mag
    exit_short = -short_exit_mag

    return {
        "tcn_lookback": rng.choice(lookback_options),
        "tcn_target_horizon": rng.choice(horizon_options),
        "tcn_hidden": rng.choice(hidden_options),
        "tcn_blocks": rng.choice(blocks_options),
        "tcn_kernel": rng.choice([3, 5]),
        "tcn_dropout": rng.uniform(0.0, 0.2),
        "enter_long": enter_long,
        "exit_long": exit_long,
        "enter_short": enter_short,
        "exit_short": exit_short,
        # Small grid for trade cost
        "tcn_trade_cost_bps": rng.choice([0.5, 1.0]),
    }


def compute_objective(run_dir: Path) -> Tuple[float, Dict[str, float]]:
    """Compute an objective (avg Sharpe) from a WFO run directory.

    Returns (score, per-metric summary).
    """
    windows_path = run_dir / "windows.jsonl"
    rows: List[Dict[str, Any]] = []
    with windows_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return 0.0, {}
    sharpes: List[float] = []
    cumrets: List[float] = []
    dds: List[float] = []
    trades: List[float] = []
    tims: List[float] = []  # time in market
    win_rates: List[float] = []
    win_losses: List[float] = []
    pfs: List[float] = []
    r2s: List[float] = []
    long_trades: List[float] = []
    short_trades: List[float] = []
    long_win_trades: List[float] = []
    long_loss_trades: List[float] = []
    short_win_trades: List[float] = []
    short_loss_trades: List[float] = []
    z_means: List[float] = []
    z_stds: List[float] = []
    pos_long_fracs: List[float] = []
    pos_short_fracs: List[float] = []
    pos_flat_fracs: List[float] = []
    for r in rows:
        # Newer WFO writes metrics at the top level; fall back to nested dict.
        m = r.get("metrics") or r
        sharpes.append(float(m.get("sharpe") or 0.0))
        cumrets.append(float(m.get("cum_return") or 0.0))
        dds.append(float(m.get("max_dd") or 0.0))
        trades.append(float(m.get("trades") or 0.0))
        tims.append(float(m.get("time_in_mkt") or 0.0))
        win_rates.append(float(m.get("win_rate") or 0.0))
        win_losses.append(float(m.get("win_loss") or 0.0))
        pfs.append(float(m.get("profit_factor") or 0.0))
        r2s.append(float(m.get("equity_r2") or 0.0))
        long_trades.append(float(m.get("long_trades") or 0.0))
        short_trades.append(float(m.get("short_trades") or 0.0))
        long_win_trades.append(float(m.get("long_win_trades") or 0.0))
        long_loss_trades.append(float(m.get("long_loss_trades") or 0.0))
        short_win_trades.append(float(m.get("short_win_trades") or 0.0))
        short_loss_trades.append(float(m.get("short_loss_trades") or 0.0))
        # Diagnostics from adapter (may be missing in older runs)
        z_means.append(float(m.get("z_mean") or 0.0))
        z_stds.append(float(m.get("z_std") or 0.0))
        pos_long_fracs.append(float(m.get("pos_frac_long") or 0.0))
        pos_short_fracs.append(float(m.get("pos_frac_short") or 0.0))
        pos_flat_fracs.append(float(m.get("pos_frac_flat") or 0.0))

    # Convert to numpy for convenience on rate-like metrics
    arr_rates = np.array(
        [sharpes, cumrets, dds, tims, win_rates, win_losses, pfs, r2s],
        dtype=float,
    )
    mean_sharpe = float(np.mean(arr_rates[0]))
    mean_cum = float(np.mean(arr_rates[1]))
    mean_dd = float(np.mean(arr_rates[2]))
    mean_tim = float(np.mean(arr_rates[3]))
    mean_wr = float(np.mean(arr_rates[4]))
    mean_wl = float(np.mean(arr_rates[5]))
    mean_pf = float(np.mean(arr_rates[6]))
    mean_r2 = float(np.mean(arr_rates[7]))

    # Counts are summed across windows; these will be integers in practice.
    total_trades = int(round(float(np.sum(trades))))
    total_long_tr = int(round(float(np.sum(long_trades))))
    total_short_tr = int(round(float(np.sum(short_trades))))
    total_long_win = int(round(float(np.sum(long_win_trades))))
    total_long_loss = int(round(float(np.sum(long_loss_trades))))
    total_short_win = int(round(float(np.sum(short_win_trades))))
    total_short_loss = int(round(float(np.sum(short_loss_trades))))

    # Hard filters to eliminate degenerate solutions
    # - Require at least a few closed trades across windows
    # - Require both long and short trades to appear
    # - Strongly penalize "always in market" styles
    mean_pos_short = float(np.mean(pos_short_fracs)) if pos_short_fracs else 0.0
    mean_pos_long = float(np.mean(pos_long_fracs)) if pos_long_fracs else 0.0
    if total_trades < 4 or total_long_tr == 0 or total_short_tr == 0 or mean_tim > 0.98:
        penalty_summary = {
            "mean_sharpe": mean_sharpe,
            "mean_cum_return": mean_cum,
            "mean_max_dd": mean_dd,
            "mean_time_in_mkt": mean_tim,
            "mean_win_rate": mean_wr,
            "mean_win_loss": mean_wl,
            "mean_profit_factor": mean_pf,
            "mean_equity_r2": mean_r2,
            "total_trades": total_trades,
            "total_long_trades": total_long_tr,
            "total_short_trades": total_short_tr,
            "total_long_win_trades": total_long_win,
            "total_long_loss_trades": total_long_loss,
            "total_short_win_trades": total_short_win,
            "total_short_loss_trades": total_short_loss,
            "mean_pos_long_frac": mean_pos_long,
            "mean_pos_short_frac": mean_pos_short,
        }
        return -1e9, penalty_summary

    # Simple objective:
    # - Primary: higher Sharpe
    # - Secondary: higher cumulative return
    # - Encourage more trades (up to a point)
    # - Lightly penalize high time-in-market (prefer more selective entries)
    #
    # Coefficients are intentionally small so Sharpe remains dominant.
    trade_term = min(float(total_trades), 20.0) / 20.0  # 0..1 for first ~20 trades
    tim_penalty = mean_tim  # 0..1
    score = (
        mean_sharpe
        + 0.1 * mean_cum
        + 0.2 * trade_term
        - 0.2 * tim_penalty
    )
    summary = {
        "mean_sharpe": mean_sharpe,
        "mean_cum_return": mean_cum,
        "mean_max_dd": mean_dd,
        "mean_time_in_mkt": mean_tim,
        "mean_win_rate": mean_wr,
        "mean_win_loss": mean_wl,
        "mean_profit_factor": mean_pf,
        "mean_equity_r2": mean_r2,
        "total_trades": total_trades,
        "total_long_trades": total_long_tr,
        "total_short_trades": total_short_tr,
        "total_long_win_trades": total_long_win,
        "total_long_loss_trades": total_long_loss,
        "total_short_win_trades": total_short_win,
        "total_short_loss_trades": total_short_loss,
        "mean_pos_long_frac": mean_pos_long,
        "mean_pos_short_frac": mean_pos_short,
        "mean_pos_flat_frac": float(np.mean(pos_flat_fracs)) if pos_flat_fracs else 0.0,
        "mean_z_mean": float(np.mean(z_means)) if z_means else 0.0,
        "mean_z_std": float(np.mean(z_stds)) if z_stds else 0.0,
    }
    return score, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Random search over TCN scalar hyperparameters using WFO")
    ap.add_argument("--base-config", required=True, help="Path to base JSON config (tcn_eurusd_example.json)")
    ap.add_argument("--trials", type=int, default=10, help="Number of random hyperparameter trials")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))
    rng = random.Random(int(args.seed))

    # Fixed WFO settings from base config
    start = str(base_cfg.get("start"))
    end = str(base_cfg.get("end"))
    if not start or not end:
        raise SystemExit("Base config must include 'start' and 'end'")

    train_n = float(base_cfg.get("train_n", 1.0))
    val_n = float(base_cfg.get("val_n", 1.0))
    step_n = float(base_cfg.get("step_n", 1.0))
    unit = str(base_cfg.get("unit", "months"))
    base_gran = str(base_cfg.get("base_gran", "M5"))
    out_dir = str(base_cfg.get("out_dir", "wfo/runs"))
    candle_cache_base = str(base_cfg.get("candle_cache_base", "http://127.0.0.1:9100"))

    fixed_adapter = {
        "instrument": str(base_cfg.get("tcn_instrument", "EUR_USD")),
        "grans": str(base_cfg.get("grans", "M5,H1,D")),
        "base_gran": str(base_cfg.get("tcn_base_gran", base_gran)),
        "epochs": int(base_cfg.get("epochs", 5)),
        "lr": float(base_cfg.get("lr", 1e-3)),
        "candle_cache_base": candle_cache_base,
    }

    best_score = -1e9
    best_cfg: Dict[str, Any] = {}
    best_summary: Dict[str, float] = {}

    for i in range(int(args.trials)):
        hp = sample_hyperparams(rng)
        adapter_kwargs = dict(fixed_adapter)
        adapter_kwargs.update(
            lookback_bars=int(hp["tcn_lookback"]),
            target_horizon=int(hp["tcn_target_horizon"]),
            hidden_channels=int(hp["tcn_hidden"]),
            num_blocks=int(hp["tcn_blocks"]),
            kernel_size=int(hp["tcn_kernel"]),
            dropout=float(hp["tcn_dropout"]),
            enter_long=float(hp["enter_long"]),
            exit_long=float(hp["exit_long"]),
            enter_short=float(hp["enter_short"]),
            exit_short=float(hp["exit_short"]),
            trade_cost_bps=float(hp["tcn_trade_cost_bps"]),
        )

        adapter = TCNScalarThresholdAdapter(**adapter_kwargs)
        cfg = WFOConfig(
            start=start,
            end=end,
            train_n=train_n,
            val_n=val_n,
            step_n=step_n,
            unit=unit,
            windows_limit=int(base_cfg.get("windows_limit", 0)),
            base_gran=base_gran,
            out_dir=out_dir,
            adapter_spec="wfo.adapters.tcn_scalar_threshold:TCNScalarThresholdAdapter",
            adapter_kwargs=adapter_kwargs,
            parallel=1,  # keep single-process per trial to reduce complexity and load
            no_chart=True,
            chart_every=1,
            quiet=True,
            mode="fast",
        )

        print(json.dumps({"event": "trial_start", "trial": i + 1, "hp": hp}), flush=True)
        run_dir = Path(run_wfo(adapter, cfg))
        score, summary = compute_objective(run_dir)
        print(json.dumps({"event": "trial_result", "trial": i + 1, "score": score, "summary": summary, "run_dir": str(run_dir)}), flush=True)

        if score > best_score:
            best_score = score
            best_cfg = hp
            best_summary = summary

    print("\nBest configuration:")
    print(json.dumps({"score": best_score, "hp": best_cfg, "summary": best_summary}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
