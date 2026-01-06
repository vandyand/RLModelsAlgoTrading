from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .core import WFOConfig, run_wfo
from .adapters.tcn_scalar_threshold import TCNActionAdapter


def load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Base config JSON must contain an object at the top level")
    return data


def sample_hyperparams(rng: random.Random) -> Dict[str, Any]:
    """Randomly sample a TCN action-adapter hyperparameter set.

    Ranges are widened for more fine-grained meta-parameter discovery,
    but capped to avoid excessively large models:
    - tcn_lookback: 12..240 (step 8)
    - tcn_target_horizon: 2..192 (step 4)
    - tcn_hidden: 8..32 (step 4)
    - tcn_blocks: fixed at 1 (for speed)
    - tcn_kernel: fixed at 1 (for speed)

    For NQ we also include causal-rule hyperparameters used by the
    TCNActionNQAdapter override:
    - nq_ret_thresh: 2e-4..2e-3 (raw horizon log-return threshold)
    - nq_vol_lookback: 32..128 (rolling volatility window, step 16)
    - nq_z_thresh: 0.5..2.5 (z-score threshold on horizon return / vol)
    """
    lookback_options = list(range(12, 241, 8))
    horizon_options = list(range(2, 193, 4))
    hidden_options = list(range(8, 33, 4))
    # For now we fix num_blocks and kernel_size to 1 to keep the
    # models small and speed up searches (especially for NQ).
    blocks_options = [1]
    kernel_options = [1]

    # Label thresholds are in raw forward-return space.
    # 0.0005 ~ 5 bp, 0.0080 ~ 80 bp.
    label_long_thresh = rng.uniform(0.0005, 0.0080)
    label_short_thresh = -rng.uniform(0.0005, 0.0080)

    # Quantile-based labeler hyperparameters (only used when label_mode == "quantile").
    # We ensure label_short_quantile < label_long_quantile by construction.
    # Example ranges:
    #   short_q ~ U(0.05, 0.35)
    #   long_q  ~ U(max(0.55, short_q + 0.1), 0.95)
    short_q_base = rng.uniform(0.05, 0.35)
    long_q_min = max(0.55, short_q_base + 0.10)
    long_q = rng.uniform(long_q_min, 0.95)
    short_q = short_q_base

    # NQ-only causal rule parameters (ignored by FX adapters, but passed
    # through config so NQ-specific adapters can consume them).
    nq_ret_thresh = rng.uniform(0.0002, 0.0020)
    nq_vol_lookback = rng.choice(list(range(32, 129, 16)))
    nq_z_thresh = rng.uniform(0.5, 2.5)

    return {
        "tcn_lookback": rng.choice(lookback_options),
        "tcn_target_horizon": rng.choice(horizon_options),
        "tcn_hidden": rng.choice(hidden_options),
        "tcn_blocks": rng.choice(blocks_options),
        "tcn_kernel": rng.choice(kernel_options),
        "tcn_dropout": rng.uniform(0.0, 0.15),
        "label_long_thresh": label_long_thresh,
        "label_short_thresh": label_short_thresh,
        "label_long_quantile": long_q,
        "label_short_quantile": short_q,
        # Hold trade cost fixed at 1.0 bps for all runs
        "tcn_trade_cost_bps": 1.0,
        "nq_ret_thresh": nq_ret_thresh,
        "nq_vol_lookback": nq_vol_lookback,
        "nq_z_thresh": nq_z_thresh,
    }


def compute_objective(run_dir: Path) -> Tuple[float, Dict[str, float]]:
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
    tims: List[float] = []
    win_rates: List[float] = []
    win_losses: List[float] = []
    pfs: List[float] = []
    r2s: List[float] = []
    trades: List[float] = []
    long_trades: List[float] = []
    short_trades: List[float] = []
    long_win_trades: List[float] = []
    long_loss_trades: List[float] = []
    short_win_trades: List[float] = []
    short_loss_trades: List[float] = []

    for r in rows:
        m = r.get("metrics") or r
        sharpes.append(float(m.get("sharpe") or 0.0))
        cumrets.append(float(m.get("cum_return") or 0.0))
        dds.append(float(m.get("max_dd") or 0.0))
        tims.append(float(m.get("time_in_mkt") or 0.0))
        win_rates.append(float(m.get("win_rate") or 0.0))
        win_losses.append(float(m.get("win_loss") or 0.0))
        pfs.append(float(m.get("profit_factor") or 0.0))
        r2s.append(float(m.get("equity_r2") or 0.0))
        trades.append(float(m.get("trades") or 0.0))
        long_trades.append(float(m.get("long_trades") or 0.0))
        short_trades.append(float(m.get("short_trades") or 0.0))
        long_win_trades.append(float(m.get("long_win_trades") or 0.0))
        long_loss_trades.append(float(m.get("long_loss_trades") or 0.0))
        short_win_trades.append(float(m.get("short_win_trades") or 0.0))
        short_loss_trades.append(float(m.get("short_loss_trades") or 0.0))

    arr_rates = np.array([sharpes, cumrets, dds, tims, win_rates, win_losses, pfs, r2s], dtype=float)
    mean_sharpe = float(np.mean(arr_rates[0]))
    mean_cum = float(np.mean(arr_rates[1]))
    mean_dd = float(np.mean(arr_rates[2]))
    mean_tim = float(np.mean(arr_rates[3]))
    mean_wr = float(np.mean(arr_rates[4]))
    mean_wl = float(np.mean(arr_rates[5]))
    mean_pf = float(np.mean(arr_rates[6]))
    mean_r2 = float(np.mean(arr_rates[7]))

    total_trades = int(round(float(np.sum(trades))))
    total_long_tr = int(round(float(np.sum(long_trades))))
    total_short_tr = int(round(float(np.sum(short_trades))))
    total_long_win = int(round(float(np.sum(long_win_trades))))
    total_long_loss = int(round(float(np.sum(long_loss_trades))))
    total_short_win = int(round(float(np.sum(short_win_trades))))
    total_short_loss = int(round(float(np.sum(short_loss_trades))))

    # Hard behavioral constraints to avoid degenerate solutions
    if total_trades < 4:
        return -1e9, {
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
        }
    if total_long_tr == 0 or total_short_tr == 0:
        return -1e9, {
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
        }

    trade_term = min(float(total_trades), 40.0) / 40.0
    tim_penalty = mean_tim

    score = (
        mean_sharpe
        + 0.1 * mean_cum
        + 0.3 * trade_term
        - 0.3 * tim_penalty
    )

    summary = {
        "mean_sharpe": mean_sharpe,
        "mean_cum_return": mean_cum,
        "mean_max_dd": mean_dd,
        "mean_cum_return_bp": mean_cum * 1e4,
        "mean_max_dd_bp": mean_dd * 1e4,
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
    }
    return score, summary


def compute_hundred_debouncer_objective(run_dir: Path) -> Tuple[float, Dict[str, float]]:
    """
    Compute the "hundred-debouncer" objective over validation windows.

    For each window with metrics m, define:

      numerator =
        (100 + 100 * cum_return      if cum_return >= 0 else 100) *
        (100 + sharpe                if sharpe >= 0     else 100) *
        (100 + sortino               if sortino >= 0    else 100) *
        (100 + trades) *
        (100 + win_rate) *
        (100 + profit_factor)

      denominator =
        (100 + abs(sharpe)           if sharpe < 0      else 100) *
        (100 + abs(cum_return * 100) if cum_return < 0  else 100) *
        (100 + abs(sortino)          if sortino < 0     else 100) *
        (100 + abs(max_dd * 100)) *
        (100 + time_in_mkt * 10)

      score_window = numerator / denominator

    The objective score is the mean(score_window) across all windows, with
    the same basic aggregate summary as `compute_objective`.
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
    tims: List[float] = []
    win_rates: List[float] = []
    win_losses: List[float] = []
    pfs: List[float] = []
    r2s: List[float] = []
    trades: List[float] = []
    long_trades: List[float] = []
    short_trades: List[float] = []
    long_win_trades: List[float] = []
    long_loss_trades: List[float] = []
    short_win_trades: List[float] = []
    short_loss_trades: List[float] = []
    hd_scores: List[float] = []

    for r in rows:
        # Skip windows whose entire validation interval falls on weekend days
        # (for typical 1-day validation windows around weekends). For longer
        # validation spans (val_n > 2 days), this condition will not trigger.
        val = r.get("val")
        if isinstance(val, list) and len(val) == 2:
            try:
                v_s = pd.Timestamp(val[0])
                v_e = pd.Timestamp(val[1])
                # Normalize to dates and iterate inclusive
                days = pd.date_range(v_s.normalize(), v_e.normalize(), freq="D")
                if len(days) <= 2 and all(d.weekday() >= 5 for d in days):
                    # Entire validation window is Sat/Sun -> skip
                    continue
            except Exception:
                # On any parsing error, fall back to keeping the window.
                pass

        m = r.get("metrics") or r

        sharpe = float(m.get("sharpe") or 0.0)
        cumret = float(m.get("cum_return") or 0.0)
        dd = float(m.get("max_dd") or 0.0)
        tim = float(m.get("time_in_mkt") or 0.0)
        win_rate = float(m.get("win_rate") or 0.0)
        win_loss = float(m.get("win_loss") or 0.0)
        pf = float(m.get("profit_factor") or 0.0)
        r2 = float(m.get("equity_r2") or 0.0)
        tr = float(m.get("trades") or 0.0)
        lt = float(m.get("long_trades") or 0.0)
        st = float(m.get("short_trades") or 0.0)
        lwin = float(m.get("long_win_trades") or 0.0)
        lloss = float(m.get("long_loss_trades") or 0.0)
        swin = float(m.get("short_win_trades") or 0.0)
        sloss = float(m.get("short_loss_trades") or 0.0)

        sharpes.append(sharpe)
        cumrets.append(cumret)
        dds.append(dd)
        tims.append(tim)
        win_rates.append(win_rate)
        win_losses.append(win_loss)
        pfs.append(pf)
        r2s.append(r2)
        trades.append(tr)
        long_trades.append(lt)
        short_trades.append(st)
        long_win_trades.append(lwin)
        long_loss_trades.append(lloss)
        short_win_trades.append(swin)
        short_loss_trades.append(sloss)

        # Per-window hundred-debouncer score
        num = 1.0
        den = 1.0

        # Numerator components
        num *= (100.0 + 100.0 * cumret) if cumret >= 0.0 else 100.0
        num *= (100.0 + sharpe) if sharpe >= 0.0 else 100.0
        num *= (100.0 + float(m.get("sortino") or 0.0)) if float(m.get("sortino") or 0.0) >= 0.0 else 100.0
        num *= (100.0 + tr)
        num *= (100.0 + win_rate)
        num *= (100.0 + pf)

        # Denominator components
        sortino = float(m.get("sortino") or 0.0)
        if sharpe < 0.0:
            den *= 100.0 + abs(sharpe)
        else:
            den *= 100.0
        if cumret < 0.0:
            den *= 100.0 + abs(cumret * 100.0)
        else:
            den *= 100.0
        if sortino < 0.0:
            den *= 100.0 + abs(sortino)
        else:
            den *= 100.0
        den *= 100.0 + abs(dd * 100.0)
        # Stronger penalty for time in market: coefficient 20 (was 10).
        den *= 100.0 + tim * 20.0

        if den <= 0.0 or not np.isfinite(den):
            score_w = 0.0
        else:
            score_w = float(num / den)
        if not np.isfinite(score_w):
            score_w = 0.0
        hd_scores.append(score_w)

    arr_rates = np.array([sharpes, cumrets, dds, tims, win_rates, win_losses, pfs, r2s], dtype=float)
    mean_sharpe = float(np.mean(arr_rates[0]))
    mean_cum = float(np.mean(arr_rates[1]))
    mean_dd = float(np.mean(arr_rates[2]))
    mean_tim = float(np.mean(arr_rates[3]))
    mean_wr = float(np.mean(arr_rates[4]))
    mean_wl = float(np.mean(arr_rates[5]))
    mean_pf = float(np.mean(arr_rates[6]))
    mean_r2 = float(np.mean(arr_rates[7]))

    total_trades = int(round(float(np.sum(trades))))
    total_long_tr = int(round(float(np.sum(long_trades))))
    total_short_tr = int(round(float(np.sum(short_trades))))
    total_long_win = int(round(float(np.sum(long_win_trades))))
    total_long_loss = int(round(float(np.sum(long_loss_trades))))
    total_short_win = int(round(float(np.sum(short_win_trades))))
    total_short_loss = int(round(float(np.sum(short_loss_trades))))

    # Objective: mean hundred-debouncer score across all non-skipped windows.
    score = float(np.mean(hd_scores)) if hd_scores else 0.0

    summary = {
        "mean_sharpe": mean_sharpe,
        "mean_cum_return": mean_cum,
        "mean_max_dd": mean_dd,
        "mean_cum_return_bp": mean_cum * 1e4,
        "mean_max_dd_bp": mean_dd * 1e4,
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
        "hd_mean_score": score,
    }
    return score, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Random search over TCN action-classifier hyperparameters using WFO")
    ap.add_argument("--base-config", required=True, help="Path to base JSON config (tcn_eurusd_example.json)")
    ap.add_argument("--trials", type=int, default=10, help="Number of random hyperparameter trials")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_cfg = load_base_config(Path(args.base_config))
    rng = random.Random(int(args.seed))

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
            label_long_thresh=float(hp["label_long_thresh"]),
            label_short_thresh=float(hp["label_short_thresh"]),
            trade_cost_bps=float(hp["tcn_trade_cost_bps"]),
            # NQ-only causal rule hyperparameters; FX paths ignore them.
            nq_ret_thresh=float(hp.get("nq_ret_thresh", base_cfg.get("nq_ret_thresh", 0.0005))),
            nq_vol_lookback=int(hp.get("nq_vol_lookback", base_cfg.get("nq_vol_lookback", 64))),
            nq_z_thresh=float(hp.get("nq_z_thresh", base_cfg.get("nq_z_thresh", 1.0))),
        )

        adapter = TCNActionAdapter(**adapter_kwargs)
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
            adapter_spec="wfo.adapters.tcn_scalar_threshold:TCNActionAdapter",
            adapter_kwargs=adapter_kwargs,
            parallel=1,
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
