from __future__ import annotations

"""
Optuna-based hyperparameter search for TCN+MLP policy configs using WFO.

Key design points:

- The search space is defined *inside the base JSON config*:
  - If a value is not a list: treated as a constant.
  - If a value is a list of length 1: treated as a constant (that element).
  - If a value is a list of length 2: interpreted as a continuous range
    [min_val, max_val].
  - If a value is a list of length 3: interpreted as a grid
    [min_val, step, max_val].

- For `mlp_layers`:
  - If it is a list of ints (e.g. [256, 128]), it is treated as constant.
  - If it is a list of lists, each inner list is interpreted as a range/grid
    using the rules above, and we sample one integer per layer.

- Fitness is computed from WFO validation metrics on a per-window basis by
  reusing the `compute_objective` function from `tcn_action_random_search`,
  which aggregates per-window metrics such as:
    - cum_return, sharpe, max_dd, time_in_mkt, trades, win_rate, etc.

Usage (from forex-rl root):

  python3 -m wfo.tcn_policy_optuna_search \\
      --base-config wfo/tcn_eurjpy_policy_debug.json \\
      --trials 40 \\
      --seed 0 \\
      --out-config wfo/tcn_eurjpy_policy_live_2025Q4.json

The resulting best-fit config (with concrete scalar hyperparameters) is written
to --out-config (or a derived path if omitted) and is suitable to plug into
`tcn_multi_policy_live_config.json` as that instrument's `base_config`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime, timezone

import optuna  # type: ignore[import-untyped]

from .core import WFOConfig, run_wfo, _instantiate_adapter
from .model_zoo_2023 import build_wfo_cfg, load_base_config
from .tcn_action_ga_search import _tcn_hp_to_adapter_kwargs
from .tcn_action_random_search import compute_hundred_debouncer_objective


JsonVal = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]


def _is_int_like(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _parse_range_spec(
    spec: List[Any],
) -> Tuple[str, float, float, float]:
    """
    Interpret a JSON list as a range/grid spec.

    Returns (kind, start, step, end), where:
      - kind: "const", "cont", or "grid"
      - start/step/end: numeric parameters (step is 0.0 for non-grid)
    """
    if len(spec) == 0:
        raise ValueError("Empty range spec")
    if len(spec) == 1:
        v = float(spec[0])
        return "const", v, 0.0, v
    if len(spec) == 2:
        a, b = float(spec[0]), float(spec[1])
        lo, hi = (a, b) if a <= b else (b, a)
        return "cont", lo, 0.0, hi
    # len >= 3: treat as [start, step, end]
    start, step, end = float(spec[0]), float(spec[1]), float(spec[2])
    if step == 0.0:
        lo, hi = (start, end) if start <= end else (end, start)
        return "cont", lo, 0.0, hi
    # Normalize order
    lo, hi = (start, end) if start <= end else (end, start)
    step_abs = abs(step)
    return "grid", lo, step_abs, hi


def _suggest_numeric_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: List[Any],
    prefer_int: bool = True,
) -> float:
    """
    Suggest a numeric value for Optuna based on a compact list spec.

    Semantics:
      - [v]                 -> constant v
      - [min, max]          -> uniform/minint range
      - [min, step, max]    -> discrete grid with given step

    If prefer_int is True and bounds/step are integral, we use integer
    suggestions; otherwise we fall back to float/categorical.
    """
    kind, start, step, end = _parse_range_spec(spec)
    if kind == "const":
        return start

    if kind == "cont":
        if prefer_int and float(int(round(start))) == start and float(int(round(end))) == end:
            lo_i = int(round(start))
            hi_i = int(round(end))
            return float(trial.suggest_int(name, lo_i, hi_i))
        return float(trial.suggest_float(name, start, end))

    # Grid: [start, step, end]
    if prefer_int and all(
        float(int(round(v))) == v for v in (start, step, end)
    ):
        lo_i = int(round(start))
        step_i = int(round(step)) if int(round(step)) != 0 else 1
        hi_i = int(round(end))
        # Optuna's suggest_int supports step
        return float(trial.suggest_int(name, lo_i, hi_i, step=abs(step_i)))

    # Float grid -> use categorical over enumerated values
    values: List[float] = []
    cur = start
    # Ensure we always include start and end (within numeric tolerance)
    if step <= 0.0:
        step = end - start
        if step == 0.0:
            values = [start]
        else:
            step = abs(step)
    if not values:
        while cur <= end + 1e-12:
            values.append(float(cur))
            cur += step
        # Numerical protection
        if abs(values[-1] - end) > 1e-6:
            values[-1] = end
    return float(trial.suggest_categorical(name, values))


def _sample_mlp_layers(
    trial: optuna.Trial,
    mlp_spec: Any,
) -> List[int]:
    """
    Sample MLP layer widths from a spec in base config.

    - If mlp_spec is a list of ints: treated as a fixed architecture.
    - If mlp_spec is a list of lists: each inner list is a range spec
      [min,max] or [min,step,max] and we sample one integer per layer.
    """
    if not isinstance(mlp_spec, list):
        return []
    if not mlp_spec:
        return []

    # Constant layout: [256, 128]
    if all(_is_int_like(x) for x in mlp_spec):
        return [int(x) for x in mlp_spec]

    # Layer-wise specs: [[64, 32, 256], [32, 32, 256]]
    layers: List[int] = []
    for i, layer_spec in enumerate(mlp_spec):
        name = f"mlp_layer_{i}"
        if isinstance(layer_spec, list) and layer_spec:
            val = int(round(_suggest_numeric_from_spec(trial, name, layer_spec, prefer_int=True)))
            layers.append(max(0, val))
        elif _is_int_like(layer_spec):
            layers.append(int(layer_spec))
    return layers


def _build_hyperparams_from_cfg(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct a TCN hyperparameter dict for a given Optuna trial by reading
    range specs from the base config.

    Keys that are present and list-valued in the base config participate in
    search; scalar values remain fixed. This function focuses on the core
    hyperparameters that materially impact cost/behavior.
    """
    hp: Dict[str, Any] = {}

    def add_param(
        key: str,
        prefer_int: bool = True,
    ) -> None:
        val = base_cfg.get(key, None)
        if isinstance(val, list):
            if len(val) == 0:
                return
            # Enum / categorical: list of strings
            if all(isinstance(x, str) for x in val):
                hp[key] = trial.suggest_categorical(key, list(val))
                return
            if len(val) == 1:
                hp[key] = val[0]
                return
            # Numeric range/grid spec
            hp[key] = _suggest_numeric_from_spec(trial, key, val, prefer_int=prefer_int)
        else:
            if val is not None:
                hp[key] = val

    # Core TCN geometry / capacity
    # Geometry knobs (train_n/val_n/step_n) may optionally be search dimensions.
    add_param("train_n", prefer_int=False)
    add_param("val_n", prefer_int=False)
    add_param("step_n", prefer_int=False)
    add_param("tcn_lookback", prefer_int=True)
    add_param("tcn_target_horizon", prefer_int=True)
    add_param("tcn_hidden", prefer_int=True)
    add_param("tcn_blocks", prefer_int=True)
    add_param("tcn_kernel", prefer_int=True)
    add_param("tcn_dropout", prefer_int=False)

    # Optional label mode (bps vs quantile, or other adapter-specific modes).
    add_param("label_mode", prefer_int=False)

    # ------------------------------------------------------------------
    # Label thresholds: enforce symmetric long/short when label_thresh is present.
    #   label_thresh > 0  ->  long = +label_thresh, short = -label_thresh
    # Fallback: treat label_long_thresh / label_short_thresh independently.
    # ------------------------------------------------------------------
    thresh_spec = base_cfg.get("label_thresh", None)
    thresh_val: float | None = None
    if isinstance(thresh_spec, list):
        if len(thresh_spec) == 1:
            thresh_val = float(thresh_spec[0])
        elif len(thresh_spec) > 1:
            thresh_val = float(
                _suggest_numeric_from_spec(
                    trial, "label_thresh", thresh_spec, prefer_int=False
                )
            )
    elif thresh_spec is not None:
        try:
            thresh_val = float(thresh_spec)
        except Exception:
            thresh_val = None

    if thresh_val is not None:
        # Ensure strictly positive to avoid degenerate zero-threshold behavior.
        t = abs(float(thresh_val))
        hp["label_long_thresh"] = t
        hp["label_short_thresh"] = -t
    else:
        # Backwards-compatible behavior.
        add_param("label_long_thresh", prefer_int=False)
        add_param("label_short_thresh", prefer_int=False)

    # ------------------------------------------------------------------
    # Label quantiles: optional symmetric mode around 0.5 via label_quantile_dist.
    #   dist in (0, 0.5) -> long = 0.5 + dist, short = 0.5 - dist
    # Fallback: treat label_long_quantile / label_short_quantile independently.
    # ------------------------------------------------------------------
    qdist_spec = base_cfg.get("label_quantile_dist", None)
    qdist_val: float | None = None
    if isinstance(qdist_spec, list):
        if len(qdist_spec) == 1:
            qdist_val = float(qdist_spec[0])
        elif len(qdist_spec) > 1:
            qdist_val = float(
                _suggest_numeric_from_spec(
                    trial, "label_quantile_dist", qdist_spec, prefer_int=False
                )
            )
    elif qdist_spec is not None:
        try:
            qdist_val = float(qdist_spec)
        except Exception:
            qdist_val = None

    if qdist_val is not None:
        d = max(0.0, min(float(qdist_val), 0.49))
        long_q = 0.5 + d
        short_q = 0.5 - d
        hp["label_long_quantile"] = long_q
        hp["label_short_quantile"] = short_q
    else:
        add_param("label_long_quantile", prefer_int=False)
        add_param("label_short_quantile", prefer_int=False)

    # Trade cost (usually fixed at 1 bp)
    add_param("tcn_trade_cost_bps", prefer_int=False)

    # ------------------------------------------------------------------
    # Entry/exit hysteresis: optional symmetric mode via enter_dist + exit_frac.
    #
    #   enter_dist > 0
    #   exit_frac in (0, 1)
    #
    #   nq_enter_long  = +enter_dist
    #   nq_enter_short = -enter_dist
    #   nq_exit_long   = +exit_frac * enter_dist
    #   nq_exit_short  = -exit_frac * enter_dist
    #
    # If enter_dist is not provided, fall back to legacy independent params.
    # If exit_frac is omitted but enter_dist is present, we fall back to the
    # previous exit_dist semantics (absolute distance in price/return space).
    # ------------------------------------------------------------------
    enter_spec = base_cfg.get("enter_dist", None)
    exit_frac_spec = base_cfg.get("exit_frac", None)
    exit_spec = base_cfg.get("exit_dist", None)  # legacy absolute-distance mode
    enter_val: float | None = None
    exit_val: float | None = None

    if isinstance(enter_spec, list):
        if len(enter_spec) == 1:
            enter_val = float(enter_spec[0])
        elif len(enter_spec) > 1:
            enter_val = float(
                _suggest_numeric_from_spec(
                    trial, "enter_dist", enter_spec, prefer_int=False
                )
            )
    elif enter_spec is not None:
        try:
            enter_val = float(enter_spec)
        except Exception:
            enter_val = None

    if enter_val is not None:
        e = abs(float(enter_val))

        # Preferred mode: exit_frac in (0, 1)
        exit_frac_val: float | None = None
        if isinstance(exit_frac_spec, list):
            if len(exit_frac_spec) == 1:
                exit_frac_val = float(exit_frac_spec[0])
            elif len(exit_frac_spec) > 1:
                exit_frac_val = float(
                    _suggest_numeric_from_spec(
                        trial, "exit_frac", exit_frac_spec, prefer_int=False
                    )
                )
        elif exit_frac_spec is not None:
            try:
                exit_frac_val = float(exit_frac_spec)
            except Exception:
                exit_frac_val = None

        if exit_frac_val is not None:
            # Clamp into a sensible open-ish interval (avoid 0 and 1).
            f = max(0.01, min(float(exit_frac_val), 0.99))
            x = f * e
        else:
            # Legacy absolute exit distance mode via exit_dist.
            if isinstance(exit_spec, list):
                if len(exit_spec) == 1:
                    exit_val = float(exit_spec[0])
                elif len(exit_spec) > 1:
                    exit_val = float(
                        _suggest_numeric_from_spec(
                            trial, "exit_dist", exit_spec, prefer_int=False
                        )
                    )
            elif exit_spec is not None:
                try:
                    exit_val = float(exit_spec)
                except Exception:
                    exit_val = None

            if exit_val is None:
                # Default: exit at half the entry distance.
                x = 0.5 * e
            else:
                # Enforce 0 < exit < enter.
                x = abs(float(exit_val))
                if x >= e:
                    x = max(0.1 * e, 0.5 * e)

        hp["nq_enter_long"] = e
        hp["nq_enter_short"] = -e
        hp["nq_exit_long"] = x
        hp["nq_exit_short"] = -x
    else:
        # Backwards-compatible independent thresholds.
        add_param("nq_enter_long", prefer_int=False)
        add_param("nq_exit_long", prefer_int=False)
        add_param("nq_enter_short", prefer_int=False)
        add_param("nq_exit_short", prefer_int=False)

    # Optional NQ causal-rule hyperparameters
    add_param("nq_ret_thresh", prefer_int=False)
    add_param("nq_vol_lookback", prefer_int=True)
    add_param("nq_z_thresh", prefer_int=False)

    # Optional MLP head layout
    mlp_spec = base_cfg.get("mlp_layers", None)
    if mlp_spec is not None:
        hp["mlp_layers"] = _sample_mlp_layers(trial, mlp_spec)

    return hp


def _run_single_trial(
    base_cfg: Dict[str, Any],
    hp: Dict[str, Any],
    trial_id: int,
) -> Tuple[float, Dict[str, float], str]:
    """
    Execute a single WFO run for a given hyperparameter set and compute fitness.

    - Instantiates the adapter specified in base_cfg (defaults to the same
      TCNActionNQAdapter used by the live multi-instrument trader).
    - Uses build_wfo_cfg to construct a WFOConfig (geometry from base_cfg).
    - Calls run_wfo(), then computes per-window objective via windows.jsonl.
    """
    adapter_spec = str(
        base_cfg.get(
            "adapter_spec",
            "wfo.adapters.tcn_nq_action:TCNActionNQAdapter",
        )
    )
    # Map TCN hyperparameters into adapter kwargs (instrument, grans, etc.)
    adapter_kwargs = _tcn_hp_to_adapter_kwargs(base_cfg, hp)
    adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)

    # Build WFOConfig from base geometry (start/end/train_n/val_n/step_n/etc.)
    cfg: WFOConfig = build_wfo_cfg(base_cfg, adapter_spec=adapter_spec, adapter_kwargs=adapter_kwargs)
    run_dir = Path(run_wfo(adapter, cfg))

    # Fitness + summary from per-window metrics (cum_return, sharpe, sortino, etc.)
    # using the "hundred-debouncer" objective.
    score, summary = compute_hundred_debouncer_objective(run_dir)
    return score, summary, str(run_dir)


def _run_study_for_cfg(
    cfg_base: Dict[str, Any],
    base_path: Path,
    trials: int,
    seed: int,
    out_path: Path,
) -> None:
    """Run an Optuna study for a single (possibly instrument-specific) base config."""
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        # Per-trial copy of config so we can annotate if needed.
        cfg_local = dict(cfg_base)
        # Hyperparameters for this trial come from range specs in cfg_local.
        hp = _build_hyperparams_from_cfg(trial, cfg_local)
        # Overlay sampled hyperparameters into local config so geometry fields
        # (train_n/val_n/step_n) and any config-level options become scalars.
        for k, v in hp.items():
            cfg_local[k] = v

        # Attach HPs to trial for later inspection.
        trial.set_user_attr("hp", hp)

        score, summary, run_dir = _run_single_trial(cfg_local, hp, trial.number)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("run_dir", run_dir)

        return float(score)

    study.optimize(objective, n_trials=int(trials))

    best = study.best_trial
    best_hp: Dict[str, Any] = best.user_attrs.get("hp", {})
    best_summary: Dict[str, float] = best.user_attrs.get("summary", {})
    best_run_dir: str = best.user_attrs.get("run_dir", "")

    # Build final concrete config: cfg_base with best_hp overlaid on top.
    best_cfg = dict(cfg_base)
    for k, v in best_hp.items():
        # For mlp_layers we ensure ints
        if k == "mlp_layers" and isinstance(v, list):
            best_cfg[k] = [int(x) for x in v]
        else:
            best_cfg[k] = v

    # Drop search-only helper keys that were used to define ranges but are not
    # consumed by the live adapter. The corresponding concrete scalar fields
    # (label_long_thresh, label_short_thresh, label_long_quantile,
    # label_short_quantile, nq_enter_*, nq_exit_*, etc.) are already present.
    for search_key in (
        "label_thresh",
        "label_quantile_dist",
        "enter_dist",
        "exit_frac",
    ):
        best_cfg.pop(search_key, None)
    # Multi-instrument search configs may define an "instruments" key to
    # enumerate targets; final per-instrument configs should not carry this.
    best_cfg.pop("instruments", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, sort_keys=True)

    # Emit a compact machine-readable summary and human-readable note.
    payload = {
        "best_score": float(best.value),
        "best_hp": best_hp,
        "best_summary": best_summary,
        "best_run_dir": best_run_dir,
        "best_config_path": str(out_path),
    }
    print(json.dumps(payload), flush=True)

    if best_summary:
        sharpe = best_summary.get("mean_sharpe", 0.0)
        cumret = best_summary.get("mean_cum_return", 0.0)
        tim = best_summary.get("mean_time_in_mkt", 0.0)
        trades = int(best_summary.get("total_trades", 0) or 0)
        print(
            f"\nBest trial: score={best.value:.4f}, sharpe={sharpe:.3f}, "
            f"cum_ret={cumret:.4f}, time_in_mkt={tim:.3f}, trades={trades}, "
            f"config={out_path}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna hyperparameter search for TCN+MLP policy configs using WFO"
    )
    ap.add_argument(
        "--base-config",
        required=True,
        help="Path to base JSON config (geometry/instrument/hyperparams search specs)",
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
        "--out-config",
        type=str,
        default=None,
        help="Where to write the best hyperparameterized config JSON "
        "(defaults to <base_stem>_optuna_best.json next to the base config).",
    )
    ap.add_argument(
        "--instruments",
        type=str,
        default=None,
        help=(
            "Comma-separated list of instruments (e.g. EUR_USD,GBP_USD,USD_JPY). "
            "When provided, runs a separate Optuna study per instrument, overriding "
            "tcn_instrument in the base config for each one."
        ),
    )
    ap.add_argument(
        "--optuna-dir",
        type=str,
        default="wfo/optuna_search_results",
        help=(
            "Directory to write per-instrument optimized configs when --instruments "
            "is used."
        ),
    )
    ap.add_argument(
        "--rolling-from-yesterday",
        dest="rolling_from_yesterday",
        action="store_true",
        help=(
            "When set, ignore explicit start/end in the base config and instead "
            "build windows_limit windows ending on yesterday using the geometry "
            "(train_n/val_n/step_n, unit)."
        ),
    )
    args = ap.parse_args()

    base_path = Path(args.base_config)
    base_cfg: Dict[str, Any] = load_base_config(base_path)
    if args.rolling_from_yesterday:
        base_cfg["rolling_from_yesterday"] = True
    if args.epochs is not None:
        base_cfg["epochs"] = int(args.epochs)
    instruments_arg = str(args.instruments or "").strip()

    # Determine per-instrument list:
    # Priority: CLI --instruments > config "instruments" key.
    cfg_instruments = base_cfg.get("instruments")

    inst_list: List[str] = []
    if instruments_arg:
        inst_list = [
            inst.strip().upper()
            for inst in instruments_arg.split(",")
            if inst.strip()
        ]
    elif isinstance(cfg_instruments, list):
        inst_list = [str(x).strip().upper() for x in cfg_instruments if str(x).strip()]
    elif isinstance(cfg_instruments, str):
        inst_list = [
            tok.strip().upper()
            for tok in cfg_instruments.split(",")
            if tok.strip()
        ]

    if inst_list:
        # Multi-instrument mode: ignore --out-config and run a separate study per
        # instrument, overriding tcn_instrument in the base config.
        optuna_dir = Path(args.optuna_dir)
        optuna_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running Optuna search for instruments: {', '.join(inst_list)}")

        for inst in inst_list:
            cfg_inst = dict(base_cfg)
            cfg_inst["tcn_instrument"] = inst

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_path = optuna_dir / f"optimal_{inst.lower()}_{ts}.json"

            print(f"\n=== Instrument {inst} ===")
            _run_study_for_cfg(
                cfg_base=cfg_inst,
                base_path=base_path,
                trials=int(args.trials),
                seed=int(args.seed),
                out_path=out_path,
            )
    else:
        # Single-instrument (legacy) mode using whatever tcn_instrument is set
        # inside the base config.
        if args.out_config:
            out_path = Path(args.out_config)
        else:
            out_path = base_path.with_name(f"{base_path.stem}_optuna_best.json")

        _run_study_for_cfg(
            cfg_base=base_cfg,
            base_path=base_path,
            trials=int(args.trials),
            seed=int(args.seed),
            out_path=out_path,
        )


if __name__ == "__main__":  # pragma: no cover
    main()

