from __future__ import annotations

import argparse
from typing import Any, Dict
from .core import WFOConfig, run_wfo

# Adapters import lazily to avoid heavy deps if unused

def main() -> None:
    p = argparse.ArgumentParser(description="Generic Walk-Forward Optimization runner")
    p.add_argument("--adapter", required=True, choices=["ac-multi20", "ga-ness"], help="Which strategy adapter to use")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--train-n", type=float, default=3.0)
    p.add_argument("--val-n", type=float, default=1.0)
    p.add_argument("--step-n", type=float, default=1.0)
    p.add_argument("--unit", default="months", choices=["months", "weeks", "days", "steps"])
    p.add_argument("--windows-limit", type=int, default=0)
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--out-dir", default="wfo/runs")
    p.add_argument("--parallel", type=int, default=0, help="Number of windows to run in parallel (0=auto cpu count)")
    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--chart-every", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--mode", default="thorough", choices=["fast", "thorough"])  # logging/metrics presets

    # Adapter-specific passthrough (flat namespace for now)
    p.add_argument("--instruments", default=",")
    p.add_argument("--grans", default="M5,H1,D")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-sigma", type=float, default=0.3)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--enter-long", type=float, default=0.80)
    p.add_argument("--exit-long", type=float, default=0.60)
    p.add_argument("--enter-short", type=float, default=0.20)
    p.add_argument("--exit-short", type=float, default=0.40)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--carry-forward", action="store_true")
    p.add_argument("--init-model", default="")

    args = p.parse_args()

    if args.adapter == "ac-multi20":
        from .adapters.ac_multi20 import ACMulti20Adapter
        adapter = ACMulti20Adapter(
            instruments=args.instruments,
            grans=args.grans,
            epochs=args.epochs,
            gamma=args.gamma,
            actor_sigma=args.actor_sigma,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            hidden=args.hidden,
            reward_scale=args.reward_scale,
            max_grad_norm=args.max_grad_norm,
            enter_long=args.enter_long,
            exit_long=args.exit_long,
            enter_short=args.enter_short,
            exit_short=args.exit_short,
            no_save=args.no_save,
            carry_forward=args.carry_forward,
            init_model=args.init_model,
        )
    elif args.adapter == "ga-ness":
        from .adapters.ga_ness import GANessAdapter
        adapter = GANessAdapter(
            csv=None,
            population=30,
            generations=10,
            elite_frac=0.1,
            mutation_prob=0.3,
            crossover_prob=0.8,
            cost_bps=1.0,
        )
    else:
        raise SystemExit("Unknown adapter")

    # Decide parallel
    par = int(args.parallel)
    if par == 0:
        import os as _os
        par = max(1, len(_os.sched_getaffinity(0)) if hasattr(_os, 'sched_getaffinity') else _os.cpu_count() or 1)

    # Build clean adapter kwargs (exclude instance internals)
    if args.adapter == "ac-multi20":
        adapter_kwargs = dict(
            instruments=args.instruments,
            grans=args.grans,
            epochs=args.epochs,
            gamma=args.gamma,
            actor_sigma=args.actor_sigma,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            hidden=args.hidden,
            reward_scale=args.reward_scale,
            max_grad_norm=args.max_grad_norm,
            enter_long=args.enter_long,
            exit_long=args.exit_long,
            enter_short=args.enter_short,
            exit_short=args.exit_short,
            no_save=args.no_save,
            carry_forward=args.carry_forward,
            init_model=args.init_model,
        )
    else:  # ga-ness
        adapter_kwargs = dict(
            csv=None,
            population=30,
            generations=10,
            elite_frac=0.1,
            mutation_prob=0.3,
            crossover_prob=0.8,
            cost_bps=1.0,
            w_sharpe=0.4,
            w_return=0.3,
            w_dd=0.2,
            w_trades=0.1,
            neg_sharpe_penalty=10.0,
            neg_return_penalty=5.0,
        )

    cfg = WFOConfig(
        start=args.start,
        end=args.end,
        train_n=args.train_n,
        val_n=args.val_n,
        step_n=args.step_n,
        unit=args.unit,
        windows_limit=args.windows_limit,
        base_gran=args.base_gran,
        out_dir=args.out_dir,
        adapter_spec=("wfo.adapters.ac_multi20:ACMulti20Adapter" if args.adapter == "ac-multi20" else "wfo.adapters.ga_ness:GANessAdapter"),
        adapter_kwargs=adapter_kwargs,
        parallel=par,
        no_chart=bool(args.no_chart),
        chart_every=int(args.chart_every),
        quiet=bool(args.quiet),
        mode=str(args.mode),
    )

    run_dir = run_wfo(adapter, cfg)
    print(run_dir)


if __name__ == "__main__":
    main()
