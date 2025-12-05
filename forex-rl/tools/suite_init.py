#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Simple interactive initializer for suite manifests (live/backtest/online)
# Friendly wizard: sensible defaults, clear explanations, presets per strategy.
# Produces YAML if PyYAML is available; otherwise uses a minimal YAML emitter.

SUPPORTED: List[Dict[str, Any]] = [
    {
        "name": "online_box",
        "script": "forex-rl/online_trader.py",
        "suite": ["live", "online"],
        "suggested_base": {
            "broker": "ipc",
            "market-data": "frb",
            "instruments": ["EUR_USD", "USD_JPY"],
            "slip-ramp-start-bps": 0,
            "slip-ramp-target-bps": 1,
            "slip-ramp-days": 5,
        },
        "base_help": {
            "broker": "Use local paper broker (ipc)",
            "market-data": "Use FRB shared-memory feed for prices",
            "instruments": "FX instruments (OANDA code, comma-separated)",
            "slip-ramp-start-bps": "Start slippage (bps) for ramp",
            "slip-ramp-target-bps": "Target slippage (bps) over ramp",
            "slip-ramp-days": "Days to ramp from start to target",
        },
        "sweep_tips": [
            "max-units: list like 500,1000,1500",
            "noise-sigma: list like 0.1,0.2",
            "slip-ramp-target-bps: range like 0.5..1.0 step=0.25",
        ],
        "sweep_keys": {
            "max-units": "Absolute units cap per instrument",
            "noise-sigma": "Exploration noise stddev",
            "slip-ramp-target-bps": "Target slippage (bps)",
        },
    },
    {
        "name": "ct_threshold",
        "script": "forex-rl/continuous-trader/live_trader.py",
        "suite": ["live"],
        "suggested_base": {
            "broker": "ipc",
            "instrument": "EUR_USD",
        },
        "base_help": {
            "instrument": "Single FX instrument",
            "broker": "Use local paper broker (ipc)",
        },
        "sweep_tips": [
            "max-units: list like 500,1000",
            "thresholds: choices like '0.7,0.6,0.3,0.4'",
        ],
        "sweep_keys": {
            "max-units": "Absolute units when in position",
            "thresholds": "Comma separated enter/exit thresholds",
        },
    },
    {
        "name": "ct_siamese",
        "script": "forex-rl/continuous-trader/live_siamese_trader.py",
        "suite": ["live"],
        "suggested_base": {
            "broker": "ipc",
            "instrument": "EUR_USD",
        },
        "base_help": {
            "instrument": "Single FX instrument",
        },
        "sweep_tips": [
            "units: list like 100,200",
            "thresholds: choices like '0.7,0.6,0.3,0.4'",
        ],
        "sweep_keys": {
            "units": "Absolute units when in position",
            "thresholds": "Comma separated enter/exit thresholds",
        },
    },
    {
        "name": "ac_multi20_live",
        "script": "forex-rl/ac-multi20/live_ac_trader.py",
        "suite": ["live"],
        "suggested_base": {
            "broker": "ipc",
            "max-units": 100,
        },
        "base_help": {
            "max-units": "Absolute units when action=±1",
        },
        "sweep_tips": [
            "max-units: list like 50,100,150",
        ],
        "sweep_keys": {"max-units": "Absolute units cap"},
    },
    {
        "name": "ga_multi20_live",
        "script": "forex-rl/ga-multi20/live_trader.py",
        "suite": ["live"],
        "suggested_base": {
            "broker": "ipc",
            "units": 100,
        },
        "base_help": {
            "units": "Absolute units when Long/Short",
        },
        "sweep_tips": [
            "units: list like 50,100,150",
        ],
        "sweep_keys": {"units": "Absolute units"},
    },
    {
        "name": "ct_train_offline",
        "script": "forex-rl/continuous-trader/train_offline.py",
        "suite": ["backtest"],
        "suggested_base": {
            "epochs": 10,
        },
        "base_help": {"epochs": "Training epochs"},
        "sweep_tips": [
            "lr: list like 1e-4,3e-4,1e-3",
            "batch-size: range 256..1024 step=256",
        ],
        "sweep_keys": {
            "lr": "Learning rate",
            "batch-size": "Mini-batch size",
        },
    },
    {
        "name": "ac_offline_multi20",
        "script": "forex-rl/actor-critic/multi20_offline_actor_critic.py",
        "suite": ["backtest"],
        "suggested_base": {
            "epochs": 6,
            "batch-size": 128,
        },
        "base_help": {
            "instruments": "Comma-separated instruments",
            "epochs": "Training epochs",
            "batch-size": "Mini-batch size",
        },
        "sweep_tips": [
            "latent: list like 64,128",
            "gamma: list like 0.95,0.99",
            "actor-sigma: list like 0.2,0.3",
            "entropy-coef: list like 0.0005,0.001",
            "trunk-hidden: choices like '2048,512'",
            "policy-hidden/value-hidden: list like 256,512",
        ],
        "sweep_keys": {
            "latent": "Latent dimension",
            "gamma": "Discount factor",
            "actor-sigma": "Policy noise sigma",
            "entropy-coef": "Entropy regularization coef",
            "trunk-hidden": "Encoder hidden sizes (comma list)",
            "policy-hidden": "Policy hidden size",
            "value-hidden": "Value hidden size",
        },
    },
    {
        "name": "ga_ness_walk_forward",
        "script": "forex-rl/ga-ness/walk_forward_gp.py",
        "suite": ["backtest"],
        "suggested_base": {
        },
        "base_help": {
            "generations": "Number of generations",
        },
        "sweep_tips": [
            "generations: list like 50,100",
        ],
        "sweep_keys": {
            "generations": "GA generations",
        },
    },
    {
        "name": "unsup_autoencoder",
        "script": "forex-rl/unsupervised-ae/unsupervised_autoencoder.py",
        "suite": ["backtest"],
        "suggested_base": {
            "epochs": 10,
            "batch-size": 128,
        },
        "base_help": {
            "epochs": "Training epochs",
            "batch-size": "Mini-batch size",
        },
        "sweep_tips": [
            "lr: list like 1e-4,3e-4,1e-3",
            "latent: list like 32,64,128",
        ],
        "sweep_keys": {
            "lr": "Learning rate",
            "latent": "Latent dimension",
        },
    },
    {
        "name": "td3_on_latent_train",
        "script": "forex-rl/unsupervised-ae/train_td3_on_latent.py",
        "suite": ["backtest"],
        "suggested_base": {
            "epochs": 5,
            "batch-size": 128,
        },
        "base_help": {
            "epochs": "Training epochs",
            "batch-size": "Mini-batch size",
        },
        "sweep_tips": [
            "lr: list like 1e-4,3e-4,1e-3",
            "actor-lr/critic-lr: list like 1e-4,3e-4",
        ],
        "sweep_keys": {
            "lr": "Learning rate",
            "actor-lr": "TD3 actor LR",
            "critic-lr": "TD3 critic LR",
        },
    },
    {
        "name": "ga_multi20_train",
        "script": "forex-rl/ga-multi20/train_ga_multi20.py",
        "suite": ["backtest"],
        "suggested_base": {
            "lookback-days": 20,
            "trade-cost": 0.0002,
        },
        "base_help": {
            "instruments": "Comma-separated instruments (empty = default set)",
            "lookback-days": "Days of data for fitness",
            "trade-cost": "Cost per flip (abs return deduction)",
            "n-jobs": "Parallel workers (0=auto, 1=serial)",
            "resume": "Path to best_genome-*.json to warm-start",
        },
        "sweep_tips": [
            "population: list like 32,64",
            "generations: list like 50,100",
            "mutation: list like 0.05,0.1",
            "weight-sigma: list like 0.1,0.2",
            "affine-sigma: list like 0.05,0.1",
            "threshold-mode: choices band|absolute",
            "band-enter/band-exit or enter/exit thresholds",
            "downsample: range 1..5"
        ],
        "sweep_keys": {
            "population": "GA population size",
            "generations": "GA generations",
            "mutation": "Mutation probability (0..1)",
            "weight-sigma": "Weight mutation sigma",
            "affine-sigma": "Input affine mutation sigma",
            "threshold-mode": "band or absolute",
            "band-enter": "Band enter",
            "band-exit": "Band exit",
            "enter-long": "Absolute enter long",
            "exit-long": "Absolute exit long",
            "enter-short": "Absolute enter short",
            "exit-short": "Absolute exit short",
            "downsample": "Evaluate every k-th bar",
            "hidden": "Hidden sizes (comma list)",
            "seed": "Random seed",
        },
    },
    {
        "name": "tpsl_trader",
        "script": "forex-rl/tpsl_trader.py",
        "suite": ["live"],
        "suggested_base": {
            "broker": "ipc",
            "instruments": ["EUR_USD"],
        },
        "base_help": {
            "instruments": "FX instruments",
        },
        "sweep_tips": ["max-units: list like 100,200"],
        "sweep_keys": {"max-units": "Absolute units cap"},
    },
]


def prompt(msg: str, default: Optional[str] = None) -> str:
    if default is not None:
        line = input(f"{msg} [{default}]: ").strip()
        return line if line != "" else default
    return input(f"{msg}: ").strip()


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    line = input(f"{msg} ({d}): ").strip().lower()
    if line == "":
        return default
    return line in ("y", "yes")


def choose_suite_type() -> str:
    print("Choose suite type:")
    print("  1) live (inference trading)")
    print("  2) backtest (offline/batch)")
    print("  3) online (online learning/trading)")
    while True:
        sel = input("Enter 1/2/3: ").strip()
        if sel in ("1", "2", "3"):
            return {"1": "live", "2": "backtest", "3": "online"}[sel]
        print("Invalid selection.")


def list_supported(suite_type: str) -> List[Dict[str, Any]]:
    avail = [s for s in SUPPORTED if suite_type in s["suite"]]
    print("\nSupported strategies for", suite_type)
    for i, s in enumerate(avail, start=1):
        print(f"  {i}) {s['name']}  ->  {s['script']}")
    print(f"  {len(avail)+1}) Custom script path")
    return avail


def parse_value_list(s: str) -> List[Any]:
    # parse comma-separated values; try int, then float, else string
    vals = []
    for t in [x.strip() for x in s.split(',') if x.strip() != ""]:
        try:
            vals.append(int(t))
            continue
        except Exception:
            pass
        try:
            vals.append(float(t))
            continue
        except Exception:
            pass
        vals.append(t)
    return vals


def add_sweep_interactive(sweep_keys_help: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("\nDefine parameters (press Enter with empty name to finish).")
    if sweep_keys_help:
        print("Common parameters:")
        for k, desc in sweep_keys_help.items():
            print(f"  - {k}: {desc}")
    print("Modes: single (one value), list (e.g., 100,200), or range (start/end with step or count).")
    sweep: Dict[str, Any] = {}
    singles: Dict[str, Any] = {}
    while True:
        key = input("Param name (blank to finish): ").strip()
        if key == "":
            break
        mode = input("Value mode: [single|list|range] (default single): ").strip().lower() or "single"
        if mode == "single":
            s = input("Enter value: ").strip()
            # coerce
            try:
                singles[key] = int(s)
            except Exception:
                try:
                    singles[key] = float(s)
                except Exception:
                    singles[key] = s
        elif mode == "list":
            s = input("Enter comma-separated values: ").strip()
            sweep[key] = parse_value_list(s)
        elif mode == "range":
            start = input(" start: ").strip()
            end = input(" end: ").strip()
            step = input(" step (optional): ").strip()
            count = input(" count (optional): ").strip()
            spec: Dict[str, Any] = {}
            try:
                spec["start"] = float(start)
                spec["end"] = float(end)
            except Exception:
                print("  Invalid start/end; skipping this parameter.")
                continue
            if step:
                try:
                    spec["step"] = float(step)
                except Exception:
                    print("  Ignoring invalid step.")
            if count:
                try:
                    spec["count"] = int(count)
                except Exception:
                    print("  Ignoring invalid count.")
            sweep[key] = spec
        else:
            print("Unknown mode; skipping.")
    return sweep, singles


def add_job_interactive(suite_type: str) -> Dict[str, Any]:
    avail = list_supported(suite_type)
    while True:
        sel = input("Choose strategy number: ").strip()
        try:
            idx = int(sel)
        except Exception:
            print("Not a number.")
            continue
        if 1 <= idx <= len(avail):
            spec = avail[idx - 1]
            name = spec["name"]
            script = spec["script"]
            base = dict(spec.get("suggested_base") or {})
            tips = spec.get("sweep_tips") or []
            break
        if idx == len(avail) + 1:
            name = input("Enter custom job name: ").strip() or "custom"
            script = input("Enter script path (relative to repo root): ").strip()
            base = {}
            tips = []
            break
        print("Out of range.")

    print(f"\nSelected: {name} -> {script}")
    # Base args edits
    if base:
        print("Suggested base args (press Enter to accept defaults):")
        base_help = spec.get("base_help") or {}
        for k in list(base.keys()):
            v = base[k]
            hint = f"  ({base_help.get(k)})" if k in base_help else ""
            nv = input(f"  {k} [{v}]{hint}: ").strip()
            if nv != "":
                try:
                    base[k] = int(nv)
                except Exception:
                    try:
                        base[k] = float(nv)
                    except Exception:
                        # list input support: a,b,c
                        if "," in nv:
                            base[k] = parse_value_list(nv)
                        else:
                            base[k] = nv
    # Allow adding base args
    if prompt_yes_no("Add more base args?", False):
        while True:
            k = input("  key (blank to finish): ").strip()
            if not k:
                break
            v = input("  value: ").strip()
            if v == "":
                continue
            # coerce
            try:
                base[k] = int(v)
            except Exception:
                try:
                    base[k] = float(v)
                except Exception:
                    if "," in v:
                        base[k] = parse_value_list(v)
                    else:
                        base[k] = v

    # Offer dynamic discovery of CLI params for custom or advanced users
    if prompt_yes_no("Discover available CLI parameters from script --help?", False):
        try:
            params = discover_cli_params(script)
            if params:
                print("Discovered parameters (flags without --):")
                # Map to names used in manifest (without --)
                for p in params:
                    print("  -", p)
                # Merge into sweep_keys help without descriptions
                spec.setdefault("sweep_keys", {})
                for p in params:
                    spec["sweep_keys"].setdefault(p, "")
        except Exception:
            print("(skip) failed to discover parameters; continuing")

    if tips:
        print("\nSweep suggestions:")
        for t in tips:
            print("  -", t)
    sweep, singles = add_sweep_interactive(spec.get("sweep_keys"))
    # Merge single-value params into base args so they are fixed (not swept)
    if singles:
        base.update(singles)

    # Auto label from first two sweep keys if present
    default_label = "{idx}"
    if sweep:
        keys = list(sweep.keys())
        if keys:
            default_label = "_".join([f"{keys[i]}{{{keys[i]}}}" for i in range(min(2, len(keys)))])
    label = input(f"Label template [default {default_label}]: ").strip() or default_label

    job: Dict[str, Any] = {
        "name": name,
        "script": script,
        "base_args": base,
        "sweep": sweep,
        "label": label,
    }

    if suite_type in ("live", "online"):
        print("Accounts configuration:")
        print("  1) auto (use account_auto_start and next IDs)")
        print("  2) start+count")
        print("  3) explicit list")
        sel = input("Choose 1/2/3 [1]: ").strip() or "1"
        if sel == "1":
            job["accounts"] = "auto"
        elif sel == "2":
            start = int(input("  start: ").strip())
            count = int(input("  count: ").strip())
            job["accounts"] = {"start": start, "count": count}
        elif sel == "3":
            lst = input("  list (comma-separated integers): ").strip()
            try:
                nums = [int(x) for x in lst.split(',') if x.strip()]
                job["accounts"] = nums
            except Exception:
                print("  invalid list; defaulting to auto")
                job["accounts"] = "auto"
        else:
            job["accounts"] = "auto"

    return job


def _yaml_dump_fallback(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines: List[str] = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(_yaml_dump_fallback(v, indent + 1))
            else:
                sval = v
                if isinstance(v, str) and (":" in v or v.strip() == "" or v.startswith("[") or v.startswith("{")):
                    sval = f'"{v}"'
                lines.append(f"{sp}{k}: {sval}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines: List[str] = []
        for it in obj:
            if isinstance(it, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_yaml_dump_fallback(it, indent + 1))
            else:
                lines.append(f"{sp}- {it}")
        return "\n".join(lines)
    return f"{sp}{obj}"


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
        with outp.open('w', encoding='utf-8') as f:
            yaml.safe_dump(manifest, f, sort_keys=False)  # type: ignore
        print(f"Wrote YAML manifest: {outp}")
        return
    except Exception:
        # Fallback to minimal YAML emitter (avoid JSON surprise)
        content = _yaml_dump_fallback(manifest)
        outp.write_text(content + "\n", encoding='utf-8')
        print(f"PyYAML not available; wrote YAML (fallback): {outp}")


def main() -> None:
    print("Suite manifest initializer")
    print("This wizard creates runnable manifests for: live (inference trading), backtest (offline), online (learning).\n")
    suite_type = choose_suite_type()

    # Top-level fields
    logs_dir_default = {
        "live": "logs/suite_live",
        "backtest": "logs/suite_backtest",
        "online": "logs/suite_online",
    }[suite_type]
    # Keep advanced hidden unless requested
    show_adv = prompt_yes_no("Show advanced settings (logs_dir, max_procs, account_auto_start)?", False)
    logs_dir = logs_dir_default
    max_procs = int(os.cpu_count() or 4)
    account_auto_start = 1
    if show_adv:
        logs_dir = prompt("logs_dir", logs_dir_default)
        try:
            max_procs = int(prompt("max_procs (parallel processes)", str(max_procs)))
        except Exception:
            pass
        if suite_type in ("live", "online"):
            try:
                account_auto_start = int(prompt("account_auto_start (first auto account id)", "1"))
            except Exception:
                account_auto_start = 1

    manifest: Dict[str, Any] = {
        "logs_dir": logs_dir,
        "max_procs": max_procs,
    }

    if suite_type in ("live", "online"):
        manifest["account_auto_start"] = account_auto_start

    jobs: List[Dict[str, Any]] = []
    while True:
        jobs.append(add_job_interactive(suite_type))
        if not prompt_yes_no("Add another job?", False):
            break

    manifest["jobs"] = jobs

    print("\nPreview manifest:")
    try:
        import yaml  # type: ignore
        print(yaml.safe_dump(manifest, sort_keys=False))  # type: ignore
    except Exception:
        print(_yaml_dump_fallback(manifest))
    out_path = prompt("Output manifest path", f"{suite_type}.yaml")
    write_manifest(out_path, manifest)
    # Print how to run
    runner = {
        "live": "forex-rl/tools/run_suite_live.py",
        "backtest": "forex-rl/tools/run_suite_backtest.py",
        "online": "forex-rl/tools/run_suite_online.py",
    }[suite_type]
    print(f"\nRun this suite:\n  python {runner} --manifest {out_path}\n")
    if suite_type in ("live", "online"):
        print("Optional NAV sampling (PnL time series):")
        print("  python forex-rl/tools/nav_sampler.py --logs-dir logs/suite_live --interval-secs 15")


def discover_cli_params(script_path: str) -> List[str]:
    """Heuristic extraction of --flags from a Python script by grepping argparse add_argument calls.
    Returns param names without leading dashes suitable for manifest keys.
    """
    p = Path(script_path)
    if not p.exists():
        # Try relative to repo root
        p = Path.cwd() / script_path
        if not p.exists():
            return []
    text = p.read_text(encoding='utf-8', errors='ignore')
    import re
    flags = re.findall(r"add_argument\(\s*['\"](--[\w-]+)['\"]", text)
    # Deduplicate and strip leading dashes
    out: List[str] = []
    seen = set()
    for f in flags:
        name = f.lstrip('-')
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
