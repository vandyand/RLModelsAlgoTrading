#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def _load_manifest(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    # Try YAML first if available
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)  # type: ignore[no-any-return]
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Try JSON
    try:
        return json.loads(text)
    except Exception:
        raise SystemExit(
            "Failed to parse manifest. Install PyYAML (pip install pyyaml) or regenerate the manifest as JSON."
        )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _kebab(k: str) -> str:
    return k.replace("_", "-")


def _cli_from_params(params: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in params.items():
        if v is None:
            continue
        flag = f"--{_kebab(str(k))}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
            else:
                # explicit false -> include flag with value
                args.extend([flag, "false"])
        elif isinstance(v, (int, float)):
            args.extend([flag, str(v)])
        elif isinstance(v, (list, tuple)):
            args.extend([flag, ",".join(str(x) for x in v)])
        else:
            args.extend([flag, str(v)])
    return args


ParamValue = Union[List[Any], Dict[str, Any], Any]


def _expand_value(key: str, spec: ParamValue) -> List[Tuple[str, Any]]:
    # Returns list of (key, value) choices
    if isinstance(spec, list):
        return [(key, v) for v in spec]
    if isinstance(spec, dict):
        # range forms: {start,end,step} or {start,end,count}
        start = spec.get("start")
        end = spec.get("end")
        step = spec.get("step")
        count = spec.get("count")
        if start is None or end is None:
            raise ValueError(f"range spec for '{key}' requires start and end")
        start_f = float(start)
        end_f = float(end)
        if step is not None:
            step_f = float(step)
            if step_f == 0:
                raise ValueError(f"step==0 for '{key}'")
            vals: List[float] = []
            x = start_f
            # inclusive end with tolerance
            if step_f > 0:
                while x <= end_f + 1e-12:
                    vals.append(x)
                    x += step_f
            else:
                while x >= end_f - 1e-12:
                    vals.append(x)
                    x += step_f
            return [(key, _coerce_number(v)) for v in vals]
        if count is not None:
            n = int(count)
            if n <= 1:
                return [(key, _coerce_number(start_f))]
            vals = [start_f + (end_f - start_f) * (i / (n - 1)) for i in range(n)]
            return [(key, _coerce_number(v)) for v in vals]
        raise ValueError(f"range spec for '{key}' requires step or count")
    # scalar
    return [(key, spec)]


def _coerce_number(x: float) -> Union[int, float]:
    ix = int(round(x))
    if abs(x - ix) < 1e-12:
        return ix
    return float(x)


@dataclass
class JobVariant:
    job_name: str
    script: str
    params: Dict[str, Any]
    label: str
    account_id: int
    log_path: Path


def _cartesian_product(sweep: Dict[str, ParamValue]) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    choices: List[List[Tuple[str, Any]]] = [
        _expand_value(k, sweep[k]) for k in keys
    ]
    variants: List[Dict[str, Any]] = []
    for prod in itertools.product(*choices):
        d: Dict[str, Any] = {}
        for k, v in prod:
            d[k] = v
        variants.append(d)
    return variants


def _render_label(template: str, params: Dict[str, Any], fallback: str) -> str:
    try:
        return template.format(**{k.replace('-', '_'): v for k, v in params.items()})
    except Exception:
        return fallback


def _rand_suffix(k: int = 7) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(k))


def _build_variants(job: Dict[str, Any], ts: str, log_dir: Path, *, auto_acct_start: int, next_auto_acct: int) -> Tuple[List[JobVariant], int]:
    name = str(job.get("name") or job.get("script"))
    script = str(job["script"])  # required
    base_args = job.get("base_args") or {}
    if isinstance(base_args, str):
        # parse CLI string to dict-like later; easier: keep as extra list
        base_list = shlex.split(base_args)
        base_params: Dict[str, Any] = {}
    elif isinstance(base_args, dict):
        base_list = []
        base_params = dict(base_args)
    else:
        base_list = []
        base_params = {}

    # Sweep expansion
    sweep = job.get("sweep") or {}
    variants = _cartesian_product(sweep)

    # Accounts allocation
    acct_cfg = job.get("accounts") or {}
    acct_list: List[int]
    if isinstance(acct_cfg, str) and acct_cfg.lower() == "auto":
        acct_list = [next_auto_acct + i for i in range(len(variants) or 1)]
        next_auto_acct += len(acct_list)
    elif isinstance(acct_cfg, dict) and ("start" in acct_cfg or acct_cfg.get("auto")):
        start = int(acct_cfg["start"])
        count = int(acct_cfg.get("count", len(variants)))
        acct_list = [start + i for i in range(count)]
    elif isinstance(acct_cfg, list):
        acct_list = [int(x) for x in acct_cfg]
    else:
        # default: global auto allocation starting from manifest-level account_auto_start
        acct_list = [max(next_auto_acct, auto_acct_start) + i for i in range(len(variants) or 1)]
        next_auto_acct = acct_list[-1] + 1

    # Clip variant count to available accounts if needed
    n = min(len(variants) or 1, len(acct_list))
    variants = variants[:n] if variants else [{}]

    label_tmpl = str(job.get("label", "{idx}"))

    out: List[JobVariant] = []
    for idx, (vparams, acct) in enumerate(zip(variants, acct_list), start=1):
        merged: Dict[str, Any] = {}
        merged.update(base_params)
        merged.update(vparams)
        # Always ensure broker ipc by default unless overridden
        merged.setdefault("broker", "ipc")
        # Build label
        label_base = _render_label(label_tmpl, {**merged, "idx": idx, "account_id": acct}, fallback=f"{name}_{idx}")
        label = f"{label_base}-{_rand_suffix()}"
        # Build log path
        log_path = log_dir / f"{name}_{label}_acct{acct}_{ts}.log"
        out.append(JobVariant(job_name=name, script=script, params=merged, label=label, account_id=acct, log_path=log_path))
    return out, next_auto_acct


def run_suite(manifest_path: str, max_procs: Optional[int] = None, python_bin: Optional[str] = None, dry_run: bool = False) -> int:
    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("manifest must contain a non-empty 'jobs' list")

    ts = _now_tag()
    logs_dir = Path(manifest.get("logs_dir") or "logs/suite_live")
    _ensure_dir(logs_dir)

    # Build variants
    all_variants: List[JobVariant] = []
    next_auto_acct = int(manifest.get("account_auto_start") or 1)
    for job in jobs:
        variants, next_auto_acct = _build_variants(job, ts, logs_dir, auto_acct_start=int(manifest.get("account_auto_start") or 1), next_auto_acct=next_auto_acct)
        all_variants.extend(variants)

    if not all_variants:
        print("No variants to run.")
        return 0

    # Concurrency
    if max_procs is None:
        max_procs = int(manifest.get("max_procs") or os.cpu_count() or 4)
    py = python_bin or sys.executable

    running: List[Tuple[subprocess.Popen, JobVariant, Any]] = []
    pending: List[JobVariant] = list(all_variants)
    rc = 0

    def spawn(v: JobVariant):
        # Compose command
        params_with_account = dict(v.params)
        params_with_account.setdefault("broker-account-id", v.account_id)
        cli = [py, v.script] + _cli_from_params(params_with_account)
        # Compose extra base list if provided in manifest as string
        # Note: we've already merged base_args dict; base string was not preserved
        stdout_f = open(v.log_path, "w", buffering=1)
        env = os.environ.copy()
        p = subprocess.Popen(cli, stdout=stdout_f, stderr=subprocess.STDOUT)
        print(f"started {v.job_name} label={v.label} acct={v.account_id} pid={p.pid} log={v.log_path}")
        return p, stdout_f

    # Handle SIGINT to terminate children
    def _terminate_all():
        for p, v, f in running:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1.0)
        for p, v, f in running:
            try:
                p.kill()
            except Exception:
                pass

    try:
        while pending or running:
            # Fill slots
            while pending and len(running) < max_procs:
                v = pending.pop(0)
                if dry_run:
                    print(f"DRYRUN would start: {v.job_name} {v.label} acct={v.account_id} -> {v.log_path}")
                    continue
                p, f = spawn(v)
                running.append((p, v, f))

            # Reap
            alive: List[Tuple[subprocess.Popen, JobVariant, Any]] = []
            for p, v, f in running:
                ret = p.poll()
                if ret is None:
                    alive.append((p, v, f))
                else:
                    try:
                        f.close()
                    except Exception:
                        pass
                    rc |= int(ret)
                    print(f"ended {v.job_name} label={v.label} acct={v.account_id} rc={ret}")
            running = alive
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted, terminating children...")
        _terminate_all()
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifest-driven live inference trading suite runner")
    ap.add_argument("--manifest", required=True, help="Path to YAML/JSON manifest")
    ap.add_argument("--max-procs", type=int, default=None)
    ap.add_argument("--python-bin", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rc = run_suite(args.manifest, max_procs=args.max_procs, python_bin=args.python_bin, dry_run=args.dry_run)
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
