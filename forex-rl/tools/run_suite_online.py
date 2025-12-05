#!/usr/bin/env python3
from __future__ import annotations

# Online learning/trading suite runner (similar to live runner but kept separate for future specialization)

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import string


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
                args.extend([flag, "false"])
        elif isinstance(v, (int, float)):
            args.extend([flag, str(v)])
        elif isinstance(v, (list, tuple)):
            args.extend([flag, ",".join(str(x) for x in v)])
        else:
            args.extend([flag, str(v)])
    return args

ParamValue = Union[List[Any], Dict[str, Any], Any]


def _expand_value(key: str, spec: ParamValue):
    if isinstance(spec, list):
        return [(key, v) for v in spec]
    if isinstance(spec, dict):
        start = spec.get("start"); end = spec.get("end"); step = spec.get("step"); count = spec.get("count")
        if start is None or end is None:
            raise ValueError(f"range spec for '{key}' requires start and end")
        start_f = float(start); end_f = float(end)
        if step is not None:
            step_f = float(step)
            vals = []
            x = start_f
            if step_f > 0:
                while x <= end_f + 1e-12:
                    vals.append(x); x += step_f
            else:
                while x >= end_f - 1e-12:
                    vals.append(x); x += step_f
            return [(key, _coerce(x)) for x in vals]
        if count is not None:
            n = int(count)
            if n <= 1:
                return [(key, _coerce(start_f))]
            vals = [start_f + (end_f - start_f) * (i/(n-1)) for i in range(n)]
            return [(key, _coerce(x)) for x in vals]
        raise ValueError(f"range spec for '{key}' requires step or count")
    return [(key, spec)]


def _coerce(x: float):
    ix = int(round(x))
    return ix if abs(x - ix) < 1e-12 else float(x)


def _cartesian(sweep: Dict[str, ParamValue]) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    choices = [_expand_value(k, sweep[k]) for k in keys]
    out: List[Dict[str, Any]] = []
    for prod in itertools.product(*choices):
        d: Dict[str, Any] = {}
        for k, v in prod:
            d[k] = v
        out.append(d)
    return out


def _rand_suffix(k: int = 7) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(k))


@dataclass
class Variant:
    name: str
    script: str
    params: Dict[str, Any]
    label: str
    account_id: int
    log_path: Path


def build_variants(job: Dict[str, Any], ts: str, logs_dir: Path, *, next_acct: int, default_start: int) -> Tuple[List[Variant], int]:
    name = str(job.get("name") or job.get("script"))
    script = str(job["script"])  # required
    base_args = job.get("base_args") or {}
    if isinstance(base_args, str):
        base_params: Dict[str, Any] = {}
        base_list = shlex.split(base_args)
    elif isinstance(base_args, dict):
        base_params = dict(base_args)
        base_list = []
    else:
        base_params = {}
        base_list = []

    sweep = job.get("sweep") or {}
    combos = _cartesian(sweep)

    # Accounts allocation (auto by default)
    acct_cfg = job.get("accounts") or "auto"
    acct_list: List[int]
    if isinstance(acct_cfg, str) and acct_cfg.lower() == "auto":
        acct_list = [max(next_acct, default_start) + i for i in range(len(combos) or 1)]
        next_acct = acct_list[-1] + 1
    elif isinstance(acct_cfg, dict) and ("start" in acct_cfg or acct_cfg.get("auto")):
        start = int(acct_cfg.get("start", max(next_acct, default_start)))
        count = int(acct_cfg.get("count", len(combos)))
        acct_list = [start + i for i in range(count)]
        next_acct = start + count
    elif isinstance(acct_cfg, list):
        acct_list = [int(x) for x in acct_cfg]
    else:
        acct_list = [max(next_acct, default_start) + i for i in range(len(combos) or 1)]
        next_acct = acct_list[-1] + 1

    label_tmpl = str(job.get("label", "{idx}"))

    out: List[Variant] = []
    for idx, (params, acct) in enumerate(zip(combos, acct_list), start=1):
        merged = dict(base_params); merged.update(params)
        merged.setdefault("broker", "ipc")
        merged.setdefault("broker-account-id", acct)
        label_base = label_tmpl.format(**{**merged, "idx": idx, "account_id": acct}) if "{" in label_tmpl else f"{label_tmpl}-{idx}"
        label = f"{label_base}-{_rand_suffix()}"
        log_path = logs_dir / f"{name}_{label}_acct{acct}_{ts}.log"
        out.append(Variant(name=name, script=script, params=merged, label=label, account_id=acct, log_path=log_path))
    return out, next_acct


def run_suite(manifest_path: str, max_procs: Optional[int], python_bin: Optional[str], dry_run: bool) -> int:
    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("manifest must contain a non-empty 'jobs' list")
    ts = _now_tag()
    logs_dir = Path(manifest.get("logs_dir") or "logs/suite_online")
    _ensure_dir(logs_dir)

    variants: List[Variant] = []
    next_acct = int(manifest.get("account_auto_start") or 1)
    for job in jobs:
        vs, next_acct = build_variants(job, ts, logs_dir, next_acct=next_acct, default_start=int(manifest.get("account_auto_start") or 1))
        variants.extend(vs)
    if not variants:
        print("No variants to run.")
        return 0

    if max_procs is None:
        max_procs = int(manifest.get("max_procs") or os.cpu_count() or 4)
    py = python_bin or sys.executable

    running: List[Tuple[subprocess.Popen, Variant, Any]] = []
    pending: List[Variant] = list(variants)
    rc = 0

    def spawn(v: Variant):
        cli = [py, v.script] + _cli_from_params(v.params)
        f = open(v.log_path, "w", buffering=1)
        p = subprocess.Popen(cli, stdout=f, stderr=subprocess.STDOUT)
        print(f"started {v.name} label={v.label} acct={v.account_id} pid={p.pid} log={v.log_path}")
        return p, f

    try:
        while pending or running:
            while pending and len(running) < max_procs:
                v = pending.pop(0)
                if dry_run:
                    print(f"DRYRUN would start: {v.name} {v.label} acct={v.account_id} -> {v.log_path}")
                    continue
                p, f = spawn(v)
                running.append((p, v, f))
            alive: List[Tuple[subprocess.Popen, Variant, Any]] = []
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
                    print(f"ended {v.name} label={v.label} acct={v.account_id} rc={ret}")
            running = alive
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted; terminating...")
        for p, v, f in running:
            try:
                p.terminate()
            except Exception:
                pass
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifest-driven online learning suite runner")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-procs", type=int, default=None)
    ap.add_argument("--python-bin", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(run_suite(args.manifest, args.max_procs, args.python_bin, args.dry_run))


if __name__ == "__main__":
    main()
