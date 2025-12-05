#!/usr/bin/env python3
from __future__ import annotations

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
    log_path: Path


def build_variants(job: Dict[str, Any], ts: str, logs_dir: Path) -> List[Variant]:
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
    label_tmpl = str(job.get("label", "{idx}"))

    out: List[Variant] = []
    for idx, vparams in enumerate(combos, start=1):
        params = dict(base_params); params.update(vparams)
        label_base = label_tmpl.format(**{**params, "idx": idx}) if "{" in label_tmpl else f"{label_tmpl}-{idx}"
        label = f"{label_base}-{_rand_suffix()}"
        log_path = logs_dir / f"{name}_{label}_{ts}.log"
        out.append(Variant(name=name, script=script, params=params, label=label, log_path=log_path))
    return out


def run_suite(manifest_path: str, max_procs: Optional[int], python_bin: Optional[str], dry_run: bool) -> int:
    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("manifest must contain a non-empty 'jobs' list")
    ts = _now_tag()
    logs_dir = Path(manifest.get("logs_dir") or "logs/suite_backtest")
    _ensure_dir(logs_dir)

    variants: List[Variant] = []
    for job in jobs:
        variants.extend(build_variants(job, ts, logs_dir))
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
        print(f"started {v.name} label={v.label} pid={p.pid} log={v.log_path}")
        return p, f

    try:
        while pending or running:
            while pending and len(running) < max_procs:
                v = pending.pop(0)
                if dry_run:
                    print(f"DRYRUN would start: {v.name} {v.label} -> {v.log_path}")
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
                    print(f"ended {v.name} label={v.label} rc={ret}")
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
    ap = argparse.ArgumentParser(description="Manifest-driven backtest suite runner")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-procs", type=int, default=None)
    ap.add_argument("--python-bin", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(run_suite(args.manifest, args.max_procs, args.python_bin, args.dry_run))


if __name__ == "__main__":
    main()
