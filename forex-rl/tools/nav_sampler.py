#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure forex-rl root is importable to access broker_ipc
TOOLS_DIR = Path(__file__).resolve().parent
FX_DIR = TOOLS_DIR.parent
if str(FX_DIR) not in sys.path:
    sys.path.append(str(FX_DIR))

try:
    import broker_ipc  # type: ignore
except Exception as e:
    broker_ipc = None  # type: ignore


def parse_accounts_arg(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    out: List[int] = []
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            step = 1 if end >= start else -1
            for x in range(start, end + step, step):
                out.append(int(x))
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return sorted(list({x for x in out}))


def scan_accounts_from_logs(logs_dir: str) -> Dict[int, str]:
    # Extract acct and label from filenames like: <job>_<label>_acct<id>_<timestamp>.log
    p = Path(logs_dir)
    if not p.exists():
        return {}
    acct_to_label: Dict[int, str] = {}
    pat = re.compile(r"^(?P<prefix>.+?)_(?P<label>.+?)_acct(?P<acct>\d+)_\d{8}_\d{6}\.log$")
    for entry in p.iterdir():
        if not entry.is_file():
            continue
        m = pat.match(entry.name)
        if not m:
            continue
        acct = int(m.group('acct'))
        label = m.group('label')
        acct_to_label.setdefault(acct, label)
    return acct_to_label


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_field(d: Dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        try:
            if isinstance(v, (int, float)):
                return float(v)
        except Exception:
            continue
    return None


def sample_nav(client: 'broker_ipc.BrokerIPCClient', account_id: int) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        'equity': None,
        'nav': None,
        'balance': None,
        'realized_pnl': None,
        'unrealized_pnl': None,
        'margin': None,
        'free_margin': None,
    }
    try:
        r = client.get_account_derived(account_id)
        jd = r.data or {}
        out['equity'] = select_field(jd, ['equity', 'Equity'])
        out['nav'] = select_field(jd, ['nav', 'NAV', 'NetAssetValue'])
        out['balance'] = select_field(jd, ['balance', 'Balance'])
        out['realized_pnl'] = select_field(jd, ['realizedPnL', 'realized_pnl'])
        out['unrealized_pnl'] = select_field(jd, ['unrealizedPnL', 'unrealized_pnl'])
        out['margin'] = select_field(jd, ['margin', 'used_margin'])
        out['free_margin'] = select_field(jd, ['free_margin', 'available_margin'])
        if out['equity'] is None or out['nav'] is None:
            # Fallback to get_account
            r2 = client.get_account(account_id)
            jd2 = r2.data or {}
            out['equity'] = out['equity'] or select_field(jd2, ['equity', 'Equity', 'balance'])
            out['nav'] = out['nav'] or select_field(jd2, ['NAV', 'nav', 'balance'])
            out['balance'] = out['balance'] or select_field(jd2, ['balance'])
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Periodic NAV sampler for tall-borker IPC")
    ap.add_argument('--ipc-socket', default=os.environ.get('PRAGMAGEN_IPC_SOCKET', '/run/pragmagen/pragmagen.sock'))
    ap.add_argument('--accounts', help='Comma-separated accounts or ranges like 1,2,10-20')
    ap.add_argument('--logs-dir', default='logs/suite_live', help='Scan this dir for acct IDs in filenames if --accounts not set')
    ap.add_argument('--interval-secs', type=float, default=15.0)
    ap.add_argument('--duration-secs', type=float, default=0.0, help='0 = run indefinitely')
    ap.add_argument('--out', default='', help='Output CSV path (default logs/nav/nav_<timestamp>.csv)')
    ap.add_argument('--append', action='store_true', help='Append to existing CSV if present')
    args = ap.parse_args()

    if broker_ipc is None:
        raise SystemExit('broker_ipc module unavailable; ensure forex-rl/broker_ipc.py exists')

    acct_list = parse_accounts_arg(args.accounts)
    labels_by_acct: Dict[int, str] = {}
    if not acct_list:
        labels_by_acct = scan_accounts_from_logs(args.logs_dir)
        acct_list = sorted(labels_by_acct.keys())
    if not acct_list:
        raise SystemExit('No accounts provided or discovered in logs-dir')

    out_dir = Path('logs/nav')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"nav_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    # Open CSV
    header = ['time_iso', 'epoch', 'account_id', 'label', 'equity', 'nav', 'balance', 'realized_pnl', 'unrealized_pnl', 'margin', 'free_margin']
    write_header = True
    if out_path.exists() and args.append:
        write_header = False
    f = out_path.open('a' if args.append else 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
        f.flush()

    client = broker_ipc.BrokerIPCClient(socket_path=args.ipc_socket)

    start_ts = time.time()
    try:
        while True:
            now_iso = iso_now()
            epoch = time.time()
            for acct in acct_list:
                navs = sample_nav(client, int(acct))
                label = labels_by_acct.get(int(acct), '')
                row = [now_iso, f"{epoch:.3f}", int(acct), label,
                       _fmt(navs.get('equity')), _fmt(navs.get('nav')), _fmt(navs.get('balance')),
                       _fmt(navs.get('realized_pnl')), _fmt(navs.get('unrealized_pnl')),
                       _fmt(navs.get('margin')), _fmt(navs.get('free_margin'))]
                writer.writerow(row)
            f.flush()
            if args.duration_secs and (time.time() - start_ts) >= float(args.duration_secs):
                break
            time.sleep(max(0.5, float(args.interval_secs)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass
    print(f"wrote {out_path}")


def _fmt(x: Optional[float]) -> str:
    try:
        if x is None:
            return ''
        return f"{float(x):.8f}"
    except Exception:
        return ''


if __name__ == '__main__':
    main()
