#!/usr/bin/env python3
"""
Lightweight QuestDB ILP client and helpers (TCP 9009 by default).
- Formats InfluxDB Line Protocol (ILP) lines
- Maintains a persistent TCP socket with auto-reconnect
- Provides utility functions to escape tags/fields and convert timestamps

Designed for high-frequency tick ingestion without external dependencies.
"""
from __future__ import annotations

import socket
import time
import threading
from typing import Any, Dict, Optional

try:
    from dateutil import parser as dateutil_parser  # type: ignore
except Exception:  # pragma: no cover
    dateutil_parser = None  # type: ignore


def _escape_measurement(name: str) -> str:
    # In ILP, measurement (table) escapes commas and spaces
    return name.replace(",", "\\,").replace(" ", "\\ ")


def _escape_tag_key_or_value(s: str) -> str:
    # Tag keys/values escape commas, equals, spaces
    return s.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _escape_string_field_value(s: str) -> str:
    # Field string values are double-quoted; escape quotes and backslashes
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


def iso_to_nanos(iso_ts: str) -> int:
    """Convert an ISO8601/RFC3339 timestamp to epoch nanoseconds.
    Falls back to current time if parsing fails.
    """
    try:
        if dateutil_parser is None:
            raise RuntimeError("python-dateutil not available")
        dt = dateutil_parser.isoparse(iso_ts)
        if dt.tzinfo is None:
            # Assume UTC if naive
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return time.time_ns()


class QuestDBILPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9009, connect_timeout: float = 3.0) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        with self._lock:
            self._close_locked()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            # Disable Nagle for lower latency
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            # Set non-blocking writes to avoid stalls; we handle errors
            s.setblocking(True)
            self._sock = s

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def write_line(self, line: str) -> None:
        data = (line.rstrip("\n") + "\n").encode("utf-8")
        with self._lock:
            for attempt in range(2):
                try:
                    if self._sock is None:
                        self._connect()
                    assert self._sock is not None
                    self._sock.sendall(data)
                    return
                except Exception:
                    # reconnect once
                    try:
                        self._connect()
                    except Exception:
                        if attempt == 1:
                            raise
                        # brief backoff
                        time.sleep(0.1)

    def build_line(self, table: str, tags: Dict[str, Any], fields: Dict[str, Any], ts_ns: Optional[int]) -> str:
        m = _escape_measurement(table)
        # tags: only keep non-empty stringable values
        tag_parts = []
        for k, v in tags.items():
            if v is None:
                continue
            ks = str(k)
            vs = str(v)
            if ks == "" or vs == "":
                continue
            tag_parts.append(f"{_escape_tag_key_or_value(ks)}={_escape_tag_key_or_value(vs)}")
        tag_set = ("," + ",".join(tag_parts)) if tag_parts else ""
        # fields: follow ILP types: int -> i, float -> as-is, bool -> t/f, str -> quoted
        field_parts = []
        for k, v in fields.items():
            if v is None:
                continue
            key = _escape_tag_key_or_value(str(k))
            if isinstance(v, bool):
                field_parts.append(f"{key}={'t' if v else 'f'}")
            elif isinstance(v, int) and not isinstance(v, bool):
                field_parts.append(f"{key}={v}i")
            elif isinstance(v, float):
                # Ensure finite representation
                if v != v or v == float("inf") or v == float("-inf"):
                    continue
                field_parts.append(f"{key}={repr(float(v))}")
            else:
                s = _escape_string_field_value(str(v))
                field_parts.append(f"{key}=\"{s}\"")
        if not field_parts:
            # ILP requires at least one field; add a dummy
            field_parts.append("ok=1i")
        field_set = ",".join(field_parts)
        if ts_ns is None:
            ts_ns = time.time_ns()
        return f"{m}{tag_set} {field_set} {int(ts_ns)}"

    def send_row(self, table: str, tags: Dict[str, Any], fields: Dict[str, Any], ts_ns: Optional[int] = None) -> None:
        line = self.build_line(table, tags, fields, ts_ns)
        self.write_line(line)


__all__ = [
    "QuestDBILPClient",
    "iso_to_nanos",
]
