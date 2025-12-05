from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_SOCKET = "/run/pragmagen/pragmagen.sock"


@dataclass
class IpcResponse:
    ok: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    raw: str


class BrokerIPCClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        self.socket_path = socket_path

    def _send(self, payload: Dict[str, Any]) -> IpcResponse:
        req = json.dumps(payload) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.socket_path)
            s.sendall(req.encode("utf-8"))
            chunks = []
            s.shutdown(socket.SHUT_WR)
            while True:
                buf = s.recv(8192)
                if not buf:
                    break
                chunks.append(buf)
        raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
        try:
            jd = json.loads(raw)
            return IpcResponse(ok=bool(jd.get("ok", False)), data=jd.get("data"), error=jd.get("error"), raw=raw)
        except Exception:
            return IpcResponse(ok=False, data=None, error="invalid_json_response", raw=raw)

    def help(self) -> IpcResponse:
        return self._send({"action": "help"})

    def get_account(self, account_id: int) -> IpcResponse:
        return self._send({"action": "get_account", "payload": {"account_id": account_id}})

    def get_account_derived(self, account_id: int) -> IpcResponse:
        return self._send({"action": "get_account_derived", "payload": {"account_id": account_id}})

    def get_instruments(self) -> IpcResponse:
        return self._send({"action": "get_instruments"})

    def get_positions(self, account_id: int) -> IpcResponse:
        return self._send({"action": "get_positions", "payload": {"account_id": account_id}})

    def place_order(
        self,
        *,
        account_id: int,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "GTC",
        sim_slippage_bps: float = 0.0,
        sim_fee_perc: float = 0.0,
        sim_fee_fixed: float = 0.0,
    ) -> IpcResponse:
        payload = {
            "action": "place_order",
            "payload": {
                "account_id": account_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "limit_price": limit_price,
                "stop_price": None,
                "time_in_force": time_in_force,
                "sim_slippage_bps": sim_slippage_bps,
                "sim_fee_perc": sim_fee_perc,
                "sim_fee_fixed": sim_fee_fixed,
            },
        }
        return self._send(payload)

    def create_account(
        self,
        *,
        name: str,
        account_type: str = "paper",
        base_currency: Optional[str] = None,
        balance: Optional[float] = None,
        equity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IpcResponse:
        data: Dict[str, Any] = {
            "name": name,
            "type": account_type,
        }
        if base_currency is not None:
            data["base_currency"] = base_currency
        if balance is not None:
            data["balance"] = balance
        if equity is not None:
            data["equity"] = equity
        if metadata is not None:
            data["metadata"] = metadata
        return self._send({"action": "create_account", "payload": data})
