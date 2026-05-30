#!/usr/bin/env python3
"""Small Manifold API client used by dry-run-safe betting-agent modules."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.manifold.markets/v0"


@dataclass
class ManifoldResponse:
    ok: bool
    status: int | None
    payload: dict[str, Any] | list[Any] | None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        if isinstance(self.payload, dict):
            keys = ("id", "betId", "contractId", "amount", "outcome", "probBefore", "probAfter", "probability")
            return {k: self.payload.get(k) for k in keys if k in self.payload}
        return {"error": self.error, "status": self.status}


class ManifoldClient:
    """Minimal wrapper around Manifold's HTTP API.

    The API key is intentionally read from the environment only. Callers should
    keep all production order gates outside this client.
    """

    def __init__(self, api_key: str | None = None, *, base_url: str = DEFAULT_BASE_URL, timeout: int = 12) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("MANIFOLD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> ManifoldResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Key {self.api_key}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(f"{self.base_url}/{path.lstrip('/')}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
                return ManifoldResponse(ok=True, status=resp.status, payload=payload)
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                payload = None
            return ManifoldResponse(ok=False, status=exc.code, payload=payload, error=raw or str(exc))
        except Exception as exc:  # pragma: no cover - exercised only by real network failures.
            return ManifoldResponse(ok=False, status=None, payload=None, error=str(exc))

    def get_market(self, market_id: str) -> ManifoldResponse:
        return self._request("GET", f"market/{market_id}")

    def place_bet(self, *, contract_id: str, amount: float, outcome: str) -> ManifoldResponse:
        return self._request(
            "POST",
            "bet",
            {
                "contractId": contract_id,
                "amount": amount,
                "outcome": outcome,
            },
        )

