from __future__ import annotations

import http.client
import json
import math
from urllib.parse import urlsplit

from .soak import _validated_base_url


def fetch_execution_preview(
    base_url: str, *, session_token: str | None = None, timeout: float = 5.0
) -> dict:
    """Read local diagnostics without proxies, redirects, or policy mutation."""
    url = urlsplit(_validated_base_url(base_url, allow_remote=False))
    if url.path not in {"", "/"}:
        raise ValueError("preview URL must not include a path")
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be finite and between 0 and 60 seconds")
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=timeout)
    headers = {"Accept": "application/json"}
    if session_token is not None:
        headers["Authorization"] = f"Bearer {session_token}"
    try:
        connection.request("GET", "/v1/execution-plan/preview", headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"preview HTTP status {response.status}")
        data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("preview response exceeds 1 MiB")
        payload = json.loads(data)
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
            raise ValueError("invalid preview envelope")
        if payload["schema_version"] != 1 or type(payload.get("available")) is not bool:
            raise ValueError("unsupported preview envelope")
        if payload["available"]:
            plan = payload.get("plan")
            if not isinstance(plan, dict) or plan.get("dry_run") is not True or payload.get("reason") is not None:
                raise ValueError("invalid preview plan")
        elif payload.get("plan") is not None or not isinstance(payload.get("reason"), str) or not payload["reason"]:
            raise ValueError("invalid unavailable preview")
        return payload
    finally:
        connection.close()
