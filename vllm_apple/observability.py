from __future__ import annotations

import re
import threading
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Iterator

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$")
_current_request_id: ContextVar[str | None] = ContextVar("vllm_apple_request_id", default=None)


def resolve_request_id(candidate: str | None) -> str:
    """Keep a safe caller ID or create an opaque ID without embedding user data."""
    if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    return _current_request_id.get()


@contextmanager
def request_scope(request_id: str) -> Iterator[None]:
    token = _current_request_id.set(request_id)
    try:
        yield
    finally:
        _current_request_id.reset(token)


@dataclass(frozen=True, slots=True)
class RequestLogRecord:
    request_id: str
    method: str
    route: str
    status: int
    duration_ms: float
    streamed: bool
    error_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StructuredRequestLog:
    """Bounded, content-free request history safe for the long-lived daemon."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("request log capacity must be positive")
        self.capacity = capacity
        self._records: deque[RequestLogRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, record: RequestLogRecord) -> None:
        with self._lock:
            self._records.append(record)

    def records(self) -> tuple[RequestLogRecord, ...]:
        with self._lock:
            return tuple(self._records)
