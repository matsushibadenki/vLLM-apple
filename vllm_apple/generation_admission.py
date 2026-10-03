"""Bounded admission for the wrapper's single generation worker."""
from __future__ import annotations

import math
import select
import socket
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator


def peer_disconnected(connection: socket.socket) -> bool:
    """Best-effort EOF check without consuming unread HTTP bytes."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class GenerationAdmissionError(RuntimeError):
    pass


class GenerationAdmission:
    def __init__(self, maximum_waiters: int = 8, timeout_seconds: float = 30) -> None:
        if type(maximum_waiters) is not int or not 0 <= maximum_waiters <= 128:
            raise ValueError("maximum waiters must be between 0 and 128")
        if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 300):
            raise ValueError("queue timeout must be finite and between 0 and 300 seconds")
        self._capacity = threading.BoundedSemaphore(maximum_waiters + 1)
        self._worker = threading.Lock()
        self._timeout = timeout_seconds

    @contextmanager
    def admit(self, cancelled: Callable[[], bool] | None = None) -> Iterator[None]:
        if not self._capacity.acquire(blocking=False):
            raise GenerationAdmissionError("generation_queue_full")
        acquired = False
        try:
            deadline = time.monotonic() + self._timeout
            while not acquired:
                if cancelled is not None and cancelled():
                    raise GenerationAdmissionError("generation_queue_cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                acquired = self._worker.acquire(timeout=min(remaining, .05))
            if not acquired:
                raise GenerationAdmissionError("generation_queue_timeout")
            if cancelled is not None and cancelled():
                raise GenerationAdmissionError("generation_queue_cancelled")
            yield
        finally:
            if acquired:
                self._worker.release()
            self._capacity.release()
