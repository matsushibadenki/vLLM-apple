"""Bounded admission for the wrapper's single generation worker."""
from __future__ import annotations

import math
import select
import socket
import threading
import time
from collections import deque
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
        self._condition = threading.Condition()
        self._ready: deque[object] = deque()
        self._busy = False
        self._timeout = timeout_seconds
        self._metrics_lock = threading.Lock()
        self._metrics = dict(inflight=0, active=0, started=0, cancelled=0, timed_out=0,
                             rejected=0, preparation_failed=0, active_disconnects=0)

    def _record(self, **changes: int) -> None:
        with self._metrics_lock:
            for key, delta in changes.items():
                self._metrics[key] += delta

    def snapshot(self) -> dict[str, int]:
        with self._metrics_lock:
            return dict(self._metrics)

    @contextmanager
    def admit(self, cancelled: Callable[[], bool] | None = None,
              prepare: Callable[[], None] | None = None) -> Iterator[None]:
        if not self._capacity.acquire(blocking=False):
            self._record(rejected=1)
            raise GenerationAdmissionError("generation_queue_full")
        self._record(inflight=1)
        acquired = False
        started = False
        ticket = None
        try:
            if prepare is not None:
                try:
                    prepare()
                except Exception:
                    self._record(preparation_failed=1)
                    raise
            deadline = time.monotonic() + self._timeout
            with self._condition:
                ticket = object()
                self._ready.append(ticket)
                while not acquired:
                    if cancelled is not None and cancelled():
                        self._record(cancelled=1)
                        raise GenerationAdmissionError("generation_queue_cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if not self._busy and self._ready[0] is ticket:
                        self._ready.popleft()
                        self._busy = acquired = True
                    else:
                        self._condition.wait(timeout=min(remaining, .05))
            if not acquired:
                self._record(timed_out=1)
                raise GenerationAdmissionError("generation_queue_timeout")
            if cancelled is not None and cancelled():
                self._record(cancelled=1)
                raise GenerationAdmissionError("generation_queue_cancelled")
            self._record(active=1, started=1)
            started = True
            try:
                yield
            except (BrokenPipeError, ConnectionResetError):
                self._record(active_disconnects=1)
                raise
        finally:
            self._record(inflight=-1, active=-int(started))
            with self._condition:
                if acquired:
                    self._busy = False
                elif ticket is not None:
                    self._ready.remove(ticket)
                self._condition.notify_all()
            self._capacity.release()
