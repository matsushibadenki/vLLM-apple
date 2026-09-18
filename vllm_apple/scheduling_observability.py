"""Fixed-cardinality, non-sensitive scheduling counters."""
from __future__ import annotations

import threading

from .execution import ExecutionBackend

_MAX_COUNT = 2_147_483_647
_WAIT_BUCKETS = ("under_1ms", "1_to_10ms", "10_to_100ms", "100ms_or_more")


class SchedulingObservability:
    """No request IDs, operators, inputs, model names, or arbitrary error strings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._assignments = {backend.value: 0 for backend in ExecutionBackend}
        self._queue_wait = {bucket: 0 for bucket in _WAIT_BUCKETS}
        self._adaptive = {status: 0 for status in ("applied", "deferred", "ignored")}
        self._fallback_attempts = 0
        self._fallback_exhausted = 0
        self._contention_rejections = 0
        self._steals = 0

    @staticmethod
    def _increment(value: int, amount: int = 1) -> int:
        return min(_MAX_COUNT, value + amount)

    def assignment(self, backend: ExecutionBackend) -> None:
        with self._lock:
            key = backend.value
            self._assignments[key] = self._increment(self._assignments[key])

    def queue_wait(self, nanoseconds: int) -> None:
        if nanoseconds < 0:
            raise ValueError("negative queue wait")
        bucket = (
            "under_1ms" if nanoseconds < 1_000_000
            else "1_to_10ms" if nanoseconds < 10_000_000
            else "10_to_100ms" if nanoseconds < 100_000_000
            else "100ms_or_more"
        )
        with self._lock:
            self._queue_wait[bucket] = self._increment(self._queue_wait[bucket])

    def fallback(self, attempts: int, *, exhausted: bool) -> None:
        if attempts < 0 or attempts > len(ExecutionBackend):
            raise ValueError("invalid fallback attempt count")
        with self._lock:
            self._fallback_attempts = self._increment(self._fallback_attempts, attempts)
            if exhausted:
                self._fallback_exhausted = self._increment(self._fallback_exhausted)

    def contention_rejection(self) -> None:
        with self._lock:
            self._contention_rejections = self._increment(self._contention_rejections)

    def steal(self) -> None:
        with self._lock:
            self._steals = self._increment(self._steals)

    def adaptive_transition(self, status: str) -> None:
        if status not in self._adaptive:
            raise ValueError("invalid adaptive transition")
        with self._lock:
            self._adaptive[status] = self._increment(self._adaptive[status])

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "assignments": dict(self._assignments),
                "queue_wait_buckets": dict(self._queue_wait),
                "fallback_attempts": self._fallback_attempts,
                "fallback_exhausted": self._fallback_exhausted,
                "contention_rejections": self._contention_rejections,
                "steals": self._steals,
                "adaptive_transitions": dict(self._adaptive),
            }
