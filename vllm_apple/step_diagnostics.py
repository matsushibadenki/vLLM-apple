"""Bounded host-wall timing of a backend scheduler step, without tensor inspection."""
from __future__ import annotations

import threading
import time
from functools import wraps

from .phase_profile import _BoundedLatency


class StepDiagnostics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies = _BoundedLatency()
        self._slow_count = 0
        self._exceptions = 0
        self._samples: list[dict] = []

    def wrap(self, method):
        @wraps(method)
        def timed(*args, **kwargs):
            wall = time.time_ns()
            start = time.monotonic_ns()
            failed = False
            try:
                return method(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = max(0, time.monotonic_ns()-start)
                with self._lock:
                    self._latencies.record(elapsed)
                    self._exceptions += int(failed)
                    if elapsed >= 250_000_000:
                        self._slow_count += 1
                        self._samples.append(dict(started_at_unix_ns=wall,
                                                  elapsed_ms=elapsed/1e6, failed=failed))
                        del self._samples[:-64]
        return timed

    def snapshot(self) -> dict:
        with self._lock:
            return dict(sample_count=self._latencies.count,
                        statistics=self._latencies.snapshot() if self._latencies.count else None,
                        slow_steps_observed=self._slow_count, exceptions=self._exceptions,
                        recent_slow_steps=[dict(s) for s in self._samples], sample_limit=64,
                        slow_threshold_ms=250,
                        scope="BatchGenerator.next host wall time; not GPU kernel or request latency")
