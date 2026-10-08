"""Bounded host-wall timing of a backend scheduler step, without tensor inspection."""
from __future__ import annotations

import threading
import time
from functools import wraps

from .phase_profile import _BoundedLatency


class StepDiagnostics:
    def __init__(self, *, sample_limit: int = 64,
                 scope: str = "BatchGenerator.next host wall time; not GPU kernel or request latency") -> None:
        if type(sample_limit) is not int or not 1 <= sample_limit <= 64:
            raise ValueError('diagnostic sample limit must be between 1 and 64')
        self._sample_limit = sample_limit
        self._scope = scope
        self._lock = threading.Lock()
        self._latencies = _BoundedLatency()
        self._cpu_latencies = _BoundedLatency()
        self._slow_count = 0
        self._exceptions = 0
        self._samples: list[dict] = []

    def wrap(self, method, *, skip_empty_tokens: bool = False):
        @wraps(method)
        def timed(*args, **kwargs):
            if skip_empty_tokens:
                tokens = args[1] if len(args) > 1 else kwargs.get('tokens')
                # The reviewed prompt method returns immediately for an empty
                # Python list. Avoid clocks/locks and don't dilute real prefill.
                if type(tokens) is list and not tokens:
                    return method(*args, **kwargs)
            wall = time.time_ns()
            cpu_start = time.thread_time_ns()
            start = time.monotonic_ns()
            failed = False
            try:
                return method(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = max(0, time.monotonic_ns()-start)
                cpu_elapsed = max(0, time.thread_time_ns()-cpu_start)
                with self._lock:
                    self._latencies.record(elapsed)
                    self._cpu_latencies.record(cpu_elapsed)
                    self._exceptions += int(failed)
                    if elapsed >= 250_000_000:
                        self._slow_count += 1
                        self._samples.append(dict(started_at_unix_ns=wall,
                                                  elapsed_ms=elapsed/1e6, thread_cpu_ms=cpu_elapsed/1e6, failed=failed))
                        del self._samples[:-self._sample_limit]
        return timed

    def snapshot(self) -> dict:
        with self._lock:
            return dict(sample_count=self._latencies.count,
                        statistics=self._latencies.snapshot() if self._latencies.count else None,
                        thread_cpu_statistics=self._cpu_latencies.snapshot() if self._cpu_latencies.count else None,
                        thread_cpu_scope="calling thread CPU only; excludes GPU and other threads",
                        slow_steps_observed=self._slow_count, exceptions=self._exceptions,
                        recent_slow_steps=[dict(s) for s in self._samples], sample_limit=self._sample_limit,
                        slow_threshold_ms=250,
                        scope=self._scope)
