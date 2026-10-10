"""Opt-in bounded host wall-time counters; nested spans are inclusive."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext

PHASES = frozenset({
    "file_stat", "checksum_read", "checksum_hash", "mlx_load", "weight_eval",
    "expert_build", "expert_eval", "router_eval", "router_to_host",
    "output_assembly", "output_eval", "acquire", "release",
})


class ExpertTimings:
    def __init__(self):
        self._values = {phase: [0, 0] for phase in PHASES}
        self._lock = threading.Lock()

    @contextmanager
    def span(self, phase):
        if phase not in PHASES:
            raise ValueError("unknown expert timing phase")
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = max(0, time.perf_counter_ns() - started)
            with self._lock:
                values = self._values[phase]
                values[0] += 1
                values[1] += elapsed

    def snapshot(self):
        with self._lock:
            return {phase: dict(count=value[0], nanoseconds=value[1])
                    for phase, value in sorted(self._values.items())}


def measure(timings, phase):
    return timings.span(phase) if timings is not None else nullcontext()
