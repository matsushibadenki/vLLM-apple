"""Bounded multi-model command submission stress qualification."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .kernel_probe import KernelMeasurement


@dataclass(frozen=True, slots=True)
class MultiModelStressReport:
    model_count: int
    iterations_per_model: int
    maximum_concurrency: int
    peak_concurrency: int
    command_count: int
    failures: int
    digest_mismatches: int
    elapsed_nanoseconds: int
    passed: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "model_count": self.model_count,
            "iterations_per_model": self.iterations_per_model,
            "maximum_concurrency": self.maximum_concurrency,
            "peak_concurrency": self.peak_concurrency,
            "command_count": self.command_count,
            "failures": self.failures,
            "digest_mismatches": self.digest_mismatches,
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "passed": self.passed,
        }


def run_multi_model_command_stress(
    model_ids: tuple[str, ...],
    measure: Callable[[str, int], KernelMeasurement],
    *,
    iterations_per_model: int = 3,
    maximum_concurrency: int = 2,
) -> MultiModelStressReport:
    """Run stable per-model command streams under bounded cross-model concurrency."""
    if (
        not isinstance(model_ids, tuple)
        or not 2 <= len(model_ids) <= 16
        or len(set(model_ids)) != len(model_ids)
        or any(not model_id or len(model_id) > 128 for model_id in model_ids)
        or not callable(measure)
        or type(iterations_per_model) is not int
        or not 1 <= iterations_per_model <= 1000
        or type(maximum_concurrency) is not int
        or not 1 <= maximum_concurrency <= 8
    ):
        raise ValueError("invalid multi-model command stress configuration")
    admission = threading.BoundedSemaphore(maximum_concurrency)
    lock = threading.Lock()
    active = 0
    peak = 0
    failures = 0
    mismatches = 0
    completed = 0
    expected: dict[str, str] = {}
    started = time.monotonic_ns()

    def worker(model_id: str) -> None:
        nonlocal active, peak, failures, mismatches, completed
        for iteration in range(iterations_per_model):
            with admission:
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    measurement = measure(model_id, iteration)
                    if not isinstance(measurement, KernelMeasurement):
                        raise TypeError("stress measurement is invalid")
                    with lock:
                        prior = expected.setdefault(model_id, measurement.output_digest)
                        if prior != measurement.output_digest:
                            mismatches += 1
                        completed += 1
                except BaseException:
                    with lock:
                        failures += 1
                finally:
                    with lock:
                        active -= 1

    threads = tuple(
        threading.Thread(target=worker, args=(model_id,), daemon=True)
        for model_id in model_ids
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = max(1, time.monotonic_ns() - started)
    commands = len(model_ids) * iterations_per_model
    passed = (
        completed == commands
        and failures == 0
        and mismatches == 0
        and active == 0
        and 1 <= peak <= maximum_concurrency
    )
    return MultiModelStressReport(
        len(model_ids), iterations_per_model, maximum_concurrency, peak,
        commands, failures, mismatches, elapsed, passed,
    )
