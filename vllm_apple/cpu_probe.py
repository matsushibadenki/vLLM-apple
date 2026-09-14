"""Bounded standard-library CPU correctness and latency probes."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .execution import ExecutionBackend
from .kernel_probe import KernelMeasurement, KernelProbeConfig, KernelProbeResult, run_kernel_probe


def _measurement(operation) -> KernelMeasurement:
    started = time.perf_counter_ns()
    output = operation()
    elapsed = max(1, time.perf_counter_ns() - started)
    digest = hashlib.sha256(
        json.dumps(output, separators=(",", ":")).encode()
    ).hexdigest()
    return KernelMeasurement(digest, elapsed)


@dataclass(frozen=True, slots=True)
class NativeCPUProbeAdapter:
    """Measures small deterministic kernels without optional dependencies."""

    def probe_suite(
        self,
        *,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        samples: int = 3,
        maximum_slowdown_ratio: float = 100,
    ) -> tuple[KernelProbeResult, ...]:
        operations = {
            "vector_add": (
                lambda: [index + index for index in range(256)],
                lambda: [left + right for left, right in zip(range(256), range(256))],
            ),
            "matmul": (self._reference_matmul, self._candidate_matmul),
            "kv_copy": (
                lambda: list(range(512))[64:192],
                lambda: [value for value in range(64, 192)],
            ),
        }
        return tuple(
            run_kernel_probe(
                KernelProbeConfig(
                    hardware_fingerprint,
                    environment_fingerprint,
                    ExecutionBackend.CPU,
                    operator,
                    samples=samples,
                    maximum_slowdown_ratio=maximum_slowdown_ratio,
                ),
                lambda reference=reference: _measurement(reference),
                lambda candidate=candidate: _measurement(candidate),
            )
            for operator, (reference, candidate) in operations.items()
        )

    @staticmethod
    def _reference_matmul() -> list[list[int]]:
        size = 8
        matrix = [[(row * size + column) % 7 for column in range(size)] for row in range(size)]
        output = [[0] * size for _ in range(size)]
        for row in range(size):
            for column in range(size):
                for inner in range(size):
                    output[row][column] += matrix[row][inner] * matrix[inner][column]
        return output

    @staticmethod
    def _candidate_matmul() -> list[list[int]]:
        size = 8
        matrix = [[(row * size + column) % 7 for column in range(size)] for row in range(size)]
        columns = list(zip(*matrix))
        return [
            [sum(left * right for left, right in zip(row, column)) for column in columns]
            for row in matrix
        ]
