from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Any


PHASE_PROFILE_SCHEMA_VERSION = 1
LATENCY_BUCKETS_NS = tuple(
    value * 1_000_000 for value in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000)
)


@dataclass(frozen=True, slots=True)
class PhaseMeasurement:
    started_ns: int
    first_token_ns: int
    completed_ns: int
    prompt_tokens: int
    output_tokens: int
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        if self.started_ns < 0 or not (
            self.started_ns <= self.first_token_ns <= self.completed_ns
        ):
            raise ValueError("measurement timestamps must be monotonic")
        if self.prompt_tokens < 0 or self.output_tokens <= 0:
            raise ValueError("measurement token counts are invalid")
        if self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must not be negative")

    @property
    def ttft_ns(self) -> int:
        return self.first_token_ns - self.started_ns

    @property
    def decode_ns(self) -> int:
        return self.completed_ns - self.first_token_ns

    @property
    def token_intervals(self) -> int:
        return max(0, self.output_tokens - 1)


class _BoundedLatency:
    def __init__(self) -> None:
        self.counts = [0] * (len(LATENCY_BUCKETS_NS) + 1)
        self.count = 0
        self.total_ns = 0
        self.maximum_ns = 0

    def record(self, duration_ns: int) -> None:
        self.count += 1
        self.total_ns += duration_ns
        self.maximum_ns = max(self.maximum_ns, duration_ns)
        index = len(LATENCY_BUCKETS_NS)
        for candidate, boundary in enumerate(LATENCY_BUCKETS_NS):
            if duration_ns <= boundary:
                index = candidate
                break
        self.counts[index] += 1

    def percentile_ms(self, percentile: float) -> float | str:
        if self.count == 0:
            return 0.0
        target = max(1, math.ceil(self.count * percentile))
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                if index < len(LATENCY_BUCKETS_NS):
                    return LATENCY_BUCKETS_NS[index] / 1_000_000
                return f">{LATENCY_BUCKETS_NS[-1] / 1_000_000:g}"
        return f">{LATENCY_BUCKETS_NS[-1] / 1_000_000:g}"

    def snapshot(self) -> dict[str, float | str]:
        return {
            "mean_ms": round(self.total_ns / self.count / 1_000_000, 3) if self.count else 0.0,
            "p50_upper_bound_ms": self.percentile_ms(0.5),
            "p95_upper_bound_ms": self.percentile_ms(0.95),
            "max_ms": round(self.maximum_ns / 1_000_000, 3),
        }


class ExecutionPhaseProfiler:
    """Constant-memory aggregate for phase-separated inference measurements."""

    def __init__(self, hardware_fingerprint: str, model_id: str, backend: str) -> None:
        if not hardware_fingerprint or not model_id or not backend:
            raise ValueError("profile identity fields must not be empty")
        self.hardware_fingerprint = hardware_fingerprint
        self.model_id = model_id
        self.backend = backend
        self._lock = threading.Lock()
        self._samples = 0
        self._prompt_tokens = 0
        self._output_tokens = 0
        self._decode_ns = 0
        self._decode_intervals = 0
        self._peak_memory_bytes = 0
        self._ttft = _BoundedLatency()
        self._tpot = _BoundedLatency()

    def record(self, measurement: PhaseMeasurement) -> None:
        with self._lock:
            self._samples += 1
            self._prompt_tokens += measurement.prompt_tokens
            self._output_tokens += measurement.output_tokens
            self._decode_ns += measurement.decode_ns
            self._decode_intervals += measurement.token_intervals
            self._peak_memory_bytes = max(
                self._peak_memory_bytes, measurement.peak_memory_bytes
            )
            self._ttft.record(measurement.ttft_ns)
            if measurement.token_intervals:
                self._tpot.record(measurement.decode_ns // measurement.token_intervals)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            identity = {
                "hardware_fingerprint": self.hardware_fingerprint,
                "model_id": self.model_id,
                "backend": self.backend,
                "samples": self._samples,
                "prompt_tokens": self._prompt_tokens,
                "output_tokens": self._output_tokens,
                "decode_ns": self._decode_ns,
                "decode_intervals": self._decode_intervals,
                "peak_memory_bytes": self._peak_memory_bytes,
                "ttft_counts": self._ttft.counts,
                "tpot_counts": self._tpot.counts,
            }
            profile_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:24]
            throughput = (
                self._decode_intervals / (self._decode_ns / 1_000_000_000)
                if self._decode_ns > 0
                else 0.0
            )
            return {
                "schema_version": PHASE_PROFILE_SCHEMA_VERSION,
                "profile_id": profile_id,
                "hardware_fingerprint": self.hardware_fingerprint,
                "model_id": self.model_id,
                "backend": self.backend,
                "sample_count": self._samples,
                "prefill": {
                    "prompt_tokens": self._prompt_tokens,
                    "ttft": self._ttft.snapshot(),
                },
                "decode": {
                    "output_tokens": self._output_tokens,
                    "token_intervals": self._decode_intervals,
                    "duration_ms": round(self._decode_ns / 1_000_000, 3),
                    "tpot": self._tpot.snapshot(),
                    "tokens_per_second": round(throughput, 3),
                },
                "peak_memory_bytes": self._peak_memory_bytes,
                "storage": {
                    "latency_bucket_count": len(self._ttft.counts) + len(self._tpot.counts),
                    "raw_sample_count": 0,
                },
            }
