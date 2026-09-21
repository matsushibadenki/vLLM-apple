"""Bounded end-to-end performance profiles for heterogeneous workload phases."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from enum import Enum

MAX_WORKLOAD_SAMPLES = 256


class EndToEndPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    VISION_ENCODER = "vision_encoder"
    AUDIO_ENCODER = "audio_encoder"
    SAMPLING = "sampling"
    DRAFT = "draft"
    VERIFY = "verify"


@dataclass(frozen=True, slots=True)
class EndToEndPerformanceSample:
    latency_nanoseconds: int
    work_units: int
    peak_unified_memory_bytes: int
    output_sha256: str
    energy_joules: float | None = None

    def __post_init__(self) -> None:
        if (not 1 <= self.latency_nanoseconds <= 24 * 3600 * 1_000_000_000
                or not 1 <= self.work_units <= 1 << 40
                or not 0 <= self.peak_unified_memory_bytes <= 1 << 50
                or len(self.output_sha256) != 64
                or any(value not in "0123456789abcdef" for value in self.output_sha256)
                or (self.energy_joules is not None
                    and (not math.isfinite(self.energy_joules)
                         or self.energy_joules < 0))):
            raise ValueError("invalid end-to-end performance sample")


@dataclass(frozen=True, slots=True)
class EndToEndPerformanceProfile:
    profile_id: str
    hardware_fingerprint: str
    model_id: str
    backend: str
    phase: EndToEndPhase
    workload_identity: str
    sample_count: int
    median_latency_nanoseconds: int
    p95_latency_nanoseconds: int
    throughput_units_per_second: float
    peak_unified_memory_bytes: int
    median_energy_joules: float | None
    output_sha256: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["schema_version"] = 1
        value["phase"] = self.phase.value
        return value


@dataclass(frozen=True, slots=True)
class EndToEndPromotionDecision:
    promoted: bool
    reason: str
    latency_improvement_ratio: float


def build_end_to_end_performance_profile(
    *,
    hardware_fingerprint: str,
    model_id: str,
    backend: str,
    phase: EndToEndPhase,
    workload_identity: str,
    samples: tuple[EndToEndPerformanceSample, ...],
) -> EndToEndPerformanceProfile:
    if (not hardware_fingerprint or not model_id or not backend
            or not workload_identity or not isinstance(phase, EndToEndPhase)
            or not 1 <= len(samples) <= MAX_WORKLOAD_SAMPLES):
        raise ValueError("invalid end-to-end performance profile identity")
    digests = {sample.output_sha256 for sample in samples}
    if len(digests) != 1:
        raise ValueError("end-to-end performance output is not deterministic")
    latencies = sorted(sample.latency_nanoseconds for sample in samples)
    median_latency = int(statistics.median(latencies))
    p95_latency = latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
    total_units = sum(sample.work_units for sample in samples)
    total_seconds = sum(sample.latency_nanoseconds for sample in samples) / 1e9
    energies = [sample.energy_joules for sample in samples if sample.energy_joules is not None]
    median_energy = (
        float(statistics.median(energies)) if len(energies) == len(samples) else None
    )
    identity = {
        "hardware_fingerprint": hardware_fingerprint,
        "model_id": model_id,
        "backend": backend,
        "phase": phase.value,
        "workload_identity": workload_identity,
        "samples": [asdict(sample) for sample in samples],
    }
    profile_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EndToEndPerformanceProfile(
        profile_id,
        hardware_fingerprint,
        model_id,
        backend,
        phase,
        workload_identity,
        len(samples),
        median_latency,
        p95_latency,
        total_units / total_seconds,
        max(sample.peak_unified_memory_bytes for sample in samples),
        median_energy,
        next(iter(digests)),
    )


def evaluate_end_to_end_promotion(
    baseline: EndToEndPerformanceProfile,
    candidate: EndToEndPerformanceProfile,
    *,
    minimum_improvement_ratio: float = 0.05,
) -> EndToEndPromotionDecision:
    if not 0 <= minimum_improvement_ratio < 1:
        raise ValueError("invalid end-to-end promotion threshold")
    def identity(value: EndToEndPerformanceProfile) -> tuple[object, ...]:
        return (
            value.hardware_fingerprint, value.model_id, value.phase,
            value.workload_identity,
        )
    if identity(baseline) != identity(candidate):
        return EndToEndPromotionDecision(False, "identity_mismatch", 0.0)
    if min(baseline.sample_count, candidate.sample_count) < 3:
        return EndToEndPromotionDecision(False, "insufficient_samples", 0.0)
    if baseline.output_sha256 != candidate.output_sha256:
        return EndToEndPromotionDecision(False, "output_mismatch", 0.0)
    improvement = 1 - (
        candidate.median_latency_nanoseconds / baseline.median_latency_nanoseconds
    )
    if candidate.peak_unified_memory_bytes > baseline.peak_unified_memory_bytes:
        return EndToEndPromotionDecision(False, "peak_memory_regression", improvement)
    if (baseline.median_energy_joules is not None
            and candidate.median_energy_joules is not None
            and candidate.median_energy_joules > baseline.median_energy_joules):
        return EndToEndPromotionDecision(False, "energy_regression", improvement)
    if improvement < minimum_improvement_ratio:
        return EndToEndPromotionDecision(False, "latency_improvement_too_small", improvement)
    return EndToEndPromotionDecision(True, "promoted", improvement)
