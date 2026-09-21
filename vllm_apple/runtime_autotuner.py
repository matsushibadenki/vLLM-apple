"""Evidence-only Mac-specific runtime configuration autotuner."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass

MAX_RUNTIME_TUNING_CANDIDATES = 256


@dataclass(frozen=True, order=True, slots=True)
class RuntimeTuningConfiguration:
    batch_size: int
    tile_size: int
    kv_block_tokens: int
    prefill_chunk_tokens: int
    kernel: str

    def __post_init__(self) -> None:
        if (type(self.batch_size) is not int or not 1 <= self.batch_size <= 256
                or type(self.tile_size) is not int or not 1 <= self.tile_size <= 4096
                or type(self.kv_block_tokens) is not int
                or self.kv_block_tokens not in {8, 16, 32, 64, 128}
                or type(self.prefill_chunk_tokens) is not int
                or not 16 <= self.prefill_chunk_tokens <= 65536
                or self.prefill_chunk_tokens % self.kv_block_tokens
                or not self.kernel or len(self.kernel) > 128):
            raise ValueError("invalid runtime tuning configuration")


@dataclass(frozen=True, slots=True)
class RuntimeTuningMeasurement:
    configuration: RuntimeTuningConfiguration
    prefill_nanoseconds: tuple[int, ...]
    decode_nanoseconds: tuple[int, ...]
    peak_memory_bytes: int
    output_digest: str

    def __post_init__(self) -> None:
        if (not isinstance(self.configuration, RuntimeTuningConfiguration)
                or not 3 <= len(self.prefill_nanoseconds) <= 64
                or not 3 <= len(self.decode_nanoseconds) <= 64
                or any(type(value) is not int or value <= 0
                       for value in (*self.prefill_nanoseconds, *self.decode_nanoseconds))
                or type(self.peak_memory_bytes) is not int or self.peak_memory_bytes <= 0
                or len(self.output_digest) != 64
                or any(character not in "0123456789abcdef" for character in self.output_digest)):
            raise ValueError("invalid runtime tuning measurement")


@dataclass(frozen=True, slots=True)
class RuntimeTuningReport:
    hardware_fingerprint: str
    phase_profile_id: str
    winner: RuntimeTuningConfiguration
    winner_score_nanoseconds: float
    qualified_candidates: int
    rejected_candidates: int
    report_id: str


def tune_runtime_configuration(
    hardware_fingerprint: str,
    phase_profile_id: str,
    measurements: tuple[RuntimeTuningMeasurement, ...],
    *,
    baseline_output_digest: str,
    maximum_peak_memory_bytes: int,
    prefill_weight: float = 1.0,
    decode_weight: float = 1.0,
) -> RuntimeTuningReport:
    if (len(hardware_fingerprint) != 24 or len(phase_profile_id) != 64
            or len(baseline_output_digest) != 64
            or not 1 <= len(measurements) <= MAX_RUNTIME_TUNING_CANDIDATES
            or any(not isinstance(item, RuntimeTuningMeasurement) for item in measurements)
            or len({item.configuration for item in measurements}) != len(measurements)
            or type(maximum_peak_memory_bytes) is not int or maximum_peak_memory_bytes <= 0
            or any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
                   or not math.isfinite(weight) or weight <= 0
                   for weight in (prefill_weight, decode_weight))):
        raise ValueError("invalid runtime tuning request")
    qualified = []
    for item in measurements:
        if (item.output_digest != baseline_output_digest
                or item.peak_memory_bytes > maximum_peak_memory_bytes):
            continue
        score = (
            statistics.median(item.prefill_nanoseconds) * prefill_weight
            + statistics.median(item.decode_nanoseconds) * decode_weight
        )
        qualified.append((score, item.peak_memory_bytes, item.configuration))
    if not qualified:
        raise ValueError("no runtime tuning candidate passed correctness and memory gates")
    qualified.sort(key=lambda item: (item[0], item[1], item[2]))
    score, _, winner = qualified[0]
    identity = {
        "hardware_fingerprint": hardware_fingerprint,
        "phase_profile_id": phase_profile_id,
        "winner": asdict(winner),
        "winner_score_nanoseconds": score,
        "qualified_candidates": len(qualified),
        "rejected_candidates": len(measurements) - len(qualified),
    }
    return RuntimeTuningReport(
        hardware_fingerprint, phase_profile_id, winner, score, len(qualified),
        len(measurements) - len(qualified),
        hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
