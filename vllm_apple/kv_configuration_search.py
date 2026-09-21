"""Bounded correctness- and memory-gated KV configuration search."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from enum import Enum

MAX_KV_SEARCH_CANDIDATES = 512


class KVSearchObjective(str, Enum):
    BALANCED = "balanced"
    CAPACITY = "capacity"
    LATENCY = "latency"


@dataclass(frozen=True, order=True, slots=True)
class KVConfiguration:
    precision: str
    context_tokens: int
    batch_size: int
    block_tokens: int

    def __post_init__(self) -> None:
        if (self.precision not in {"fp32", "fp16", "bf16", "int8"}
                or type(self.context_tokens) is not int
                or not 128 <= self.context_tokens <= 16_777_216
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= 256
                or type(self.block_tokens) is not int
                or self.block_tokens not in {8, 16, 32, 64, 128}
                or self.context_tokens % self.block_tokens):
            raise ValueError("invalid KV configuration")


@dataclass(frozen=True, slots=True)
class KVConfigurationMeasurement:
    configuration: KVConfiguration
    bytes_per_token: int
    peak_memory_bytes: int
    maximum_absolute_error: float
    rmse: float
    cosine_similarity: float
    decode_nanoseconds: tuple[int, ...]
    output_digest: str

    def __post_init__(self) -> None:
        if (not isinstance(self.configuration, KVConfiguration)
                or type(self.bytes_per_token) is not int or self.bytes_per_token <= 0
                or type(self.peak_memory_bytes) is not int or self.peak_memory_bytes <= 0
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in (
                           self.maximum_absolute_error, self.rmse, self.cosine_similarity
                       ))
                or self.maximum_absolute_error < 0 or self.rmse < 0
                or not -1 <= self.cosine_similarity <= 1
                or not 3 <= len(self.decode_nanoseconds) <= 64
                or any(type(value) is not int or value <= 0 for value in self.decode_nanoseconds)
                or len(self.output_digest) != 64):
            raise ValueError("invalid KV configuration measurement")
        required = (
            self.bytes_per_token * self.configuration.context_tokens
            * self.configuration.batch_size
        )
        if self.peak_memory_bytes < required:
            raise ValueError("KV peak memory is below its declared state size")


@dataclass(frozen=True, slots=True)
class KVConfigurationSearchReport:
    objective: KVSearchObjective
    selected: KVConfiguration
    feasible_candidates: int
    rejected_candidates: int
    selected_median_decode_nanoseconds: int
    selected_peak_memory_bytes: int
    report_id: str


def search_kv_configuration(
    measurements: tuple[KVConfigurationMeasurement, ...],
    *,
    objective: KVSearchObjective,
    maximum_peak_memory_bytes: int,
    maximum_absolute_error: float,
    maximum_rmse: float,
    minimum_cosine_similarity: float,
    baseline_output_digest: str,
) -> KVConfigurationSearchReport:
    if (not isinstance(objective, KVSearchObjective)
            or not 1 <= len(measurements) <= MAX_KV_SEARCH_CANDIDATES
            or any(not isinstance(item, KVConfigurationMeasurement) for item in measurements)
            or len({item.configuration for item in measurements}) != len(measurements)
            or type(maximum_peak_memory_bytes) is not int or maximum_peak_memory_bytes <= 0
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in (
                       maximum_absolute_error, maximum_rmse, minimum_cosine_similarity
                   ))
            or maximum_absolute_error < 0 or maximum_rmse < 0
            or not -1 <= minimum_cosine_similarity <= 1
            or len(baseline_output_digest) != 64):
        raise ValueError("invalid KV configuration search")
    feasible = []
    for item in measurements:
        if (item.peak_memory_bytes > maximum_peak_memory_bytes
                or item.maximum_absolute_error > maximum_absolute_error
                or item.rmse > maximum_rmse
                or item.cosine_similarity < minimum_cosine_similarity
                or item.output_digest != baseline_output_digest):
            continue
        median = int(statistics.median(item.decode_nanoseconds))
        capacity = item.configuration.context_tokens * item.configuration.batch_size
        if objective is KVSearchObjective.CAPACITY:
            key = (-capacity, median, item.peak_memory_bytes, item.configuration)
        elif objective is KVSearchObjective.LATENCY:
            key = (median, -capacity, item.peak_memory_bytes, item.configuration)
        else:
            normalized_latency = median / min(item.decode_nanoseconds)
            utilization = item.peak_memory_bytes / maximum_peak_memory_bytes
            balanced_score = normalized_latency * utilization / max(1, capacity)
            key = (balanced_score, median, -capacity, item.configuration)
        feasible.append((key, item, median))
    if not feasible:
        raise ValueError("no KV configuration passed quality and memory gates")
    feasible.sort(key=lambda value: value[0])
    _, selected, median = feasible[0]
    identity = {
        "objective": objective.value,
        "selected": {
            "precision": selected.configuration.precision,
            "context_tokens": selected.configuration.context_tokens,
            "batch_size": selected.configuration.batch_size,
            "block_tokens": selected.configuration.block_tokens,
        },
        "feasible": len(feasible),
        "rejected": len(measurements) - len(feasible),
        "median": median,
        "peak": selected.peak_memory_bytes,
    }
    return KVConfigurationSearchReport(
        objective, selected.configuration, len(feasible), len(measurements) - len(feasible),
        median, selected.peak_memory_bytes,
        hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
