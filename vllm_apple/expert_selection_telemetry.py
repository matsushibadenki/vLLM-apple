"""Bounded, content-free telemetry for mixture-of-experts routing."""
from __future__ import annotations

import math
import statistics
import threading
from collections import Counter, deque
from dataclasses import dataclass

from .expert_residency import ExpertKey

MAX_EXPERT_SELECTION_SAMPLES = 65_536
MAX_SELECTED_EXPERTS = 64


@dataclass(frozen=True, slots=True)
class ExpertSelectionSample:
    layer: int
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]
    latency_nanoseconds: int
    cache_hit_experts: tuple[int, ...]

    def __post_init__(self) -> None:
        if (type(self.layer) is not int or not 0 <= self.layer < 1_000_000
                or not 1 <= len(self.selected_experts) <= MAX_SELECTED_EXPERTS
                or len(self.selected_experts) != len(self.routing_weights)
                or len(set(self.selected_experts)) != len(self.selected_experts)
                or any(type(expert) is not int or not 0 <= expert < 1_000_000
                       for expert in self.selected_experts)
                or any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
                       or not math.isfinite(weight) or weight < 0
                       for weight in self.routing_weights)
                or sum(self.routing_weights) <= 0
                or type(self.latency_nanoseconds) is not int
                or not 1 <= self.latency_nanoseconds <= 3600 * 1_000_000_000
                or len(set(self.cache_hit_experts)) != len(self.cache_hit_experts)
                or not set(self.cache_hit_experts).issubset(self.selected_experts)):
            raise ValueError("invalid expert selection sample")


class ExpertSelectionTelemetry:
    def __init__(self, maximum_samples: int = 4096) -> None:
        if not 1 <= maximum_samples <= MAX_EXPERT_SELECTION_SAMPLES:
            raise ValueError("invalid expert selection telemetry bound")
        self._samples: deque[ExpertSelectionSample] = deque(maxlen=maximum_samples)
        self._dropped = 0
        self._lock = threading.Lock()

    def record(self, sample: ExpertSelectionSample) -> None:
        if not isinstance(sample, ExpertSelectionSample):
            raise ValueError("invalid expert selection sample")
        with self._lock:
            if len(self._samples) == self._samples.maxlen:
                self._dropped += 1
            self._samples.append(sample)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            samples = tuple(self._samples)
            dropped = self._dropped
        selections: Counter[ExpertKey] = Counter()
        cache_hits: Counter[ExpertKey] = Counter()
        for sample in samples:
            for expert in sample.selected_experts:
                key = ExpertKey(sample.layer, expert)
                selections[key] += 1
                if expert in sample.cache_hit_experts:
                    cache_hits[key] += 1
        latencies = sorted(sample.latency_nanoseconds for sample in samples)
        per_expert = tuple(
            {
                "layer": key.layer,
                "expert": key.expert,
                "selections": selections[key],
                "cache_hits": cache_hits[key],
            }
            for key in sorted(selections)
        )
        return {
            "schema_version": 1,
            "samples": len(samples),
            "dropped": dropped,
            "selected_experts": sum(len(item.selected_experts) for item in samples),
            "cache_hits": sum(len(item.cache_hit_experts) for item in samples),
            "latency_nanoseconds": {
                "median": statistics.median(latencies) if latencies else None,
                "p95": latencies[max(0, math.ceil(len(latencies) * .95) - 1)]
                if latencies else None,
                "maximum": latencies[-1] if latencies else None,
            },
            "per_expert": per_expert,
        }
