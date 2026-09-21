"""Bounded non-content resource and modality telemetry."""
from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .types import ThermalState

MAX_TELEMETRY_SAMPLES = 1024


class ModalityKind(str, Enum):
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class RuntimeResourceSample:
    timestamp_nanoseconds: int
    cpu_utilization_percent: float
    gpu_utilization_percent: float | None
    unified_memory_bandwidth_bytes_per_second: int | None
    power_watts: float | None
    thermal_state: ThermalState

    def __post_init__(self) -> None:
        if (type(self.timestamp_nanoseconds) is not int or self.timestamp_nanoseconds < 0
                or not _percent(self.cpu_utilization_percent)
                or (self.gpu_utilization_percent is not None
                    and not _percent(self.gpu_utilization_percent))
                or (self.unified_memory_bandwidth_bytes_per_second is not None
                    and (type(self.unified_memory_bandwidth_bytes_per_second) is not int
                         or self.unified_memory_bandwidth_bytes_per_second < 0))
                or (self.power_watts is not None
                    and (not math.isfinite(self.power_watts) or self.power_watts < 0))
                or not isinstance(self.thermal_state, ThermalState)):
            raise ValueError("invalid runtime resource sample")


@dataclass(frozen=True, slots=True)
class ModalitySample:
    modality: ModalityKind
    operation: str
    latency_nanoseconds: int
    input_units: int
    output_units: int
    peak_memory_bytes: int
    succeeded: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.modality, ModalityKind)
                or not self.operation or len(self.operation) > 128
                or type(self.latency_nanoseconds) is not int
                or not 1 <= self.latency_nanoseconds <= 24 * 3600 * 1_000_000_000
                or any(type(value) is not int or value < 0 for value in (
                    self.input_units, self.output_units, self.peak_memory_bytes
                )) or type(self.succeeded) is not bool):
            raise ValueError("invalid modality telemetry sample")


class WorkloadTelemetry:
    def __init__(self, maximum_samples: int = 256) -> None:
        if not 1 <= maximum_samples <= MAX_TELEMETRY_SAMPLES:
            raise ValueError("invalid telemetry sample bound")
        self._resources: deque[RuntimeResourceSample] = deque(maxlen=maximum_samples)
        self._modalities: deque[ModalitySample] = deque(maxlen=maximum_samples)
        self._resource_dropped = self._modality_dropped = 0
        self._lock = threading.Lock()

    def record_resource(self, sample: RuntimeResourceSample) -> None:
        if not isinstance(sample, RuntimeResourceSample):
            raise ValueError("invalid runtime resource sample")
        with self._lock:
            if len(self._resources) == self._resources.maxlen:
                self._resource_dropped += 1
            self._resources.append(sample)

    def record_modality(self, sample: ModalitySample) -> None:
        if not isinstance(sample, ModalitySample):
            raise ValueError("invalid modality telemetry sample")
        with self._lock:
            if len(self._modalities) == self._modalities.maxlen:
                self._modality_dropped += 1
            self._modalities.append(sample)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            resources = tuple(self._resources)
            modalities = tuple(self._modalities)
            resource_dropped = self._resource_dropped
            modality_dropped = self._modality_dropped
        return {
            "schema_version": 1,
            "resource": {
                "samples": len(resources),
                "dropped": resource_dropped,
                "cpu_utilization_percent": _summary(
                    tuple(item.cpu_utilization_percent for item in resources)
                ),
                "gpu_utilization_percent": _summary(tuple(
                    item.gpu_utilization_percent for item in resources
                    if item.gpu_utilization_percent is not None
                )),
                "bandwidth_bytes_per_second": _summary(tuple(
                    item.unified_memory_bandwidth_bytes_per_second for item in resources
                    if item.unified_memory_bandwidth_bytes_per_second is not None
                )),
                "power_watts": _summary(tuple(
                    item.power_watts for item in resources if item.power_watts is not None
                )),
                "thermal_counts": {
                    state.value: sum(item.thermal_state is state for item in resources)
                    for state in ThermalState
                },
            },
            "modalities": {
                modality.value: _modality_summary(modalities, modality)
                for modality in ModalityKind
            },
            "modality_dropped": modality_dropped,
        }


def _modality_summary(
    samples: tuple[ModalitySample, ...], modality: ModalityKind
) -> dict[str, object]:
    selected = tuple(item for item in samples if item.modality is modality)
    return {
        "samples": len(selected),
        "failures": sum(not item.succeeded for item in selected),
        "latency_nanoseconds": _summary(tuple(item.latency_nanoseconds for item in selected)),
        "input_units": sum(item.input_units for item in selected),
        "output_units": sum(item.output_units for item in selected),
        "maximum_peak_memory_bytes": max(
            (item.peak_memory_bytes for item in selected), default=0
        ),
    }


def _summary(values: tuple[int | float, ...]) -> dict[str, int | float | None]:
    if not values:
        return {"median": None, "p95": None, "maximum": None}
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "median": statistics.median(ordered),
        "p95": p95,
        "maximum": ordered[-1],
    }


def _percent(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and 0 <= value <= 100
