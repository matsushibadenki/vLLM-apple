from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .generative_evaluation import (
    MAX_DURATION_MS,
    MEMORY_PRESSURES,
    THERMAL_STATES,
    GenerativeSampleEvidence,
)
from .generative_qualification import GenerativeQualificationPlan


MAX_TELEMETRY_EVENTS = 4096
_MEMORY_SEVERITY = {"normal": 0, "warning": 1, "unknown": 2, "critical": 3}
_THERMAL_SEVERITY = {"nominal": 0, "fair": 1, "unknown": 2, "serious": 3, "critical": 4}


@dataclass(frozen=True, slots=True)
class GenerationTelemetryEvent:
    kind: str
    elapsed_ms: float
    process_rss_bytes: int
    memory_pressure: str
    thermal_state: str
    output_width: int | None = None
    output_height: int | None = None
    output_frames: int | None = None
    output_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"started", "progress", "first_output", "completed"}:
            raise ValueError("unsupported generation telemetry event kind")
        if not math.isfinite(self.elapsed_ms) or not 0 <= self.elapsed_ms <= MAX_DURATION_MS:
            raise ValueError("generation telemetry elapsed time is outside the supported range")
        if self.kind != "started" and self.elapsed_ms == 0:
            raise ValueError("non-start telemetry events require positive elapsed time")
        if not 0 < self.process_rss_bytes <= 16_384 * 1024**3:
            raise ValueError("generation telemetry RSS is outside the supported range")
        if self.memory_pressure not in MEMORY_PRESSURES:
            raise ValueError("unsupported memory pressure")
        if self.thermal_state not in THERMAL_STATES:
            raise ValueError("unsupported thermal state")
        output_values = (
            self.output_width,
            self.output_height,
            self.output_frames,
            self.output_sha256,
        )
        if self.kind == "completed":
            if any(value is None for value in output_values):
                raise ValueError("completed telemetry event requires output metadata")
            if (
                self.output_width <= 0
                or self.output_height <= 0
                or self.output_frames <= 0
                or len(self.output_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.output_sha256)
            ):
                raise ValueError("completed telemetry output metadata is invalid")
        elif any(value is not None for value in output_values):
            raise ValueError("output metadata is only allowed on completed telemetry events")


def collect_generative_sample(
    plan: GenerativeQualificationPlan,
    *,
    sample_index: int,
    events: Iterable[GenerationTelemetryEvent],
) -> GenerativeSampleEvidence:
    if not plan.eligible:
        raise ValueError("cannot collect a sample for an ineligible generation plan")
    event_count = 0
    previous_elapsed_ms = -1.0
    peak_rss_bytes = 0
    worst_memory_pressure = "normal"
    worst_thermal_state = "nominal"
    first_output_ms: float | None = None
    completed: GenerationTelemetryEvent | None = None

    iterator = iter(events)
    try:
        for event in iterator:
            event_count += 1
            if event_count > MAX_TELEMETRY_EVENTS:
                raise ValueError("generation telemetry event limit exceeded")
            if event.elapsed_ms < previous_elapsed_ms:
                raise ValueError("generation telemetry time must be monotonic")
            if event_count == 1 and event.kind != "started":
                raise ValueError("generation telemetry must start with a started event")
            if completed is not None:
                raise ValueError("generation telemetry contains events after completion")
            if event.kind == "started" and event_count != 1:
                raise ValueError("generation telemetry contains duplicate started event")
            if event.kind == "first_output":
                if first_output_ms is not None:
                    raise ValueError("generation telemetry contains duplicate first output")
                first_output_ms = event.elapsed_ms
            if event.kind == "completed":
                completed = event
            previous_elapsed_ms = event.elapsed_ms
            peak_rss_bytes = max(peak_rss_bytes, event.process_rss_bytes)
            if _MEMORY_SEVERITY[event.memory_pressure] > _MEMORY_SEVERITY[worst_memory_pressure]:
                worst_memory_pressure = event.memory_pressure
            if _THERMAL_SEVERITY[event.thermal_state] > _THERMAL_SEVERITY[worst_thermal_state]:
                worst_thermal_state = event.thermal_state
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    if completed is None:
        raise ValueError("generation telemetry is missing completion")
    if first_output_ms is None:
        first_output_ms = completed.elapsed_ms
    return GenerativeSampleEvidence(
        sample_index=sample_index,
        wall_time_ms=completed.elapsed_ms,
        first_output_ms=first_output_ms,
        peak_rss_bytes=peak_rss_bytes,
        memory_pressure=worst_memory_pressure,
        thermal_state=worst_thermal_state,
        output_width=completed.output_width,
        output_height=completed.output_height,
        output_frames=completed.output_frames,
        output_sha256=completed.output_sha256,
        stores_prompt=False,
        stores_output=False,
    )
