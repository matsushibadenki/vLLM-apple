from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable

from .memory_telemetry import MemoryTelemetrySnapshot
from .scheduler import ScheduleRequest
from .types import MemoryPressure, Priority


class MemoryPressureAdmissionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryAdmissionSnapshot:
    effective_pressure: str
    admitted: int
    rejected: int
    last_rejection_reason: str | None
    recovery_stage: int | None
    recovery_max_batch_size: int | None
    recovery_max_context_tokens: int | None
    recovery_memory_fraction: float | None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


class MemoryPressureAdmissionGate:
    """Fail-closed admission without double-counting Unified Memory views."""

    _STAGES = (
        (1, 4_096, 0.125),
        (2, 8_192, 0.25),
        (4, 32_768, 0.5),
    )

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        recovery_boundaries: tuple[float, float, float] = (5.0, 15.0, 30.0),
    ) -> None:
        if any(value <= 0 for value in recovery_boundaries) or tuple(
            sorted(recovery_boundaries)
        ) != recovery_boundaries:
            raise ValueError("invalid recovery boundaries")
        self._lock = threading.Lock()
        self._clock = clock
        self._recovery_boundaries = recovery_boundaries
        self._effective_pressure = MemoryPressure.UNKNOWN
        self._admitted = 0
        self._rejected = 0
        self._last_rejection_reason: str | None = None
        self._recovery_started: float | None = None

    def _recovery_stage_locked(self) -> int | None:
        if self._recovery_started is None:
            return None
        elapsed = max(0.0, self._clock() - self._recovery_started)
        for stage, boundary in enumerate(self._recovery_boundaries):
            if elapsed < boundary:
                return stage
        self._recovery_started = None
        return None

    @staticmethod
    def effective_pressure(memory: MemoryTelemetrySnapshot) -> MemoryPressure:
        native = MemoryPressure(memory.pressure)
        available_ratio = memory.unified_available_bytes / memory.unified_total_bytes
        observed = max(memory.backend_resident_bytes or 0, memory.iogpu_bytes or 0)
        observed_ratio = observed / memory.unified_total_bytes
        derived = MemoryPressure.NORMAL
        if available_ratio < 0.08 or observed_ratio >= 0.92:
            derived = MemoryPressure.CRITICAL
        elif available_ratio < 0.18 or observed_ratio >= 0.82:
            derived = MemoryPressure.WARNING
        ranks = {
            MemoryPressure.UNKNOWN: -1,
            MemoryPressure.NORMAL: 0,
            MemoryPressure.WARNING: 1,
            MemoryPressure.CRITICAL: 2,
        }
        return native if ranks[native] >= ranks[derived] else derived

    def admit(self, request: ScheduleRequest, memory: MemoryTelemetrySnapshot) -> None:
        pressure = self.effective_pressure(memory)
        reason = None
        if pressure == MemoryPressure.CRITICAL and request.priority not in {
            Priority.REALTIME,
            Priority.INTERACTIVE,
        }:
            reason = "critical_memory_pressure"
        elif pressure == MemoryPressure.WARNING and request.priority == Priority.BACKGROUND:
            reason = "warning_memory_pressure_background"
        elif request.estimated_memory_bytes > memory.unified_available_bytes:
            reason = "insufficient_unified_memory"
        with self._lock:
            self._effective_pressure = pressure
            stage = self._recovery_stage_locked() if pressure == MemoryPressure.NORMAL else None
            if stage is not None and reason is None:
                max_batch, max_context, memory_fraction = self._STAGES[stage]
                if request.batch_size > max_batch:
                    reason = "recovery_batch_limited"
                elif request.estimated_context_tokens > max_context:
                    reason = "recovery_context_limited"
                elif request.estimated_memory_bytes > int(
                    memory.unified_available_bytes * memory_fraction
                ):
                    reason = "recovery_memory_limited"
            if reason is None:
                self._admitted += 1
            else:
                self._rejected += 1
                self._last_rejection_reason = reason
        if reason is not None:
            raise MemoryPressureAdmissionError(reason)

    def refresh(self, memory: MemoryTelemetrySnapshot) -> None:
        pressure = self.effective_pressure(memory)
        with self._lock:
            previous = self._effective_pressure
            self._effective_pressure = pressure
            if pressure == MemoryPressure.NORMAL and previous in {
                MemoryPressure.WARNING,
                MemoryPressure.CRITICAL,
            }:
                self._recovery_started = self._clock()
            elif pressure in {MemoryPressure.WARNING, MemoryPressure.CRITICAL}:
                self._recovery_started = None
            if pressure == MemoryPressure.NORMAL:
                self._last_rejection_reason = None

    def snapshot(self) -> MemoryAdmissionSnapshot:
        with self._lock:
            stage = self._recovery_stage_locked()
            limits = self._STAGES[stage] if stage is not None else (None, None, None)
            return MemoryAdmissionSnapshot(
                self._effective_pressure.value,
                self._admitted,
                self._rejected,
                self._last_rejection_reason,
                stage,
                limits[0],
                limits[1],
                limits[2],
            )
