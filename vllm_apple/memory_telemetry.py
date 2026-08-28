from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from .types import MemoryPressure


@dataclass(frozen=True, slots=True)
class MemoryTelemetrySnapshot:
    unified_total_bytes: int
    unified_available_bytes: int
    control_resident_bytes: int
    control_resident_peak_bytes: int
    backend_resident_bytes: int | None
    backend_resident_peak_bytes: int | None
    allocator_current_bytes: int | None
    allocator_peak_bytes: int | None
    allocator_resident_peak_delta_bytes: int | None
    kv_used_bytes: int | None
    kv_capacity_bytes: int | None
    kv_usage_ratio: float | None
    iogpu_bytes: int | None
    iogpu_peak_bytes: int | None
    pressure: str
    pressure_notifications: int
    os_source: str
    allocator_source: str | None
    kv_source: str | None
    iogpu_source: str | None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


class UnifiedMemoryTelemetry:
    """Thread-safe two-layer ledger; it samples nothing and retains no history."""

    def __init__(
        self,
        total_bytes: int,
        available_bytes: int,
        os_source: str,
        initial_pressure: MemoryPressure = MemoryPressure.UNKNOWN,
    ) -> None:
        if total_bytes <= 0 or not 0 <= available_bytes <= total_bytes:
            raise ValueError("invalid unified memory capacity")
        self._lock = threading.Lock()
        self._total = total_bytes
        self._available = available_bytes
        self._os_source = os_source
        self._control_resident = 0
        self._control_peak = 0
        self._backend_resident: int | None = None
        self._backend_peak: int | None = None
        self._allocator_current: int | None = None
        self._allocator_peak: int | None = None
        self._allocator_source: str | None = None
        self._kv_used: int | None = None
        self._kv_capacity: int | None = None
        self._kv_ratio: float | None = None
        self._kv_source: str | None = None
        self._iogpu: int | None = None
        self._iogpu_peak: int | None = None
        self._iogpu_source: str | None = None
        if not isinstance(initial_pressure, MemoryPressure):
            raise ValueError("invalid initial memory pressure")
        self._pressure = initial_pressure
        self._pressure_notifications = 0

    @staticmethod
    def _validate_bytes(*values: int | None) -> None:
        if any(value is not None and value < 0 for value in values):
            raise ValueError("memory telemetry values must not be negative")

    def update_os(
        self,
        *,
        available_bytes: int,
        control_resident_bytes: int,
        backend_resident_bytes: int | None = None,
        source: str | None = None,
    ) -> None:
        self._validate_bytes(available_bytes, control_resident_bytes, backend_resident_bytes)
        if available_bytes > self._total:
            raise ValueError("available memory exceeds unified memory")
        with self._lock:
            self._available = available_bytes
            self._control_resident = control_resident_bytes
            self._control_peak = max(self._control_peak, control_resident_bytes)
            self._backend_resident = backend_resident_bytes
            if backend_resident_bytes is not None:
                self._backend_peak = max(self._backend_peak or 0, backend_resident_bytes)
            if source is not None:
                self._os_source = source

    def update_allocator(
        self, current_bytes: int, *, source: str, peak_bytes: int | None = None
    ) -> None:
        self._validate_bytes(current_bytes, peak_bytes)
        if not source:
            raise ValueError("allocator source must not be empty")
        with self._lock:
            self._allocator_current = current_bytes
            self._allocator_peak = max(self._allocator_peak or 0, current_bytes, peak_bytes or 0)
            self._allocator_source = source

    def update_backend_resident(self, resident_bytes: int, *, source: str | None = None) -> None:
        self._validate_bytes(resident_bytes)
        with self._lock:
            self._backend_resident = resident_bytes
            self._backend_peak = max(self._backend_peak or 0, resident_bytes)
            if source is not None:
                self._os_source = source

    def update_kv_cache(self, used_bytes: int, capacity_bytes: int, *, source: str) -> None:
        self._validate_bytes(used_bytes, capacity_bytes)
        if used_bytes > capacity_bytes or not source:
            raise ValueError("invalid KV cache telemetry")
        with self._lock:
            self._kv_used = used_bytes
            self._kv_capacity = capacity_bytes
            self._kv_ratio = None
            self._kv_source = source

    def update_kv_ratio(self, usage_ratio: float, *, source: str) -> None:
        if not 0 <= usage_ratio <= 1 or not source:
            raise ValueError("invalid KV cache ratio telemetry")
        with self._lock:
            self._kv_used = None
            self._kv_capacity = None
            self._kv_ratio = usage_ratio
            self._kv_source = source

    def update_iogpu(self, current_bytes: int, *, source: str) -> None:
        self._validate_bytes(current_bytes)
        if not source:
            raise ValueError("IOGPU source must not be empty")
        with self._lock:
            self._iogpu = current_bytes
            self._iogpu_peak = max(self._iogpu_peak or 0, current_bytes)
            self._iogpu_source = source

    def update_pressure(self, pressure: MemoryPressure) -> None:
        if not isinstance(pressure, MemoryPressure):
            raise ValueError("invalid memory pressure")
        with self._lock:
            if pressure != self._pressure:
                self._pressure_notifications += 1
            self._pressure = pressure

    def snapshot(self) -> MemoryTelemetrySnapshot:
        with self._lock:
            resident_peak = self._backend_peak
            delta = None
            if self._allocator_peak is not None and resident_peak is not None:
                delta = resident_peak - self._allocator_peak
            ratio = self._kv_ratio
            if self._kv_used is not None and self._kv_capacity is not None:
                ratio = self._kv_used / self._kv_capacity if self._kv_capacity else 0.0
            return MemoryTelemetrySnapshot(
                unified_total_bytes=self._total,
                unified_available_bytes=self._available,
                control_resident_bytes=self._control_resident,
                control_resident_peak_bytes=self._control_peak,
                backend_resident_bytes=self._backend_resident,
                backend_resident_peak_bytes=self._backend_peak,
                allocator_current_bytes=self._allocator_current,
                allocator_peak_bytes=self._allocator_peak,
                allocator_resident_peak_delta_bytes=delta,
                kv_used_bytes=self._kv_used,
                kv_capacity_bytes=self._kv_capacity,
                kv_usage_ratio=ratio,
                iogpu_bytes=self._iogpu,
                iogpu_peak_bytes=self._iogpu_peak,
                pressure=self._pressure.value,
                pressure_notifications=self._pressure_notifications,
                os_source=self._os_source,
                allocator_source=self._allocator_source,
                kv_source=self._kv_source,
                iogpu_source=self._iogpu_source,
            )
