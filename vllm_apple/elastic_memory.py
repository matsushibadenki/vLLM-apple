from __future__ import annotations

import threading
from dataclasses import dataclass

from .semantic_state import SemanticStateCoordinator
from .types import MemoryPressure


@dataclass(frozen=True, slots=True)
class ElasticMemoryDecision:
    pressure: MemoryPressure
    status: str
    target_entries: int
    target_bytes: int
    evicted_entries: int

    def __post_init__(self) -> None:
        if self.status not in {"applied", "deferred", "ignored"}:
            raise ValueError("elastic memory decision status is invalid")
        if self.target_entries < 0 or self.target_bytes < 0 or self.evicted_entries < 0:
            raise ValueError("elastic memory decision values are invalid")


class ElasticMemoryController:
    """Applies cache budget changes only at scheduler safe points."""

    def __init__(self, semantic_state: SemanticStateCoordinator) -> None:
        initial = semantic_state.snapshot()
        entries = int(initial["capacity_entries"])
        state_bytes = int(initial["capacity_bytes"])
        if entries <= 0 or state_bytes <= 0:
            raise ValueError("elastic memory controller requires an enabled cache")
        self._semantic_state = semantic_state
        self._normal_entries = entries
        self._normal_bytes = state_bytes
        self._current_pressure = MemoryPressure.NORMAL
        self._pending_pressure: MemoryPressure | None = None
        self._adjustments = 0
        self._deferred = 0
        self._last_evicted = 0
        self._lock = threading.Lock()

    def request(
        self,
        pressure: MemoryPressure,
        *,
        safe_to_apply: bool,
    ) -> ElasticMemoryDecision:
        if not isinstance(pressure, MemoryPressure):
            raise ValueError("memory pressure is invalid")
        target_entries, target_bytes = self._target(pressure)
        if pressure == MemoryPressure.UNKNOWN:
            return ElasticMemoryDecision(pressure, "ignored", target_entries, target_bytes, 0)
        if not safe_to_apply:
            with self._lock:
                self._pending_pressure = pressure
                self._deferred += 1
            return ElasticMemoryDecision(pressure, "deferred", target_entries, target_bytes, 0)
        evicted = self._semantic_state.resize(target_entries, target_bytes)
        with self._lock:
            self._current_pressure = pressure
            self._pending_pressure = None
            self._adjustments += 1
            self._last_evicted = evicted
        return ElasticMemoryDecision(pressure, "applied", target_entries, target_bytes, evicted)

    def apply_pending(self, *, safe_to_apply: bool) -> ElasticMemoryDecision | None:
        with self._lock:
            pending = self._pending_pressure
        if pending is None:
            return None
        if not safe_to_apply:
            target_entries, target_bytes = self._target(pending)
            return ElasticMemoryDecision(
                pending,
                "deferred",
                target_entries,
                target_bytes,
                0,
            )
        return self.request(pending, safe_to_apply=safe_to_apply)

    def snapshot(self) -> dict[str, int | bool | str | None]:
        cache = self._semantic_state.snapshot()
        with self._lock:
            return {
                "enabled": True,
                "current_pressure": self._current_pressure.value,
                "pending_pressure": (
                    self._pending_pressure.value if self._pending_pressure is not None else None
                ),
                "normal_capacity_entries": self._normal_entries,
                "normal_capacity_bytes": self._normal_bytes,
                "current_capacity_entries": int(cache["capacity_entries"]),
                "current_capacity_bytes": int(cache["capacity_bytes"]),
                "adjustments": self._adjustments,
                "deferred_adjustments": self._deferred,
                "last_evicted_entries": self._last_evicted,
            }

    def _target(self, pressure: MemoryPressure) -> tuple[int, int]:
        if pressure in {MemoryPressure.NORMAL, MemoryPressure.UNKNOWN}:
            divisor = 1
        elif pressure == MemoryPressure.WARNING:
            divisor = 2
        else:
            divisor = 8
        return (
            max(1, self._normal_entries // divisor),
            max(1, self._normal_bytes // divisor),
        )


def disabled_elastic_memory_snapshot() -> dict[str, int | bool | str | None]:
    return {
        "enabled": False,
        "current_pressure": "unknown",
        "pending_pressure": None,
        "normal_capacity_entries": 0,
        "normal_capacity_bytes": 0,
        "current_capacity_entries": 0,
        "current_capacity_bytes": 0,
        "adjustments": 0,
        "deferred_adjustments": 0,
        "last_evicted_entries": 0,
    }
