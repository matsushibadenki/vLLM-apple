from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


COMPONENT_NAMES = (
    "weights",
    "kv",
    "recurrent",
    "prefix",
    "window",
    "experts",
    "scratch",
    "metal_heap",
    "coreml",
)
ADDITIVE_COMPONENTS = frozenset(COMPONENT_NAMES) - {"metal_heap"}


@dataclass(frozen=True, slots=True)
class MemoryBudgetComponent:
    current_bytes: int | None
    peak_bytes: int | None
    source: str | None
    accounting: str


@dataclass(frozen=True, slots=True)
class MemoryBudgetSnapshot:
    capacity_bytes: int
    known_component_bytes: int
    known_remaining_bytes: int
    overcommitted_bytes: int
    unknown_components: tuple[str, ...]
    overlap_envelope_bytes: int | None
    components: dict[str, MemoryBudgetComponent]

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "known_component_bytes": self.known_component_bytes,
            "known_remaining_bytes": self.known_remaining_bytes,
            "overcommitted_bytes": self.overcommitted_bytes,
            "unknown_components": list(self.unknown_components),
            "overlap_envelope_bytes": self.overlap_envelope_bytes,
            "components": {name: asdict(component) for name, component in self.components.items()},
        }


class UnifiedMemoryBudgetLedger:
    """Constant-space component ledger with explicit overlap accounting."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("memory budget capacity must be positive")
        self._capacity = capacity_bytes
        self._lock = threading.Lock()
        self._components = {
            name: MemoryBudgetComponent(
                current_bytes=None,
                peak_bytes=None,
                source=None,
                accounting="additive" if name in ADDITIVE_COMPONENTS else "overlap_envelope",
            )
            for name in COMPONENT_NAMES
        }

    def update(self, name: str, current_bytes: int | None, *, source: str | None) -> None:
        if name not in self._components:
            raise ValueError("unknown memory budget component")
        if current_bytes is not None and current_bytes < 0:
            raise ValueError("memory budget bytes must not be negative")
        if (current_bytes is None) != (source is None) or source == "":
            raise ValueError("memory budget value and source must both be known or unknown")
        with self._lock:
            previous = self._components[name]
            peak = previous.peak_bytes
            if current_bytes is not None:
                peak = max(peak or 0, current_bytes)
            self._components[name] = MemoryBudgetComponent(
                current_bytes=current_bytes,
                peak_bytes=peak,
                source=source,
                accounting=previous.accounting,
            )

    def snapshot(self) -> MemoryBudgetSnapshot:
        with self._lock:
            components = dict(self._components)
        known = sum(
            component.current_bytes or 0
            for name, component in components.items()
            if name in ADDITIVE_COMPONENTS
        )
        unknown = tuple(
            name
            for name in COMPONENT_NAMES
            if name in ADDITIVE_COMPONENTS and components[name].current_bytes is None
        )
        envelope = components["metal_heap"].current_bytes
        return MemoryBudgetSnapshot(
            capacity_bytes=self._capacity,
            known_component_bytes=known,
            known_remaining_bytes=max(0, self._capacity - known),
            overcommitted_bytes=max(0, known - self._capacity),
            unknown_components=unknown,
            overlap_envelope_bytes=envelope,
            components=components,
        )
