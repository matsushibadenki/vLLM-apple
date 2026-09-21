"""Deterministic age/pressure policy for backend-owned inference state."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Protocol

from .types import MemoryPressure


MAX_STATE_RECORDS = 65_536
_PRECISION_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}


class AdaptiveStateKind(str, Enum):
    KV = "kv"
    RECURRENT = "recurrent"
    PREFIX = "prefix"
    ATTENTION_WINDOW = "attention_window"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class AdaptiveStateRecord:
    state_id: str
    kind: AdaptiveStateKind
    resident_bytes: int
    age_seconds: float
    precision: str
    promoted_precisions: tuple[str, ...]
    pinned: bool = False

    def __post_init__(self) -> None:
        if (not self.state_id or len(self.state_id) > 256
                or not isinstance(self.kind, AdaptiveStateKind)
                or not 1 <= self.resident_bytes <= 1 << 40
                or not 0 <= self.age_seconds <= 365 * 24 * 3600
                or self.precision not in _PRECISION_BYTES
                or not self.promoted_precisions
                or self.precision not in self.promoted_precisions
                or len(set(self.promoted_precisions)) != len(self.promoted_precisions)
                or any(value not in _PRECISION_BYTES for value in self.promoted_precisions)):
            raise ValueError("invalid adaptive state record")


@dataclass(frozen=True, slots=True)
class AdaptiveStateAction:
    state_id: str
    action: str
    source_precision: str
    target_precision: str | None
    source_bytes: int
    target_bytes: int
    reason: str

    def __post_init__(self) -> None:
        if self.action not in {"retain", "reprecision", "evict"}:
            raise ValueError("invalid adaptive state action")
        if self.source_precision not in _PRECISION_BYTES:
            raise ValueError("invalid adaptive state source precision")
        if self.action == "evict":
            if self.target_precision is not None or self.target_bytes != 0:
                raise ValueError("evicted adaptive state must have zero target")
        elif self.target_precision not in _PRECISION_BYTES or self.target_bytes <= 0:
            raise ValueError("invalid adaptive state target")


@dataclass(frozen=True, slots=True)
class AdaptiveStatePlan:
    pressure: MemoryPressure
    source_bytes: int
    target_bytes: int
    actions: tuple[AdaptiveStateAction, ...]

    @property
    def bytes_released(self) -> int:
        return self.source_bytes - self.target_bytes


class AdaptiveStateAllocator:
    """Choose only explicitly promoted precision transitions, oldest state first."""

    def plan(
        self,
        records: tuple[AdaptiveStateRecord, ...],
        pressure: MemoryPressure,
    ) -> AdaptiveStatePlan:
        if (not isinstance(pressure, MemoryPressure) or len(records) > MAX_STATE_RECORDS
                or len({record.state_id for record in records}) != len(records)):
            raise ValueError("invalid adaptive state allocation input")
        source = sum(record.resident_bytes for record in records)
        target_limit = source
        if pressure is MemoryPressure.WARNING:
            target_limit = source * 3 // 4
        elif pressure is MemoryPressure.CRITICAL:
            target_limit = source // 2
        elif pressure is MemoryPressure.UNKNOWN:
            pressure = MemoryPressure.NORMAL

        selected: dict[str, tuple[str, int, str]] = {}
        target = source
        candidates = sorted(
            (record for record in records if not record.pinned),
            key=lambda record: (-record.age_seconds, record.state_id),
        )
        for record in candidates:
            if target <= target_limit:
                break
            promoted = sorted(
                record.promoted_precisions,
                key=lambda value: (_PRECISION_BYTES[value], value),
            )[0]
            current_width = _PRECISION_BYTES[record.precision]
            target_width = _PRECISION_BYTES[promoted]
            if target_width < current_width:
                size = max(1, record.resident_bytes * target_width // current_width)
                selected[record.state_id] = (promoted, size, "pressure_precision")
                target -= record.resident_bytes - size

        if target > target_limit and pressure is MemoryPressure.CRITICAL:
            for record in candidates:
                if target <= target_limit:
                    break
                if record.state_id in selected:
                    _, size, _ = selected.pop(record.state_id)
                    target += record.resident_bytes - size
                selected[record.state_id] = ("", 0, "critical_age_eviction")
                target -= record.resident_bytes

        actions = []
        for record in records:
            choice = selected.get(record.state_id)
            if choice is None:
                actions.append(AdaptiveStateAction(
                    record.state_id, "retain", record.precision, record.precision,
                    record.resident_bytes, record.resident_bytes,
                    "pinned" if record.pinned else "within_budget",
                ))
            elif choice[1] == 0:
                actions.append(AdaptiveStateAction(
                    record.state_id, "evict", record.precision, None,
                    record.resident_bytes, 0, choice[2],
                ))
            else:
                actions.append(AdaptiveStateAction(
                    record.state_id, "reprecision", record.precision, choice[0],
                    record.resident_bytes, choice[1], choice[2],
                ))
        return AdaptiveStatePlan(pressure, source, target, tuple(actions))


class AdaptiveStateTransaction(Protocol):
    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class AdaptiveStateBackend(Protocol):
    def adaptive_state_records(self) -> tuple[AdaptiveStateRecord, ...]: ...

    def begin_adaptive_state(self, plan: AdaptiveStatePlan) -> AdaptiveStateTransaction: ...


@dataclass(frozen=True, slots=True)
class AdaptiveStateDecision:
    pressure: MemoryPressure
    status: str
    source_bytes: int
    target_bytes: int
    reprecisioned: int
    evicted: int

    def __post_init__(self) -> None:
        if (self.status not in {"applied", "deferred", "ignored"}
                or min(self.source_bytes, self.target_bytes,
                       self.reprecisioned, self.evicted) < 0):
            raise ValueError("invalid adaptive state decision")


class AdaptiveStateCoordinator:
    """Apply a backend-owned state plan atomically at scheduler safe points."""

    def __init__(
        self,
        backend: AdaptiveStateBackend,
        allocator: AdaptiveStateAllocator | None = None,
    ) -> None:
        self._backend = backend
        self._allocator = allocator or AdaptiveStateAllocator()
        self._pending: MemoryPressure | None = None
        self._applied = 0
        self._rollbacks = 0
        self._last_released = 0
        self._lock = threading.Lock()

    def request(
        self, pressure: MemoryPressure, *, safe_to_apply: bool
    ) -> AdaptiveStateDecision:
        records = self._backend.adaptive_state_records()
        plan = self._allocator.plan(records, pressure)
        if not safe_to_apply:
            with self._lock:
                self._pending = pressure
            return self._decision(plan, "deferred")
        changed = tuple(action for action in plan.actions if action.action != "retain")
        if not changed:
            with self._lock:
                self._pending = None
            return self._decision(plan, "ignored")
        transaction = self._backend.begin_adaptive_state(plan)
        try:
            transaction.commit()
        except BaseException:
            try:
                transaction.rollback()
            finally:
                with self._lock:
                    self._rollbacks += 1
            raise
        with self._lock:
            self._pending = None
            self._applied += 1
            self._last_released = plan.bytes_released
        return self._decision(plan, "applied")

    def apply_pending(self, *, safe_to_apply: bool) -> AdaptiveStateDecision | None:
        with self._lock:
            pressure = self._pending
        if pressure is None:
            return None
        return self.request(pressure, safe_to_apply=safe_to_apply)

    def snapshot(self) -> dict[str, int | str | None | bool]:
        with self._lock:
            return {
                "adaptive_state_enabled": True,
                "adaptive_state_pending_pressure": (
                    self._pending.value if self._pending is not None else None
                ),
                "adaptive_state_applied": self._applied,
                "adaptive_state_rollbacks": self._rollbacks,
                "adaptive_state_last_released_bytes": self._last_released,
            }

    @staticmethod
    def _decision(plan: AdaptiveStatePlan, status: str) -> AdaptiveStateDecision:
        return AdaptiveStateDecision(
            plan.pressure,
            status,
            plan.source_bytes,
            plan.target_bytes,
            sum(action.action == "reprecision" for action in plan.actions),
            sum(action.action == "evict" for action in plan.actions),
        )
