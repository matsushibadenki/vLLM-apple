from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass

from .types import ThermalState


@dataclass(frozen=True, slots=True)
class ContextReevaluationSnapshot:
    enabled: bool
    status: str
    configured_context_tokens: int | None
    effective_context_tokens: int | None
    capacity_context_tokens: int | None
    kv_capacity_bytes: int | None
    kv_bytes_per_token: int | None
    weights_bytes: int | None
    source: str | None
    reevaluations: int

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)


class ContextCapacityReevaluator:
    """Reduces admission context after an exact backend KV capacity sample."""

    def __init__(
        self,
        configured_context_tokens: int,
        kv_bytes_per_token: int,
        weights_bytes: int,
    ) -> None:
        if min(configured_context_tokens, kv_bytes_per_token) <= 0 or weights_bytes < 0:
            raise ValueError("invalid context reevaluation inputs")
        self._configured = configured_context_tokens
        self._kv_bytes_per_token = kv_bytes_per_token
        self._weights_bytes = weights_bytes
        self._capacity: int | None = None
        self._capacity_tokens: int | None = None
        self._source: str | None = None
        self._reevaluations = 0
        self._thermal_state = ThermalState.UNKNOWN
        self._workload_tokens: deque[int] = deque(maxlen=64)
        self._lock = threading.Lock()

    def observe_workload(self, context_tokens: int) -> bool:
        """Record bounded admitted demand used by the thermal context policy."""
        if type(context_tokens) is not int or context_tokens <= 0:
            raise ValueError("invalid context workload sample")
        with self._lock:
            previous = self._effective_tokens_locked()
            self._workload_tokens.append(min(context_tokens, self._configured))
            return previous != self._effective_tokens_locked()

    def update_thermal_state(self, thermal_state: ThermalState) -> bool:
        if not isinstance(thermal_state, ThermalState):
            raise TypeError("thermal state must use the declared enum")
        with self._lock:
            if thermal_state is self._thermal_state:
                return False
            self._thermal_state = thermal_state
            return True

    def update(self, capacity_bytes: int, *, source: str) -> bool:
        if capacity_bytes < 0 or not source:
            raise ValueError("invalid backend KV capacity")
        with self._lock:
            if self._capacity == capacity_bytes and self._source == source:
                return False
            self._capacity = capacity_bytes
            self._capacity_tokens = capacity_bytes // self._kv_bytes_per_token
            self._source = source
            self._reevaluations += 1
            return True

    def snapshot(self) -> ContextReevaluationSnapshot:
        with self._lock:
            capacity_tokens = self._capacity_tokens
            effective = self._effective_tokens_locked()
            base = min(self._configured, capacity_tokens) if capacity_tokens is not None else self._configured
            status = (
                "pending"
                if capacity_tokens is None
                else "dynamic_reduced"
                if effective < base
                else "reduced"
                if capacity_tokens < self._configured
                else "sufficient"
            )
            return ContextReevaluationSnapshot(
                True,
                status,
                self._configured,
                effective,
                capacity_tokens,
                self._capacity,
                self._kv_bytes_per_token,
                self._weights_bytes,
                self._source,
                self._reevaluations,
            )

    def _effective_tokens_locked(self) -> int:
        base = (
            min(self._configured, self._capacity_tokens)
            if self._capacity_tokens is not None
            else self._configured
        )
        if self._thermal_state in {ThermalState.NOMINAL, ThermalState.UNKNOWN}:
            return base
        upper_ratio, lower_ratio = {
            ThermalState.FAIR: ((7, 8), (3, 4)),
            ThermalState.SERIOUS: ((3, 4), (1, 2)),
            ThermalState.CRITICAL: ((1, 2), (1, 2)),
        }[self._thermal_state]
        upper = base * upper_ratio[0] // upper_ratio[1]
        lower = base * lower_ratio[0] // lower_ratio[1]
        demand = max(self._workload_tokens, default=upper)
        target = max(lower, min(upper, (demand * 5 + 3) // 4))
        return max(1, target // 16 * 16)


def disabled_context_reevaluation_snapshot() -> dict[str, int | str | bool | None]:
    return ContextReevaluationSnapshot(
        False, "disabled", None, None, None, None, None, None, None, 0
    ).to_dict()
