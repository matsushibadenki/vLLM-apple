"""Quality-gated backend-owned KV and recurrent state precision adapter."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import threading

from .adaptive_state_allocation import (
    AdaptiveStateAction,
    AdaptiveStateBackend,
    AdaptiveStateKind,
    AdaptiveStatePlan,
    AdaptiveStateRecord,
    AdaptiveStateTransaction,
)


MAX_STATE_VALUES = 16_777_216


@dataclass(frozen=True, slots=True)
class StatePrecisionGate:
    maximum_absolute_error: float
    maximum_rmse: float
    minimum_cosine_similarity: float
    minimum_memory_reduction_ratio: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.maximum_absolute_error)
                or not math.isfinite(self.maximum_rmse)
                or self.maximum_absolute_error < 0
                or self.maximum_rmse < 0
                or not 0 <= self.minimum_cosine_similarity <= 1
                or not 0 < self.minimum_memory_reduction_ratio < 1):
            raise ValueError("invalid state precision gate")


@dataclass(frozen=True, slots=True)
class StatePrecisionMeasurement:
    source_precision: str
    target_precision: str
    maximum_absolute_error: float
    rmse: float
    cosine_similarity: float
    source_bytes: int
    target_bytes: int
    promoted: bool


@dataclass(frozen=True, slots=True)
class _EncodedState:
    state_id: str
    kind: AdaptiveStateKind
    precision: str
    payload: bytes
    scale: float
    value_count: int
    age_seconds: float
    pinned: bool
    promoted_precisions: tuple[str, ...]


class MeasuredStateBackendAdapter(AdaptiveStateBackend):
    """Owns real encoded buffers and atomically applies measured conversions."""

    def __init__(self, gate: StatePrecisionGate) -> None:
        self._gate = gate
        self._states: dict[str, _EncodedState] = {}
        self._measurements: dict[tuple[str, str, str], StatePrecisionMeasurement] = {}
        self._lock = threading.RLock()

    def register(
        self,
        state_id: str,
        kind: AdaptiveStateKind,
        values: tuple[float, ...],
        *,
        age_seconds: float = 0,
        pinned: bool = False,
    ) -> None:
        if (not state_id or len(state_id) > 256 or not values
                or len(values) > MAX_STATE_VALUES
                or not all(math.isfinite(value) for value in values)):
            raise ValueError("invalid backend state")
        encoded = _encode(values, "fp32")
        with self._lock:
            if state_id in self._states:
                raise ValueError("duplicate backend state")
            self._states[state_id] = _EncodedState(
                state_id, kind, "fp32", encoded[0], encoded[1], len(values),
                age_seconds, pinned, ("fp32",),
            )

    def qualify_precision(self, state_id: str, target_precision: str) -> StatePrecisionMeasurement:
        if target_precision not in {"fp16", "int8"}:
            raise ValueError("unsupported state precision")
        with self._lock:
            state = self._states[state_id]
            source = _decode(state)
            payload, scale = _encode(source, target_precision)
            candidate = _EncodedState(
                state.state_id, state.kind, target_precision, payload, scale,
                state.value_count, state.age_seconds, state.pinned,
                state.promoted_precisions,
            )
            restored = _decode(candidate)
            errors = tuple(abs(left - right) for left, right in zip(source, restored))
            rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
            source_norm = math.sqrt(sum(value * value for value in source))
            target_norm = math.sqrt(sum(value * value for value in restored))
            cosine = 1.0 if source_norm == target_norm == 0 else (
                sum(left * right for left, right in zip(source, restored))
                / (source_norm * target_norm)
                if source_norm and target_norm else 0.0
            )
            reduction = 1 - len(payload) / len(state.payload)
            promoted = (
                max(errors) <= self._gate.maximum_absolute_error
                and rmse <= self._gate.maximum_rmse
                and cosine >= self._gate.minimum_cosine_similarity
                and reduction >= self._gate.minimum_memory_reduction_ratio
            )
            measurement = StatePrecisionMeasurement(
                state.precision, target_precision, max(errors), rmse, cosine,
                len(state.payload), len(payload), promoted,
            )
            self._measurements[(state_id, state.precision, target_precision)] = measurement
            if promoted:
                promoted_precisions = tuple(dict.fromkeys(
                    (*state.promoted_precisions, target_precision)
                ))
                self._states[state_id] = _EncodedState(
                    state.state_id, state.kind, state.precision, state.payload,
                    state.scale, state.value_count, state.age_seconds,
                    state.pinned, promoted_precisions,
                )
            return measurement

    def adaptive_state_records(self) -> tuple[AdaptiveStateRecord, ...]:
        with self._lock:
            return tuple(
                AdaptiveStateRecord(
                    state.state_id, state.kind, len(state.payload), state.age_seconds,
                    state.precision, state.promoted_precisions, state.pinned,
                )
                for state in self._states.values()
            )

    def begin_adaptive_state(self, plan: AdaptiveStatePlan) -> AdaptiveStateTransaction:
        with self._lock:
            staged = dict(self._states)
            for action in plan.actions:
                self._stage_action(staged, action)
        return _StateTransaction(self, staged)

    def values(self, state_id: str) -> tuple[float, ...]:
        with self._lock:
            return _decode(self._states[state_id])

    def _stage_action(
        self, staged: dict[str, _EncodedState], action: AdaptiveStateAction
    ) -> None:
        state = staged.get(action.state_id)
        if state is None or state.precision != action.source_precision:
            raise ValueError("adaptive state plan is stale")
        if action.action == "retain":
            return
        if action.action == "evict":
            del staged[action.state_id]
            return
        assert action.target_precision is not None
        measurement = self._measurements.get(
            (action.state_id, action.source_precision, action.target_precision)
        )
        if measurement is None or not measurement.promoted:
            raise ValueError("state precision was not promoted")
        payload, scale = _encode(_decode(state), action.target_precision)
        if len(payload) != action.target_bytes:
            raise ValueError("adaptive state byte estimate mismatch")
        staged[action.state_id] = _EncodedState(
            state.state_id, state.kind, action.target_precision, payload, scale,
            state.value_count, state.age_seconds, state.pinned,
            tuple(value for value in state.promoted_precisions
                  if value == action.target_precision),
        )


class _StateTransaction(AdaptiveStateTransaction):
    def __init__(self, backend: MeasuredStateBackendAdapter, staged: dict[str, _EncodedState]) -> None:
        self._backend = backend
        self._staged = staged
        self._closed = False

    def commit(self) -> None:
        with self._backend._lock:
            if self._closed:
                raise RuntimeError("adaptive state transaction is closed")
            self._backend._states = self._staged
            self._closed = True

    def rollback(self) -> None:
        self._closed = True


def _encode(values: tuple[float, ...], precision: str) -> tuple[bytes, float]:
    if precision == "fp32":
        return struct.pack(f"<{len(values)}f", *values), 1.0
    if precision == "fp16":
        return struct.pack(f"<{len(values)}e", *values), 1.0
    if precision == "int8":
        maximum = max(abs(value) for value in values)
        scale = maximum / 127 if maximum else 1.0
        quantized = tuple(max(-127, min(127, round(value / scale))) for value in values)
        return struct.pack(f"<{len(values)}b", *quantized), scale
    raise ValueError("unsupported state precision")


def _decode(state: _EncodedState) -> tuple[float, ...]:
    if state.precision == "fp32":
        return tuple(struct.unpack(f"<{state.value_count}f", state.payload))
    if state.precision == "fp16":
        return tuple(struct.unpack(f"<{state.value_count}e", state.payload))
    if state.precision == "int8":
        return tuple(
            value * state.scale
            for value in struct.unpack(f"<{state.value_count}b", state.payload)
        )
    raise ValueError("unsupported state precision")
