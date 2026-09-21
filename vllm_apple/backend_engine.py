"""Versioned exchange contract for isolated CPU/GPU/ANE backend engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .execution import ExecutionBackend, WorkloadPhase
from .inference_request import InferenceRequestContext
from .fault_injection import (
    DeterministicFaultInjector,
    FaultAction,
    FaultPoint,
    InjectedFault,
)


BACKEND_ENGINE_SCHEMA_VERSION = 1
MAX_BACKEND_METADATA_VALUES = 128
_Result = TypeVar("_Result")


def _values(values: tuple[str, ...], label: str) -> None:
    if (not values or len(values) > MAX_BACKEND_METADATA_VALUES
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or not 1 <= len(value) <= 128
                   or any(ord(character) < 0x20 for character in value)
                   for value in values)):
        raise ValueError(f"invalid backend engine {label}")


@dataclass(frozen=True, slots=True)
class BackendEngineDescriptor:
    backend: ExecutionBackend
    version: str
    model_architectures: tuple[str, ...]
    precisions: tuple[str, ...]
    phases: tuple[WorkloadPhase, ...]
    operators: tuple[str, ...]
    isolation: str
    schema_version: int = BACKEND_ENGINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.schema_version != BACKEND_ENGINE_SCHEMA_VERSION
                or not isinstance(self.backend, ExecutionBackend)):
            raise ValueError("invalid backend engine descriptor version")
        _values((self.version,), "version")
        _values(self.model_architectures, "model architectures")
        _values(self.precisions, "precisions")
        _values(self.operators, "operators")
        if (not self.phases or len(set(self.phases)) != len(self.phases)
                or any(not isinstance(phase, WorkloadPhase) for phase in self.phases)):
            raise ValueError("invalid backend engine phases")
        if self.isolation not in {"in_process", "dedicated_thread", "subprocess"}:
            raise ValueError("invalid backend engine isolation")


@dataclass(frozen=True, slots=True)
class BackendEngineRequest:
    operator: str
    phase: WorkloadPhase
    precision: str
    model_architecture: str
    candidates: tuple[ExecutionBackend, ...]

    def __post_init__(self) -> None:
        _values((self.operator,), "request operator")
        _values((self.precision,), "request precision")
        _values((self.model_architecture,), "request architecture")
        if (not isinstance(self.phase, WorkloadPhase) or not self.candidates
                or len(self.candidates) > len(ExecutionBackend)
                or len(set(self.candidates)) != len(self.candidates)):
            raise ValueError("invalid backend engine request")


class BackendEngine(Protocol[_Result]):
    @property
    def descriptor(self) -> BackendEngineDescriptor: ...

    @property
    def ready(self) -> bool: ...

    def start(self) -> None: ...

    def execute(
        self, request: BackendEngineRequest, context: InferenceRequestContext
    ) -> _Result: ...

    def stop(self) -> None: ...


class BackendEngineFailure(RuntimeError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        _values((code,), "failure code")
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class BackendEngineAttempt:
    backend: ExecutionBackend
    status: str
    reason: str | None

    def __post_init__(self) -> None:
        if self.status not in {"rejected", "failed", "succeeded"}:
            raise ValueError("invalid backend engine attempt status")
        if self.status == "succeeded" and self.reason is not None:
            raise ValueError("successful backend engine attempt cannot have a reason")
        if self.status != "succeeded":
            _values((self.reason,), "attempt reason")


@dataclass(frozen=True, slots=True)
class BackendEngineResult(Generic[_Result]):
    value: _Result
    backend: ExecutionBackend
    attempts: tuple[BackendEngineAttempt, ...]


class BackendEngineRegistry(Generic[_Result]):
    """Owns exchangeable engines and executes only an explicit candidate chain."""

    def __init__(
        self,
        engines: tuple[BackendEngine[_Result], ...],
        *,
        fault_injector: DeterministicFaultInjector | None = None,
    ) -> None:
        if (not engines or len(engines) > len(ExecutionBackend)
                or len({engine.descriptor.backend for engine in engines}) != len(engines)):
            raise ValueError("backend engine registry entries are invalid")
        self._engines = {engine.descriptor.backend: engine for engine in engines}
        self._fault_injector = fault_injector

    def execute(
        self, request: BackendEngineRequest, context: InferenceRequestContext
    ) -> BackendEngineResult[_Result]:
        if not isinstance(context, InferenceRequestContext):
            raise ValueError("backend engine request context is invalid")
        attempts: list[BackendEngineAttempt] = []
        for backend in request.candidates:
            engine = self._engines.get(backend)
            reason = self._ineligible_reason(engine, request)
            if reason is not None:
                attempts.append(BackendEngineAttempt(backend, "rejected", reason))
                continue
            assert engine is not None
            context.raise_if_cancelled()
            try:
                if self._fault_injector is not None:
                    self._fault_injector.hit(FaultPoint.BACKEND_EXECUTE)
                value = engine.execute(request, context)
            except InjectedFault as error:
                if error.action is FaultAction.TIMEOUT:
                    code, retryable = "injected_backend_timeout", True
                else:
                    code = f"injected_backend_{error.action.value}"
                    retryable = error.action is FaultAction.RETRYABLE
                attempts.append(BackendEngineAttempt(backend, "failed", code))
                if not retryable:
                    raise BackendEngineFailure(code, retryable=False) from error
                continue
            except BackendEngineFailure as error:
                attempts.append(BackendEngineAttempt(backend, "failed", error.code))
                if not error.retryable:
                    raise
                continue
            context.raise_if_cancelled()
            attempts.append(BackendEngineAttempt(backend, "succeeded", None))
            return BackendEngineResult(value, backend, tuple(attempts))
        raise BackendEngineFailure("backend_engine_fallback_exhausted", retryable=False)

    @staticmethod
    def _ineligible_reason(
        engine: BackendEngine[_Result] | None, request: BackendEngineRequest
    ) -> str | None:
        if engine is None:
            return "backend_unregistered"
        descriptor = engine.descriptor
        if not engine.ready:
            return "backend_not_ready"
        if request.operator not in descriptor.operators:
            return "operator_unsupported"
        if request.phase not in descriptor.phases:
            return "phase_unsupported"
        if request.precision not in descriptor.precisions:
            return "precision_unsupported"
        if request.model_architecture not in descriptor.model_architectures:
            return "architecture_unsupported"
        return None

    def stop_all(self) -> None:
        failures = []
        for backend in reversed(tuple(self._engines)):
            try:
                if self._fault_injector is not None:
                    self._fault_injector.hit(FaultPoint.BACKEND_STOP)
                self._engines[backend].stop()
            except Exception as error:  # lifecycle aggregation is intentionally bounded
                failures.append((backend.value, type(error).__name__))
        if failures:
            raise RuntimeError(f"backend engine shutdown failures: {failures}")
