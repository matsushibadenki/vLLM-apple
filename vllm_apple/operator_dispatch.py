from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .execution import ExecutionBackend
from .kernel_probe import KernelCapabilityRegistry

_ExecutionResult = TypeVar("_ExecutionResult")


class BackendExecutionError(RuntimeError):
    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        if not error_code or len(error_code) > 128:
            raise ValueError("backend error code must contain 1 to 128 characters")
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class OperatorFallbackExhaustedError(RuntimeError):
    def __init__(self, attempts: tuple["BackendExecutionAttempt", ...]) -> None:
        super().__init__("operator backend fallback exhausted")
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class BackendExecutionAttempt:
    backend: ExecutionBackend
    status: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"failed", "succeeded"}:
            raise ValueError("invalid backend execution attempt status")


@dataclass(frozen=True, slots=True)
class OperatorExecutionResult(Generic[_ExecutionResult]):
    value: _ExecutionResult
    backend: ExecutionBackend
    attempts: tuple[BackendExecutionAttempt, ...]


@dataclass(frozen=True, slots=True)
class OperatorDispatchRequest:
    operator: str
    candidates: tuple[ExecutionBackend, ...]

    def __post_init__(self) -> None:
        if not self.operator or len(self.operator) > 128:
            raise ValueError("operator must contain 1 to 128 characters")
        if not self.candidates or len(self.candidates) > len(ExecutionBackend):
            raise ValueError("dispatch requires bounded backend candidates")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("dispatch candidates must be unique")


@dataclass(frozen=True, slots=True)
class OperatorDispatchDecision:
    operator: str
    selected: ExecutionBackend
    fallback_chain: tuple[ExecutionBackend, ...]
    quarantined: tuple[ExecutionBackend, ...]
    unprobed: tuple[ExecutionBackend, ...]
    reason: str


class OperatorDispatcher:
    """Fail-closed dispatcher: accelerators require a passing profile-bound probe."""

    def __init__(self, registry: KernelCapabilityRegistry) -> None:
        self.registry = registry

    def dispatch(self, request: OperatorDispatchRequest) -> OperatorDispatchDecision:
        usable: list[ExecutionBackend] = []
        quarantined: list[ExecutionBackend] = []
        unprobed: list[ExecutionBackend] = []
        results = {
            (result.backend, result.operator): result
            for result in self.registry.snapshot()
        }
        for backend in request.candidates:
            if backend is ExecutionBackend.CPU:
                usable.append(backend)
                continue
            result = results.get((backend, request.operator))
            if result is None:
                unprobed.append(backend)
            elif result.quarantined:
                quarantined.append(backend)
            elif result.passed:
                usable.append(backend)
        if not usable:
            raise RuntimeError("no probed backend or CPU fallback is available")
        selected = usable[0]
        return OperatorDispatchDecision(
            operator=request.operator,
            selected=selected,
            fallback_chain=tuple(usable[1:]),
            quarantined=tuple(quarantined),
            unprobed=tuple(unprobed),
            reason="preferred_probe_passed"
            if selected is not ExecutionBackend.CPU
            else "accelerator_unavailable_cpu_fallback",
        )


class OperatorFallbackExecutor:
    """Execute only the dispatcher's bounded, probe-approved fallback chain."""

    def execute(
        self,
        decision: OperatorDispatchDecision,
        operation: Callable[[ExecutionBackend], _ExecutionResult],
    ) -> OperatorExecutionResult[_ExecutionResult]:
        backends = (decision.selected, *decision.fallback_chain)
        if not backends or len(backends) > len(ExecutionBackend) or len(set(backends)) != len(backends):
            raise ValueError("invalid operator fallback chain")
        attempts: list[BackendExecutionAttempt] = []
        for backend in backends:
            try:
                value = operation(backend)
            except BackendExecutionError as error:
                attempts.append(BackendExecutionAttempt(backend, "failed", error.error_code))
                if not error.retryable:
                    raise
                continue
            attempts.append(BackendExecutionAttempt(backend, "succeeded"))
            return OperatorExecutionResult(value, backend, tuple(attempts))
        raise OperatorFallbackExhaustedError(tuple(attempts))
