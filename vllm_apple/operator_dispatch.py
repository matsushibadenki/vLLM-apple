from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionBackend
from .kernel_probe import KernelCapabilityRegistry


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
