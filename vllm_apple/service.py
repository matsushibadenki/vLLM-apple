from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, BinaryIO, Protocol

from .elastic_memory import (
    ElasticMemoryController,
    ElasticMemoryDecision,
    disabled_elastic_memory_snapshot,
)
from .events import EventBus
from .execution import AppleExecutionPlan, WorkloadPhase
from .kernel_context import InferenceKernelContext, PagedAttentionKernelSelection
from .kernel_profile import PagedAttentionShape
from .metal_probe import MetalThreadConfiguration
from .metal_tuning import MetalTuningReport
from .memory_telemetry import UnifiedMemoryTelemetry
from .memory_admission import MemoryPressureAdmissionError, MemoryPressureAdmissionGate
from .operator_dispatch import OperatorDispatcher
from .profile import build_profile
from .runtime_errors import RuntimeFailure, classify_runtime_failure
from .scheduler import BasicScheduler, PlanApplicationDecision, Reservation, ScheduleRequest
from .semantic_cache import SemanticAnchor, SemanticAnchorKind
from .semantic_state import (
    SemanticRestoreResult,
    SemanticStateCoordinator,
    disabled_semantic_state_snapshot,
)
from .types import GIB, MemoryPressure, RuntimeProfile, RuntimeState


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceEngine(Protocol):
    @property
    def ready(self) -> bool: ...

    def models(self) -> list[dict[str, Any]]: ...

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def open_chat_stream(self, request: dict[str, Any]) -> AbstractContextManager[BinaryIO]: ...


class UnavailableInferenceEngine:
    @property
    def ready(self) -> bool:
        return False

    def models(self) -> list[dict[str, Any]]:
        return []

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailableError("no inference backend is loaded")

    def open_chat_stream(self, request: dict[str, Any]) -> AbstractContextManager[BinaryIO]:
        raise InferenceUnavailableError("no inference backend is loaded")


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    state: RuntimeState
    control_ready: bool
    inference_ready: bool
    profile: RuntimeProfile
    scheduler: dict[str, int]
    last_error: str | None
    failure: dict[str, object] | None
    events: dict[str, int]
    semantic_cache: dict[str, int | bool]
    elastic_memory: dict[str, int | bool | str | None]
    execution_plan: dict[str, str | int | bool | None]
    memory_telemetry: dict[str, int | float | str | None]
    memory_admission: dict[str, int | float | str | None]
    token_estimation: dict[str, int | str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "control_ready": self.control_ready,
            "inference_ready": self.inference_ready,
            "profile": self.profile.to_dict(),
            "scheduler": self.scheduler,
            "last_error": self.last_error,
            "failure": self.failure,
            "events": self.events,
            "semantic_cache": self.semantic_cache,
            "elastic_memory": self.elastic_memory,
            "execution_plan": self.execution_plan,
            "memory_telemetry": self.memory_telemetry,
            "memory_admission": self.memory_admission,
            "token_estimation": self.token_estimation,
        }


class RuntimeService:
    def __init__(
        self,
        engine: InferenceEngine | None = None,
        profile: RuntimeProfile | None = None,
        event_bus: EventBus | None = None,
        semantic_state: SemanticStateCoordinator | None = None,
        memory_telemetry: UnifiedMemoryTelemetry | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state = RuntimeState.STARTING
        self._last_error: str | None = None
        self._failure: RuntimeFailure | None = None
        self._pending_operator_dispatcher: OperatorDispatcher | None = None
        self._active_metal_tuning: MetalTuningReport | None = None
        self._pending_metal_tuning: MetalTuningReport | None = None
        self._tokenizer_fallbacks = 0
        self._last_prompt_tokens: int | None = None
        self.events = event_bus or EventBus()
        self.profile = profile or build_profile()
        memory = self.profile.hardware.memory
        self.memory_telemetry = memory_telemetry or UnifiedMemoryTelemetry(
            memory.total_bytes, memory.available_bytes, memory.source, memory.pressure
        )
        self.memory_telemetry.update_os(
            available_bytes=memory.available_bytes,
            control_resident_bytes=memory.process_resident_bytes,
        )
        self.memory_admission = MemoryPressureAdmissionGate()
        self.memory_admission.refresh(self.memory_telemetry.snapshot())
        # Scheduler reservations cover transient work only. Retain the larger of
        # 1 GiB or 8% as an unreservable emergency margin.
        emergency_margin = max(GIB, int(memory.total_bytes * 0.08))
        scheduler_capacity = max(0, memory.available_bytes - emergency_margin)
        self.scheduler = BasicScheduler(self.profile.hardware, scheduler_capacity)
        self.engine = engine or UnavailableInferenceEngine()
        self.semantic_state = semantic_state
        self.elastic_memory = (
            ElasticMemoryController(semantic_state) if semantic_state is not None else None
        )
        self._state = RuntimeState.READY if self.engine.ready else RuntimeState.DEGRADED
        self.events.publish(
            "runtime.state",
            {"state": self._state.value, "inference_ready": self.engine.ready},
        )

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def set_state(self, state: RuntimeState) -> None:
        with self._lock:
            self._state = state
            if state != RuntimeState.FAILED:
                self._last_error = None
                self._failure = None
        self.events.publish(
            "runtime.state",
            {"state": state.value, "inference_ready": self.engine.ready},
        )

    def set_failure(self, error: BaseException | str | RuntimeFailure) -> RuntimeFailure:
        failure = error if isinstance(error, RuntimeFailure) else classify_runtime_failure(error)
        with self._lock:
            self._last_error = failure.message_key
            self._failure = failure
            self._state = RuntimeState.FAILED
        self.events.publish(
            "runtime.failure",
            {"state": RuntimeState.FAILED.value, "failure": failure.to_dict()},
        )
        return failure

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                state=self._state,
                control_ready=self._state not in {RuntimeState.FAILED, RuntimeState.STOPPED},
                inference_ready=self.engine.ready,
                profile=self.profile,
                scheduler=self.scheduler.memory.snapshot(),
                last_error=self._last_error,
                failure=self._failure.to_dict() if self._failure is not None else None,
                events=self.events.snapshot(),
                semantic_cache=(
                    self.semantic_state.snapshot()
                    if self.semantic_state is not None
                    else disabled_semantic_state_snapshot()
                ),
                elastic_memory=(
                    self.elastic_memory.snapshot()
                    if self.elastic_memory is not None
                    else disabled_elastic_memory_snapshot()
                ),
                execution_plan=self.scheduler.execution_plan_snapshot(),
                memory_telemetry=self.memory_telemetry.snapshot().to_dict(),
                memory_admission=self.memory_admission.snapshot().to_dict(),
                token_estimation=self.token_estimation_snapshot(),
            )

    def token_estimation_snapshot(self) -> dict[str, int | str | None]:
        tokenizer_snapshot = getattr(self.engine, "tokenizer_snapshot", None)
        backend = tokenizer_snapshot() if callable(tokenizer_snapshot) else {}
        with self._lock:
            return {
                "source": "backend_tokenizer" if self._last_prompt_tokens is not None else "fallback",
                "measured": int(backend.get("measured", 0)),
                "failures": int(backend.get("failures", 0)),
                "fallbacks": self._tokenizer_fallbacks,
                "last_prompt_tokens": self._last_prompt_tokens,
                "cache_capacity": int(backend.get("cache_capacity", 0)),
                "cache_entries": int(backend.get("cache_entries", 0)),
                "cache_hits": int(backend.get("cache_hits", 0)),
                "cache_misses": int(backend.get("cache_misses", 0)),
                "cache_evictions": int(backend.get("cache_evictions", 0)),
                "cache_expirations": int(backend.get("cache_expirations", 0)),
            }

    def chat_schedule_request(self, request: dict[str, Any]) -> ScheduleRequest:
        raw_batch = request.get("n", 1)
        batch_size = (
            raw_batch
            if isinstance(raw_batch, int) and not isinstance(raw_batch, bool) and raw_batch > 0
            else 1
        )
        raw_output = request.get("max_completion_tokens", request.get("max_tokens", 0))
        output_tokens = (
            raw_output
            if isinstance(raw_output, int) and not isinstance(raw_output, bool) and raw_output > 0
            else 0
        )
        estimator = getattr(self.engine, "estimate_prompt_tokens", None)
        prompt_tokens = estimator(request) if callable(estimator) else None
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < 0:
            prompt_tokens = None
        with self._lock:
            if prompt_tokens is None:
                self._tokenizer_fallbacks += 1
                self._last_prompt_tokens = None
            else:
                self._last_prompt_tokens = prompt_tokens
        return ScheduleRequest(
            "paged_attention",
            0,
            batch_size=batch_size,
            phase=WorkloadPhase.DECODE,
            estimated_context_tokens=(prompt_tokens or 0) + output_tokens,
        )

    def record_framework_memory(
        self, current_bytes: int, *, source: str, peak_bytes: int | None = None
    ) -> None:
        self.memory_telemetry.update_allocator(current_bytes, source=source, peak_bytes=peak_bytes)

    def record_kv_cache_memory(
        self, used_bytes: int, capacity_bytes: int, *, source: str
    ) -> None:
        self.memory_telemetry.update_kv_cache(used_bytes, capacity_bytes, source=source)

    def record_kv_cache_ratio(self, usage_ratio: float, *, source: str) -> None:
        self.memory_telemetry.update_kv_ratio(usage_ratio, source=source)

    def record_backend_resident_memory(self, resident_bytes: int, *, source: str) -> None:
        self.memory_telemetry.update_backend_resident(resident_bytes, source=source)
        self.memory_admission.refresh(self.memory_telemetry.snapshot())

    def record_iogpu_memory(self, current_bytes: int, *, source: str) -> None:
        self.memory_telemetry.update_iogpu(current_bytes, source=source)
        self.memory_admission.refresh(self.memory_telemetry.snapshot())

    def record_os_memory(
        self,
        *,
        available_bytes: int,
        control_resident_bytes: int,
        backend_resident_bytes: int | None = None,
        source: str | None = None,
    ) -> None:
        self.memory_telemetry.update_os(
            available_bytes=available_bytes,
            control_resident_bytes=control_resident_bytes,
            backend_resident_bytes=backend_resident_bytes,
            source=source,
        )
        self.memory_admission.refresh(self.memory_telemetry.snapshot())

    def admit_schedule(self, request: ScheduleRequest) -> Reservation:
        try:
            self.memory_admission.admit(request, self.memory_telemetry.snapshot())
        except MemoryPressureAdmissionError as error:
            self.events.publish(
                "memory.admission",
                {"status": "rejected", "reason": str(error)},
            )
            raise
        reservation = self.scheduler.admit(request)
        with self._lock:
            tuning_id = (
                self._active_metal_tuning.tuning_id
                if self._active_metal_tuning is not None
                else None
            )
        return replace(reservation, kernel_tuning_id=tuning_id)

    def complete_schedule(
        self, reservation: Reservation
    ) -> tuple[PlanApplicationDecision | None, ElasticMemoryDecision | None]:
        self.scheduler.complete(reservation)
        return self.apply_pending_runtime_policy()

    def request_execution_plan(self, plan: AppleExecutionPlan) -> PlanApplicationDecision:
        decision = self.scheduler.request_execution_plan(plan)
        self._publish_plan_decision(decision)
        return decision

    def apply_pending_runtime_policy(
        self,
    ) -> tuple[PlanApplicationDecision | None, ElasticMemoryDecision | None]:
        # Cache eviction/restoration precedes plan activation at the same safe point.
        # This prevents a new plan from admitting work while state resizing is active.
        def apply() -> tuple[PlanApplicationDecision | None, ElasticMemoryDecision | None]:
            with self._lock:
                dispatcher = self._pending_operator_dispatcher
                self._pending_operator_dispatcher = None
            if dispatcher is not None:
                self.scheduler.install_operator_dispatcher(dispatcher)
                self.events.publish("runtime.operator_dispatcher", {"status": "applied"})
            with self._lock:
                tuning = self._pending_metal_tuning
                self._pending_metal_tuning = None
                if tuning is not None:
                    self._active_metal_tuning = tuning
            if tuning is not None:
                self.events.publish(
                    "runtime.metal_tuning",
                    {"status": "applied", "tuning_id": tuning.tuning_id},
                )
            elastic = (
                self.elastic_memory.apply_pending(safe_to_apply=True)
                if self.elastic_memory is not None
                else None
            )
            plan = self.scheduler.apply_pending_execution_plan()
            return plan, elastic

        applied, result = self.scheduler.at_safe_point(apply)
        if applied:
            assert result is not None
            plan, elastic = result
        else:
            elastic = (
                self.elastic_memory.apply_pending(safe_to_apply=False)
                if self.elastic_memory is not None
                else None
            )
            plan = self.scheduler.apply_pending_execution_plan()
        if elastic is not None:
            self._publish_elastic_decision(elastic)
        if plan is not None:
            self._publish_plan_decision(plan)
        return plan, elastic

    def install_operator_dispatcher(self, dispatcher: OperatorDispatcher) -> bool:
        with self._lock:
            self._pending_operator_dispatcher = dispatcher

        def install() -> None:
            self.scheduler.install_operator_dispatcher(dispatcher)
            with self._lock:
                if self._pending_operator_dispatcher is dispatcher:
                    self._pending_operator_dispatcher = None

        applied, _ = self.scheduler.at_safe_point(install)
        self.events.publish(
            "runtime.operator_dispatcher",
            {"status": "applied" if applied else "deferred"},
        )
        return applied

    def install_metal_tuning(self, report: MetalTuningReport) -> bool:
        with self._lock:
            self._pending_metal_tuning = report

        def install() -> None:
            with self._lock:
                self._active_metal_tuning = report
                if self._pending_metal_tuning is report:
                    self._pending_metal_tuning = None

        applied, _ = self.scheduler.at_safe_point(install)
        self.events.publish(
            "runtime.metal_tuning",
            {"status": "applied" if applied else "deferred", "tuning_id": report.tuning_id},
        )
        return applied

    def metal_tuning_snapshot(self) -> dict[str, object]:
        with self._lock:
            active = self._active_metal_tuning
            pending = self._pending_metal_tuning
            return {
                "active_tuning_id": active.tuning_id if active else None,
                "pending_tuning_id": pending.tuning_id if pending else None,
                "active_profile_id": active.profile_id if active else None,
                "winner_count": len(active.decisions) if active else 0,
            }

    def metal_thread_configuration(
        self, shape: PagedAttentionShape
    ) -> MetalThreadConfiguration | None:
        with self._lock:
            report = self._active_metal_tuning
            if report is None:
                return None
            for decision in report.decisions:
                if decision.shape == shape:
                    return decision.winner
        return None

    def inference_kernel_context(self, reservation: Reservation) -> InferenceKernelContext | None:
        """Return only the tuning snapshot captured when this request was admitted."""
        with self._lock:
            report = self._active_metal_tuning
            if report is None or reservation.kernel_tuning_id != report.tuning_id:
                return None
            return InferenceKernelContext(
                tuning_id=report.tuning_id,
                profile_id=report.profile_id,
                paged_attention=tuple(
                    PagedAttentionKernelSelection(decision.shape, decision.winner)
                    for decision in report.decisions
                ),
            )

    def chat_completions(self, request: dict[str, Any], reservation: Reservation) -> dict[str, Any]:
        context_method = getattr(self.engine, "chat_completions_with_context", None)
        if callable(context_method):
            return context_method(request, self.inference_kernel_context(reservation))
        return self.engine.chat_completions(request)

    def open_chat_stream(
        self, request: dict[str, Any], reservation: Reservation
    ) -> AbstractContextManager[BinaryIO]:
        context_method = getattr(self.engine, "open_chat_stream_with_context", None)
        if callable(context_method):
            return context_method(request, self.inference_kernel_context(reservation))
        return self.engine.open_chat_stream(request)

    def capture_semantic_state(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
        token_position: int,
        kind: SemanticAnchorKind,
    ) -> SemanticAnchor | None:
        if self.semantic_state is None:
            return None
        return self.semantic_state.capture(
            session_fingerprint,
            prefix_fingerprint,
            token_position,
            kind,
        )

    def restore_semantic_state(
        self,
        session_fingerprint: str,
        boundaries: tuple[tuple[int, str], ...],
    ) -> SemanticRestoreResult:
        if self.semantic_state is None:
            return SemanticRestoreResult(False)
        return self.semantic_state.restore_deepest(session_fingerprint, boundaries)

    def resize_semantic_cache(self, capacity_entries: int, capacity_bytes: int) -> int:
        if self.semantic_state is None:
            return 0
        return self.semantic_state.resize(capacity_entries, capacity_bytes)

    def apply_memory_pressure(self, pressure: MemoryPressure) -> ElasticMemoryDecision | None:
        self.memory_telemetry.update_pressure(pressure)
        self.memory_admission.refresh(self.memory_telemetry.snapshot())
        if self.elastic_memory is None:
            self.events.publish(
                "memory.pressure",
                {"pressure": pressure.value, "elastic_memory_enabled": False},
            )
            return None
        applied, result = self.scheduler.at_safe_point(
            lambda: self.elastic_memory.request(pressure, safe_to_apply=True)
        )
        decision = result if applied else self.elastic_memory.request(pressure, safe_to_apply=False)
        assert decision is not None
        self._publish_elastic_decision(decision)
        return decision

    def apply_pending_memory_pressure(self) -> ElasticMemoryDecision | None:
        if self.elastic_memory is None:
            return None
        applied, result = self.scheduler.at_safe_point(
            lambda: self.elastic_memory.apply_pending(safe_to_apply=True)
        )
        decision = result if applied else self.elastic_memory.apply_pending(safe_to_apply=False)
        if decision is not None:
            self._publish_elastic_decision(decision)
        return decision

    def _publish_elastic_decision(self, decision: ElasticMemoryDecision) -> None:
        self.events.publish(
            "memory.elastic_budget",
            {
                "pressure": decision.pressure.value,
                "status": decision.status,
                "target_entries": decision.target_entries,
                "target_bytes": decision.target_bytes,
                "evicted_entries": decision.evicted_entries,
            },
        )

    def _publish_plan_decision(self, decision: PlanApplicationDecision) -> None:
        self.events.publish(
            "execution.plan",
            {
                "plan_id": decision.plan_id,
                "status": decision.status,
                "replaced_plan_id": decision.replaced_plan_id,
            },
        )
