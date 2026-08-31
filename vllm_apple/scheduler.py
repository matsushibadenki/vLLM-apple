from __future__ import annotations

import heapq
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from .execution import AppleExecutionPlan, ExecutionBackend, WorkloadPhase
from .operator_dispatch import (
    OperatorDispatchDecision,
    OperatorDispatcher,
    OperatorDispatchRequest,
    OperatorExecutionResult,
    OperatorFallbackExecutor,
)
from .types import Backend, HardwareInfo, Priority

_SafePointResult = TypeVar("_SafePointResult")


class MemoryCapacityError(RuntimeError):
    """Raised when admitting work would exceed the runtime memory budget."""


class ExecutionPlanAdmissionError(RuntimeError):
    """Raised when work violates the active execution plan."""


class MaintenanceInProgressError(RuntimeError):
    """Raised while an exclusive idle maintenance operation owns the scheduler."""


class ScheduleQueueFullError(RuntimeError):
    """Raised before enqueueing beyond the bounded scheduler queue."""


class QueuedAdmissionError(RuntimeError):
    def __init__(self, token: str, request: "ScheduleRequest", error: Exception) -> None:
        super().__init__("queued request admission failed")
        self.token = token
        self.request = request
        self.error = error


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    operator: str
    estimated_memory_bytes: int
    priority: Priority = Priority.NORMAL
    batch_size: int = 1
    phase: WorkloadPhase | None = None
    estimated_context_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            self.estimated_memory_bytes < 0
            or self.batch_size <= 0
            or self.estimated_context_tokens < 0
        ):
            raise ValueError("invalid schedule request")


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_id: str
    bytes: int
    backend: Backend
    priority: Priority
    created_at_monotonic: float
    execution_plan_id: str | None = None
    kernel_tuning_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueuedAdmission:
    token: str
    request: ScheduleRequest
    reservation: Reservation


class PriorityScheduleQueue:
    _RANK = {
        Priority.REALTIME: 0,
        Priority.INTERACTIVE: 1,
        Priority.NORMAL: 2,
        Priority.BACKGROUND: 3,
    }

    def __init__(self, maximum_requests: int = 1024) -> None:
        if maximum_requests <= 0 or maximum_requests > 65_536:
            raise ValueError("maximum_requests must be between 1 and 65536")
        self._maximum = maximum_requests
        self._sequence = 0
        self._heap: list[tuple[int, int, str]] = []
        self._requests: dict[str, ScheduleRequest] = {}
        self._claimed: set[str] = set()
        self._condition = threading.Condition()

    def enqueue(self, request: ScheduleRequest) -> str:
        with self._condition:
            if len(self._requests) + len(self._claimed) >= self._maximum:
                raise ScheduleQueueFullError("scheduler queue is full")
            token = uuid.uuid4().hex
            sequence = self._sequence
            self._sequence += 1
            self._requests[token] = request
            heapq.heappush(self._heap, (self._RANK[request.priority], sequence, token))
            self._condition.notify()
            return token

    def cancel(self, token: str) -> bool:
        with self._condition:
            if self._requests.pop(token, None) is not None:
                return True
            if token in self._claimed:
                self._claimed.remove(token)
                return True
            return False

    def finish_claim(self, token: str) -> bool:
        with self._condition:
            if token not in self._claimed:
                return False
            self._claimed.remove(token)
            return True

    def dequeue(self, timeout: float | None = None) -> tuple[str, ScheduleRequest] | None:
        if timeout is not None and timeout < 0:
            raise ValueError("queue timeout must not be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                while self._heap:
                    _, _, token = heapq.heappop(self._heap)
                    request = self._requests.pop(token, None)
                    if request is not None:
                        self._claimed.add(token)
                        return token, request
                if timeout == 0:
                    return None
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            counts = {priority.value: 0 for priority in Priority}
            for request in self._requests.values():
                counts[request.priority.value] += 1
            return {
                "queued": len(self._requests),
                "dispatching": len(self._claimed),
                "capacity": self._maximum,
                **counts,
            }


@dataclass(frozen=True, slots=True)
class PlanApplicationDecision:
    plan_id: str
    status: str
    replaced_plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"applied", "deferred", "ignored"}:
            raise ValueError("invalid plan application status")


class MemoryAdmissionController:
    """Thread-safe hard memory limit for work admitted to the runtime.

    Reservations are explicit and never trigger allocation themselves. Keeping the
    accounting separate avoids a failed allocation becoming the pressure signal.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes cannot be negative")
        self._capacity = capacity_bytes
        self._reserved = 0
        self._reservations: dict[str, Reservation] = {}
        self._lock = threading.Lock()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved

    @property
    def available_bytes(self) -> int:
        with self._lock:
            return self._capacity - self._reserved

    def reserve(self, request: ScheduleRequest, backend: Backend) -> Reservation:
        with self._lock:
            remaining = self._capacity - self._reserved
            if request.estimated_memory_bytes > remaining:
                raise MemoryCapacityError(
                    f"request needs {request.estimated_memory_bytes} bytes; {remaining} available"
                )
            reservation = Reservation(
                reservation_id=uuid.uuid4().hex,
                bytes=request.estimated_memory_bytes,
                backend=backend,
                priority=request.priority,
                created_at_monotonic=monotonic(),
            )
            self._reservations[reservation.reservation_id] = reservation
            self._reserved += reservation.bytes
            return reservation

    def release(self, reservation_id: str) -> bool:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return False
            self._reserved -= reservation.bytes
            return True

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "capacity_bytes": self._capacity,
                "reserved_bytes": self._reserved,
                "available_bytes": self._capacity - self._reserved,
                "active_reservations": len(self._reservations),
            }


class BasicScheduler:
    _CPU_OPERATORS = {"sampling", "tokenization", "routing", "state_read", "state_write"}
    _METAL_OPERATORS = {"paged_attention", "mla"}
    _GPU_OPERATORS = {
        "matmul",
        "gemm",
        "gemv",
        "attention",
        "convolution",
        "image_resize",
        "video_decode",
    }

    def __init__(
        self,
        hardware: HardwareInfo,
        memory_capacity_bytes: int,
        operator_dispatcher: OperatorDispatcher | None = None,
        maximum_queued_requests: int = 1024,
    ) -> None:
        self.hardware = hardware
        self.memory = MemoryAdmissionController(memory_capacity_bytes)
        self._policy_lock = threading.RLock()
        self._active_plan: AppleExecutionPlan | None = None
        self._pending_plan: AppleExecutionPlan | None = None
        self._operator_dispatcher = operator_dispatcher
        self._maintenance_owner: str | None = None
        self._queue = PriorityScheduleQueue(maximum_queued_requests)
        self._queued_active: dict[str, Reservation] = {}

    def choose_backend(self, request: ScheduleRequest) -> Backend:
        decision = self.dispatch_decision(request)
        return {
            ExecutionBackend.CPU: Backend.CPU,
            ExecutionBackend.NATIVE_MLX: Backend.MLX_GPU,
            ExecutionBackend.NATIVE_METAL: Backend.METAL,
        }.get(decision.selected, Backend.CPU)

    def dispatch_decision(self, request: ScheduleRequest) -> OperatorDispatchDecision:
        operator = request.operator.lower()
        candidates = self._dispatch_candidates(operator, request.batch_size)
        if self._operator_dispatcher is not None:
            return self._operator_dispatcher.dispatch(OperatorDispatchRequest(operator, candidates))
        if not self.hardware.is_apple_silicon:
            candidates = (ExecutionBackend.CPU,)
        return OperatorDispatchDecision(
            operator,
            candidates[0],
            candidates[1:],
            (),
            (),
            "legacy_policy",
        )

    def execute_with_fallback(
        self,
        request: ScheduleRequest,
        operation: Callable[[ExecutionBackend], _SafePointResult],
    ) -> OperatorExecutionResult[_SafePointResult]:
        return OperatorFallbackExecutor().execute(self.dispatch_decision(request), operation)

    def _dispatch_candidates(
        self, operator: str, batch_size: int
    ) -> tuple[ExecutionBackend, ...]:
        if operator in self._CPU_OPERATORS:
            return (ExecutionBackend.CPU,)
        if operator in {"gemv", "matmul"} and batch_size == 1:
            return (ExecutionBackend.CPU, ExecutionBackend.NATIVE_MLX)
        if operator in self._METAL_OPERATORS:
            return (
                ExecutionBackend.NATIVE_METAL,
                ExecutionBackend.NATIVE_MLX,
                ExecutionBackend.CPU,
            )
        if operator in self._GPU_OPERATORS:
            return (ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU)
        return (ExecutionBackend.CPU,)

    def admit(self, request: ScheduleRequest) -> Reservation:
        with self._policy_lock:
            if self._maintenance_owner is not None:
                raise MaintenanceInProgressError(
                    f"scheduler maintenance is active: {self._maintenance_owner}"
                )
            self._validate_plan_admission(request)
            reservation = self.memory.reserve(request, self.choose_backend(request))
            if self._active_plan is None:
                return reservation
            return Reservation(
                reservation_id=reservation.reservation_id,
                bytes=reservation.bytes,
                backend=reservation.backend,
                priority=reservation.priority,
                created_at_monotonic=reservation.created_at_monotonic,
                execution_plan_id=self._active_plan.plan_id,
            )

    def complete(self, reservation: Reservation) -> None:
        with self._policy_lock:
            self.memory.release(reservation.reservation_id)

    def submit(self, request: ScheduleRequest) -> str:
        return self._queue.enqueue(request)

    def admit_next(self, timeout: float | None = None) -> QueuedAdmission | None:
        queued = self._queue.dequeue(timeout)
        if queued is None:
            return None
        token, request = queued
        try:
            reservation = self.admit(request)
        except BaseException as error:
            self._queue.finish_claim(token)
            if isinstance(error, Exception):
                raise QueuedAdmissionError(token, request, error) from error
            raise
        with self._policy_lock:
            self._queued_active[token] = reservation
            if not self._queue.finish_claim(token):
                self._queued_active.pop(token, None)
                self.memory.release(reservation.reservation_id)
                return None
        return QueuedAdmission(token, request, reservation)

    def complete_queued(self, token: str) -> bool:
        with self._policy_lock:
            reservation = self._queued_active.pop(token, None)
            if reservation is None:
                return False
            self.memory.release(reservation.reservation_id)
            return True

    def cancel(self, token: str) -> bool:
        if self._queue.cancel(token):
            return True
        return self.complete_queued(token)

    def queue_snapshot(self) -> dict[str, int]:
        snapshot = self._queue.snapshot()
        with self._policy_lock:
            snapshot["active"] = len(self._queued_active)
        return snapshot

    def request_execution_plan(self, plan: AppleExecutionPlan) -> PlanApplicationDecision:
        self._validate_execution_plan(plan)
        with self._policy_lock:
            current_id = self._active_plan.plan_id if self._active_plan else None
            if current_id == plan.plan_id:
                self._pending_plan = None
                return PlanApplicationDecision(plan.plan_id, "ignored", current_id)
            if self.memory.snapshot()["active_reservations"]:
                replaced = self._pending_plan.plan_id if self._pending_plan else None
                self._pending_plan = plan
                return PlanApplicationDecision(plan.plan_id, "deferred", replaced)
            self._active_plan = plan
            self._pending_plan = None
            return PlanApplicationDecision(plan.plan_id, "applied", current_id)

    def apply_pending_execution_plan(self) -> PlanApplicationDecision | None:
        with self._policy_lock:
            pending = self._pending_plan
            if pending is None:
                return None
            if self.memory.snapshot()["active_reservations"]:
                return PlanApplicationDecision(pending.plan_id, "deferred")
            current_id = self._active_plan.plan_id if self._active_plan else None
            self._active_plan = pending
            self._pending_plan = None
            return PlanApplicationDecision(pending.plan_id, "applied", current_id)

    def execution_plan_snapshot(self) -> dict[str, str | int | bool | None]:
        with self._policy_lock:
            plan = self._active_plan
            pending = self._pending_plan
            return {
                "enabled": plan is not None,
                "active_plan_id": plan.plan_id if plan else None,
                "pending_plan_id": pending.plan_id if pending else None,
                "context_tokens": plan.context_tokens if plan else 0,
                "prefill_batch_size": plan.prefill.batch_size if plan else 0,
                "decode_batch_size": plan.decode.batch_size if plan else 0,
                "state_precision": plan.decode.state_precision if plan else None,
            }

    def install_operator_dispatcher(self, dispatcher: OperatorDispatcher) -> None:
        with self._policy_lock:
            if self.memory.snapshot()["active_reservations"]:
                raise RuntimeError("operator dispatcher requires a scheduler safe point")
            self._operator_dispatcher = dispatcher

    def at_safe_point(
        self, operation: Callable[[], _SafePointResult]
    ) -> tuple[bool, _SafePointResult | None]:
        """Run an operation while new admissions are blocked and no work is active."""
        with self._policy_lock:
            if self.memory.snapshot()["active_reservations"]:
                return False, None
            return True, operation()

    def begin_idle_maintenance(self, owner: str) -> bool:
        """Acquire an exclusive lease only when no request or maintenance is active."""
        if not owner or len(owner) > 64:
            raise ValueError("maintenance owner must contain 1 to 64 characters")
        with self._policy_lock:
            if self._maintenance_owner is not None:
                return False
            if self.memory.snapshot()["active_reservations"]:
                return False
            self._maintenance_owner = owner
            return True

    def end_idle_maintenance(self, owner: str) -> None:
        with self._policy_lock:
            if self._maintenance_owner != owner:
                raise RuntimeError("scheduler maintenance owner mismatch")
            self._maintenance_owner = None

    def maintenance_snapshot(self) -> dict[str, str | bool | None]:
        with self._policy_lock:
            return {
                "active": self._maintenance_owner is not None,
                "owner": self._maintenance_owner,
            }

    def _validate_plan_admission(self, request: ScheduleRequest) -> None:
        plan = self._active_plan
        if plan is None:
            return
        phase = request.phase
        if phase is None:
            try:
                phase = WorkloadPhase(request.operator.lower())
            except ValueError:
                phase = WorkloadPhase.AUXILIARY
        limit = None
        if phase == WorkloadPhase.PREFILL:
            limit = plan.prefill.batch_size
        elif phase == WorkloadPhase.DECODE:
            limit = plan.decode.batch_size
        if limit is not None and request.batch_size > limit:
            raise ExecutionPlanAdmissionError(
                f"{phase.value} batch {request.batch_size} exceeds active plan limit {limit}"
            )

    @staticmethod
    def _validate_execution_plan(plan: AppleExecutionPlan) -> None:
        if plan.dry_run:
            raise ValueError("dry-run execution plans cannot be activated")
        if plan.estimated_peak_bytes > plan.memory_ceiling_bytes:
            raise ValueError("execution plan exceeds its memory ceiling")
