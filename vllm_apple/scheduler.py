from __future__ import annotations

import heapq
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from .device_pipeline import (
    DevicePipelineExecutor,
    DevicePipelineResult,
    DevicePipelineStage,
)
from .device_placement import DevicePlacementPlan
from .device_resources import (
    DeviceResourceCapacityError,
    DeviceResourceRequest,
    UnifiedDeviceResourceLedger,
    contention_profile_id,
)
from .execution import AppleExecutionPlan, ExecutionBackend, WorkloadPhase
from .operator_dispatch import (
    BackendExecutionError,
    OperatorDispatchDecision,
    OperatorDispatcher,
    OperatorDispatchRequest,
    OperatorExecutionResult,
    OperatorFallbackExecutor,
    OperatorFallbackExhaustedError,
)
from .scheduling_observability import SchedulingObservability
from .types import Backend, HardwareInfo, MemoryPressure, PowerMode, Priority, ThermalState

_SafePointResult = TypeVar("_SafePointResult")


class MemoryCapacityError(RuntimeError):
    """Raised when admitting work would exceed the runtime memory budget."""


class ExecutionPlanAdmissionError(RuntimeError):
    """Raised when work violates the active execution plan."""


class MaintenanceInProgressError(RuntimeError):
    """Raised while an exclusive idle maintenance operation owns the scheduler."""


class AdaptiveScheduleCapacityError(RuntimeError):
    """Raised when a new request exceeds the current operating-state policy."""


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
    precision: str | None = None
    dimensions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.estimated_memory_bytes < 0
            or self.batch_size <= 0
            or self.estimated_context_tokens < 0
            or (self.precision is not None and not self.precision)
            or len(self.dimensions) > 8
            or any(value <= 0 for value in self.dimensions)
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
    device_placement_plan_id: str | None = None
    device_resource_reservation_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueuedAdmission:
    token: str
    request: ScheduleRequest
    reservation: Reservation


@dataclass(frozen=True, slots=True)
class AdaptiveSchedulingPolicy:
    pressure: MemoryPressure
    thermal: ThermalState
    power: PowerMode
    level: int
    maximum_active_requests: int
    maximum_batch_size: int

    @classmethod
    def from_inputs(
        cls, pressure: MemoryPressure, thermal: ThermalState, power: PowerMode,
        preference: str = "automatic",
    ) -> "AdaptiveSchedulingPolicy":
        if not (
            isinstance(pressure, MemoryPressure)
            and isinstance(thermal, ThermalState)
            and isinstance(power, PowerMode)
        ):
            raise ValueError("invalid adaptive scheduling inputs")
        if preference not in {"automatic", "low_power", "high_performance"}:
            raise ValueError("invalid scheduling preference")
        level = max(
            {MemoryPressure.CRITICAL: 2, MemoryPressure.WARNING: 1}.get(pressure, 0),
            {ThermalState.CRITICAL: 2, ThermalState.SERIOUS: 1}.get(thermal, 0),
            1 if preference == "low_power" or (
                preference == "automatic" and power is PowerMode.LOW_POWER
            ) else 0,
        )
        return cls(
            pressure, thermal, power, level,
            (1024, 2, 1)[level], (2_147_483_647, 4, 1)[level],
        )


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
        self._enqueued_ns: dict[str, int] = {}
        self._claimed: set[str] = set()
        self._claim_entries: dict[str, tuple[int, int, ScheduleRequest]] = {}
        self._condition = threading.Condition()

    def enqueue(self, request: ScheduleRequest) -> str:
        with self._condition:
            if len(self._requests) + len(self._claimed) >= self._maximum:
                raise ScheduleQueueFullError("scheduler queue is full")
            token = uuid.uuid4().hex
            sequence = self._sequence
            self._sequence += 1
            self._requests[token] = request
            self._enqueued_ns[token] = time.monotonic_ns()
            heapq.heappush(self._heap, (self._RANK[request.priority], sequence, token))
            self._condition.notify()
            return token

    def cancel(self, token: str) -> bool:
        with self._condition:
            if self._requests.pop(token, None) is not None:
                self._enqueued_ns.pop(token, None)
                self._compact_cancelled_locked()
                return True
            if token in self._claimed:
                self._claimed.remove(token)
                self._claim_entries.pop(token, None)
                self._enqueued_ns.pop(token, None)
                return True
            return False

    def _compact_cancelled_locked(self) -> None:
        """Bound tombstones even when no consumer drains the priority heap.

        Preserve the original rank/sequence so restored claims and FIFO ties
        remain ordered. Batch cleanup amortizes heap rebuilding across cancels.
        The caller holds the condition lock, including during heap replacement.
        """
        if not self._requests:
            self._heap.clear()
        elif len(self._heap) > 2 * len(self._requests) + 64:
            self._heap = [entry for entry in self._heap if entry[2] in self._requests]
            heapq.heapify(self._heap)

    def finish_claim(self, token: str) -> bool:
        with self._condition:
            if token not in self._claimed:
                return False
            self._claimed.remove(token)
            self._claim_entries.pop(token, None)
            self._enqueued_ns.pop(token, None)
            return True

    def claimed_wait_nanoseconds(self, token: str) -> int:
        with self._condition:
            started = self._enqueued_ns.get(token)
            return max(0, time.monotonic_ns() - started) if started is not None else 0

    def restore_claim(self, token: str) -> bool:
        """Return a capacity-blocked claim to its original priority/FIFO position."""
        with self._condition:
            entry = self._claim_entries.pop(token, None)
            if entry is None or token not in self._claimed:
                return False
            rank, sequence, request = entry
            self._claimed.remove(token)
            self._requests[token] = request
            heapq.heappush(self._heap, (rank, sequence, token))
            self._condition.notify()
            return True

    def dequeue(self, timeout: float | None = None) -> tuple[str, ScheduleRequest] | None:
        if timeout is not None and timeout < 0:
            raise ValueError("queue timeout must not be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                while self._heap:
                    rank, sequence, token = heapq.heappop(self._heap)
                    request = self._requests.pop(token, None)
                    if request is not None:
                        self._claimed.add(token)
                        self._claim_entries[token] = (rank, sequence, request)
                        return token, request
                if timeout == 0:
                    return None
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def peek(self) -> tuple[str, ScheduleRequest] | None:
        """Observe only the highest-priority pending request, without reordering it."""
        with self._condition:
            while self._heap and self._heap[0][2] not in self._requests:
                heapq.heappop(self._heap)
            if not self._heap:
                return None
            token = self._heap[0][2]
            return token, self._requests[token]

    def claim_head(self, token: str) -> tuple[str, ScheduleRequest] | None:
        """Claim only if the observed request is still the priority head."""
        with self._condition:
            while self._heap and self._heap[0][2] not in self._requests:
                heapq.heappop(self._heap)
            if not self._heap or self._heap[0][2] != token:
                return None
            rank, sequence, _ = heapq.heappop(self._heap)
            request = self._requests.pop(token)
            self._claimed.add(token)
            self._claim_entries[token] = (rank, sequence, request)
            return token, request

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
        self.device_resources = UnifiedDeviceResourceLedger(
            unified_memory_bytes=memory_capacity_bytes,
            cpu_threads=max(1, hardware.logical_cpu_count),
            gpu_command_queues=4 if hardware.is_apple_silicon else 0,
            ane_tasks=2 if hardware.is_apple_silicon else 0,
            bandwidth_slots=4 if hardware.is_apple_silicon else 1,
            contention_profile_id=contention_profile_id(
                hardware.soc, hardware.os_version, hardware.architecture
            ),
        )
        self._policy_lock = threading.RLock()
        self._active_plan: AppleExecutionPlan | None = None
        self._pending_plan: AppleExecutionPlan | None = None
        self._active_device_placement_plan: DevicePlacementPlan | None = None
        self._pending_device_placement_plan: DevicePlacementPlan | None = None
        self._operator_dispatcher = operator_dispatcher
        self._maintenance_owner: str | None = None
        self._queue = PriorityScheduleQueue(maximum_queued_requests)
        self._queued_active: dict[str, Reservation] = {}
        self._device_pipeline = DevicePipelineExecutor(self.device_resources)
        self._observability = SchedulingObservability()
        self._scheduling_preference = "automatic"
        self._adaptive_policy = AdaptiveSchedulingPolicy.from_inputs(
            hardware.memory.pressure, hardware.thermal_state, hardware.power_mode,
            self._scheduling_preference,
        )
        self._pending_adaptive_policy: AdaptiveSchedulingPolicy | None = None

    def update_adaptive_inputs(
        self,
        *,
        pressure: MemoryPressure | None = None,
        thermal: ThermalState | None = None,
        power: PowerMode | None = None,
    ) -> str:
        """Restrict new work immediately; relax only after active work drains."""
        if pressure is not None and not isinstance(pressure, MemoryPressure):
            raise ValueError("invalid memory pressure")
        if thermal is not None and not isinstance(thermal, ThermalState):
            raise ValueError("invalid thermal state")
        if power is not None and not isinstance(power, PowerMode):
            raise ValueError("invalid power mode")
        with self._policy_lock:
            current = self._adaptive_policy
            latest = self._pending_adaptive_policy or current
            if pressure is MemoryPressure.UNKNOWN:
                pressure = latest.pressure
            if thermal is ThermalState.UNKNOWN:
                thermal = latest.thermal
            if power is PowerMode.UNKNOWN:
                power = latest.power
            proposed = AdaptiveSchedulingPolicy.from_inputs(
                pressure if pressure is not None else latest.pressure,
                thermal if thermal is not None else latest.thermal,
                power if power is not None else latest.power,
                self._scheduling_preference,
            )
            if proposed == current:
                self._pending_adaptive_policy = None
                self._observability.adaptive_transition("ignored")
                return "ignored"
            if proposed.level < current.level and self._has_active_work():
                self._pending_adaptive_policy = proposed
                self._observability.adaptive_transition("deferred")
                return "deferred"
            self._adaptive_policy = proposed
            self._pending_adaptive_policy = None
            self._observability.adaptive_transition("applied")
            return "applied"

    def set_scheduling_preference(self, preference: str) -> str:
        if type(preference) is not str or preference not in {
            "automatic", "low_power", "high_performance"
        }:
            raise ValueError("invalid scheduling preference")
        with self._policy_lock:
            self._scheduling_preference = preference
            return self.update_adaptive_inputs()

    def apply_pending_adaptive_policy(self) -> str | None:
        with self._policy_lock:
            pending = self._pending_adaptive_policy
            if pending is None:
                return None
            if self._has_active_work():
                return "deferred"
            self._adaptive_policy = pending
            self._pending_adaptive_policy = None
            self._observability.adaptive_transition("applied")
            return "applied"

    def scheduling_observability_snapshot(self) -> dict[str, object]:
        return {
            **self._observability.snapshot(),
            "adaptive_policy": self.adaptive_scheduling_snapshot(),
        }

    def _has_active_work(self) -> bool:
        return bool(
            self.memory.snapshot()["active_reservations"]
            or self.device_resources.snapshot()["active_reservations"]
        )

    def adaptive_scheduling_snapshot(self) -> dict[str, str | int | None]:
        with self._policy_lock:
            active = self._adaptive_policy
            return {
                "level": active.level,
                "maximum_active_requests": active.maximum_active_requests,
                "maximum_batch_size": active.maximum_batch_size,
                "pressure": active.pressure.value,
                "thermal": active.thermal.value,
                "power": active.power.value,
                "preference": self._scheduling_preference,
                "pending_level": (
                    self._pending_adaptive_policy.level
                    if self._pending_adaptive_policy is not None else None
                ),
            }

    def execute_device_pipeline(
        self, stages: tuple[DevicePipelineStage[_SafePointResult], ...]
    ) -> DevicePipelineResult[_SafePointResult]:
        """Execute a bounded pipeline only when its device pairs are qualified."""
        with self._policy_lock:
            if self._adaptive_policy.level > 0:
                raise AdaptiveScheduleCapacityError(
                    "parallel device pipeline disabled under adaptive pressure"
                )
        try:
            return self._device_pipeline.execute(stages)
        except DeviceResourceCapacityError as error:
            if "contention_unqualified" in str(error):
                self._observability.contention_rejection()
            raise
        finally:
            self.apply_pending_adaptive_policy()

    def choose_backend(self, request: ScheduleRequest) -> Backend:
        decision = self.dispatch_decision(request)
        return {
            ExecutionBackend.CPU: Backend.CPU,
            ExecutionBackend.VLLM_METAL: Backend.METAL,
            ExecutionBackend.NATIVE_MLX: Backend.MLX_GPU,
            ExecutionBackend.NATIVE_METAL: Backend.METAL,
            ExecutionBackend.COREML_DRAFT: Backend.COREML,
        }.get(decision.selected, Backend.CPU)

    def dispatch_decision(self, request: ScheduleRequest) -> OperatorDispatchDecision:
        operator = request.operator.lower()
        candidates = self._dispatch_candidates(operator, request.batch_size)
        placement = self._matching_device_placement(request)
        if placement is not None:
            candidates = (placement.backend,) + tuple(
                backend for backend in candidates if backend is not placement.backend
            )
        if self._adaptive_policy.level == 2:
            candidates = (ExecutionBackend.CPU,)
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
        validate: Callable[[_SafePointResult, ExecutionBackend], bool] | None = None,
        reservation: Reservation | None = None,
    ) -> OperatorExecutionResult[_SafePointResult]:
        def prepare(backend: ExecutionBackend) -> None:
            if reservation is None:
                return
            resource_id = reservation.device_resource_reservation_id
            if resource_id is None:
                raise BackendExecutionError("resource_reservation_missing", retryable=False)
            try:
                self.device_resources.transfer(
                    resource_id,
                    DeviceResourceRequest.for_backend(backend, reservation.bytes),
                )
            except DeviceResourceCapacityError as error:
                raise BackendExecutionError(
                    "fallback_resource_unavailable", retryable=True
                ) from error
        try:
            result = OperatorFallbackExecutor().execute(
                self.dispatch_decision(request), operation, validate, prepare
            )
        except OperatorFallbackExhaustedError as error:
            self._observability.fallback(len(error.attempts), exhausted=True)
            raise
        self._observability.fallback(
            sum(attempt.status == "failed" for attempt in result.attempts),
            exhausted=False,
        )
        return result

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
            decision = self.dispatch_decision(request)
            return self._admit_selected(request, decision.selected)

    def _admit_selected(
        self,
        request: ScheduleRequest,
        execution_backend: ExecutionBackend,
        *,
        steal_from: ExecutionBackend | None = None,
    ) -> Reservation:
        with self._policy_lock:
            if self._maintenance_owner is not None:
                raise MaintenanceInProgressError(
                    f"scheduler maintenance is active: {self._maintenance_owner}"
                )
            self._validate_plan_admission(request)
            policy = self._adaptive_policy
            if request.batch_size > policy.maximum_batch_size:
                raise ExecutionPlanAdmissionError(
                    "request batch exceeds adaptive scheduling limit"
                )
            if (
                self.memory.snapshot()["active_reservations"]
                >= policy.maximum_active_requests
            ):
                raise AdaptiveScheduleCapacityError(
                    "adaptive scheduling concurrency limit reached"
                )
            backend = {
                ExecutionBackend.CPU: Backend.CPU,
                ExecutionBackend.VLLM_METAL: Backend.METAL,
                ExecutionBackend.NATIVE_MLX: Backend.MLX_GPU,
                ExecutionBackend.NATIVE_METAL: Backend.METAL,
                ExecutionBackend.COREML_DRAFT: Backend.COREML,
            }[execution_backend]
            reservation = self.memory.reserve(request, backend)
            try:
                device_reservation = self.device_resources.reserve(
                    DeviceResourceRequest.for_backend(
                        execution_backend, request.estimated_memory_bytes
                    ),
                    steal_from=steal_from,
                )
            except BaseException as error:
                self.memory.release(reservation.reservation_id)
                if isinstance(error, DeviceResourceCapacityError) and (
                    "contention_unqualified" in str(error)
                    or "work_steal_unqualified_or_busy" in str(error)
                ):
                    self._observability.contention_rejection()
                raise
            self._observability.assignment(execution_backend)
            if self._active_plan is None and self._active_device_placement_plan is None:
                return Reservation(
                    reservation.reservation_id, reservation.bytes, reservation.backend,
                    reservation.priority, reservation.created_at_monotonic,
                    device_resource_reservation_id=device_reservation.reservation_id,
                )
            return Reservation(
                reservation_id=reservation.reservation_id,
                bytes=reservation.bytes,
                backend=reservation.backend,
                priority=reservation.priority,
                created_at_monotonic=reservation.created_at_monotonic,
                execution_plan_id=(
                    self._active_plan.plan_id if self._active_plan is not None else None
                ),
                device_placement_plan_id=(
                    self._active_device_placement_plan.plan_id
                    if self._active_device_placement_plan is not None else None
                ),
                device_resource_reservation_id=device_reservation.reservation_id,
            )

    def complete(self, reservation: Reservation) -> None:
        with self._policy_lock:
            self.memory.release(reservation.reservation_id)
            if reservation.device_resource_reservation_id is not None:
                self.device_resources.release(
                    reservation.device_resource_reservation_id
                )
            self.apply_pending_adaptive_policy()

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
            wait_nanoseconds = self._queue.claimed_wait_nanoseconds(token)
            if not self._queue.finish_claim(token):
                self._queued_active.pop(token, None)
                self.memory.release(reservation.reservation_id)
                if reservation.device_resource_reservation_id is not None:
                    self.device_resources.release(
                        reservation.device_resource_reservation_id
                    )
                return None
        self._observability.queue_wait(wait_nanoseconds)
        return QueuedAdmission(token, request, reservation)

    def steal_next(
        self, target_backend: ExecutionBackend
    ) -> QueuedAdmission | None:
        """Steal at most the queue head onto a probed, pair-qualified idle backend.

        No lower-priority request is bypassed. Capacity failure restores the same
        token and FIFO position, so another worker may still admit it normally.
        """
        if not isinstance(target_backend, ExecutionBackend):
            raise ValueError("invalid work-stealing backend")
        with self._policy_lock:
            if (
                self._maintenance_owner is not None
                or self._operator_dispatcher is None
                or self._adaptive_policy.level > 0
            ):
                return None
            head = self._queue.peek()
            if head is None:
                return None
            token, request = head
            decision = self.dispatch_decision(request)
            if (
                target_backend is decision.selected
                or target_backend not in decision.fallback_chain
                or not self.device_resources.can_steal(
                    decision.selected, target_backend
                )
            ):
                return None
            if self._queue.claim_head(token) is None:
                return None
            try:
                reservation = self._admit_selected(
                    request, target_backend, steal_from=decision.selected
                )
            except (MemoryCapacityError, DeviceResourceCapacityError):
                self._queue.restore_claim(token)
                return None
            except BaseException as error:
                self._queue.finish_claim(token)
                if isinstance(error, Exception):
                    raise QueuedAdmissionError(token, request, error) from error
                raise
            self._queued_active[token] = reservation
            wait_nanoseconds = self._queue.claimed_wait_nanoseconds(token)
            if not self._queue.finish_claim(token):
                self._queued_active.pop(token, None)
                self.complete(reservation)
                return None
            self._observability.queue_wait(wait_nanoseconds)
            self._observability.steal()
            return QueuedAdmission(token, request, reservation)

    def complete_queued(self, token: str) -> bool:
        with self._policy_lock:
            reservation = self._queued_active.pop(token, None)
            if reservation is None:
                return False
            self.memory.release(reservation.reservation_id)
            if reservation.device_resource_reservation_id is not None:
                self.device_resources.release(
                    reservation.device_resource_reservation_id
                )
            self.apply_pending_adaptive_policy()
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

    def request_device_placement_plan(
        self, plan: DevicePlacementPlan
    ) -> PlanApplicationDecision:
        if not isinstance(plan, DevicePlacementPlan):
            raise ValueError("invalid device placement plan")
        if int(time.time()) >= plan.valid_until_unix_seconds:
            raise ValueError("expired device placement plans cannot be activated")
        with self._policy_lock:
            if self._operator_dispatcher is None:
                raise RuntimeError("device placement requires a probe-gated dispatcher")
            registry = self._operator_dispatcher.registry
            if (
                plan.hardware_fingerprint != registry.hardware_fingerprint
                or plan.environment_fingerprint != registry.environment_fingerprint
            ):
                raise ValueError("device placement plan does not match dispatcher profile")
            current_id = (
                self._active_device_placement_plan.plan_id
                if self._active_device_placement_plan else None
            )
            if current_id == plan.plan_id:
                self._pending_device_placement_plan = None
                return PlanApplicationDecision(plan.plan_id, "ignored", current_id)
            if self.memory.snapshot()["active_reservations"]:
                replaced = (
                    self._pending_device_placement_plan.plan_id
                    if self._pending_device_placement_plan else None
                )
                self._pending_device_placement_plan = plan
                return PlanApplicationDecision(plan.plan_id, "deferred", replaced)
            self._active_device_placement_plan = plan
            self._pending_device_placement_plan = None
            return PlanApplicationDecision(plan.plan_id, "applied", current_id)

    def apply_pending_device_placement_plan(self) -> PlanApplicationDecision | None:
        with self._policy_lock:
            pending = self._pending_device_placement_plan
            if pending is None:
                return None
            if self.memory.snapshot()["active_reservations"]:
                return PlanApplicationDecision(pending.plan_id, "deferred")
            current_id = (
                self._active_device_placement_plan.plan_id
                if self._active_device_placement_plan else None
            )
            self._active_device_placement_plan = pending
            self._pending_device_placement_plan = None
            return PlanApplicationDecision(pending.plan_id, "applied", current_id)

    def device_placement_snapshot(self) -> dict[str, object]:
        with self._policy_lock:
            active = self._active_device_placement_plan
            pending = self._pending_device_placement_plan
            return {
                "enabled": active is not None,
                "active_plan_id": active.plan_id if active else None,
                "pending_plan_id": pending.plan_id if pending else None,
                "placement_count": len(active.placements) if active else 0,
                "valid_until_unix_seconds": (
                    active.valid_until_unix_seconds if active else None
                ),
                "placements": [] if active is None else [
                    {
                        "operator": value.operator,
                        "phase": value.phase.value,
                        "precision": value.precision,
                        "dimensions": list(value.dimensions),
                        "batch_size": value.batch_size,
                        "backend": value.backend.value,
                        "improvement_ratio": value.improvement_ratio,
                    }
                    for value in active.placements
                ],
            }

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

    def _matching_device_placement(self, request: ScheduleRequest):
        plan = self._active_device_placement_plan
        if plan is None or request.phase is None or request.precision is None:
            return None
        operator = request.operator.lower()
        return next(
            (
                placement
                for placement in plan.placements
                if placement.operator == operator
                and placement.phase is request.phase
                and placement.precision == request.precision
                and placement.dimensions == request.dimensions
                and placement.batch_size == request.batch_size
            ),
            None,
        )

    @staticmethod
    def _validate_execution_plan(plan: AppleExecutionPlan) -> None:
        if plan.dry_run:
            raise ValueError("dry-run execution plans cannot be activated")
        if plan.estimated_peak_bytes > plan.memory_ceiling_bytes:
            raise ValueError("execution plan exceeds its memory ceiling")
