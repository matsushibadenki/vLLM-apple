from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from time import monotonic

from .types import Backend, HardwareInfo, Priority


class MemoryCapacityError(RuntimeError):
    """Raised when admitting work would exceed the runtime memory budget."""


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    operator: str
    estimated_memory_bytes: int
    priority: Priority = Priority.NORMAL
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.estimated_memory_bytes < 0 or self.batch_size <= 0:
            raise ValueError("invalid schedule request")


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_id: str
    bytes: int
    backend: Backend
    priority: Priority
    created_at_monotonic: float


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

    def __init__(self, hardware: HardwareInfo, memory_capacity_bytes: int) -> None:
        self.hardware = hardware
        self.memory = MemoryAdmissionController(memory_capacity_bytes)

    def choose_backend(self, request: ScheduleRequest) -> Backend:
        operator = request.operator.lower()
        if operator in self._CPU_OPERATORS:
            return Backend.CPU
        if self.hardware.is_apple_silicon and operator in self._METAL_OPERATORS:
            return Backend.METAL
        if self.hardware.is_apple_silicon and operator in self._GPU_OPERATORS:
            # Tiny single-item GEMV often loses to GPU launch overhead. The profiler
            # will replace this threshold in Phase 2.
            if operator in {"gemv", "matmul"} and request.batch_size == 1:
                return Backend.CPU
            return Backend.MLX_GPU
        return Backend.CPU

    def admit(self, request: ScheduleRequest) -> Reservation:
        return self.memory.reserve(request, self.choose_backend(request))

    def complete(self, reservation: Reservation) -> None:
        self.memory.release(reservation.reservation_id)
