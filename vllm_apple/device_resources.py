"""Atomic bounded resource accounting across CPU, GPU, ANE, and Unified Memory."""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass

from .execution import ExecutionBackend


class DeviceResourceCapacityError(RuntimeError):
    """Raised before dispatch when a heterogeneous resource would be overcommitted."""


@dataclass(frozen=True, slots=True)
class DeviceResourceRequest:
    backend: ExecutionBackend
    unified_memory_bytes: int
    cpu_threads: int = 0
    gpu_command_queues: int = 0
    ane_tasks: int = 0
    bandwidth_slots: int = 1

    def __post_init__(self) -> None:
        values = (
            self.unified_memory_bytes, self.cpu_threads, self.gpu_command_queues,
            self.ane_tasks, self.bandwidth_slots,
        )
        if (
            not isinstance(self.backend, ExecutionBackend)
            or any(type(value) is not int or value < 0 for value in values)
            or self.bandwidth_slots < 1
            or self.cpu_threads > 1024
            or self.gpu_command_queues > 64
            or self.ane_tasks > 64
            or self.bandwidth_slots > 64
        ):
            raise ValueError("invalid device resource request")

    @classmethod
    def for_backend(
        cls, backend: ExecutionBackend, unified_memory_bytes: int
    ) -> "DeviceResourceRequest":
        return cls(
            backend,
            unified_memory_bytes,
            cpu_threads=1 if backend is ExecutionBackend.CPU else 0,
            gpu_command_queues=1 if backend in {
                ExecutionBackend.VLLM_METAL,
                ExecutionBackend.NATIVE_MLX,
                ExecutionBackend.NATIVE_METAL,
            } else 0,
            ane_tasks=1 if backend is ExecutionBackend.COREML_DRAFT else 0,
        )


@dataclass(frozen=True, slots=True)
class DeviceResourceReservation:
    reservation_id: str
    request: DeviceResourceRequest


class UnifiedDeviceResourceLedger:
    def __init__(
        self,
        *,
        unified_memory_bytes: int,
        cpu_threads: int,
        gpu_command_queues: int,
        ane_tasks: int,
        bandwidth_slots: int,
    ) -> None:
        capacities = {
            "unified_memory_bytes": unified_memory_bytes,
            "cpu_threads": cpu_threads,
            "gpu_command_queues": gpu_command_queues,
            "ane_tasks": ane_tasks,
            "bandwidth_slots": bandwidth_slots,
        }
        if any(type(value) is not int or value < 0 for value in capacities.values()):
            raise ValueError("invalid device resource capacity")
        if bandwidth_slots < 1 or cpu_threads < 1:
            raise ValueError("CPU and bandwidth capacities must be positive")
        self._capacity = capacities
        self._used = {key: 0 for key in capacities}
        self._reservations: dict[str, DeviceResourceRequest] = {}
        self._lock = threading.Lock()

    def reserve(self, request: DeviceResourceRequest) -> DeviceResourceReservation:
        demand = {
            "unified_memory_bytes": request.unified_memory_bytes,
            "cpu_threads": request.cpu_threads,
            "gpu_command_queues": request.gpu_command_queues,
            "ane_tasks": request.ane_tasks,
            "bandwidth_slots": request.bandwidth_slots,
        }
        with self._lock:
            exhausted = tuple(
                key for key, value in demand.items()
                if self._used[key] + value > self._capacity[key]
            )
            if exhausted:
                raise DeviceResourceCapacityError(
                    "device resources unavailable: " + ",".join(exhausted)
                )
            reservation_id = uuid.uuid4().hex
            for key, value in demand.items():
                self._used[key] += value
            self._reservations[reservation_id] = request
            return DeviceResourceReservation(reservation_id, request)

    def release(self, reservation_id: str) -> bool:
        with self._lock:
            request = self._reservations.pop(reservation_id, None)
            if request is None:
                return False
            for key, value in asdict(request).items():
                if key != "backend":
                    self._used[key] -= value
            return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "capacity": dict(self._capacity),
                "used": dict(self._used),
                "available": {
                    key: self._capacity[key] - self._used[key]
                    for key in self._capacity
                },
                "active_reservations": len(self._reservations),
            }
