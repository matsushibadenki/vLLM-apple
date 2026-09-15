"""Atomic bounded resource accounting across CPU, GPU, ANE, and Unified Memory."""
from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import asdict, dataclass

from .execution import ExecutionBackend


def contention_profile_id(soc: str, os_version: str, architecture: str) -> str:
    if any(not isinstance(value, str) or not value for value in (soc, os_version, architecture)):
        raise ValueError("contention profile identity is invalid")
    return hashlib.sha256(f"{soc}\0{os_version}\0{architecture}".encode()).hexdigest()


class DeviceResourceCapacityError(RuntimeError):
    """Raised before dispatch when a heterogeneous resource would be overcommitted."""


@dataclass(frozen=True, slots=True)
class BandwidthContentionEvidence:
    profile_id: str
    first_backend: ExecutionBackend
    second_backend: ExecutionBackend
    sequential_latency_nanoseconds: int
    parallel_latency_nanoseconds: int
    sample_count: int
    outputs_match: bool

    def __post_init__(self) -> None:
        if (
            not self.profile_id or len(self.profile_id) > 128
            or self.first_backend is self.second_backend
            or type(self.sequential_latency_nanoseconds) is not int
            or type(self.parallel_latency_nanoseconds) is not int
            or self.sequential_latency_nanoseconds < 1
            or self.parallel_latency_nanoseconds < 1
            or type(self.sample_count) is not int or not 3 <= self.sample_count <= 1024
            or type(self.outputs_match) is not bool
        ):
            raise ValueError("invalid bandwidth contention evidence")

    @property
    def qualified(self) -> bool:
        return (
            self.outputs_match
            and self.parallel_latency_nanoseconds * 100
            <= self.sequential_latency_nanoseconds * 95
        )


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
        contention_profile_id: str = "runtime",
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
        if not contention_profile_id or len(contention_profile_id) > 128:
            raise ValueError("contention profile ID is invalid")
        self._contention_profile_id = contention_profile_id
        self._capacity = capacities
        self._used = {key: 0 for key in capacities}
        self._reservations: dict[str, DeviceResourceRequest] = {}
        self._qualified_pairs: set[frozenset[ExecutionBackend]] = set()
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
            active_backends = {value.backend for value in self._reservations.values()}
            if any(
                active is not request.backend
                and frozenset((active, request.backend)) not in self._qualified_pairs
                for active in active_backends
            ):
                raise DeviceResourceCapacityError(
                    "device resources unavailable: bandwidth_contention_unqualified"
                )
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

    def transfer(
        self, reservation_id: str, request: DeviceResourceRequest
    ) -> DeviceResourceReservation:
        """Atomically replace a reservation for a fallback backend."""
        demand = {
            "unified_memory_bytes": request.unified_memory_bytes,
            "cpu_threads": request.cpu_threads,
            "gpu_command_queues": request.gpu_command_queues,
            "ane_tasks": request.ane_tasks,
            "bandwidth_slots": request.bandwidth_slots,
        }
        with self._lock:
            previous = self._reservations.get(reservation_id)
            if previous is None:
                raise ValueError("device resource reservation does not exist")
            other_backends = {
                value.backend for identifier, value in self._reservations.items()
                if identifier != reservation_id
            }
            if any(
                active is not request.backend
                and frozenset((active, request.backend)) not in self._qualified_pairs
                for active in other_backends
            ):
                raise DeviceResourceCapacityError(
                    "device resources unavailable: bandwidth_contention_unqualified"
                )
            previous_demand = {
                key: value for key, value in asdict(previous).items() if key != "backend"
            }
            exhausted = tuple(
                key for key, value in demand.items()
                if self._used[key] - previous_demand[key] + value > self._capacity[key]
            )
            if exhausted:
                raise DeviceResourceCapacityError(
                    "device resources unavailable: " + ",".join(exhausted)
                )
            for key, value in demand.items():
                self._used[key] += value - previous_demand[key]
            self._reservations[reservation_id] = request
            return DeviceResourceReservation(reservation_id, request)

    def install_contention_evidence(self, evidence: BandwidthContentionEvidence) -> bool:
        if not isinstance(evidence, BandwidthContentionEvidence):
            raise ValueError("invalid bandwidth contention evidence")
        if evidence.profile_id != self._contention_profile_id:
            raise ValueError("bandwidth contention evidence profile mismatch")
        if not evidence.qualified:
            return False
        with self._lock:
            self._qualified_pairs.add(frozenset((
                evidence.first_backend, evidence.second_backend,
            )))
        return True

    def replace_contention_evidence(
        self, evidence: tuple[BandwidthContentionEvidence, ...]
    ) -> None:
        if (
            not 1 <= len(evidence) <= 16
            or any(
                not isinstance(item, BandwidthContentionEvidence)
                or item.profile_id != self._contention_profile_id
                or not item.qualified
                for item in evidence
            )
        ):
            raise ValueError("invalid replacement contention evidence")
        pairs = {
            frozenset((item.first_backend, item.second_backend)) for item in evidence
        }
        if len(pairs) != len(evidence):
            raise ValueError("duplicate replacement contention evidence")
        with self._lock:
            self._qualified_pairs = pairs

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
                "contention_profile_id": self._contention_profile_id,
                "qualified_contention_pairs": len(self._qualified_pairs),
                "contention_profile_loaded": bool(self._qualified_pairs),
            }
