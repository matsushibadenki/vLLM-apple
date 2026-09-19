"""Bounded parallel execution for contention-qualified heterogeneous stages."""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Generic, TypeVar

from .device_capability import DeviceCapabilityRegistry, DeviceEligibilityRequest
from .device_resources import DeviceResourceRequest, UnifiedDeviceResourceLedger
from .execution import ExecutionBackend, WorkloadPhase

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class DevicePipelineStage(Generic[_Result]):
    name: str
    backend: ExecutionBackend
    unified_memory_bytes: int
    operation: Callable[[], _Result]

    def __post_init__(self) -> None:
        if (
            not self.name or len(self.name) > 128
            or not isinstance(self.backend, ExecutionBackend)
            or type(self.unified_memory_bytes) is not int
            or self.unified_memory_bytes < 0
            or not callable(self.operation)
        ):
            raise ValueError("invalid device pipeline stage")


@dataclass(frozen=True, slots=True)
class DevicePipelineResult(Generic[_Result]):
    outputs: tuple[_Result, ...]
    backends: tuple[ExecutionBackend, ...]
    elapsed_nanoseconds: int


class DevicePipelineExecutor:
    """Runs only groups admitted atomically by the shared resource ledger."""

    def __init__(self, ledger: UnifiedDeviceResourceLedger) -> None:
        self._ledger = ledger

    def execute(
        self, stages: tuple[DevicePipelineStage[_Result], ...]
    ) -> DevicePipelineResult[_Result]:
        if (
            not 2 <= len(stages) <= 3
            or any(not isinstance(stage, DevicePipelineStage) for stage in stages)
            or len({stage.name for stage in stages}) != len(stages)
            or len({stage.backend for stage in stages}) != len(stages)
        ):
            raise ValueError("pipeline requires two or three unique stages and backends")
        reservations = self._ledger.reserve_many(tuple(
            DeviceResourceRequest.for_backend(stage.backend, stage.unified_memory_bytes)
            for stage in stages
        ))
        started = time.monotonic_ns()
        try:
            with ThreadPoolExecutor(max_workers=len(stages), thread_name_prefix="device-pipeline") as pool:
                futures = tuple(pool.submit(stage.operation) for stage in stages)
                outputs = tuple(future.result() for future in futures)
            return DevicePipelineResult(
                outputs=outputs,
                backends=tuple(stage.backend for stage in stages),
                elapsed_nanoseconds=max(1, time.monotonic_ns() - started),
            )
        finally:
            for reservation in reservations:
                self._ledger.release(reservation.reservation_id)


class ANEAuxiliaryWorkload(str, Enum):
    """Fixed-graph workloads that may be promoted to the ANE."""

    VISION_ENCODER = "vision_encoder"
    AUDIO_ENCODER = "audio_encoder"
    EMBEDDING = "embedding"
    CLASSIFIER = "classifier"


@dataclass(frozen=True, slots=True)
class ANEAuxiliaryRoute:
    workload: ANEAuxiliaryWorkload
    operator: str
    precision: str
    capability_id: str


def require_ane_auxiliary_route(
    registry: DeviceCapabilityRegistry,
    *,
    workload: ANEAuxiliaryWorkload,
    operator: str,
    precision: str,
) -> ANEAuxiliaryRoute:
    """Require exact, probe-passing ANE evidence for one auxiliary graph."""
    if not isinstance(registry, DeviceCapabilityRegistry):
        raise ValueError("invalid device capability registry")
    if not isinstance(workload, ANEAuxiliaryWorkload):
        raise ValueError("invalid ANE auxiliary workload")
    if not isinstance(operator, str) or not operator.startswith(f"{workload.value}@"):
        raise ValueError("ANE auxiliary operator does not match workload")
    decision = registry.decide(DeviceEligibilityRequest(
        operator,
        WorkloadPhase.AUXILIARY,
        precision,
        (ExecutionBackend.COREML_DRAFT,),
    ))
    capability = next(
        (
            item for item in registry.snapshot()
            if item.backend is decision.selected and item.operators == (operator,)
        ),
        None,
    )
    if capability is None or capability.status != "available":
        raise RuntimeError("ANE auxiliary capability evidence is unavailable")
    return ANEAuxiliaryRoute(
        workload, operator, precision, capability.capability_id
    )


@dataclass(frozen=True, slots=True)
class EncoderLLMPipelineResult(Generic[_Result]):
    output: _Result
    encoder_backend: ExecutionBackend
    llm_backend: ExecutionBackend
    capability_id: str
    elapsed_nanoseconds: int


class AsyncEncoderLLMPipeline:
    """Bounded ANE encoder -> GPU LLM dependency pipeline.

    The stages are deliberately sequential: the GPU callback receives the encoder
    output only after ANE completion. Resource reservations therefore do not claim
    unmeasured ANE/GPU overlap, while submission remains asynchronous to the caller.
    """

    _GPU_BACKENDS = frozenset({
        ExecutionBackend.VLLM_METAL,
        ExecutionBackend.NATIVE_MLX,
        ExecutionBackend.NATIVE_METAL,
    })

    def __init__(
        self,
        ledger: UnifiedDeviceResourceLedger,
        capability_registry: DeviceCapabilityRegistry,
        *,
        maximum_pending: int = 8,
    ) -> None:
        if (
            not isinstance(ledger, UnifiedDeviceResourceLedger)
            or not isinstance(capability_registry, DeviceCapabilityRegistry)
            or type(maximum_pending) is not int
            or not 1 <= maximum_pending <= 64
        ):
            raise ValueError("invalid asynchronous encoder pipeline configuration")
        self._ledger = ledger
        self._capability_registry = capability_registry
        self._slots = threading.BoundedSemaphore(maximum_pending)
        self._pool = ThreadPoolExecutor(
            max_workers=min(maximum_pending, 8), thread_name_prefix="encoder-llm-pipeline"
        )
        self._lock = threading.Lock()
        self._closed = False

    def submit(
        self,
        route: ANEAuxiliaryRoute,
        *,
        encoder_memory_bytes: int,
        llm_backend: ExecutionBackend,
        llm_memory_bytes: int,
        encode: Callable[[], object],
        consume: Callable[[object], _Result],
    ) -> Future[EncoderLLMPipelineResult[_Result]]:
        if (
            not isinstance(route, ANEAuxiliaryRoute)
            or llm_backend not in self._GPU_BACKENDS
            or type(encoder_memory_bytes) is not int
            or type(llm_memory_bytes) is not int
            or encoder_memory_bytes < 0
            or llm_memory_bytes < 0
            or not callable(encode)
            or not callable(consume)
        ):
            raise ValueError("invalid encoder LLM pipeline request")
        current_route = require_ane_auxiliary_route(
            self._capability_registry,
            workload=route.workload,
            operator=route.operator,
            precision=route.precision,
        )
        if current_route.capability_id != route.capability_id:
            raise RuntimeError("ANE auxiliary capability evidence is stale")
        with self._lock:
            if self._closed:
                raise RuntimeError("encoder LLM pipeline is closed")
            if not self._slots.acquire(blocking=False):
                raise RuntimeError("encoder LLM pipeline queue is full")
            try:
                future = self._pool.submit(
                    self._execute,
                    route,
                    encoder_memory_bytes,
                    llm_backend,
                    llm_memory_bytes,
                    encode,
                    consume,
                )
            except BaseException:
                self._slots.release()
                raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future

    def _execute(
        self,
        route: ANEAuxiliaryRoute,
        encoder_memory_bytes: int,
        llm_backend: ExecutionBackend,
        llm_memory_bytes: int,
        encode: Callable[[], object],
        consume: Callable[[object], _Result],
    ) -> EncoderLLMPipelineResult[_Result]:
        started = time.monotonic_ns()
        encoder_reservation = self._ledger.reserve(DeviceResourceRequest.for_backend(
            ExecutionBackend.COREML_DRAFT, encoder_memory_bytes
        ))
        try:
            encoded = encode()
        finally:
            self._ledger.release(encoder_reservation.reservation_id)
        llm_reservation = self._ledger.reserve(DeviceResourceRequest.for_backend(
            llm_backend, llm_memory_bytes
        ))
        try:
            output = consume(encoded)
        finally:
            self._ledger.release(llm_reservation.reservation_id)
        return EncoderLLMPipelineResult(
            output,
            ExecutionBackend.COREML_DRAFT,
            llm_backend,
            route.capability_id,
            max(1, time.monotonic_ns() - started),
        )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "AsyncEncoderLLMPipeline":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
