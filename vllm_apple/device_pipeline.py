"""Bounded parallel execution for contention-qualified heterogeneous stages."""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

from .device_resources import DeviceResourceRequest, UnifiedDeviceResourceLedger
from .execution import ExecutionBackend

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
