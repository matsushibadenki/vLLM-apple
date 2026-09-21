"""Bounded execution of a validated Multi-Mac plan through explicit node clients."""
from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .multi_mac import FabricStage, FabricTransfer, MultiMacExecutionPlan

MAX_MULTI_MAC_RESULT_BYTES = 256 * 1024 * 1024


class MultiMacNodeClient(Protocol):
    def execute(self, stage: FabricStage, inputs: tuple[bytes, ...]) -> bytes: ...


class MultiMacTransferClient(Protocol):
    def transfer(self, planned: FabricTransfer, payload: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class MultiMacStageResult:
    stage_id: str
    node_id: str
    output_bytes: int
    output_sha256: str
    elapsed_nanoseconds: int


@dataclass(frozen=True, slots=True)
class MultiMacExecutionReport:
    plan_id: str
    results: tuple[MultiMacStageResult, ...]
    peak_result_bytes: int
    elapsed_nanoseconds: int
    passed: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "results": [
                {
                    "stage_id": value.stage_id,
                    "node_id": value.node_id,
                    "output_bytes": value.output_bytes,
                    "output_sha256": value.output_sha256,
                    "elapsed_nanoseconds": value.elapsed_nanoseconds,
                }
                for value in self.results
            ],
            "peak_result_bytes": self.peak_result_bytes,
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "passed": self.passed,
            "stores_payload": False,
        }


class MultiMacExecutionCancelled(RuntimeError):
    pass


class MultiMacExecutionCoordinator:
    def __init__(
        self,
        plan: MultiMacExecutionPlan,
        stages: tuple[FabricStage, ...],
        node_clients: dict[str, MultiMacNodeClient],
        transfer_client: MultiMacTransferClient,
        *,
        maximum_concurrency: int = 8,
        maximum_result_bytes: int = MAX_MULTI_MAC_RESULT_BYTES,
    ) -> None:
        if (
            not isinstance(plan, MultiMacExecutionPlan)
            or not 1 <= maximum_concurrency <= 64
            or not 1 <= maximum_result_bytes <= MAX_MULTI_MAC_RESULT_BYTES
        ):
            raise ValueError("invalid Multi-Mac execution limits")
        stage_by_id = {stage.stage_id: stage for stage in stages}
        placement_by_id = {value.stage_id: value for value in plan.placements}
        if (
            not stages
            or len(stage_by_id) != len(stages)
            or set(stage_by_id) != set(placement_by_id)
            or any(
                placement.reserved_bytes != stage_by_id[stage_id].resident_bytes
                for stage_id, placement in placement_by_id.items()
            )
            or any(placement.node_id not in node_clients for placement in plan.placements)
            or sum(stage.output_bytes for stage in stages) > maximum_result_bytes
        ):
            raise ValueError("execution inputs do not match the Multi-Mac plan")
        transfer_by_edge = {
            (value.source_stage, value.destination_stage): value
            for value in plan.transfers
        }
        if len(transfer_by_edge) != len(plan.transfers):
            raise ValueError("duplicate Multi-Mac transfer edge")
        for stage in stages:
            for dependency in stage.dependencies:
                source = placement_by_id.get(dependency)
                if source is None:
                    raise ValueError("execution stage dependency is missing")
                cross_node = source.node_id != placement_by_id[stage.stage_id].node_id
                if cross_node != ((dependency, stage.stage_id) in transfer_by_edge):
                    raise ValueError("execution transfer edges do not match placements")
        self.plan = plan
        self._stages = stage_by_id
        self._placements = placement_by_id
        self._transfers = transfer_by_edge
        self._node_clients = dict(node_clients)
        self._transfer_client = transfer_client
        self._maximum_concurrency = maximum_concurrency
        self._maximum_result_bytes = maximum_result_bytes

    def execute(
        self,
        *,
        cancellation: threading.Event | None = None,
        deadline: float | None = None,
    ) -> MultiMacExecutionReport:
        if deadline is not None and (
            not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
        ):
            raise ValueError("invalid Multi-Mac execution deadline")
        cancellation = cancellation or threading.Event()
        started = time.monotonic_ns()
        pending = set(self._stages)
        outputs: dict[str, bytes] = {}
        results: dict[str, MultiMacStageResult] = {}
        peak = 0
        with ThreadPoolExecutor(max_workers=self._maximum_concurrency) as executor:
            while pending:
                self._raise_if_cancelled(cancellation, deadline)
                ready = sorted(
                    stage_id
                    for stage_id in pending
                    if all(dependency in outputs for dependency in self._stages[stage_id].dependencies)
                )
                if not ready:
                    raise RuntimeError("Multi-Mac execution graph cannot make progress")
                futures = {
                    stage_id: executor.submit(
                        self._execute_stage,
                        self._stages[stage_id],
                        outputs,
                        cancellation,
                        deadline,
                    )
                    for stage_id in ready
                }
                wave = {}
                try:
                    for stage_id in ready:
                        wave[stage_id] = futures[stage_id].result()
                except BaseException:
                    cancellation.set()
                    for future in futures.values():
                        future.cancel()
                    raise
                self._raise_if_cancelled(cancellation, deadline)
                for stage_id in ready:
                    payload, result = wave[stage_id]
                    outputs[stage_id] = payload
                    results[stage_id] = result
                    pending.remove(stage_id)
                resident = sum(len(value) for value in outputs.values())
                peak = max(peak, resident)
                if resident > self._maximum_result_bytes:
                    raise MemoryError("Multi-Mac result memory exceeded its bound")
        elapsed = time.monotonic_ns() - started
        return MultiMacExecutionReport(
            self.plan.plan_id,
            tuple(results[stage_id] for stage_id in sorted(results)),
            peak,
            elapsed,
        )

    def _execute_stage(
        self,
        stage: FabricStage,
        outputs: dict[str, bytes],
        cancellation: threading.Event,
        deadline: float | None,
    ) -> tuple[bytes, MultiMacStageResult]:
        self._raise_if_cancelled(cancellation, deadline)
        placement = self._placements[stage.stage_id]
        inputs = []
        for dependency in stage.dependencies:
            payload = outputs[dependency]
            transfer = self._transfers.get((dependency, stage.stage_id))
            if transfer is not None:
                moved = self._transfer_client.transfer(transfer, payload)
                if (
                    not isinstance(moved, bytes)
                    or len(moved) != len(payload)
                    or hashlib.sha256(moved).digest() != hashlib.sha256(payload).digest()
                ):
                    raise ValueError("Multi-Mac transfer changed its payload")
                payload = moved
            inputs.append(payload)
        self._raise_if_cancelled(cancellation, deadline)
        started = time.monotonic_ns()
        output = self._node_clients[placement.node_id].execute(stage, tuple(inputs))
        elapsed = time.monotonic_ns() - started
        self._raise_if_cancelled(cancellation, deadline)
        if not isinstance(output, bytes) or len(output) != stage.output_bytes:
            raise ValueError("Multi-Mac stage output does not match its contract")
        return output, MultiMacStageResult(
            stage.stage_id,
            placement.node_id,
            len(output),
            hashlib.sha256(output).hexdigest(),
            elapsed,
        )

    @staticmethod
    def _raise_if_cancelled(cancellation: threading.Event, deadline: float | None) -> None:
        if cancellation.is_set():
            raise MultiMacExecutionCancelled("Multi-Mac execution was cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise MultiMacExecutionCancelled("Multi-Mac execution deadline expired")
