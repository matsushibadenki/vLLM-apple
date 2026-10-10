"""Backend-owned MoE execution retaining leases until synchronous completion.

The backend must materialize its result before returning. Lazy GPU results are
not sufficient: completion and resource lifetime remain backend responsibilities.
"""
from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Callable, TypeVar

from .expert_residency import ExpertKey, ExpertResidencyManager, ExpertResource
from .expert_selection_telemetry import ExpertSelectionSample, ExpertSelectionTelemetry
from .expert_timing import measure

_Result = TypeVar("_Result")


class ResidentExpertExecutor:
    """Connect authoritative router selections to residency and phase telemetry."""

    def __init__(self, manager: ExpertResidencyManager, *, maximum_samples: int = 4096,
                 timings=None):
        self.manager = manager
        self.timings = timings
        self.telemetry = {
            phase: ExpertSelectionTelemetry(maximum_samples)
            for phase in ("prefill", "decode")
        }

    def execute(
        self, *, phase: str, layer: int, selected_experts: tuple[int, ...],
        routing_weights: tuple[float, ...],
        consume: Callable[[tuple[ExpertResource, ...], tuple[float, ...]], _Result],
    ) -> _Result:
        if phase not in self.telemetry or not callable(consume):
            raise ValueError("invalid expert execution phase or consumer")
        # Validate the full router decision before loading any resources.
        ExpertSelectionSample(layer, selected_experts, routing_weights, 1, ())
        started = time.perf_counter_ns()
        hits = []
        def release(lease):
            with measure(self.timings, "release"):
                lease.release()
        with ExitStack() as stack:
            resources = []
            for expert in selected_experts:
                with measure(self.timings, "acquire"):
                    lease = self.manager.acquire(ExpertKey(layer, expert))
                stack.callback(release, lease)
                resources.append(lease.resource)
                if lease.cache_hit:
                    hits.append(expert)
            result = consume(tuple(resources), routing_weights)
        self.telemetry[phase].record(ExpertSelectionSample(
            layer, selected_experts, routing_weights,
            min(3600 * 1_000_000_000, max(1, time.perf_counter_ns() - started)),
            tuple(hits),
        ))
        return result
