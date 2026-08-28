from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .context import ContextPolicy, recommend_state_context
from .types import MemoryInfo, MemoryPressure, StateMemorySpec


EXECUTION_SCHEMA_VERSION = 1


class WorkloadPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    AUXILIARY = "auxiliary"


class ExecutionBackend(str, Enum):
    VLLM_METAL = "vllm_metal"
    NATIVE_MLX = "native_mlx"
    NATIVE_METAL = "native_metal"
    COREML_DRAFT = "coreml_draft"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class AppleChipProfile:
    profile_version: int
    hardware_fingerprint: str
    soc: str
    total_memory_bytes: int
    backends: tuple[ExecutionBackend, ...]
    precisions: tuple[str, ...] = ("fp16",)
    platform: str = "unknown"
    architecture: str = "unknown"
    os_version: str = "unknown"
    gpu_core_count: int | None = None
    capability_issues: tuple[str, ...] = ()
    measured_memory_bandwidth_bytes_per_second: int | None = None
    metal_launch_latency_nanoseconds: int | None = None

    def __post_init__(self) -> None:
        if self.profile_version <= 0 or not self.hardware_fingerprint or not self.soc:
            raise ValueError("invalid chip profile identity")
        if self.total_memory_bytes <= 0 or not self.backends:
            raise ValueError("chip profile requires memory and at least one backend")
        if not self.precisions:
            raise ValueError("chip profile requires at least one precision")
        if self.gpu_core_count is not None and self.gpu_core_count <= 0:
            raise ValueError("gpu_core_count must be positive")
        inference_backends = {
            ExecutionBackend.VLLM_METAL,
            ExecutionBackend.NATIVE_MLX,
            ExecutionBackend.NATIVE_METAL,
            ExecutionBackend.CPU,
        }
        if not any(backend in inference_backends for backend in self.backends):
            raise ValueError("chip profile requires a main-model inference backend")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["backends"] = [backend.value for backend in self.backends]
        result["precisions"] = list(self.precisions)
        result["capability_issues"] = list(self.capability_issues)
        return result


@dataclass(frozen=True, slots=True)
class PhaseExecutionPlan:
    phase: WorkloadPhase
    backend: ExecutionBackend
    batch_size: int
    state_precision: str


@dataclass(frozen=True, slots=True)
class AppleExecutionPlan:
    schema_version: int
    plan_id: str
    model_id: str
    hardware_fingerprint: str
    context_tokens: int
    memory_ceiling_bytes: int
    estimated_peak_bytes: int
    prefill: PhaseExecutionPlan
    decode: PhaseExecutionPlan
    fallback_chain: tuple[ExecutionBackend, ...]
    decision_reasons: tuple[str, ...]
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prefill"]["phase"] = self.prefill.phase.value
        result["prefill"]["backend"] = self.prefill.backend.value
        result["decode"]["phase"] = self.decode.phase.value
        result["decode"]["backend"] = self.decode.backend.value
        result["fallback_chain"] = [backend.value for backend in self.fallback_chain]
        result["decision_reasons"] = list(self.decision_reasons)
        return result


class AppleExecutionPlanner:
    """Conservative, deterministic planner; it performs no implicit benchmarks."""

    def __init__(self, context_policy: ContextPolicy = ContextPolicy()) -> None:
        self._context_policy = context_policy

    def plan(
        self,
        *,
        model: StateMemorySpec,
        memory: MemoryInfo,
        chip: AppleChipProfile,
        requested_context_tokens: int | None = None,
        dry_run: bool = True,
    ) -> AppleExecutionPlan:
        if memory.total_bytes != chip.total_memory_bytes:
            raise ValueError("memory and chip profile totals must match")
        recommendation = recommend_state_context(memory, model, self._context_policy)
        balanced = next(tier for tier in recommendation.tiers if tier.name == "balanced")
        context_tokens = balanced.max_tokens
        reasons = [f"context:{recommendation.limiting_factor}"]
        if requested_context_tokens is not None:
            if requested_context_tokens <= 0:
                raise ValueError("requested_context_tokens must be positive")
            context_tokens = min(context_tokens, requested_context_tokens)
            reasons.append(
                "context:requested" if context_tokens == requested_context_tokens else "context:clamped"
            )

        backend = self._preferred_backend(chip.backends)
        precision = self._state_precision(memory.pressure, chip.precisions)
        pressure_batch = 1 if memory.pressure in {MemoryPressure.WARNING, MemoryPressure.CRITICAL} else 4
        prefill_batch = pressure_batch if backend is not ExecutionBackend.CPU else 1
        estimated_peak = model.total_bytes(context_tokens)
        ceiling = recommendation.allocatable_bytes
        while context_tokens > 0 and estimated_peak > ceiling:
            context_tokens = max(0, context_tokens - self._context_policy.token_block_size)
            estimated_peak = model.total_bytes(context_tokens)
        if estimated_peak > ceiling:
            raise ValueError("model fixed memory exceeds the safe memory ceiling")

        fallback = tuple(candidate for candidate in chip.backends if candidate is not backend)
        seed = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "model": model.to_dict(),
            "memory": memory.to_dict(),
            "chip": self._chip_dict(chip),
            "context_tokens": context_tokens,
            "backend": backend.value,
            "precision": precision,
        }
        plan_id = hashlib.sha256(
            json.dumps(seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return AppleExecutionPlan(
            schema_version=EXECUTION_SCHEMA_VERSION,
            plan_id=plan_id,
            model_id=model.model_id,
            hardware_fingerprint=chip.hardware_fingerprint,
            context_tokens=context_tokens,
            memory_ceiling_bytes=ceiling,
            estimated_peak_bytes=estimated_peak,
            prefill=PhaseExecutionPlan(
                WorkloadPhase.PREFILL, backend, prefill_batch, precision
            ),
            decode=PhaseExecutionPlan(WorkloadPhase.DECODE, backend, 1, precision),
            fallback_chain=fallback,
            decision_reasons=tuple(reasons),
            dry_run=dry_run,
        )

    @staticmethod
    def _preferred_backend(backends: tuple[ExecutionBackend, ...]) -> ExecutionBackend:
        order = (
            ExecutionBackend.VLLM_METAL,
            ExecutionBackend.NATIVE_MLX,
            ExecutionBackend.NATIVE_METAL,
            ExecutionBackend.CPU,
        )
        return next(candidate for candidate in order if candidate in backends)

    @staticmethod
    def _state_precision(pressure: MemoryPressure, precisions: tuple[str, ...]) -> str:
        if pressure in {MemoryPressure.WARNING, MemoryPressure.CRITICAL}:
            for candidate in ("int8", "q8", "fp16"):
                if candidate in precisions:
                    return candidate
        return "fp16" if "fp16" in precisions else precisions[0]

    @staticmethod
    def _chip_dict(chip: AppleChipProfile) -> dict[str, Any]:
        return chip.to_dict()
