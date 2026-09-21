"""Fail-closed numeric format eligibility and measured conversion routing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from .execution import ExecutionBackend, WorkloadPhase


MAX_NUMERIC_CAPABILITIES = 512
MAX_ROUTE_PROFILES = 128


class NumericFormat(str, Enum):
    NVFP4_E2M1 = "nvfp4_e2m1"
    MXFP4_E2M1 = "mxfp4_e2m1"
    MXFP6_E2M3 = "mxfp6_e2m3"
    MXFP8_E4M3 = "mxfp8_e4m3"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E5M2 = "fp8_e5m2"
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    UINT4 = "uint4"
    INT2 = "int2"
    UINT2 = "uint2"
    NF4 = "nf4"


class NumericTensorRole(str, Enum):
    WEIGHT = "weight"
    ACTIVATION = "activation"
    KV_STATE = "kv_state"
    RECURRENT_STATE = "recurrent_state"
    EXPERT = "expert"
    VISION = "vision"
    AUDIO = "audio"
    DIFFUSION = "diffusion"


class NumericRouteStrategy(str, Enum):
    LOAD_CONVERT = "load_convert"
    FIRST_USE_CONVERT = "first_use_convert"
    CACHED_CONVERT = "cached_convert"
    FUSED_EVERY_USE = "fused_every_use"


@dataclass(frozen=True, slots=True)
class NumericCapability:
    source_format: NumericFormat
    compute_format: NumericFormat
    tensor_role: NumericTensorRole
    backend: ExecutionBackend
    operator: str
    minimum_elements: int
    maximum_elements: int
    recipe_id: str
    evidence_id: str
    qualified: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.source_format, NumericFormat)
                or not isinstance(self.compute_format, NumericFormat)
                or not isinstance(self.tensor_role, NumericTensorRole)
                or not isinstance(self.backend, ExecutionBackend)
                or not self.operator or len(self.operator) > 128
                or not 1 <= self.minimum_elements <= self.maximum_elements <= 1 << 40
                or not _identifier(self.recipe_id) or not _digest(self.evidence_id)
                or type(self.qualified) is not bool):
            raise ValueError("invalid numeric capability")

    @property
    def capability_id(self) -> str:
        value = {
            "source": self.source_format.value,
            "compute": self.compute_format.value,
            "role": self.tensor_role.value,
            "backend": self.backend.value,
            "operator": self.operator,
            "minimum_elements": self.minimum_elements,
            "maximum_elements": self.maximum_elements,
            "recipe_id": self.recipe_id,
            "evidence_id": self.evidence_id,
            "qualified": self.qualified,
        }
        return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class NumericEligibilityRequest:
    source_format: NumericFormat
    compute_format: NumericFormat
    tensor_role: NumericTensorRole
    backend: ExecutionBackend
    operator: str
    elements: int
    recipe_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.source_format, NumericFormat)
                or not isinstance(self.compute_format, NumericFormat)
                or not isinstance(self.tensor_role, NumericTensorRole)
                or not isinstance(self.backend, ExecutionBackend)
                or not self.operator or len(self.operator) > 128
                or not 1 <= self.elements <= 1 << 40
                or not _identifier(self.recipe_id)):
            raise ValueError("invalid numeric eligibility request")


@dataclass(frozen=True, slots=True)
class NumericEligibilityDecision:
    eligible: bool
    reason: str
    capability_id: str | None
    evidence_id: str | None


class NumericEligibilityMatrix:
    def __init__(self, capabilities: tuple[NumericCapability, ...]) -> None:
        if (not capabilities or len(capabilities) > MAX_NUMERIC_CAPABILITIES
                or len({item.capability_id for item in capabilities}) != len(capabilities)):
            raise ValueError("invalid numeric capability matrix")
        self._capabilities = capabilities

    def decide(self, request: NumericEligibilityRequest) -> NumericEligibilityDecision:
        if not isinstance(request, NumericEligibilityRequest):
            raise ValueError("invalid numeric eligibility request")
        candidates = tuple(item for item in self._capabilities if (
            item.source_format is request.source_format
            and item.compute_format is request.compute_format
            and item.tensor_role is request.tensor_role
            and item.backend is request.backend
            and item.operator == request.operator
            and item.recipe_id == request.recipe_id
            and item.minimum_elements <= request.elements <= item.maximum_elements
        ))
        if not candidates:
            return NumericEligibilityDecision(False, "unsupported_recipe", None, None)
        if len(candidates) != 1:
            return NumericEligibilityDecision(False, "ambiguous_capability", None, None)
        capability = candidates[0]
        if not capability.qualified:
            return NumericEligibilityDecision(
                False, "capability_not_qualified", capability.capability_id,
                capability.evidence_id,
            )
        return NumericEligibilityDecision(
            True, "qualified", capability.capability_id, capability.evidence_id
        )


@dataclass(frozen=True, slots=True)
class NumericRouteProfile:
    capability_id: str
    phase: WorkloadPhase
    strategy: NumericRouteStrategy
    conversion_nanoseconds: int
    synchronization_nanoseconds: int
    compute_nanoseconds: int
    peak_memory_bytes: int
    amortized_uses: int
    output_digest: str

    def __post_init__(self) -> None:
        if (not _digest(self.capability_id) or not _digest(self.output_digest)
                or not isinstance(self.phase, WorkloadPhase)
                or not isinstance(self.strategy, NumericRouteStrategy)
                or min(self.conversion_nanoseconds, self.synchronization_nanoseconds,
                       self.compute_nanoseconds) < 0
                or self.compute_nanoseconds == 0
                or not 0 <= self.peak_memory_bytes <= 1 << 50
                or not 1 <= self.amortized_uses <= 1_000_000):
            raise ValueError("invalid numeric route profile")

    @property
    def amortized_nanoseconds(self) -> int:
        one_time = self.conversion_nanoseconds
        if self.strategy is NumericRouteStrategy.FUSED_EVERY_USE:
            one_time *= self.amortized_uses
        total = one_time + self.synchronization_nanoseconds + (
            self.compute_nanoseconds * self.amortized_uses
        )
        return (total + self.amortized_uses - 1) // self.amortized_uses


@dataclass(frozen=True, slots=True)
class NumericRouteDecision:
    strategy: NumericRouteStrategy
    amortized_nanoseconds: int
    peak_memory_bytes: int


def choose_numeric_route(
    profiles: tuple[NumericRouteProfile, ...],
    *,
    capability_id: str,
    phase: WorkloadPhase,
    memory_ceiling_bytes: int,
) -> NumericRouteDecision:
    if (not profiles or len(profiles) > MAX_ROUTE_PROFILES
            or not _digest(capability_id)
            or not 1 <= memory_ceiling_bytes <= 1 << 50):
        raise ValueError("invalid numeric route selection")
    matching = tuple(item for item in profiles if (
        item.capability_id == capability_id and item.phase is phase
        and item.peak_memory_bytes <= memory_ceiling_bytes
    ))
    if not matching:
        raise ValueError("no qualified numeric route within memory ceiling")
    digests = {item.output_digest for item in matching}
    if len(digests) != 1:
        raise ValueError("numeric route output mismatch")
    selected = min(
        matching,
        key=lambda item: (
            item.amortized_nanoseconds, item.peak_memory_bytes, item.strategy.value
        ),
    )
    return NumericRouteDecision(
        selected.strategy, selected.amortized_nanoseconds,
        selected.peak_memory_bytes,
    )


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value
    )


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
