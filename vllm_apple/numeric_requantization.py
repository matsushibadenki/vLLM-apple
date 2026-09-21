"""Compare representation-preserving scaled INT8 with general requantization."""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from .numeric_formats import ScaledInt8Tensor

MAX_REQUANTIZATION_ELEMENTS = 65_536


@dataclass(frozen=True, slots=True)
class Int8ExecutionCapability:
    accumulator: str
    scale_granularities: tuple[str, ...]
    signed: bool = True

    def __post_init__(self) -> None:
        allowed = {"tensor", "group", "nvfp4_block"}
        if (self.accumulator not in {"int32", "fp32"}
                or not self.scale_granularities
                or len(set(self.scale_granularities)) != len(self.scale_granularities)
                or any(value not in allowed for value in self.scale_granularities)
                or type(self.signed) is not bool):
            raise ValueError("invalid INT8 execution capability")


@dataclass(frozen=True, slots=True)
class SymmetricInt8Tensor:
    payload: bytes
    scales: tuple[float, ...]
    group_size: int
    elements: int
    output_sha256: str

    def values(self) -> tuple[float, ...]:
        codes = struct.unpack(f"<{self.elements}b", self.payload)
        return tuple(
            code * self.scales[index // self.group_size]
            for index, code in enumerate(codes)
        )


@dataclass(frozen=True, slots=True)
class Int8RouteMetrics:
    route: str
    scale_granularity: str
    maximum_absolute_error: float
    rmse: float
    storage_bytes: int
    compatible: bool
    incompatibility_reason: str | None
    output_sha256: str


@dataclass(frozen=True, slots=True)
class Int8RouteComparison:
    preserving: Int8RouteMetrics
    requantized: Int8RouteMetrics
    accumulator: str


def requantize_symmetric_int8(
    values: tuple[float, ...], *, group_size: int
) -> SymmetricInt8Tensor:
    if (not values or len(values) > MAX_REQUANTIZATION_ELEMENTS
            or type(group_size) is not int or not 1 <= group_size <= len(values)
            or any(not math.isfinite(value) for value in values)):
        raise ValueError("invalid symmetric INT8 requantization")
    codes: list[int] = []
    scales = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        maximum = max(abs(value) for value in group)
        scale = maximum / 127 if maximum else 1.0
        scales.append(scale)
        codes.extend(max(-127, min(127, round(value / scale))) for value in group)
    payload = struct.pack(f"<{len(codes)}b", *codes)
    digest = hashlib.sha256(
        b"vllm-apple-symmetric-int8-v1\0" + payload
        + b"".join(struct.pack("<f", value) for value in scales)
    ).hexdigest()
    return SymmetricInt8Tensor(payload, tuple(scales), group_size, len(values), digest)


def compare_nvfp4_int8_routes(
    preserving: ScaledInt8Tensor,
    *,
    requantization_group_size: int,
    capability: Int8ExecutionCapability,
) -> Int8RouteComparison:
    if not isinstance(preserving, ScaledInt8Tensor) or not isinstance(
        capability, Int8ExecutionCapability
    ):
        raise ValueError("invalid NVFP4 INT8 route comparison")
    reference = preserving.reference_values()
    requantized = requantize_symmetric_int8(
        reference, group_size=requantization_group_size
    )
    preserving_digest = hashlib.sha256(
        b"vllm-apple-preserving-int8-v1\0" + preserving.payload
        + preserving.block_scales + struct.pack("<d", preserving.global_scale)
    ).hexdigest()
    preserving_metrics = Int8RouteMetrics(
        "nvfp4_scaled_int8_preserving", "nvfp4_block", 0.0, 0.0,
        len(preserving.payload) + len(preserving.block_scales) + 8,
        capability.signed and "nvfp4_block" in capability.scale_granularities,
        _compatibility_reason(capability, "nvfp4_block"), preserving_digest,
    )
    restored = requantized.values()
    errors = tuple(abs(left - right) for left, right in zip(reference, restored))
    granularity = (
        "tensor" if requantization_group_size == len(reference) else "group"
    )
    requantized_metrics = Int8RouteMetrics(
        "symmetric_int8_requantized", granularity, max(errors),
        math.sqrt(sum(value * value for value in errors) / len(errors)),
        len(requantized.payload) + 4 * len(requantized.scales),
        capability.signed and granularity in capability.scale_granularities,
        _compatibility_reason(capability, granularity), requantized.output_sha256,
    )
    return Int8RouteComparison(
        preserving_metrics, requantized_metrics, capability.accumulator
    )


def _compatibility_reason(
    capability: Int8ExecutionCapability, granularity: str
) -> str | None:
    if not capability.signed:
        return "signed_int8_unsupported"
    if granularity not in capability.scale_granularities:
        return "scale_granularity_unsupported"
    return None
