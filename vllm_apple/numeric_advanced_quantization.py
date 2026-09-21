"""Bounded double-quantized scales, mixed precision and sparse residuals."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .numeric_codecs import GroupwiseAffineTensor, quantize_groupwise_affine

MAX_ADVANCED_QUANTIZATION_ELEMENTS = 65_536


@dataclass(frozen=True, slots=True)
class DoubleQuantizedScales:
    payload: bytes
    offset: float
    step: float
    count: int

    def values(self) -> tuple[float, ...]:
        return tuple(self.offset + code * self.step for code in self.payload)


@dataclass(frozen=True, slots=True)
class MixedPrecisionGroup:
    start: int
    tensor: GroupwiseAffineTensor


@dataclass(frozen=True, slots=True)
class MixedPrecisionTensor:
    groups: tuple[MixedPrecisionGroup, ...]
    elements: int
    group_size: int

    def values(self) -> tuple[float, ...]:
        output = []
        expected = 0
        for group in self.groups:
            if group.start != expected:
                raise ValueError("mixed precision group order is invalid")
            values = group.tensor.values()
            output.extend(values)
            expected += len(values)
        if expected != self.elements:
            raise ValueError("mixed precision element count mismatch")
        return tuple(output)


@dataclass(frozen=True, slots=True)
class SparseResidualTensor:
    base: GroupwiseAffineTensor
    indices: tuple[int, ...]
    residuals: tuple[float, ...]
    threshold: float

    def values(self) -> tuple[float, ...]:
        values = list(self.base.values())
        if (len(self.indices) != len(self.residuals)
                or tuple(sorted(self.indices)) != self.indices
                or len(set(self.indices)) != len(self.indices)
                or any(not 0 <= index < self.base.elements for index in self.indices)
                or any(not math.isfinite(value) for value in self.residuals)):
            raise ValueError("sparse residual representation is invalid")
        for index, residual in zip(self.indices, self.residuals):
            values[index] += residual
        return tuple(values)


def double_quantize_scales(scales: tuple[float, ...]) -> DoubleQuantizedScales:
    _values(scales)
    lower, upper = min(scales), max(scales)
    step = (upper - lower) / 255 if upper != lower else 1.0
    payload = bytes(max(0, min(255, round((value - lower) / step))) for value in scales)
    return DoubleQuantizedScales(payload, lower, step, len(scales))


def quantize_mixed_precision(
    values: tuple[float, ...],
    *,
    group_size: int,
    high_precision_threshold: float,
    low_bits: int = 4,
    high_bits: int = 8,
) -> MixedPrecisionTensor:
    _values(values)
    if (type(group_size) is not int or not 1 <= group_size <= len(values)
            or not math.isfinite(high_precision_threshold)
            or high_precision_threshold < 0
            or low_bits not in {2, 4, 8} or high_bits not in {2, 4, 8}
            or low_bits > high_bits):
        raise ValueError("invalid mixed precision configuration")
    groups = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        bits = high_bits if max(abs(value) for value in group) > high_precision_threshold else low_bits
        groups.append(MixedPrecisionGroup(
            start,
            quantize_groupwise_affine(
                group, bits=bits, signed=True, group_size=len(group)
            ),
        ))
    return MixedPrecisionTensor(tuple(groups), len(values), group_size)


def quantize_with_sparse_residual(
    values: tuple[float, ...],
    *,
    bits: int,
    group_size: int,
    outlier_threshold: float,
) -> SparseResidualTensor:
    _values(values)
    if not math.isfinite(outlier_threshold) or outlier_threshold <= 0:
        raise ValueError("invalid sparse residual threshold")
    clipped = tuple(max(-outlier_threshold, min(outlier_threshold, value)) for value in values)
    base = quantize_groupwise_affine(
        clipped, bits=bits, signed=True, group_size=group_size
    )
    base_values = base.values()
    indices = tuple(index for index, value in enumerate(values)
                    if abs(value) > outlier_threshold)
    residuals = tuple(values[index] - base_values[index] for index in indices)
    return SparseResidualTensor(base, indices, residuals, outlier_threshold)


def _values(values: tuple[float, ...]) -> None:
    if (not isinstance(values, tuple)
            or not 1 <= len(values) <= MAX_ADVANCED_QUANTIZATION_ELEMENTS
            or any(type(value) not in {int, float} or not math.isfinite(value)
                   for value in values)):
        raise ValueError("invalid advanced quantization values")
