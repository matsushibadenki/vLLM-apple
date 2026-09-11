"""CPU reference contracts; no native execution or performance qualification implied."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

MAX_REFERENCE_ELEMENTS = 65_536
_E2M1_TWICE = (0, 1, 2, 3, 4, 6, 8, 12)


@dataclass(frozen=True, slots=True)
class NumericFormatDescriptor:
    encoding: str
    elements: int
    block_size: int = 16
    packing: str = "low_nibble_first"
    scale_encoding: str = "e4m3fn"
    layout: str = "contiguous_1d"
    value_multiplier: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported descriptor version")
        if type(self.elements) is not int or not 0 < self.elements <= MAX_REFERENCE_ELEMENTS:
            raise ValueError("reference element count must be in 1..65536")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("invalid block size")
        if type(self.value_multiplier) not in (int, float) or not math.isfinite(self.value_multiplier) or self.value_multiplier <= 0:
            raise ValueError("invalid value multiplier")
        for value in (self.encoding, self.packing, self.scale_encoding, self.layout):
            if not isinstance(value, str) or not 0 < len(value) <= 64:
                raise ValueError("invalid format identifier")


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    source: NumericFormatDescriptor
    target: NumericFormatDescriptor
    adapter: str = "nvfp4_int8_cpu_reference_v1"
    schema_version: int = 1

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def conversion_plan(source: NumericFormatDescriptor) -> ConversionPlan:
    """Registry entry for the sole qualified reference conversion, not an executor."""
    if (source.encoding, source.block_size, source.packing, source.scale_encoding, source.layout, source.value_multiplier) != (
        "nvfp4_e2m1", 16, "low_nibble_first", "e4m3fn", "contiguous_1d", 1.0
    ):
        raise ValueError("unsupported numeric format variant")
    return ConversionPlan(source, NumericFormatDescriptor(
        "scaled_int8", source.elements, packing="signed_byte", value_multiplier=0.5
    ))


def _scale(code: int) -> float:
    if code >= 127:
        raise ValueError("block scale must be nonnegative finite E4M3FN")
    exponent, mantissa = code >> 3, code & 7
    return math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)


def _validate_inputs(descriptor, packed, scales, global_scale) -> ConversionPlan:
    plan = conversion_plan(descriptor)
    if not isinstance(packed, bytes) or len(packed) != (descriptor.elements + 1) // 2:
        raise ValueError("packed payload size mismatch")
    if not isinstance(scales, bytes) or len(scales) != (descriptor.elements + 15) // 16:
        raise ValueError("block scale count mismatch")
    if type(global_scale) not in (float, int) or not math.isfinite(global_scale) or global_scale < 0:
        raise ValueError("invalid global scale")
    for code in scales:
        if not math.isfinite(_scale(code) * global_scale * 6):
            raise ValueError("reference scale overflow")
    if descriptor.elements % 2 and packed[-1] >> 4:
        raise ValueError("unused padding nibble must be zero")
    return plan


def _codes(packed: bytes, elements: int):
    for index in range(elements):
        yield (packed[index // 2] >> (4 * (index % 2))) & 15


def decode_nvfp4(
    descriptor: NumericFormatDescriptor, packed: bytes, scales: bytes, global_scale: float
) -> tuple[float, ...]:
    _validate_inputs(descriptor, packed, scales, global_scale)
    return tuple(
        math.copysign(_E2M1_TWICE[code & 7] / 2, -1 if code & 8 else 1)
        * _scale(scales[index // 16]) * global_scale
        for index, code in enumerate(_codes(packed, descriptor.elements))
    )


@dataclass(frozen=True, slots=True)
class ScaledInt8Tensor:
    plan: ConversionPlan
    payload: bytes
    block_scales: bytes
    global_scale: float

    def reference_values(self) -> tuple[float, ...]:
        # Preserve the original scale factors; divide the small integer first to
        # avoid introducing underflow by halving a very small global scale.
        return tuple(
            (value if value < 128 else value - 256) / 2
            * _scale(self.block_scales[index // 16]) * self.global_scale
            for index, value in enumerate(self.payload)
        )


def convert_nvfp4_to_int8(
    descriptor: NumericFormatDescriptor, packed: bytes, scales: bytes, global_scale: float
) -> ScaledInt8Tensor:
    plan = _validate_inputs(descriptor, packed, scales, global_scale)
    payload = bytes(
        ((-1 if code & 8 else 1) * _E2M1_TWICE[code & 7]) & 255
        for code in _codes(packed, descriptor.elements)
    )
    # Signed zero is canonicalized by integer storage; numeric equality only.
    return ScaledInt8Tensor(plan, payload, scales, global_scale)
