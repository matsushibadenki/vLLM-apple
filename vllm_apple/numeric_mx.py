"""OCP MX v1.0 CPU reference adapters with explicit bit packing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .numeric_formats import MAX_REFERENCE_ELEMENTS
from .numeric_requantization import SymmetricInt8Tensor, requantize_symmetric_int8

MX_BLOCK_SIZE = 32


class MXElementFormat(str, Enum):
    FP4_E2M1 = "fp4_e2m1"
    FP6_E2M3 = "fp6_e2m3"
    FP6_E3M2 = "fp6_e3m2"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


_FIELDS = {
    MXElementFormat.FP4_E2M1: (4, 2, 1, 1),
    MXElementFormat.FP6_E2M3: (6, 2, 3, 1),
    MXElementFormat.FP6_E3M2: (6, 3, 2, 3),
    MXElementFormat.FP8_E4M3: (8, 4, 3, 7),
    MXElementFormat.FP8_E5M2: (8, 5, 2, 15),
}


@dataclass(frozen=True, slots=True)
class MXFormatAdapter:
    element_format: MXElementFormat
    elements: int
    block_size: int = MX_BLOCK_SIZE
    scale_format: str = "e8m0"
    packing: str = "lsb_first_contiguous"
    overflow: str = "reject_nonfinite_or_f32_overflow"
    specification: str = "ocp-mx-v1.0"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.element_format, MXElementFormat)
            or type(self.elements) is not int
            or not 1 <= self.elements <= MAX_REFERENCE_ELEMENTS
            or self.block_size != MX_BLOCK_SIZE
            or self.scale_format != "e8m0"
            or self.packing != "lsb_first_contiguous"
            or self.overflow != "reject_nonfinite_or_f32_overflow"
            or self.specification != "ocp-mx-v1.0"
        ):
            raise ValueError("unsupported MX format adapter")

    @property
    def element_bits(self) -> int:
        return _FIELDS[self.element_format][0]

    @property
    def packed_bytes(self) -> int:
        return (self.elements * self.element_bits + 7) // 8

    @property
    def scale_count(self) -> int:
        return (self.elements + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE


def decode_mx(
    adapter: MXFormatAdapter,
    packed: bytes,
    scales: bytes,
) -> tuple[float, ...]:
    if not isinstance(adapter, MXFormatAdapter):
        raise ValueError("invalid MX adapter")
    if not isinstance(packed, bytes) or len(packed) != adapter.packed_bytes:
        raise ValueError("MX packed payload size mismatch")
    if not isinstance(scales, bytes) or len(scales) != adapter.scale_count:
        raise ValueError("MX scale count mismatch")
    unused_bits = len(packed) * 8 - adapter.elements * adapter.element_bits
    if unused_bits and packed[-1] >> (8 - unused_bits):
        raise ValueError("MX unused packing bits must be zero")
    decoded_scales = tuple(_decode_e8m0(code) for code in scales)
    result = []
    for index, code in enumerate(_unpack_lsb(packed, adapter.element_bits, adapter.elements)):
        value = _decode_element(adapter.element_format, code)
        scaled = value * decoded_scales[index // MX_BLOCK_SIZE]
        if not math.isfinite(scaled) or abs(scaled) > 3.4028234663852886e38:
            raise ValueError("MX decoded value exceeds finite F32 range")
        result.append(scaled)
    return tuple(result)


def convert_mx_to_int8(
    adapter: MXFormatAdapter,
    packed: bytes,
    scales: bytes,
    *,
    group_size: int = MX_BLOCK_SIZE,
) -> SymmetricInt8Tensor:
    """Decode one published MX variant and requantize through the shared INT8 path."""
    return requantize_symmetric_int8(
        decode_mx(adapter, packed, scales),
        group_size=group_size,
    )


def _decode_e8m0(code: int) -> float:
    if code == 0xFF:
        raise ValueError("MX E8M0 NaN scale is not executable")
    return math.ldexp(1.0, code - 127)


def _decode_element(element_format: MXElementFormat, code: int) -> float:
    bits, exponent_bits, mantissa_bits, bias = _FIELDS[element_format]
    sign = -1.0 if code >> (bits - 1) else 1.0
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa = code & mantissa_mask
    exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1)
    maximum_exponent = (1 << exponent_bits) - 1
    if element_format is MXElementFormat.FP8_E5M2 and exponent == maximum_exponent:
        raise ValueError("MX FP8 E5M2 Inf/NaN element is not executable")
    if (
        element_format is MXElementFormat.FP8_E4M3
        and exponent == maximum_exponent
        and mantissa == mantissa_mask
    ):
        raise ValueError("MX FP8 E4M3 NaN element is not executable")
    if exponent == 0:
        magnitude = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
    else:
        magnitude = math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
    return math.copysign(magnitude, sign)


def _unpack_lsb(packed: bytes, bits: int, elements: int):
    accumulator = 0
    available = 0
    offset = 0
    mask = (1 << bits) - 1
    for _ in range(elements):
        while available < bits:
            accumulator |= packed[offset] << available
            available += 8
            offset += 1
        yield accumulator & mask
        accumulator >>= bits
        available -= bits
