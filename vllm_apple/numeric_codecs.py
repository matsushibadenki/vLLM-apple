"""Bounded CPU reference codecs for non-native Apple numeric formats."""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_CODEC_ELEMENTS = 65_536
NF4_CODEBOOK = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)


@dataclass(frozen=True, slots=True)
class GroupwiseAffineTensor:
    payload: bytes
    bits: int
    signed: bool
    group_size: int
    elements: int
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]

    def values(self) -> tuple[float, ...]:
        codes = _unpack(self.payload, self.bits, self.elements)
        return tuple(
            (code - self.zero_points[index // self.group_size])
            * self.scales[index // self.group_size]
            for index, code in enumerate(codes)
        )


def decode_fp8(payload: bytes, variant: str) -> tuple[float, ...]:
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid FP8 payload")
    exponent_bits, mantissa_bits, bias, finite_nan = _fp8_parameters(variant)
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    values = []
    for code in payload:
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> mantissa_bits) & exponent_mask
        mantissa = code & mantissa_mask
        if exponent == 0:
            value = math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
        elif (finite_nan and exponent == exponent_mask and mantissa == mantissa_mask):
            raise ValueError("FP8 payload contains NaN")
        elif not finite_nan and exponent == exponent_mask:
            raise ValueError("FP8 payload contains nonfinite value")
        else:
            value = math.ldexp(1 + mantissa / (1 << mantissa_bits), exponent - bias)
        values.append(sign * value)
    return tuple(values)


def encode_fp8(values: tuple[float, ...], variant: str) -> bytes:
    _validate_values(values)
    codebook_values = []
    for code in range(256):
        try:
            decoded = decode_fp8(bytes((code,)), variant)[0]
        except ValueError:
            continue
        codebook_values.append((code, decoded))
    codebook = tuple(codebook_values)
    maximum = max(abs(value) for _, value in codebook)
    result = bytearray()
    for value in values:
        if abs(value) > maximum:
            raise ValueError("FP8 value overflows finite range")
        if value == 0:
            result.append(0x80 if math.copysign(1.0, value) < 0 else 0)
            continue
        same_sign = (
            (code, decoded) for code, decoded in codebook
            if math.copysign(1.0, decoded) == math.copysign(1.0, value)
        )
        code, _ = min(same_sign, key=lambda item: (abs(item[1] - value), item[0] & 1, item[0]))
        result.append(code)
    return bytes(result)


def encode_nf4(values: tuple[float, ...]) -> bytes:
    _validate_values(values)
    codes = tuple(
        min(range(16), key=lambda code: (abs(NF4_CODEBOOK[code] - value), code))
        for value in values
    )
    return _pack(codes, 4)


def decode_nf4(payload: bytes, elements: int) -> tuple[float, ...]:
    if type(elements) is not int or not 1 <= elements <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid NF4 element count")
    return tuple(NF4_CODEBOOK[code] for code in _unpack(payload, 4, elements))


def quantize_groupwise_affine(
    values: tuple[float, ...], *, bits: int, signed: bool, group_size: int
) -> GroupwiseAffineTensor:
    _validate_values(values)
    if (bits not in {2, 4, 8} or type(signed) is not bool
            or type(group_size) is not int or not 1 <= group_size <= len(values)):
        raise ValueError("invalid groupwise affine configuration")
    qmin, qmax = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                  if signed else (0, (1 << bits) - 1))
    stored_offset = -qmin
    codes = []
    scales = []
    zero_points = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        lower, upper = min(group), max(group)
        scale = (upper - lower) / (qmax - qmin) if upper != lower else 1.0
        zero_point = max(qmin, min(qmax, round(qmin - lower / scale)))
        scales.append(scale)
        zero_points.append(zero_point + stored_offset)
        codes.extend(
            max(qmin, min(qmax, round(value / scale) + zero_point)) + stored_offset
            for value in group
        )
    return GroupwiseAffineTensor(
        _pack(tuple(codes), bits), bits, signed, group_size, len(values),
        tuple(scales), tuple(zero_points),
    )


def _fp8_parameters(variant: str) -> tuple[int, int, int, bool]:
    if variant == "e4m3fn":
        return 4, 3, 7, True
    if variant == "e5m2":
        return 5, 2, 15, False
    raise ValueError("unsupported FP8 variant")


def _validate_values(values: tuple[float, ...]) -> None:
    if (not isinstance(values, tuple) or not 1 <= len(values) <= MAX_CODEC_ELEMENTS
            or any(type(value) not in {int, float} or not math.isfinite(value)
                   for value in values)):
        raise ValueError("invalid numeric codec values")


def _pack(codes: tuple[int, ...], bits: int) -> bytes:
    mask = (1 << bits) - 1
    result = bytearray((len(codes) * bits + 7) // 8)
    for index, code in enumerate(codes):
        if not 0 <= code <= mask:
            raise ValueError("numeric code is out of range")
        bit = index * bits
        result[bit // 8] |= code << (bit % 8)
    return bytes(result)


def _unpack(payload: bytes, bits: int, elements: int) -> tuple[int, ...]:
    expected = (elements * bits + 7) // 8
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise ValueError("packed numeric payload size mismatch")
    codes = tuple(
        (payload[index * bits // 8] >> (index * bits % 8)) & ((1 << bits) - 1)
        for index in range(elements)
    )
    used = elements * bits % 8
    if used and payload[-1] >> used:
        raise ValueError("packed numeric payload padding is nonzero")
    return codes
