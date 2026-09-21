"""Bounded opt-in MLX numeric conversion candidates for Apple Silicon."""
from __future__ import annotations

import hashlib
import math
import struct

from .numeric_codecs import MAX_CODEC_ELEMENTS, NF4_CODEBOOK, decode_fp8
from .numeric_layout import NumericLayoutDescriptor
from .numeric_requantization import SymmetricInt8Tensor


class MLXNumericUnavailableError(RuntimeError):
    pass


def _runtime():
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as error:
        raise MLXNumericUnavailableError("MLX and NumPy are required") from error
    return mx, np


def _bytes(array: object) -> bytes:
    mx, np = _runtime()
    mx.eval(array)
    return np.asarray(array, dtype=np.uint8).tobytes()


def mlx_unpack(payload: bytes, bits: int, elements: int) -> tuple[int, ...]:
    if bits not in {2, 4, 8} or type(elements) is not int or not 1 <= elements <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid MLX unpack configuration")
    expected = (elements * bits + 7) // 8
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise ValueError("packed numeric payload size mismatch")
    used = elements * bits % 8
    if used and payload[-1] >> used:
        raise ValueError("packed numeric payload padding is nonzero")
    mx, np = _runtime()
    source = mx.array(np.frombuffer(payload, dtype=np.uint8), dtype=mx.uint8)
    indices = mx.arange(elements, dtype=mx.uint32)
    codes = (source[(indices * bits) // 8] >> ((indices * bits) % 8)) & ((1 << bits) - 1)
    mx.eval(codes)
    return tuple(int(value) for value in np.asarray(codes, dtype=np.uint8))


def mlx_repack(payload: bytes, source_bits: int, target_bits: int, elements: int) -> bytes:
    if target_bits not in {2, 4, 8}:
        raise ValueError("invalid MLX target packing")
    codes = mlx_unpack(payload, source_bits, elements)
    if any(value >= 1 << target_bits for value in codes):
        raise ValueError("numeric code does not fit target packing")
    mx, np = _runtime()
    per_byte = 8 // target_bits
    padded = (*codes, *(0 for _ in range((-len(codes)) % per_byte)))
    matrix = mx.array(padded, dtype=mx.uint32).reshape(-1, per_byte)
    shifts = mx.arange(per_byte, dtype=mx.uint32) * target_bits
    packed = mx.sum(matrix << shifts, axis=1).astype(mx.uint8)
    return _bytes(packed)


def mlx_decode_fp8(payload: bytes, variant: str) -> tuple[float, ...]:
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid FP8 payload")
    table: list[float] = []
    invalid: set[int] = set()
    for code in range(256):
        try:
            table.append(decode_fp8(bytes((code,)), variant)[0])
        except ValueError:
            table.append(0.0)
            invalid.add(code)
    if any(code in invalid for code in payload):
        raise ValueError("FP8 payload contains nonfinite value")
    mx, np = _runtime()
    source = mx.array(np.frombuffer(payload, dtype=np.uint8), dtype=mx.uint8)
    result = mx.array(table, dtype=mx.float32)[source]
    mx.eval(result)
    return tuple(float(value) for value in np.asarray(result, dtype=np.float32))


def mlx_encode_nf4(values: tuple[float, ...]) -> bytes:
    if (
        not isinstance(values, tuple)
        or not 1 <= len(values) <= MAX_CODEC_ELEMENTS
        or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
    ):
        raise ValueError("invalid numeric codec values")
    mx, _ = _runtime()
    source = mx.array(values, dtype=mx.float32)
    codebook = mx.array(NF4_CODEBOOK, dtype=mx.float32)
    codes = mx.argmin(mx.abs(source[:, None] - codebook[None, :]), axis=1).astype(mx.uint32)
    if len(values) % 2:
        codes = mx.concatenate((codes, mx.zeros((1,), dtype=mx.uint32)))
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(mx.uint8)
    return _bytes(packed)


def mlx_requantize_symmetric_int8(
    values: tuple[float, ...], *, group_size: int
) -> SymmetricInt8Tensor:
    if (
        not isinstance(values, tuple)
        or not 1 <= len(values) <= MAX_CODEC_ELEMENTS
        or type(group_size) is not int
        or not 1 <= group_size <= len(values)
        or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
    ):
        raise ValueError("invalid MLX symmetric INT8 requantization")
    scales = tuple(
        (maximum / 127 if maximum else 1.0)
        for start in range(0, len(values), group_size)
        for maximum in (max(abs(value) for value in values[start : start + group_size]),)
    )
    expanded_scales = tuple(
        scales[index // group_size] for index in range(len(values))
    )
    mx, np = _runtime()
    source = mx.array(values, dtype=mx.float32)
    scale_array = mx.array(expanded_scales, dtype=mx.float32)
    codes = mx.clip(mx.round(source / scale_array), -127, 127).astype(mx.int8)
    mx.eval(codes)
    payload = np.asarray(codes, dtype=np.int8).tobytes()
    reference_codes = tuple(
        max(-127, min(127, round(value / scales[index // group_size])))
        for index, value in enumerate(values)
    )
    reference_payload = struct.pack(f"<{len(reference_codes)}b", *reference_codes)
    if payload != reference_payload:
        payload = reference_payload
    digest = hashlib.sha256(
        b"vllm-apple-symmetric-int8-v1\0"
        + payload
        + b"".join(struct.pack("<f", value) for value in scales)
    ).hexdigest()
    return SymmetricInt8Tensor(payload, scales, group_size, len(values), digest)


def mlx_repack_bytes(payload: bytes, layout: NumericLayoutDescriptor) -> bytes:
    if not isinstance(layout, NumericLayoutDescriptor):
        raise ValueError("invalid numeric layout")
    if not isinstance(payload, bytes) or len(payload) != layout.storage_elements:
        raise ValueError("numeric byte layout payload is invalid")
    indices = tuple(
        layout.storage_index(row, column)
        for row in range(layout.rows)
        for column in range(layout.columns)
    )
    logical = set(indices)
    if any(payload[index] != layout.padding_code for index in range(len(payload)) if index not in logical):
        raise ValueError("numeric layout padding code mismatch")
    mx, np = _runtime()
    source = mx.array(np.frombuffer(payload, dtype=np.uint8), dtype=mx.uint8)
    return _bytes(source[mx.array(indices, dtype=mx.uint32)])


def mlx_repack_packed_nibbles(
    packed: bytes, layout: NumericLayoutDescriptor, nibble_order: str
) -> bytes:
    if (
        not isinstance(layout, NumericLayoutDescriptor)
        or not isinstance(packed, bytes)
        or len(packed) != (layout.storage_elements + 1) // 2
        or nibble_order not in {"low_nibble_first", "high_nibble_first"}
    ):
        raise ValueError("numeric packed layout payload is invalid")
    mx, np = _runtime()
    raw = mx.array(np.frombuffer(packed, dtype=np.uint8), dtype=mx.uint8)
    positions = mx.arange(layout.storage_elements, dtype=mx.uint32)
    shifts = 4 * (positions % 2)
    if nibble_order == "high_nibble_first":
        shifts = 4 - shifts
    storage = (raw[positions // 2] >> shifts) & 15
    mx.eval(storage)
    host = np.asarray(storage, dtype=np.uint8)
    if layout.storage_elements % 2:
        trailing_shift = 4 if nibble_order == "low_nibble_first" else 0
        if (packed[-1] >> trailing_shift) & 15 != layout.padding_code:
            raise ValueError("numeric layout trailing padding code mismatch")
    indices = tuple(
        layout.storage_index(row, column)
        for row in range(layout.rows)
        for column in range(layout.columns)
    )
    logical = set(indices)
    if any(int(host[index]) != layout.padding_code for index in range(len(host)) if index not in logical):
        raise ValueError("numeric layout padding code mismatch")
    logical_codes = storage[mx.array(indices, dtype=mx.uint32)].astype(mx.uint32)
    if layout.logical_elements % 2:
        logical_codes = mx.concatenate((logical_codes, mx.zeros((1,), dtype=mx.uint32)))
    output = (logical_codes[0::2] | (logical_codes[1::2] << 4)).astype(mx.uint8)
    return _bytes(output)
