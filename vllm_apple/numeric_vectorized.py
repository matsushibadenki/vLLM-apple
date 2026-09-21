"""Optional bounded NumPy vectorized numeric conversion candidates.

These functions are acceleration candidates, not an implicit replacement for the
scalar reference codecs. Callers must probe equality before promotion.
"""
from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache

from .numeric_codecs import MAX_CODEC_ELEMENTS, NF4_CODEBOOK, decode_fp8
from .numeric_layout import NumericLayoutDescriptor
from .numeric_requantization import SymmetricInt8Tensor


class VectorizedNumericUnavailableError(RuntimeError):
    pass


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise VectorizedNumericUnavailableError("NumPy is not installed") from error
    return np


def vectorized_unpack(payload: bytes, bits: int, elements: int) -> tuple[int, ...]:
    if bits not in {2, 4, 8} or type(elements) is not int or not 1 <= elements <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid vectorized unpack configuration")
    expected = (elements * bits + 7) // 8
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise ValueError("packed numeric payload size mismatch")
    used = elements * bits % 8
    if used and payload[-1] >> used:
        raise ValueError("packed numeric payload padding is nonzero")
    np = _numpy()
    source = np.frombuffer(payload, dtype=np.uint8)
    indices = np.arange(elements, dtype=np.int64)
    shifts = (indices * bits) & 7
    codes = (source[(indices * bits) >> 3] >> shifts) & ((1 << bits) - 1)
    return tuple(int(value) for value in codes)


def vectorized_repack(payload: bytes, source_bits: int, target_bits: int, elements: int) -> bytes:
    if target_bits not in {2, 4, 8}:
        raise ValueError("invalid vectorized target packing")
    codes = vectorized_unpack(payload, source_bits, elements)
    if any(value >= 1 << target_bits for value in codes):
        raise ValueError("numeric code does not fit target packing")
    np = _numpy()
    result = np.zeros((elements * target_bits + 7) // 8, dtype=np.uint8)
    indices = np.arange(elements, dtype=np.int64)
    np.bitwise_or.at(
        result,
        (indices * target_bits) >> 3,
        np.asarray(codes, dtype=np.uint8) << ((indices * target_bits) & 7),
    )
    return result.tobytes()


@lru_cache(maxsize=2)
def _fp8_table(variant: str) -> tuple[tuple[float, ...], frozenset[int]]:
    values: list[float] = []
    invalid: set[int] = set()
    for code in range(256):
        try:
            values.append(decode_fp8(bytes((code,)), variant)[0])
        except ValueError:
            values.append(0.0)
            invalid.add(code)
    return tuple(values), frozenset(invalid)


def vectorized_decode_fp8(payload: bytes, variant: str) -> tuple[float, ...]:
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_CODEC_ELEMENTS:
        raise ValueError("invalid FP8 payload")
    np = _numpy()
    table, invalid = _fp8_table(variant)
    source = np.frombuffer(payload, dtype=np.uint8)
    if invalid and np.isin(source, np.fromiter(invalid, dtype=np.uint8)).any():
        raise ValueError("FP8 payload contains nonfinite value")
    decoded = np.asarray(table, dtype=np.float64)[source]
    return tuple(float(value) for value in decoded)


def vectorized_encode_nf4(values: tuple[float, ...]) -> bytes:
    if (
        not isinstance(values, tuple)
        or not 1 <= len(values) <= MAX_CODEC_ELEMENTS
        or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
    ):
        raise ValueError("invalid numeric codec values")
    np = _numpy()
    source = np.asarray(values, dtype=np.float64)
    codebook = np.asarray(NF4_CODEBOOK, dtype=np.float64)
    codes = np.argmin(np.abs(source[:, None] - codebook[None, :]), axis=1).astype(np.uint8)
    result = np.zeros((len(values) + 1) // 2, dtype=np.uint8)
    result[: len(codes[0::2])] |= codes[0::2]
    result[: len(codes[1::2])] |= codes[1::2] << 4
    return result.tobytes()


def vectorized_requantize_symmetric_int8(
    values: tuple[float, ...], *, group_size: int
) -> SymmetricInt8Tensor:
    if (
        not values
        or len(values) > MAX_CODEC_ELEMENTS
        or type(group_size) is not int
        or not 1 <= group_size <= len(values)
        or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
    ):
        raise ValueError("invalid symmetric INT8 requantization")
    np = _numpy()
    source = np.asarray(values, dtype=np.float64)
    codes = np.empty(len(values), dtype=np.int8)
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        group = source[start : start + group_size]
        maximum = float(np.max(np.abs(group)))
        scale = maximum / 127 if maximum else 1.0
        scales.append(scale)
        codes[start : start + len(group)] = np.clip(np.rint(group / scale), -127, 127).astype(
            np.int8
        )
    payload = codes.tobytes()
    digest = hashlib.sha256(
        b"vllm-apple-symmetric-int8-v1\0"
        + payload
        + b"".join(struct.pack("<f", value) for value in scales)
    ).hexdigest()
    return SymmetricInt8Tensor(payload, tuple(scales), group_size, len(values), digest)


def vectorized_repack_bytes(payload: bytes, layout: NumericLayoutDescriptor) -> bytes:
    if not isinstance(layout, NumericLayoutDescriptor):
        raise ValueError("invalid numeric layout")
    if not isinstance(payload, bytes) or len(payload) != layout.storage_elements:
        raise ValueError("numeric byte layout payload is invalid")
    np = _numpy()
    indices = np.fromiter(
        (
            layout.storage_index(row, column)
            for row in range(layout.rows)
            for column in range(layout.columns)
        ),
        dtype=np.int64,
        count=layout.logical_elements,
    )
    source = np.frombuffer(payload, dtype=np.uint8)
    logical_mask = np.zeros(layout.storage_elements, dtype=np.bool_)
    logical_mask[indices] = True
    if np.any(source[~logical_mask] != layout.padding_code):
        raise ValueError("numeric layout padding code mismatch")
    return source[indices].tobytes()


def vectorized_repack_packed_nibbles(
    packed: bytes, layout: NumericLayoutDescriptor, nibble_order: str
) -> bytes:
    if (
        not isinstance(layout, NumericLayoutDescriptor)
        or not isinstance(packed, bytes)
        or len(packed) != (layout.storage_elements + 1) // 2
        or nibble_order not in {"low_nibble_first", "high_nibble_first"}
    ):
        raise ValueError("numeric packed layout payload is invalid")
    np = _numpy()
    raw = np.frombuffer(packed, dtype=np.uint8)
    positions = np.arange(layout.storage_elements, dtype=np.int64)
    shifts = 4 * (positions & 1)
    if nibble_order == "high_nibble_first":
        shifts = 4 - shifts
    storage = (raw[positions >> 1] >> shifts) & 15
    if layout.storage_elements % 2:
        trailing_shift = 4 if nibble_order == "low_nibble_first" else 0
        if int((raw[-1] >> trailing_shift) & 15) != layout.padding_code:
            raise ValueError("numeric layout trailing padding code mismatch")
    logical_indices = np.fromiter(
        (
            layout.storage_index(row, column)
            for row in range(layout.rows)
            for column in range(layout.columns)
        ),
        dtype=np.int64,
        count=layout.logical_elements,
    )
    logical_mask = np.zeros(layout.storage_elements, dtype=np.bool_)
    logical_mask[logical_indices] = True
    if np.any(storage[~logical_mask] != layout.padding_code):
        raise ValueError("numeric layout padding code mismatch")
    logical = storage[logical_indices].astype(np.uint8)
    output = np.zeros((layout.logical_elements + 1) // 2, dtype=np.uint8)
    output[: len(logical[0::2])] |= logical[0::2]
    output[: len(logical[1::2])] |= logical[1::2] << 4
    return output.tobytes()
