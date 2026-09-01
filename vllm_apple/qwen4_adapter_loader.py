from __future__ import annotations

import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

from .qwen4_adapter_contract import build_qwen4_adapter_contract
from .qwen4_shard_stager import _safe_regular
from .qwen4_weight_map import _bounded_index


DEFAULT_MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSOR_RANK = 16
MAX_DIMENSION = 1 << 30
MAX_TENSORS_PER_SHARD = 16384
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Qwen4 safetensors header contains a duplicate key")
        result[key] = value
    return result


def _tensor_bytes(dtype: str, shape: list[int], *, maximum_bytes: int) -> int:
    if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or len(shape) > MAX_TENSOR_RANK:
        raise ValueError("Qwen4 safetensors header has an unsupported dtype or rank")
    elements = 1
    for dimension in shape:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            or dimension > MAX_DIMENSION
        ):
            raise ValueError("Qwen4 safetensors header has an invalid shape")
        elements *= dimension
        if elements * _DTYPE_BYTES[dtype] > maximum_bytes:
            raise ValueError("Qwen4 safetensors tensor exceeds the shard boundary")
    return elements * _DTYPE_BYTES[dtype]


def _inspect_header(
    path: Path,
    *,
    maximum_artifact_bytes: int,
    maximum_header_bytes: int,
) -> dict[str, dict[str, object]]:
    info = _safe_regular(path, maximum_bytes=maximum_artifact_bytes)
    with path.open("rb", buffering=0) as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Qwen4 safetensors shard is missing its header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= maximum_header_bytes or header_bytes > info.st_size - 8:
            raise ValueError("Qwen4 safetensors header length is outside the bounded policy")
        encoded = handle.read(header_bytes)
        if len(encoded) != header_bytes:
            raise ValueError("Qwen4 safetensors header is truncated")
    after = path.stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Qwen4 safetensors shard changed while reading its header")
    try:
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen4 safetensors header JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Qwen4 safetensors header must be an object")
    metadata = payload.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        raise ValueError("Qwen4 safetensors metadata is invalid")
    data_bytes = info.st_size - 8 - header_bytes
    if not 1 <= len(payload) <= MAX_TENSORS_PER_SHARD:
        raise ValueError("Qwen4 safetensors tensor count is outside the bounded policy")
    intervals: list[tuple[int, int]] = []
    tensors: dict[str, dict[str, object]] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not name or len(name.encode()) > 1024:
            raise ValueError("Qwen4 safetensors tensor name is invalid")
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Qwen4 safetensors tensor descriptor is invalid")
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if (
            not isinstance(dtype, str)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
        ):
            raise ValueError("Qwen4 safetensors tensor descriptor is invalid")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise ValueError("Qwen4 safetensors tensor offset is outside the shard")
        expected_bytes = _tensor_bytes(dtype, shape, maximum_bytes=data_bytes)
        if end - start != expected_bytes:
            raise ValueError("Qwen4 safetensors tensor byte length does not match its shape")
        intervals.append((start, end))
        tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "rank": len(shape),
            "bytes": expected_bytes,
            "file_offset": 8 + header_bytes + start,
        }
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != data_bytes:
        raise ValueError("Qwen4 safetensors data region contains an unassigned boundary")
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError("Qwen4 safetensors tensor ranges overlap")
        if previous[1] != current[0]:
            raise ValueError("Qwen4 safetensors data region contains an unassigned gap")
    return tensors


def inspect_qwen4_adapter_headers(
    stage_root: str | Path,
    *,
    maximum_artifact_bytes: int,
    requested_modes: tuple[str, ...] = ("text",),
    maximum_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> dict[str, object]:
    if (
        not isinstance(maximum_header_bytes, int)
        or isinstance(maximum_header_bytes, bool)
        or not 2 <= maximum_header_bytes <= DEFAULT_MAX_HEADER_BYTES
    ):
        raise ValueError("Qwen4 maximum safetensors header bytes are invalid")
    root = Path(stage_root).expanduser().resolve(strict=True)
    contract = build_qwen4_adapter_contract(
        root,
        maximum_artifact_bytes=maximum_artifact_bytes,
        requested_modes=requested_modes,
    )
    weight_map, _, _ = _bounded_index(root / "model.safetensors.index.json")
    expected: defaultdict[str, set[str]] = defaultdict(set)
    for tensor_name, shard_name in weight_map.items():
        expected[shard_name].add(tensor_name)
    dtype_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    tensor_count = 0
    for shard_name in sorted(expected):
        tensors = _inspect_header(
            root / shard_name,
            maximum_artifact_bytes=maximum_artifact_bytes,
            maximum_header_bytes=maximum_header_bytes,
        )
        if set(tensors) != expected[shard_name]:
            raise ValueError("Qwen4 safetensors header does not match the weight index")
        tensor_count += len(tensors)
        for descriptor in tensors.values():
            dtype_counts[descriptor["dtype"]] += 1
            rank_counts[descriptor["rank"]] += 1
    return {
        "schema_version": 1,
        "passed": True,
        "contract_id": contract["contract_id"],
        "config_fingerprint": contract["config_fingerprint"],
        "index_sha256": contract["index_sha256"],
        "requested_modes": list(requested_modes),
        "shard_count": len(expected),
        "tensor_count": tensor_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "rank_counts": {str(rank): count for rank, count in sorted(rank_counts.items())},
        "maximum_header_bytes": maximum_header_bytes,
        "peak_open_shards": 1,
        "reads_tensor_data": False,
        "allocates_model_or_metal": False,
    }
