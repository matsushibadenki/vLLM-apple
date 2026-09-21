"""Selective BF16/F32 safetensors reader for one MFLUX Qwen text layer."""
from __future__ import annotations

import json
import math
import os
import stat
import struct
from pathlib import Path

from .mflux_qwen_streaming_plan import inspect_mflux_qwen_text_encoder_staging

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_LAYER_BYTES = 1024 * 1024 * 1024
_STORAGE_DTYPES = {"BF16": (2, "<u2"), "F16": (2, "<u2"), "F32": (4, "<f4")}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("safetensors header contains a duplicate key")
        result[name] = value
    return result


def _read_selected_tensors(
    path: Path, names: tuple[str, ...], *, maximum_bytes: int = MAX_LAYER_BYTES,
) -> dict[str, object]:
    """Read exact tensor byte ranges; no whole-shard weight materialization."""
    import mlx.core as mx
    import numpy as np

    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 2 * 1024**3:
        raise ValueError("selected Qwen text byte budget is invalid")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 16:
            raise ValueError("Qwen text shard must be a regular file")
        prefix = os.pread(descriptor, 8, 0)
        if len(prefix) != 8:
            raise ValueError("safetensors header length is missing")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= MAX_HEADER_BYTES or 8 + header_size > info.st_size:
            raise ValueError("safetensors header is outside the bounded policy")
        encoded = os.pread(descriptor, header_size, 8)
        if len(encoded) != header_size:
            raise ValueError("safetensors header is truncated")
        try:
            header = json.loads(encoded, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("safetensors header JSON is invalid") from error
        if not isinstance(header, dict):
            raise ValueError("safetensors header root is invalid")
        data_start = 8 + header_size
        result = {}
        total = 0
        for name in names:
            entry = header.get(name)
            if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("selected Qwen text tensor is missing or malformed")
            dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
            if (
                dtype not in _STORAGE_DTYPES
                or not isinstance(shape, list)
                or not 1 <= len(shape) <= 8
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
            ):
                raise ValueError("selected Qwen text tensor descriptor is invalid")
            tensor_bytes = math.prod(shape) * _STORAGE_DTYPES[dtype][0]
            if (
                offsets[1] - offsets[0] != tensor_bytes
                or data_start + offsets[1] > info.st_size
                or tensor_bytes > maximum_bytes
            ):
                raise ValueError("selected Qwen text tensor bounds are invalid")
            total += tensor_bytes
            if total > maximum_bytes:
                raise ValueError("selected Qwen text layer exceeds the byte budget")
            raw = os.pread(descriptor, tensor_bytes, data_start + offsets[0])
            if len(raw) != tensor_bytes:
                raise ValueError("selected Qwen text tensor is truncated")
            values = np.frombuffer(raw, dtype=_STORAGE_DTYPES[dtype][1])
            tensor = mx.array(values)
            if dtype == "BF16":
                tensor = tensor.view(mx.bfloat16)
            elif dtype == "F16":
                tensor = tensor.view(mx.float16)
            tensor = tensor.reshape(shape)
            mx.eval(tensor)
            result[name] = tensor
        after = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("Qwen text shard changed while reading")
        return result
    finally:
        os.close(descriptor)


def load_mflux_qwen_text_layer(model_root: Path, layer_index: int):
    """Load exactly one verified BF16 Qwen text layer into an MLX module."""
    if type(layer_index) is not int or not 0 <= layer_index < 28:
        raise ValueError("Qwen text layer index is outside the fixed profile")
    plan = inspect_mflux_qwen_text_encoder_staging(model_root)
    if plan.quantized_weight_tensor_count or set(plan.tensor_dtype_counts) != {"BF16", "F32"}:
        raise ValueError("Qwen text layer requires the verified BF16/F32 artifact")
    component = model_root / "text_encoder"
    payload = json.loads((component / "model.safetensors.index.json").read_text())
    prefix = f"encoder.layers.{layer_index}."
    grouped: dict[str, list[str]] = {}
    for name, shard_name in payload["weight_map"].items():
        if name.startswith(prefix):
            grouped.setdefault(shard_name, []).append(name)
    if not grouped:
        raise ValueError("Qwen text layer has no indexed weights")

    from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
    from mlx.utils import tree_unflatten

    flattened = []
    for shard_name, names in sorted(grouped.items()):
        for name, tensor in _read_selected_tensors(
            component / shard_name, tuple(sorted(names))
        ).items():
            flattened.append((name.removeprefix(prefix), tensor))
    layer = QwenEncoderLayer()
    layer.update(tree_unflatten(flattened), strict=True)
    return layer
