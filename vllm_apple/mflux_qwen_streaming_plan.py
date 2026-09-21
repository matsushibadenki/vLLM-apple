"""Load-free, shard-bound text-encoder staging inventory for MFLUX Qwen Image."""
from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

_LAYER_NAME = re.compile(r"^encoder\.layers\.(\d+)\.")
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
    "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
_MAX_INDEX_BYTES = 4 * 1024 * 1024
_MAX_SHARDS = 128
_MAX_TENSORS = 100_000


@dataclass(frozen=True, slots=True)
class MFluxQwenStreamingPlan:
    shard_count: int
    tensor_count: int
    layer_count: int
    artifact_quantization_bits: int
    tensor_dtype_counts: dict[str, int]
    quantized_weight_tensor_count: int
    static_payload_bytes: int
    layer_payload_bytes: tuple[int, ...]
    total_payload_bytes: int
    maximum_layer_payload_bytes: int
    static_plus_maximum_layer_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_mflux_qwen_text_encoder_staging(
    model_root: Path, *, expected_layers: int = 28,
) -> MFluxQwenStreamingPlan:
    """Inspect only safetensors headers; never materialize weights or claim RSS fit."""
    if type(expected_layers) is not int or not 1 <= expected_layers <= 128:
        raise ValueError("expected Qwen text-encoder layers are invalid")
    component = model_root / "text_encoder"
    index = component / "model.safetensors.index.json"
    index_info = index.lstat()
    if not stat.S_ISREG(index_info.st_mode) or not 1 <= index_info.st_size <= _MAX_INDEX_BYTES:
        raise ValueError("Qwen text-encoder index must be a bounded regular file")
    payload = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"metadata", "weight_map"}:
        raise ValueError("Qwen text-encoder index schema is invalid")
    metadata, weight_map = payload["metadata"], payload["weight_map"]
    if not isinstance(metadata, dict) or metadata.get("quantization_level") != "4":
        raise ValueError("Qwen text encoder requires a 4-bit MFLUX artifact")
    if not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= _MAX_TENSORS:
        raise ValueError("Qwen text-encoder weight map is invalid")
    shard_names = set(weight_map.values())
    if (
        not 1 <= len(shard_names) <= _MAX_SHARDS
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[0-9]+\.safetensors", name)
            for name in shard_names
        )
    ):
        raise ValueError("Qwen text-encoder shard names are invalid")
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ValueError("Qwen text-encoder tensor names are invalid")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for load-free staging inspection") from error

    layer_bytes = [0] * expected_layers
    static_bytes = 0
    dtype_counts: dict[str, int] = {}
    quantized_weights = 0
    observed: set[str] = set()
    for shard_name in sorted(shard_names):
        shard = component / shard_name
        shard_info = shard.lstat()
        if not stat.S_ISREG(shard_info.st_mode) or shard_info.st_size < 16:
            raise ValueError("Qwen text-encoder shard must be a regular file")
        with safe_open(str(shard), framework="numpy") as handle:
            for name in handle.keys():
                if name in observed or weight_map.get(name) != shard_name:
                    raise ValueError("Qwen text-encoder index and shard headers disagree")
                observed.add(name)
                tensor = handle.get_slice(name)
                dtype = tensor.get_dtype()
                shape = tensor.get_shape()
                if dtype not in _DTYPE_BYTES or not 1 <= len(shape) <= 8 or any(
                    type(dimension) is not int or dimension <= 0 for dimension in shape
                ):
                    raise ValueError("Qwen text-encoder tensor descriptor is invalid")
                size = math.prod(shape) * _DTYPE_BYTES[dtype]
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                if name.endswith(".scales") or (
                    name.endswith(".weight") and dtype in {"U32", "U64"}
                ):
                    quantized_weights += 1
                layer = _LAYER_NAME.match(name)
                if layer is None:
                    static_bytes += size
                else:
                    index_number = int(layer.group(1))
                    if index_number >= expected_layers:
                        raise ValueError("Qwen text-encoder layer exceeds expected depth")
                    layer_bytes[index_number] += size
        after = shard.stat()
        if (shard_info.st_dev, shard_info.st_ino, shard_info.st_size, shard_info.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("Qwen text-encoder shard changed during inspection")
    if observed != set(weight_map) or any(size <= 0 for size in layer_bytes):
        raise ValueError("Qwen text-encoder shard coverage is incomplete")
    maximum = max(layer_bytes)
    return MFluxQwenStreamingPlan(
        len(shard_names), len(observed), expected_layers, 4,
        dict(sorted(dtype_counts.items())), quantized_weights, static_bytes,
        tuple(layer_bytes), static_bytes + sum(layer_bytes), maximum,
        static_bytes + maximum,
    )
