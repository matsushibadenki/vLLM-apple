"""Load-free, shard-bound inventory of MFLUX Qwen Image transformer blocks."""
from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .mflux_qwen_streaming_plan import _DTYPE_BYTES

_BLOCK_NAME = re.compile(r"^transformer_blocks\.(\d+)\.")
_SHARD_NAME = re.compile(r"[0-9]+\.safetensors\Z")
_MAX_INDEX_BYTES = 8 * 1024 * 1024
_MAX_TENSORS = 20_000
_MAX_SHARDS = 128


@dataclass(frozen=True, slots=True)
class MFluxQwenTransformerPlan:
    shard_count: int
    tensor_count: int
    block_count: int
    tensor_dtype_counts: dict[str, int]
    quantized_weight_tensor_count: int
    quantization_aux_tensor_count: int
    quantized_block_weight_counts: tuple[int, ...]
    static_payload_bytes: int
    block_payload_bytes: tuple[int, ...]
    total_payload_bytes: int
    maximum_block_payload_bytes: int
    static_plus_maximum_block_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_mflux_qwen_transformer_staging(
    model_root: Path, *, expected_blocks: int = 60,
) -> MFluxQwenTransformerPlan:
    """Inspect indexed safetensors descriptors without loading transformer weights."""
    if type(expected_blocks) is not int or not 1 <= expected_blocks <= 128:
        raise ValueError("expected Qwen transformer block count is invalid")
    component = model_root / "transformer"
    index = component / "model.safetensors.index.json"
    info = index.lstat()
    if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_INDEX_BYTES:
        raise ValueError("Qwen transformer index must be a bounded regular file")
    payload = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"metadata", "weight_map"}:
        raise ValueError("Qwen transformer index schema is invalid")
    metadata, weight_map = payload["metadata"], payload["weight_map"]
    if not isinstance(metadata, dict) or metadata.get("quantization_level") != "4":
        raise ValueError("Qwen transformer requires a 4-bit MFLUX artifact")
    if not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= _MAX_TENSORS:
        raise ValueError("Qwen transformer weight map is invalid")
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ValueError("Qwen transformer tensor names are invalid")
    shards = set(weight_map.values())
    if (
        not 1 <= len(shards) <= _MAX_SHARDS
        or any(not isinstance(name, str) or not _SHARD_NAME.fullmatch(name) for name in shards)
    ):
        raise ValueError("Qwen transformer shard names are invalid")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for transformer inspection") from error

    block_bytes = [0] * expected_blocks
    static_bytes = 0
    dtype_counts: dict[str, int] = {}
    quantized_weights = quantization_aux = 0
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    quantized_block_weights = [0] * expected_blocks
    observed: set[str] = set()
    for shard_name in sorted(shards):
        shard = component / shard_name
        before = shard.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size < 16:
            raise ValueError("Qwen transformer shard must be a regular file")
        with safe_open(str(shard), framework="numpy") as handle:
            for name in handle.keys():
                if name in observed or weight_map.get(name) != shard_name:
                    raise ValueError("Qwen transformer index and shard headers disagree")
                observed.add(name)
                descriptor = handle.get_slice(name)
                dtype, shape = descriptor.get_dtype(), descriptor.get_shape()
                if dtype not in _DTYPE_BYTES or not 1 <= len(shape) <= 8 or any(
                    type(dimension) is not int or dimension <= 0 for dimension in shape
                ):
                    raise ValueError("Qwen transformer tensor descriptor is invalid")
                size = math.prod(shape) * _DTYPE_BYTES[dtype]
                descriptors[name] = (dtype, tuple(shape))
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                if name.endswith(".weight") and dtype in {"U32", "U64"}:
                    quantized_weights += 1
                if name.endswith((".scales", ".biases")):
                    quantization_aux += 1
                match = _BLOCK_NAME.match(name)
                if match is None:
                    if name.startswith("transformer_blocks."):
                        raise ValueError("Qwen transformer block name is invalid")
                    static_bytes += size
                else:
                    block_index = int(match.group(1))
                    if block_index >= expected_blocks:
                        raise ValueError("Qwen transformer block exceeds expected depth")
                    block_bytes[block_index] += size
                    if name.endswith(".weight") and dtype == "U32":
                        quantized_block_weights[block_index] += 1
        after = shard.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("Qwen transformer shard changed during inspection")
    if (
        observed != set(weight_map)
        or any(size <= 0 for size in block_bytes)
        or quantized_weights <= 0
        or quantization_aux <= 0
    ):
        raise ValueError("Qwen transformer shard coverage is incomplete")
    if any(count <= 0 for count in quantized_block_weights):
        raise ValueError("Qwen transformer block quantization coverage is incomplete")
    for name, (dtype, shape) in descriptors.items():
        if dtype == "U32":
            if not name.endswith(".weight") or len(shape) != 2:
                raise ValueError("Qwen transformer packed weight layout is invalid")
            base = name[:-len(".weight")]
            scales = descriptors.get(base + ".scales")
            biases = descriptors.get(base + ".biases")
            if (
                scales is None or biases is None
                or scales[0] not in {"BF16", "F16"}
                or biases[0] != scales[0]
                or len(scales[1]) != 2
                or scales[1] != biases[1]
                or shape[0] != scales[1][0]
                or shape[1] != scales[1][1] * 8
            ):
                raise ValueError("Qwen transformer packed 4-bit layout is invalid")
        elif name.endswith((".scales", ".biases")):
            suffix = ".scales" if name.endswith(".scales") else ".biases"
            weight = descriptors.get(name[:-len(suffix)] + ".weight")
            if weight is None or weight[0] != "U32":
                raise ValueError("Qwen transformer quantization auxiliary is orphaned")
    if quantization_aux != 2 * quantized_weights:
        raise ValueError("Qwen transformer quantization auxiliary coverage is incomplete")
    maximum = max(block_bytes)
    return MFluxQwenTransformerPlan(
        len(shards), len(observed), expected_blocks, dict(sorted(dtype_counts.items())),
        quantized_weights, quantization_aux, tuple(quantized_block_weights),
        static_bytes, tuple(block_bytes),
        static_bytes + sum(block_bytes), maximum, static_bytes + maximum,
    )
