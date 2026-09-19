"""Fail-closed Qwen3-VL vision-tower admission for future Core ML routing."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .qwen4_adapter_loader import _inspect_header

MAX_METADATA_BYTES = 1024 * 1024
MAX_WEIGHT_ENTRIES = 100_000
QWEN3_VL_ARCHITECTURE = "Qwen3VLForConditionalGeneration"


@dataclass(frozen=True, slots=True)
class Qwen3VLVisionANEAdapterSpec:
    model_revision: str
    artifact_fingerprint: str
    operator: str
    depth: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    output_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]
    vision_tensor_count: int
    artifact_bytes: int
    source_precision: str
    coreml_artifact_ready: bool

    def __post_init__(self) -> None:
        if (
            len(self.model_revision) != 40
            or any(value not in "0123456789abcdef" for value in self.model_revision)
            or len(self.artifact_fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in self.artifact_fingerprint)
            or self.operator != f"vision_encoder@{self.artifact_fingerprint[:16]}"
            or not 1 <= self.depth <= 256
            or not 1 <= self.hidden_size <= 65_536
            or not 1 <= self.intermediate_size <= 262_144
            or not 1 <= self.attention_heads <= 1024
            or self.hidden_size % self.attention_heads
            or not 1 <= self.patch_size <= 256
            or not 1 <= self.temporal_patch_size <= 256
            or not 1 <= self.spatial_merge_size <= 256
            or not 1 <= self.output_hidden_size <= 65_536
            or not self.deepstack_visual_indexes
            or tuple(sorted(set(self.deepstack_visual_indexes)))
            != self.deepstack_visual_indexes
            or any(not 0 <= value < self.depth for value in self.deepstack_visual_indexes)
            or self.vision_tensor_count < self.depth
            or self.artifact_bytes < 1
            or self.source_precision not in {"affine-int4", "bf16", "fp16", "fp32"}
            or type(self.coreml_artifact_ready) is not bool
        ):
            raise ValueError("invalid Qwen3-VL ANE adapter specification")


def inspect_qwen3_vl_vision_for_ane(
    model_path: Path, *, model_revision: str
) -> Qwen3VLVisionANEAdapterSpec:
    """Bind Qwen3-VL vision metadata and its complete indexed tensor inventory.

    This is static admission only. The MLX checkpoint is never reported as an
    executable Core ML artifact; conversion and numerical qualification remain
    separate gates.
    """
    if (
        not isinstance(model_revision, str)
        or len(model_revision) != 40
        or any(value not in "0123456789abcdef" for value in model_revision)
    ):
        raise ValueError("Qwen3-VL model revision must be a lowercase commit SHA")
    root = model_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Qwen3-VL model path must be a directory")
    config_raw, config = _read_json(root / "config.json")
    processor_raw, processor = _read_json(root / "preprocessor_config.json")
    index_raw, index = _read_json(root / "model.safetensors.index.json")
    if (
        config.get("model_type") != "qwen3_vl"
        or config.get("architectures") != [QWEN3_VL_ARCHITECTURE]
        or processor.get("processor_class") != "Qwen3VLProcessor"
    ):
        raise ValueError("model is not a supported Qwen3-VL vision artifact")
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("Qwen3-VL vision configuration is missing")
    fields = {
        name: vision.get(name)
        for name in (
            "depth", "hidden_size", "intermediate_size", "num_heads",
            "patch_size", "temporal_patch_size", "spatial_merge_size",
            "out_hidden_size",
        )
    }
    if any(type(value) is not int for value in fields.values()):
        raise ValueError("Qwen3-VL vision shape is incomplete")
    deepstack = vision.get("deepstack_visual_indexes")
    if not isinstance(deepstack, list) or any(type(value) is not int for value in deepstack):
        raise ValueError("Qwen3-VL deep-stack metadata is invalid")
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not 1 <= len(weight_map) <= MAX_WEIGHT_ENTRIES
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items())
    ):
        raise ValueError("Qwen3-VL weight index is invalid")
    vision_keys = frozenset(key for key in weight_map if key.startswith("vision_tower."))
    _require_vision_tensors(vision_keys, fields["depth"], tuple(deepstack))
    shards = frozenset(weight_map[key] for key in vision_keys)
    artifact_bytes = 0
    vision_dtypes: set[str] = set()
    for shard in sorted(shards):
        if Path(shard).name != shard:
            raise ValueError("Qwen3-VL weight shard path is unsafe")
        shard_path = root / shard
        shard_bytes = _regular_file_size(shard_path)
        artifact_bytes += shard_bytes
        tensors = _inspect_header(
            shard_path,
            maximum_artifact_bytes=shard_bytes,
            maximum_header_bytes=64 * 1024 * 1024,
        )
        expected_shard_keys = {key for key, value in weight_map.items() if value == shard}
        if set(tensors) != expected_shard_keys:
            raise ValueError("Qwen3-VL safetensors header does not match weight index")
        vision_dtypes.update(tensors[key]["dtype"] for key in vision_keys if key in tensors)
    if len(vision_dtypes) != 1 or next(iter(vision_dtypes)) not in {"BF16", "F16", "F32"}:
        raise ValueError("Qwen3-VL vision tensor precision is unsupported")
    source_precision = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}[
        next(iter(vision_dtypes))
    ]
    fingerprint = hashlib.sha256(
        b"vllm-apple-qwen3-vl-ane-v1\0"
        + model_revision.encode("ascii") + b"\0"
        + config_raw + b"\0" + processor_raw + b"\0" + index_raw
    ).hexdigest()
    return Qwen3VLVisionANEAdapterSpec(
        model_revision=model_revision,
        artifact_fingerprint=fingerprint,
        operator=f"vision_encoder@{fingerprint[:16]}",
        depth=fields["depth"],
        hidden_size=fields["hidden_size"],
        intermediate_size=fields["intermediate_size"],
        attention_heads=fields["num_heads"],
        patch_size=fields["patch_size"],
        temporal_patch_size=fields["temporal_patch_size"],
        spatial_merge_size=fields["spatial_merge_size"],
        output_hidden_size=fields["out_hidden_size"],
        deepstack_visual_indexes=tuple(deepstack),
        vision_tensor_count=len(vision_keys),
        artifact_bytes=artifact_bytes,
        source_precision=source_precision,
        coreml_artifact_ready=False,
    )


def _require_vision_tensors(
    keys: frozenset[str], depth: int, deepstack: tuple[int, ...]
) -> None:
    required = {
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.pos_embed.weight",
        "vision_tower.merger.linear_fc1.weight",
        "vision_tower.merger.linear_fc2.weight",
    }
    for layer in range(depth):
        required.update({
            f"vision_tower.blocks.{layer}.attn.qkv.weight",
            f"vision_tower.blocks.{layer}.attn.proj.weight",
            f"vision_tower.blocks.{layer}.mlp.linear_fc1.weight",
            f"vision_tower.blocks.{layer}.mlp.linear_fc2.weight",
            f"vision_tower.blocks.{layer}.norm1.weight",
            f"vision_tower.blocks.{layer}.norm2.weight",
        })
    for position, _layer in enumerate(deepstack):
        required.update({
            f"vision_tower.deepstack_merger_list.{position}.linear_fc1.weight",
            f"vision_tower.deepstack_merger_list.{position}.linear_fc2.weight",
        })
    missing = required - keys
    if missing:
        raise ValueError("Qwen3-VL vision tensor inventory is incomplete")


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError("Qwen3-VL metadata is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_METADATA_BYTES:
            raise ValueError("Qwen3-VL metadata exceeds size limit")
        chunks = bytearray()
        while chunk := os.read(descriptor, min(64 * 1024, MAX_METADATA_BYTES + 1 - len(chunks))):
            chunks.extend(chunk)
            if len(chunks) > MAX_METADATA_BYTES:
                raise ValueError("Qwen3-VL metadata exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("Qwen3-VL metadata changed while reading")
    raw = bytes(chunks)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen3-VL metadata is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Qwen3-VL metadata must be an object")
    return raw, value


def _regular_file_size(path: Path) -> int:
    try:
        attributes = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("Qwen3-VL artifact file is unavailable") from error
    if not stat.S_ISREG(attributes.st_mode) or attributes.st_size < 1:
        raise ValueError("Qwen3-VL artifact file must be regular")
    return attributes.st_size
