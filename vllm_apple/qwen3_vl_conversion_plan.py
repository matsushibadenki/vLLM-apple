"""Bounded Core ML conversion plan for a Qwen3-VL BF16 vision tower."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec, _read_json, _regular_file_size
from .qwen4_adapter_loader import _inspect_header


_BLOCK = re.compile(
    r"^vision_tower\.blocks\.(\d+)\."
    r"(attn\.(?:qkv|proj)|mlp\.linear_fc[12]|norm[12])\.(weight|bias)$"
)
_MERGER = re.compile(
    r"^vision_tower\.(merger|deepstack_merger_list\.(\d+))\."
    r"(linear_fc[12]|norm)\.(weight|bias)$"
)


@dataclass(frozen=True, slots=True)
class Qwen3VLCoreMLConversionPlan:
    source_artifact_fingerprint: str
    model_revision: str
    source_precision: str
    target_precision: str
    tensor_count: int
    source_tensor_bytes: int
    target_tensor_bytes: int
    fixed_grid_profiles: tuple[tuple[int, int, int], ...]
    transform_counts: tuple[tuple[str, int], ...]
    plan_id: str


def build_qwen3_vl_coreml_conversion_plan(
    model_path: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    *,
    fixed_grid_profiles: tuple[tuple[int, int, int], ...],
    target_precision: str = "fp16",
) -> Qwen3VLCoreMLConversionPlan:
    """Inspect every vision tensor without materializing tensor payloads."""
    if (
        not isinstance(source, Qwen3VLVisionANEAdapterSpec)
        or source.source_precision not in {"bf16", "fp16", "fp32"}
        or target_precision not in {"fp16", "fp32"}
        or not 1 <= len(fixed_grid_profiles) <= 16
        or len(set(fixed_grid_profiles)) != len(fixed_grid_profiles)
    ):
        raise ValueError("invalid Qwen3-VL Core ML conversion plan request")
    for frames, height, width in fixed_grid_profiles:
        if (
            any(type(value) is not int or value <= 0 for value in (frames, height, width))
            or height % source.spatial_merge_size
            or width % source.spatial_merge_size
            or height * width > 2304
        ):
            raise ValueError("Qwen3-VL Core ML grid profile is invalid")
    root = model_path.expanduser().resolve(strict=True)
    _, index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Qwen3-VL weight index is invalid")
    vision_names = {name for name in weight_map if name.startswith("vision_tower.")}
    if len(vision_names) != source.vision_tensor_count:
        raise ValueError("Qwen3-VL conversion tensor inventory changed")
    descriptors: dict[str, dict[str, object]] = {}
    for shard in sorted({weight_map[name] for name in vision_names}):
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Qwen3-VL conversion shard path is unsafe")
        shard_path = root / shard
        shard_size = _regular_file_size(shard_path)
        inspected = _inspect_header(
            shard_path,
            maximum_artifact_bytes=shard_size,
            maximum_header_bytes=64 * 1024 * 1024,
        )
        for name in vision_names:
            if weight_map[name] == shard:
                if name not in inspected:
                    raise ValueError("Qwen3-VL conversion tensor is absent from shard")
                descriptors[name] = inspected[name]
    transforms = Counter()
    source_bytes = 0
    for name in sorted(vision_names):
        descriptor = descriptors[name]
        dtype = descriptor["dtype"]
        expected_dtype = {"bf16": "BF16", "fp16": "F16", "fp32": "F32"}[
            source.source_precision
        ]
        if dtype != expected_dtype:
            raise ValueError("Qwen3-VL conversion tensor precision changed")
        transform = _tensor_transform(name, descriptor["shape"], source)
        transforms[transform] += 1
        source_bytes += descriptor["bytes"]
    target_bytes_per_element = 2 if target_precision == "fp16" else 4
    source_bytes_per_element = {"bf16": 2, "fp16": 2, "fp32": 4}[
        source.source_precision
    ]
    target_bytes = source_bytes * target_bytes_per_element // source_bytes_per_element
    identity = {
        "schema_version": 1,
        "source_artifact_fingerprint": source.artifact_fingerprint,
        "model_revision": source.model_revision,
        "source_precision": source.source_precision,
        "target_precision": target_precision,
        "tensor_count": len(vision_names),
        "source_tensor_bytes": source_bytes,
        "target_tensor_bytes": target_bytes,
        "fixed_grid_profiles": [list(value) for value in fixed_grid_profiles],
        "transform_counts": dict(sorted(transforms.items())),
        "requires_full_model_load": False,
        "peak_open_source_shards": 1,
        "preserves_deepstack_outputs": True,
    }
    plan_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen3VLCoreMLConversionPlan(
        source.artifact_fingerprint,
        source.model_revision,
        source.source_precision,
        target_precision,
        len(vision_names),
        source_bytes,
        target_bytes,
        fixed_grid_profiles,
        tuple(sorted(transforms.items())),
        plan_id,
    )


def _tensor_transform(
    name: str, shape: object, source: Qwen3VLVisionANEAdapterSpec
) -> str:
    if not isinstance(shape, list) or any(type(value) is not int for value in shape):
        raise ValueError("Qwen3-VL conversion tensor shape is invalid")
    if name == "vision_tower.patch_embed.proj.weight":
        expected = [
            source.hidden_size, source.temporal_patch_size, source.patch_size,
            source.patch_size, 3,
        ]
        if shape != expected:
            raise ValueError("Qwen3-VL patch embedding shape changed")
        return "conv3d_mlx_otwci_to_coreml_oictw"
    if name == "vision_tower.patch_embed.proj.bias":
        return _vector(shape, source.hidden_size, "patch_bias")
    if name == "vision_tower.pos_embed.weight":
        if len(shape) != 2 or shape[1] != source.hidden_size:
            raise ValueError("Qwen3-VL position embedding shape changed")
        return "position_embedding"
    block = _BLOCK.fullmatch(name)
    if block:
        layer, component, kind = int(block.group(1)), block.group(2), block.group(3)
        if layer >= source.depth:
            raise ValueError("Qwen3-VL conversion block index is outside model depth")
        dimensions = {
            "attn.qkv": (source.hidden_size * 3, source.hidden_size),
            "attn.proj": (source.hidden_size, source.hidden_size),
            "mlp.linear_fc1": (source.intermediate_size, source.hidden_size),
            "mlp.linear_fc2": (source.hidden_size, source.intermediate_size),
        }
        if component.startswith("norm"):
            return _vector(shape, source.hidden_size, "layer_norm_parameter")
        output, input_size = dimensions[component]
        expected = [output, input_size] if kind == "weight" else [output]
        if shape != expected:
            raise ValueError("Qwen3-VL conversion block tensor shape changed")
        return "linear_weight_out_in" if kind == "weight" else "linear_bias"
    merger = _MERGER.fullmatch(name)
    if merger:
        group, deepstack_index, component, kind = merger.groups()
        if group.startswith("deepstack") and int(deepstack_index) >= len(
            source.deepstack_visual_indexes
        ):
            raise ValueError("Qwen3-VL deep-stack merger index is invalid")
        merged = source.hidden_size * source.spatial_merge_size**2
        norm_size = merged if group.startswith("deepstack") else source.hidden_size
        if component == "norm":
            return _vector(shape, norm_size, "merger_norm_parameter")
        output, input_size = (
            (merged, merged) if component == "linear_fc1"
            else (source.output_hidden_size, merged)
        )
        expected = [output, input_size] if kind == "weight" else [output]
        if shape != expected:
            raise ValueError("Qwen3-VL merger tensor shape changed")
        return "linear_weight_out_in" if kind == "weight" else "linear_bias"
    raise ValueError("Qwen3-VL conversion contains an unclassified vision tensor")


def _vector(shape: list[int], size: int, label: str) -> str:
    if shape != [size]:
        raise ValueError(f"Qwen3-VL {label} shape changed")
    return label
