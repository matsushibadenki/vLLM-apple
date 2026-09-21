"""Strict fixed-shape graph contract for Qwen3-VL Core ML compilation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_conversion_plan import Qwen3VLCoreMLConversionPlan

MAX_STAGING_MANIFEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Qwen3VLCoreMLGraphProfile:
    grid_thw: tuple[int, int, int]
    pixel_values_shape: tuple[int, int]
    hidden_states_shape: tuple[int, int]
    deepstack_output_shapes: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Qwen3VLCoreMLGraphSpec:
    source_artifact_fingerprint: str
    model_revision: str
    conversion_plan_id: str
    staged_weights_sha256: str
    compute_precision: str
    depth: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    head_dimension: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    output_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]
    profiles: tuple[Qwen3VLCoreMLGraphProfile, ...]
    graph_semantics: tuple[str, ...]
    graph_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_qwen3_vl_coreml_graph_spec(
    staged_weights: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
) -> Qwen3VLCoreMLGraphSpec:
    """Validate staged weights and freeze every dynamic dimension for compilation."""
    if (
        not isinstance(source, Qwen3VLVisionANEAdapterSpec)
        or not isinstance(plan, Qwen3VLCoreMLConversionPlan)
        or plan.source_artifact_fingerprint != source.artifact_fingerprint
        or plan.model_revision != source.model_revision
        or plan.target_precision != "fp16"
    ):
        raise ValueError("Qwen3-VL graph source does not match conversion plan")
    root = staged_weights.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Qwen3-VL staged weights must be a directory")
    manifest = _load_staging_manifest(root / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("plan_id") != plan.plan_id
        or manifest.get("source_artifact_fingerprint") != source.artifact_fingerprint
        or manifest.get("model_revision") != source.model_revision
        or manifest.get("tensor_count") != plan.tensor_count
        or manifest.get("total_bytes") != plan.target_tensor_bytes
    ):
        raise ValueError("Qwen3-VL staged weights do not match graph source")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != plan.tensor_count:
        raise ValueError("Qwen3-VL staged weight records are invalid")
    staged_digest = _verify_staged_weights(root, records, plan.target_tensor_bytes)
    profiles = tuple(_profile(source, grid) for grid in plan.fixed_grid_profiles)
    semantics = (
        "patch_embedding_conv3d_oictw",
        "position_embedding_bilinear_interpolation",
        "vision_rope_theta_10000",
        "pre_norm_attention_scaled_dot_product",
        "pre_norm_mlp_gelu_tanh",
        "residual_connections",
        "deepstack_postshuffle_merger_gelu",
        "final_preshuffle_merger_gelu",
    )
    identity = {
        "schema_version": 1,
        "source_artifact_fingerprint": source.artifact_fingerprint,
        "model_revision": source.model_revision,
        "conversion_plan_id": plan.plan_id,
        "staged_weights_sha256": staged_digest,
        "compute_precision": plan.target_precision,
        "depth": source.depth,
        "hidden_size": source.hidden_size,
        "intermediate_size": source.intermediate_size,
        "attention_heads": source.attention_heads,
        "head_dimension": source.hidden_size // source.attention_heads,
        "patch_size": source.patch_size,
        "temporal_patch_size": source.temporal_patch_size,
        "spatial_merge_size": source.spatial_merge_size,
        "output_hidden_size": source.output_hidden_size,
        "deepstack_visual_indexes": list(source.deepstack_visual_indexes),
        "profiles": [asdict(profile) for profile in profiles],
        "graph_semantics": list(semantics),
    }
    graph_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen3VLCoreMLGraphSpec(
        source.artifact_fingerprint,
        source.model_revision,
        plan.plan_id,
        staged_digest,
        plan.target_precision,
        source.depth,
        source.hidden_size,
        source.intermediate_size,
        source.attention_heads,
        source.hidden_size // source.attention_heads,
        source.patch_size,
        source.temporal_patch_size,
        source.spatial_merge_size,
        source.output_hidden_size,
        source.deepstack_visual_indexes,
        profiles,
        semantics,
        graph_id,
    )


def _profile(
    source: Qwen3VLVisionANEAdapterSpec, grid: tuple[int, int, int]
) -> Qwen3VLCoreMLGraphProfile:
    frames, height, width = grid
    tokens = frames * height * width
    divisor = source.spatial_merge_size**2
    if tokens % divisor:
        raise ValueError("Qwen3-VL graph profile cannot be spatially merged")
    patch_elements = 3 * source.temporal_patch_size * source.patch_size**2
    output = (tokens // divisor, source.output_hidden_size)
    return Qwen3VLCoreMLGraphProfile(
        grid,
        (tokens, patch_elements),
        output,
        tuple(output for _ in source.deepstack_visual_indexes),
    )


def _load_staging_manifest(path: Path) -> dict[str, object]:
    try:
        attributes = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(attributes.st_mode)
            or not 1 <= attributes.st_size <= MAX_STAGING_MANIFEST_BYTES
        ):
            raise ValueError("Qwen3-VL staging manifest is invalid")
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen3-VL staging manifest is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Qwen3-VL staging manifest must be an object")
    return payload


def _verify_staged_weights(root: Path, records: list, expected_bytes: int) -> str:
    expected_fields = {"name", "file", "dtype", "shape", "bytes", "sha256", "transform"}
    canonical = []
    filenames: set[str] = set()
    names: set[str] = set()
    total = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("Qwen3-VL staged weight record fields are invalid")
        name, filename = record["name"], record["file"]
        size, digest = record["bytes"], record["sha256"]
        shape = record["shape"]
        if (
            not isinstance(name, str)
            or not name.startswith("vision_tower.")
            or name in names
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in filenames
            or record["dtype"] != "F16"
            or type(size) is not int
            or size < 2
            or not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value < 1 for value in shape)
            or math.prod(shape) * 2 != size
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
            or not isinstance(record["transform"], str)
        ):
            raise ValueError("Qwen3-VL staged weight record is invalid")
        path = root / filename
        attributes = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_size != size:
            raise ValueError("Qwen3-VL staged weight file is invalid")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            file_hash = hashlib.sha256()
            while chunk := handle.read(8 * 1024 * 1024):
                file_hash.update(chunk)
            actual = file_hash.hexdigest()
        if actual != digest:
            raise ValueError("Qwen3-VL staged weight digest does not match")
        names.add(name)
        filenames.add(filename)
        total += size
        canonical.append(record)
    if total != expected_bytes:
        raise ValueError("Qwen3-VL staged weight total does not match")
    actual_files = {path.name for path in root.iterdir()}
    if actual_files != filenames | {"manifest.json"}:
        raise ValueError("Qwen3-VL staged weight directory contains unexpected files")
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
