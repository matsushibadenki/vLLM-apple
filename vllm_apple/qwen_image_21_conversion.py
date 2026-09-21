from __future__ import annotations

import json
import math
import os
import shutil
import struct
from pathlib import Path

from .qwen_image_21_residency import build_qwen_image_21_residency_plan
from .types import HardwareInfo


MAX_ARTIFACT_FILES = 4096
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
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
}


def _largest_safetensors_shard(root: Path) -> int:
    largest = 0
    count = 0
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = [name for name in directory_names if name != ".git"]
        for name in file_names:
            count += 1
            if count > MAX_ARTIFACT_FILES:
                raise ValueError("artifact file count exceeds bounded limit")
            if not name.endswith(".safetensors"):
                continue
            path = Path(directory, name)
            if path.is_symlink():
                raise ValueError("conversion source must not contain symlinks")
            largest = max(largest, path.stat(follow_symlinks=False).st_size)
    if largest <= 0:
        raise ValueError("conversion source has no safetensors shards")
    return largest


def _largest_safetensors_tensor(root: Path) -> int:
    largest = 0
    count = 0
    for path in sorted(root.glob("*.safetensors")):
        count += 1
        if count > MAX_ARTIFACT_FILES:
            raise ValueError("artifact file count exceeds bounded limit")
        if path.is_symlink() or not path.is_file():
            raise ValueError("conversion source must contain regular safetensors files")
        file_size = path.stat(follow_symlinks=False).st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("safetensors header is truncated")
            header_size = struct.unpack("<Q", raw_length)[0]
            if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("safetensors header is outside the bounded limit")
            try:
                header = json.loads(handle.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("safetensors header is invalid") from error
        if not isinstance(header, dict):
            raise ValueError("safetensors header root is invalid")
        data_size = file_size - 8 - header_size
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise ValueError("safetensors tensor metadata is invalid")
            dtype, shape, offsets = (
                metadata.get("dtype"),
                metadata.get("shape"),
                metadata.get("data_offsets"),
            )
            if dtype not in SAFETENSORS_DTYPE_BYTES or not isinstance(shape, list):
                raise ValueError("safetensors tensor dtype or shape is invalid")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError("safetensors tensor offsets are invalid")
            if any(not isinstance(value, int) or value < 0 for value in (*shape, *offsets)):
                raise ValueError("safetensors tensor dimensions are invalid")
            tensor_bytes = SAFETENSORS_DTYPE_BYTES[dtype]
            for dimension in shape:
                tensor_bytes *= dimension
            if offsets[1] < offsets[0] or offsets[1] > data_size:
                raise ValueError("safetensors tensor offsets are outside the file")
            if offsets[1] - offsets[0] != tensor_bytes:
                raise ValueError("safetensors tensor byte span does not match metadata")
            largest = max(largest, tensor_bytes)
    if largest <= 0:
        raise ValueError("conversion source has no safetensors tensors")
    return largest


def build_qwen_image_21_conversion_plan(
    artifact: dict[str, object],
    hardware: HardwareInfo,
    *,
    source: str | Path,
    output: str | Path,
    conversion_ready: bool,
) -> dict[str, object]:
    source_root = Path(source).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("conversion source must be a directory")
    output_path = Path(output).expanduser().absolute()
    if output_path.exists():
        raise ValueError("conversion output must not already exist")
    output_parent = output_path.parent.resolve(strict=True)
    if output_parent == source_root or output_parent.is_relative_to(source_root):
        raise ValueError("conversion output must be outside the source artifact")

    residency = build_qwen_image_21_residency_plan(artifact, hardware, target_bits=8)
    largest_shard = _largest_safetensors_shard(source_root)
    transient_peak = residency["projected_resident_bytes"] + largest_shard
    memory_ceiling = residency["dynamic_safe_ceiling_bytes"]
    disk_required = math.ceil(residency["projected_weight_bytes"] * 1.15)
    disk_free = shutil.disk_usage(output_parent).free
    issues: list[str] = []
    if not conversion_ready:
        issues.append("torchao_conversion_runtime_not_ready")
    if transient_peak > memory_ceiling:
        issues.append("conversion_peak_exceeds_dynamic_memory_ceiling")
    if disk_required > disk_free:
        issues.append("conversion_output_exceeds_free_disk")
    return {
        "schema_version": 1,
        "candidate_id": "qwen-image-2.1",
        "source": str(source_root),
        "output": str(output_path),
        "target_quantization": "int8-weight-only",
        "largest_source_shard_bytes": largest_shard,
        "projected_output_weight_bytes": residency["projected_weight_bytes"],
        "projected_steady_resident_bytes": residency["projected_resident_bytes"],
        "estimated_conversion_peak_bytes": transient_peak,
        "dynamic_memory_ceiling_bytes": memory_ceiling,
        "minimum_available_memory_bytes": transient_peak
        + residency["emergency_reserve_bytes"],
        "disk_required_bytes": disk_required,
        "disk_free_bytes": disk_free,
        "conversion_ready": conversion_ready,
        "weights_loaded": False,
        "issues": issues,
        "eligible": not issues,
    }


def build_qwen_image_21_streaming_conversion_plan(
    artifact: dict[str, object],
    hardware: HardwareInfo,
    *,
    source: str | Path,
    output: str | Path,
    conversion_ready: bool,
) -> dict[str, object]:
    source_root = Path(source).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("conversion source must be a directory")
    output_path = Path(output).expanduser().absolute()
    if output_path.exists():
        raise ValueError("conversion output must not already exist")
    output_parent = output_path.parent.resolve(strict=True)
    if output_parent == source_root or output_parent.is_relative_to(source_root):
        raise ValueError("conversion output must be outside the source artifact")

    residency = build_qwen_image_21_residency_plan(artifact, hardware, target_bits=8)
    raw_components = artifact.get("components")
    assert isinstance(raw_components, list)
    by_role = {
        component["role"]: component
        for component in raw_components
        if isinstance(component, dict) and component.get("role") in {"denoiser", "text_encoder"}
    }
    phases = []
    allocator_margin = 1024**3
    for component_name, role in (("transformer", "denoiser"), ("text_encoder", "text_encoder")):
        component = by_role.get(role)
        if not isinstance(component, dict) or not isinstance(component.get("artifact_bytes"), int):
            raise ValueError(f"missing conversion component role: {role}")
        source_bytes = component["artifact_bytes"]
        projected_bytes = (source_bytes * 8 + 15) // 16
        largest_shard = _largest_safetensors_shard(source_root / component_name)
        largest_tensor = _largest_safetensors_tensor(source_root / component_name)
        peak = largest_tensor + projected_bytes + allocator_margin
        phases.append(
            {
                "component": component_name,
                "role": role,
                "source_bytes": source_bytes,
                "projected_bytes": projected_bytes,
                "largest_source_shard_bytes": largest_shard,
                "largest_source_tensor_bytes": largest_tensor,
                "estimated_peak_bytes": peak,
            }
        )
    conversion_peak = max(phase["estimated_peak_bytes"] for phase in phases)
    emergency_reserve = residency["emergency_reserve_bytes"]
    memory_ceiling = residency["dynamic_safe_ceiling_bytes"]
    disk_required = math.ceil(residency["projected_weight_bytes"] * 1.15)
    disk_free = shutil.disk_usage(output_parent).free
    issues: list[str] = []
    if not conversion_ready:
        issues.append("torchao_conversion_runtime_not_ready")
    if conversion_peak > memory_ceiling:
        issues.append("streaming_conversion_peak_exceeds_dynamic_memory_ceiling")
    if disk_required > disk_free:
        issues.append("conversion_output_exceeds_free_disk")
    return {
        "schema_version": 1,
        "candidate_id": "qwen-image-2.1",
        "strategy": "component-streaming-int8-weight-only",
        "source": str(source_root),
        "output": str(output_path),
        "phases": phases,
        "estimated_conversion_peak_bytes": conversion_peak,
        "minimum_available_memory_bytes": conversion_peak + emergency_reserve,
        "dynamic_memory_ceiling_bytes": memory_ceiling,
        "projected_output_weight_bytes": residency["projected_weight_bytes"],
        "disk_required_bytes": disk_required,
        "disk_free_bytes": disk_free,
        "conversion_ready": conversion_ready,
        "weights_loaded": False,
        "issues": issues,
        "eligible": not issues,
    }
