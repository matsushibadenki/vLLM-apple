"""Bounded BF16-to-FP16 staging worker for Qwen3-VL Core ML export."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec, _read_json, _regular_file_size
from .qwen3_vl_conversion_plan import Qwen3VLCoreMLConversionPlan
from .qwen4_adapter_loader import _inspect_header


CONVERSION_CHUNK_BYTES = 8 * 1024 * 1024


def stage_qwen3_vl_coreml_weights(
    model_path: Path,
    destination: Path,
    source: Qwen3VLVisionANEAdapterSpec,
    plan: Qwen3VLCoreMLConversionPlan,
    *,
    maximum_output_bytes: int,
) -> Path:
    """Create a private atomic weight package without loading the full model."""
    if (
        not isinstance(source, Qwen3VLVisionANEAdapterSpec)
        or not isinstance(plan, Qwen3VLCoreMLConversionPlan)
        or plan.source_artifact_fingerprint != source.artifact_fingerprint
        or plan.model_revision != source.model_revision
        or plan.source_precision != "bf16"
        or plan.target_precision != "fp16"
        or type(maximum_output_bytes) is not int
        or maximum_output_bytes < plan.target_tensor_bytes
    ):
        raise ValueError("Qwen3-VL Core ML staging request does not match its plan")
    root = model_path.expanduser().resolve(strict=True)
    output = destination.expanduser().resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Qwen3-VL Core ML staging destination must be new")
    _, index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Qwen3-VL staging weight index is invalid")
    names = sorted(name for name in weight_map if name.startswith("vision_tower."))
    if len(names) != plan.tensor_count:
        raise ValueError("Qwen3-VL staging tensor inventory changed")

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for Qwen3-VL Core ML weight staging; "
            "install vllm-apple[coreml] or run inside the vLLM-Metal environment"
        ) from error

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    records = []
    written = 0
    try:
        descriptors = _descriptors(root, weight_map, names)
        for name in names:
            shard_name = weight_map[name]
            descriptor = descriptors[name]
            if descriptor["dtype"] != "BF16":
                raise ValueError("Qwen3-VL staging source precision changed")
            target_shape = list(descriptor["shape"])
            transpose = name == "vision_tower.patch_embed.proj.weight"
            if transpose:
                target_shape = [
                    target_shape[0], target_shape[4], target_shape[1],
                    target_shape[2], target_shape[3],
                ]
            target_bytes = descriptor["bytes"]
            if written + target_bytes > maximum_output_bytes:
                raise ValueError("Qwen3-VL Core ML staging exceeds output limit")
            filename = hashlib.sha256(name.encode()).hexdigest() + ".fp16"
            target = temporary / filename
            digest = _convert_tensor(
                root / shard_name,
                descriptor,
                target,
                transpose_conv3d=transpose,
                numpy=np,
            )
            written += target_bytes
            records.append({
                "name": name,
                "file": filename,
                "dtype": "F16",
                "shape": target_shape,
                "bytes": target_bytes,
                "sha256": digest,
                "transform": (
                    "conv3d_mlx_otwci_to_coreml_oictw" if transpose else "bf16_to_fp16"
                ),
            })
        if written != plan.target_tensor_bytes:
            raise ValueError("Qwen3-VL staging output size does not match plan")
        manifest = {
            "schema_version": 1,
            "plan_id": plan.plan_id,
            "source_artifact_fingerprint": source.artifact_fingerprint,
            "model_revision": source.model_revision,
            "tensor_count": len(records),
            "total_bytes": written,
            "records": records,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.chmod(manifest_path, 0o600)
        _fsync_directory(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _descriptors(root: Path, weight_map: dict, names: list[str]) -> dict[str, dict]:
    result = {}
    for shard in sorted({weight_map[name] for name in names}):
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Qwen3-VL staging shard path is unsafe")
        path = root / shard
        size = _regular_file_size(path)
        inspected = _inspect_header(
            path, maximum_artifact_bytes=size, maximum_header_bytes=64 * 1024 * 1024
        )
        expected = {name for name, value in weight_map.items() if value == shard}
        if set(inspected) != expected:
            raise ValueError("Qwen3-VL staging header does not match index")
        result.update({name: inspected[name] for name in names if weight_map[name] == shard})
    return result


def _convert_tensor(
    source: Path,
    descriptor: dict,
    destination: Path,
    *,
    transpose_conv3d: bool,
    numpy,
) -> str:
    offset = descriptor["file_offset"]
    remaining = descriptor["bytes"]
    digest = hashlib.sha256()
    descriptor_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor_fd)
        with destination.open("wb") as target:
            os.chmod(destination, 0o600)
            if transpose_conv3d:
                if remaining > CONVERSION_CHUNK_BYTES:
                    raise ValueError("Qwen3-VL Conv3D tensor exceeds bounded transpose limit")
                raw = os.pread(descriptor_fd, remaining, offset)
                if len(raw) != remaining:
                    raise ValueError("Qwen3-VL staging tensor is truncated")
                values = _bf16_to_fp16(raw, numpy)
                values = values.reshape(descriptor["shape"]).transpose(0, 4, 1, 2, 3)
                encoded = values.tobytes(order="C")
                target.write(encoded)
                digest.update(encoded)
            else:
                position = offset
                while remaining:
                    count = min(remaining, CONVERSION_CHUNK_BYTES)
                    count -= count % 2
                    raw = os.pread(descriptor_fd, count, position)
                    if len(raw) != count:
                        raise ValueError("Qwen3-VL staging tensor is truncated")
                    encoded = _bf16_to_fp16(raw, numpy).tobytes(order="C")
                    target.write(encoded)
                    digest.update(encoded)
                    position += count
                    remaining -= count
            target.flush()
            os.fsync(target.fileno())
        after = os.fstat(descriptor_fd)
    finally:
        os.close(descriptor_fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("Qwen3-VL source changed during staging")
    return digest.hexdigest()


def _bf16_to_fp16(raw: bytes, numpy):
    if len(raw) % 2:
        raise ValueError("Qwen3-VL BF16 tensor byte count is invalid")
    words = numpy.frombuffer(raw, dtype="<u2").astype(numpy.uint32)
    values = (words << 16).view(numpy.float32)
    if not numpy.isfinite(values).all():
        raise ValueError("Qwen3-VL source contains non-finite values")
    converted = values.astype(numpy.float16)
    if not numpy.isfinite(converted).all():
        raise ValueError("Qwen3-VL FP16 conversion overflowed")
    return converted


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
