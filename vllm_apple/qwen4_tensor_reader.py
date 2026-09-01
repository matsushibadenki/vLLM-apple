from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

from .qwen4_adapter_contract import build_qwen4_adapter_contract
from .qwen4_adapter_loader import DEFAULT_MAX_HEADER_BYTES, _inspect_header
from .qwen4_conversion_plan import _component
from .qwen4_shard_stager import COPY_CHUNK_BYTES, MANIFEST_NAME, _load_manifest
from .qwen4_weight_map import _bounded_index


MAX_TENSOR_CHUNK_BYTES = 8 * 1024 * 1024


def build_qwen4_tensor_catalog(
    stage_root: str | Path,
    *,
    maximum_artifact_bytes: int,
    requested_modes: tuple[str, ...] = ("text",),
    maximum_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> dict[str, object]:
    root = Path(stage_root).expanduser().resolve(strict=True)
    contract = build_qwen4_adapter_contract(
        root,
        maximum_artifact_bytes=maximum_artifact_bytes,
        requested_modes=requested_modes,
    )
    weight_map, _, _ = _bounded_index(root / "model.safetensors.index.json")
    expected_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
    enabled = set(contract["enabled_components"])
    tensors: dict[str, dict[str, object]] = {}
    for shard_name in sorted(expected_by_shard):
        descriptors = _inspect_header(
            root / shard_name,
            maximum_artifact_bytes=maximum_artifact_bytes,
            maximum_header_bytes=maximum_header_bytes,
        )
        if set(descriptors) != expected_by_shard[shard_name]:
            raise ValueError("Qwen4 tensor catalog header does not match the weight index")
        for tensor_name, descriptor in descriptors.items():
            component = _component(tensor_name)
            if component is None:
                raise ValueError("Qwen4 tensor catalog contains an unclassified tensor")
            tensors[tensor_name] = {
                **descriptor,
                "shard": shard_name,
                "component": component,
                "active": component in enabled,
            }
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "requested_modes": list(requested_modes),
        "maximum_artifact_bytes": maximum_artifact_bytes,
        "tensors": tensors,
    }


class Qwen4TensorReader:
    def __init__(
        self,
        stage_root: str | Path,
        *,
        maximum_artifact_bytes: int,
        requested_modes: tuple[str, ...] = ("text",),
        maximum_chunk_bytes: int = MAX_TENSOR_CHUNK_BYTES,
    ) -> None:
        if (
            not isinstance(maximum_chunk_bytes, int)
            or isinstance(maximum_chunk_bytes, bool)
            or not 1 <= maximum_chunk_bytes <= MAX_TENSOR_CHUNK_BYTES
        ):
            raise ValueError("Qwen4 tensor reader chunk size is invalid")
        self.root = Path(stage_root).expanduser().resolve(strict=True)
        self.maximum_artifact_bytes = maximum_artifact_bytes
        self.maximum_chunk_bytes = maximum_chunk_bytes
        self._catalog = build_qwen4_tensor_catalog(
            self.root,
            maximum_artifact_bytes=maximum_artifact_bytes,
            requested_modes=requested_modes,
        )
        manifest = _load_manifest(self.root / MANIFEST_NAME)
        evidence = manifest.get("shards")
        if not isinstance(evidence, dict):
            raise ValueError("Qwen4 tensor reader requires shard digest evidence")
        self._evidence = evidence

    def descriptor(self, tensor_name: str) -> dict[str, object]:
        descriptor = self._catalog["tensors"].get(tensor_name)
        if not isinstance(descriptor, dict):
            raise KeyError("unknown Qwen4 tensor")
        return dict(descriptor)

    def iter_tensor_chunks(self, tensor_name: str) -> Iterator[bytes]:
        descriptor = self.descriptor(tensor_name)
        if descriptor["active"] is not True:
            raise ValueError("Qwen4 tensor is disabled for the requested mode")
        tensor_bytes = descriptor.get("bytes")
        if not isinstance(tensor_bytes, int) or isinstance(tensor_bytes, bool):
            raise ValueError("Qwen4 tensor reader descriptor is invalid")
        return self._iter_descriptor_range(descriptor, relative_offset=0, length=tensor_bytes)

    def iter_tensor_axis0_slice(
        self,
        tensor_name: str,
        *,
        start: int,
        count: int,
    ) -> Iterator[bytes]:
        descriptor = self.descriptor(tensor_name)
        if descriptor["active"] is not True:
            raise ValueError("Qwen4 tensor is disabled for the requested mode")
        shape = descriptor.get("shape")
        tensor_bytes = descriptor.get("bytes")
        if (
            not isinstance(shape, list)
            or not shape
            or not isinstance(tensor_bytes, int)
            or isinstance(tensor_bytes, bool)
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or start < 0
            or count <= 0
            or not isinstance(shape[0], int)
            or shape[0] <= 0
            or start + count > shape[0]
            or tensor_bytes % shape[0] != 0
        ):
            raise ValueError("Qwen4 tensor axis-0 slice is invalid")
        row_bytes = tensor_bytes // shape[0]
        return self._iter_descriptor_range(
            descriptor,
            relative_offset=start * row_bytes,
            length=count * row_bytes,
        )

    def _iter_descriptor_range(
        self,
        descriptor: dict[str, object],
        *,
        relative_offset: int,
        length: int,
    ) -> Iterator[bytes]:
        shard_name = descriptor["shard"]
        evidence = self._evidence.get(shard_name)
        if not isinstance(shard_name, str) or not isinstance(evidence, dict):
            raise ValueError("Qwen4 tensor reader shard evidence is invalid")
        path = self.root / shard_name
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor_fd = os.open(path, flags)
        try:
            before = os.fstat(descriptor_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_size > self.maximum_artifact_bytes
            ):
                raise ValueError("Qwen4 tensor reader shard is unsafe")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor_fd, COPY_CHUNK_BYTES):
                digest.update(chunk)
            if evidence != {"bytes": before.st_size, "sha256": digest.hexdigest()}:
                raise ValueError("Qwen4 tensor reader shard digest verification failed")
            tensor_bytes = descriptor["bytes"]
            file_offset = descriptor["file_offset"]
            if (
                not isinstance(tensor_bytes, int)
                or not isinstance(file_offset, int)
                or relative_offset < 0
                or length < 0
                or relative_offset + length > tensor_bytes
            ):
                raise ValueError("Qwen4 tensor reader descriptor is invalid")
            remaining = length
            offset = file_offset + relative_offset
            while remaining:
                size = min(remaining, self.maximum_chunk_bytes)
                chunk = os.pread(descriptor_fd, size, offset)
                if len(chunk) != size:
                    raise ValueError("Qwen4 tensor reader encountered a truncated tensor")
                current = os.fstat(descriptor_fd)
                if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ino,
                    current.st_dev,
                ):
                    raise ValueError("Qwen4 tensor reader shard changed during reading")
                yield chunk
                remaining -= size
                offset += size
        finally:
            os.close(descriptor_fd)
