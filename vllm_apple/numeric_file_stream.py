"""Incremental NVFP4-to-scaled-INT8 tile provider over safe local files."""
from __future__ import annotations

import hashlib
import math
import os
import stat
from pathlib import Path

from .numeric_formats import (
    _E2M1_TWICE,
    _scale,
    DEFAULT_CONVERSION_REGISTRY,
    NumericFormatDescriptor,
    TensorGeometry,
    conversion_plan,
    numeric_content_digest_from_hashes,
)
from .numeric_streaming import NumericStreamingPlan


_HASH_CHUNK_BYTES = 1024 * 1024


class NVFP4FileTileProvider:
    """Owns immutable packed/scale descriptors and converts requested tiles on demand."""

    def __init__(
        self,
        packed_path: str | Path,
        scales_path: str | Path,
        descriptor: NumericFormatDescriptor,
        *,
        geometry: TensorGeometry | None,
        global_scale: float,
        packed_sha256: str,
        scales_sha256: str,
        scaled_payload_sha256: str,
        source_digest: str,
        target_digest: str,
    ) -> None:
        if not isinstance(descriptor, NumericFormatDescriptor):
            raise ValueError("file-backed NVFP4 descriptor is invalid")
        self.descriptor = descriptor
        self.geometry = geometry
        self.plan = (
            conversion_plan(descriptor)
            if geometry is None
            else DEFAULT_CONVERSION_REGISTRY.plan_tensor(descriptor, geometry)
        )
        if type(global_scale) not in (float, int) or not math.isfinite(global_scale) or global_scale < 0:
            raise ValueError("file-backed NVFP4 global scale is invalid")
        self.global_scale = float(global_scale)
        self._packed_fd, self._packed_identity = self._open_region(
            packed_path, (descriptor.elements + 1) // 2, "packed")
        self._scales_fd = None
        try:
            scale_count = (
                (descriptor.elements + descriptor.block_size - 1) // descriptor.block_size
                if geometry is None else geometry.scale_count
            )
            self._scales_fd, self._scales_identity = self._open_region(
                scales_path, scale_count, "scale")
            actual_packed = self._hash_fd(self._packed_fd, self._packed_identity)
            actual_scales = self._hash_fd(self._scales_fd, self._scales_identity, validate_scales=True)
            if actual_packed != packed_sha256 or actual_scales != scales_sha256:
                raise ValueError("file-backed NVFP4 source hash mismatch")
            if descriptor.elements % 2:
                last = os.pread(self._packed_fd, 1, self._packed_identity[2] - 1)
                padding = ((last[0] >> 4) if descriptor.packing == "low_nibble_first"
                           else (last[0] & 15)) if len(last) == 1 else 1
                if padding:
                    raise ValueError("file-backed NVFP4 padding nibble is invalid")
            if (
                numeric_content_digest_from_hashes(
                    self.plan, actual_packed, actual_scales, self.global_scale, "source"
                ) != source_digest
                or numeric_content_digest_from_hashes(
                    self.plan, scaled_payload_sha256, actual_scales,
                    self.global_scale, "target"
                ) != target_digest
            ):
                raise ValueError("file-backed NVFP4 content binding mismatch")
            self.scaled_payload_sha256 = scaled_payload_sha256
            self.source_digest = source_digest
            self.target_digest = target_digest
            self._next_index = 0
            self._closed = False
        except BaseException:
            self.close()
            raise

    def streaming_plan(
        self, tile_bytes: int, *, buffer_count: int = 2
    ) -> NumericStreamingPlan:
        return NumericStreamingPlan(
            self.scaled_payload_sha256,
            self.descriptor.elements,
            min(tile_bytes, self.descriptor.elements),
            buffer_count,
            alignment_bytes=1,
        )

    def read_tile(self, index: int, offset: int, length: int) -> bytes:
        if self._closed:
            raise ValueError("file-backed NVFP4 provider is closed")
        if (
            type(index) is not int or index != self._next_index
            or type(offset) is not int or type(length) is not int
            or offset < 0 or length <= 0 or offset + length > self.descriptor.elements
        ):
            raise ValueError("file-backed NVFP4 tile request is invalid or out of order")
        self._check_identity(self._packed_fd, self._packed_identity, "packed")
        self._check_identity(self._scales_fd, self._scales_identity, "scale")
        first_byte = offset // 2
        final_byte = (offset + length + 1) // 2
        packed = os.pread(self._packed_fd, final_byte - first_byte, first_byte)
        if len(packed) != final_byte - first_byte:
            raise ValueError("file-backed NVFP4 packed tile is truncated")
        converted = bytes(
            ((-1 if code & 8 else 1) * _E2M1_TWICE[code & 7]) & 255
            for code in (
                (packed[(element // 2) - first_byte] >> (
                    4 * (element % 2) if self.descriptor.packing == "low_nibble_first"
                    else 4 * (1 - element % 2)
                )) & 15
                for element in range(offset, offset + length)
            )
        )
        self._next_index += 1
        return converted

    def scale_code(self, element_index: int) -> int:
        if self._closed or type(element_index) is not int or not 0 <= element_index < self.descriptor.elements:
            raise ValueError("file-backed NVFP4 scale request is invalid")
        scale_index = (
            element_index // self.descriptor.block_size
            if self.geometry is None else self.geometry.scale_index(element_index)
        )
        value = os.pread(self._scales_fd, 1, scale_index)
        if len(value) != 1:
            raise ValueError("file-backed NVFP4 scale is truncated")
        return value[0]

    def verify_unchanged(self) -> None:
        """Revalidate both open files before accepting backend evidence."""
        if self._closed:
            raise ValueError("file-backed NVFP4 provider is closed")
        self._check_identity(self._packed_fd, self._packed_identity, "packed")
        self._check_identity(self._scales_fd, self._scales_identity, "scale")

    def close(self) -> None:
        if getattr(self, "_packed_fd", None) is not None:
            os.close(self._packed_fd)
            self._packed_fd = None
        if getattr(self, "_scales_fd", None) is not None:
            os.close(self._scales_fd)
            self._scales_fd = None
        self._closed = True

    @staticmethod
    def _open_region(path: str | Path, expected_size: int, label: str):
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor_fd = os.open(Path(path).expanduser().absolute(), flags)
        try:
            info = os.fstat(descriptor_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size != expected_size
            ):
                raise ValueError(f"file-backed NVFP4 {label} file is unsafe")
            return descriptor_fd, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        except BaseException:
            os.close(descriptor_fd)
            raise

    @staticmethod
    def _check_identity(descriptor_fd: int, identity: tuple[int, int, int, int], label: str):
        info = os.fstat(descriptor_fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != identity:
            raise ValueError(f"file-backed NVFP4 {label} file changed during streaming")

    def _hash_fd(
        self,
        descriptor_fd: int,
        identity: tuple[int, int, int, int],
        *,
        validate_scales: bool = False,
    ) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < identity[2]:
            chunk = os.pread(descriptor_fd, min(_HASH_CHUNK_BYTES, identity[2] - offset), offset)
            if not chunk:
                raise ValueError("file-backed NVFP4 source is truncated")
            if validate_scales:
                for code in chunk:
                    if not math.isfinite(_scale(code) * self.global_scale * 6):
                        raise ValueError("file-backed NVFP4 scale is invalid")
            digest.update(chunk)
            offset += len(chunk)
        self._check_identity(descriptor_fd, identity, "source")
        return digest.hexdigest()
