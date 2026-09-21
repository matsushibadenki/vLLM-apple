"""Page-aligned fast-load artifact and environment-bound kernel index."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import ExecutionBackend
from .numeric_routing import NumericFormat

MAX_FAST_LOAD_TENSORS = 65_536
MAX_FAST_LOAD_BYTES = 32 * 1024**3


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class FastLoadSourceTensor:
    name: str
    source_path: Path
    source_offset: int
    byte_count: int
    sha256: str
    numeric_format: NumericFormat
    layout_id: str

    def __post_init__(self) -> None:
        if (not _identifier(self.name) or type(self.source_offset) is not int
                or type(self.byte_count) is not int or self.source_offset < 0
                or not 1 <= self.byte_count <= MAX_FAST_LOAD_BYTES
                or not _digest(self.sha256) or not isinstance(self.numeric_format, NumericFormat)
                or not _digest(self.layout_id)):
            raise ValueError("invalid fast-load source tensor")


@dataclass(frozen=True, slots=True)
class FastLoadTensorEntry:
    name: str
    offset: int
    byte_count: int
    sha256: str
    numeric_format: str
    layout_id: str

    def __post_init__(self) -> None:
        if (not _identifier(self.name) or type(self.offset) is not int or self.offset < 0
                or type(self.byte_count) is not int
                or not 1 <= self.byte_count <= MAX_FAST_LOAD_BYTES
                or not _digest(self.sha256) or not _digest(self.layout_id)):
            raise ValueError("invalid fast-load tensor entry")
        try:
            NumericFormat(self.numeric_format)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid fast-load tensor numeric format") from error


@dataclass(frozen=True, slots=True)
class FastLoadArtifact:
    artifact_id: str
    root: Path
    data_path: Path
    page_size: int
    artifact_bytes: int
    entries: tuple[FastLoadTensorEntry, ...]


def build_fast_load_artifact(
    sources: tuple[FastLoadSourceTensor, ...], destination: Path, *, page_size: int
) -> FastLoadArtifact:
    if (not sources or len(sources) > MAX_FAST_LOAD_TENSORS
            or len({item.name for item in sources}) != len(sources)
            or type(page_size) is not int or not 4096 <= page_size <= 65536
            or page_size & (page_size - 1)):
        raise ValueError("invalid fast-load artifact plan")
    destination = destination.expanduser().resolve(strict=False)
    if destination.exists():
        raise ValueError("fast-load artifact destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_attributes = destination.parent.lstat()
    if (not stat.S_ISDIR(parent_attributes.st_mode)
            or parent_attributes.st_uid != os.getuid()
            or parent_attributes.st_mode & 0o077):
        raise ValueError("fast-load artifact parent must be private")
    temporary = Path(tempfile.mkdtemp(prefix=".fast-load-", dir=destination.parent))
    os.chmod(temporary, 0o700)
    entries = []
    offset = 0
    try:
        data_path = temporary / "data.bin"
        target_fd = os.open(data_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            for source in sources:
                offset = _align(offset, page_size)
                if offset + source.byte_count > MAX_FAST_LOAD_BYTES:
                    raise ValueError("fast-load artifact exceeds byte limit")
                os.lseek(target_fd, offset, os.SEEK_SET)
                digest = _copy_source(source, target_fd)
                if digest != source.sha256:
                    raise ValueError("fast-load source digest mismatch")
                entries.append(FastLoadTensorEntry(
                    source.name, offset, source.byte_count, digest,
                    source.numeric_format.value, source.layout_id,
                ))
                offset += source.byte_count
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        identity = {
            "schema_version": 1,
            "page_size": page_size,
            "artifact_bytes": offset,
            "entries": [asdict(entry) for entry in entries],
        }
        artifact_id = hashlib.sha256(_canonical(identity)).hexdigest()
        _write_private_json(temporary / "manifest.json", {
            **identity, "artifact_id": artifact_id,
        })
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_fast_load_artifact(destination)


def load_fast_load_artifact(root: Path) -> FastLoadArtifact:
    root = root.expanduser().resolve(strict=True)
    root_attributes = root.lstat()
    if (not stat.S_ISDIR(root_attributes.st_mode) or root_attributes.st_uid != os.getuid()
            or root_attributes.st_mode & 0o077):
        raise ValueError("fast-load artifact directory is unsafe")
    manifest_path = root / "manifest.json"
    manifest_fd = os.open(manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        manifest_size = os.fstat(manifest_fd).st_size
        if manifest_size > 16 * 1024 * 1024:
            raise ValueError("fast-load artifact manifest is too large")
        manifest = json.loads(os.read(manifest_fd, manifest_size + 1))
    finally:
        os.close(manifest_fd)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "artifact_id", "page_size", "artifact_bytes", "entries"
    } or manifest["schema_version"] != 1:
        raise ValueError("invalid fast-load artifact manifest")
    identity = dict(manifest)
    artifact_id = identity.pop("artifact_id")
    if not _digest(artifact_id) or hashlib.sha256(_canonical(identity)).hexdigest() != artifact_id:
        raise ValueError("fast-load artifact identity mismatch")
    page_size = manifest["page_size"]
    entries = tuple(FastLoadTensorEntry(**entry) for entry in manifest["entries"])
    if (type(page_size) is not int or not 4096 <= page_size <= 65536
            or page_size & (page_size - 1)
            or not entries or len(entries) > MAX_FAST_LOAD_TENSORS
            or len({entry.name for entry in entries}) != len(entries)):
        raise ValueError("fast-load artifact plan is invalid")
    data = root / "data.bin"
    attributes = data.lstat()
    if (not stat.S_ISREG(attributes.st_mode) or attributes.st_uid != os.getuid()
            or attributes.st_mode & 0o077 or attributes.st_size != manifest["artifact_bytes"]):
        raise ValueError("fast-load artifact data is unsafe")
    descriptor = os.open(data, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        previous_end = 0
        for entry in entries:
            if (entry.offset % page_size or entry.offset + entry.byte_count > attributes.st_size
                    or entry.offset < previous_end
                    or not _identifier(entry.name) or not _digest(entry.sha256)
                    or not _digest(entry.layout_id)):
                raise ValueError("fast-load tensor entry is invalid")
            os.lseek(descriptor, entry.offset, os.SEEK_SET)
            digest = hashlib.sha256()
            remaining = entry.byte_count
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("fast-load tensor is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != entry.sha256:
                raise ValueError("fast-load tensor digest mismatch")
            previous_end = entry.offset + entry.byte_count
    finally:
        os.close(descriptor)
    return FastLoadArtifact(
        artifact_id, root, data, page_size, attributes.st_size, entries
    )


@dataclass(frozen=True, slots=True)
class KernelCompatibility:
    backend: ExecutionBackend
    backend_version: str
    environment_sha256: str
    operator: str
    numeric_format: NumericFormat
    layout_id: str
    minimum_alignment: int
    maximum_tensor_bytes: int

    def __post_init__(self) -> None:
        if (not isinstance(self.backend, ExecutionBackend)
                or not _identifier(self.backend_version) or not _digest(self.environment_sha256)
                or not _identifier(self.operator) or not isinstance(self.numeric_format, NumericFormat)
                or not _digest(self.layout_id) or type(self.minimum_alignment) is not int
                or self.minimum_alignment < 1 or self.minimum_alignment & (self.minimum_alignment - 1)
                or not 1 <= self.maximum_tensor_bytes <= MAX_FAST_LOAD_BYTES):
            raise ValueError("invalid kernel compatibility")

    @property
    def kernel_id(self) -> str:
        value = asdict(self)
        value["backend"] = self.backend.value
        value["numeric_format"] = self.numeric_format.value
        return hashlib.sha256(_canonical(value)).hexdigest()


class KernelCompatibilityIndex:
    def __init__(self, kernels: tuple[KernelCompatibility, ...]) -> None:
        if (not kernels or len(kernels) > 1024
                or len({item.kernel_id for item in kernels}) != len(kernels)):
            raise ValueError("invalid kernel compatibility index")
        self._kernels = kernels

    def select(
        self, entry: FastLoadTensorEntry, *, backend: ExecutionBackend,
        backend_version: str, environment_sha256: str, operator: str,
    ) -> KernelCompatibility:
        candidates = tuple(item for item in self._kernels if (
            item.backend is backend and item.backend_version == backend_version
            and item.environment_sha256 == environment_sha256
            and item.operator == operator and item.numeric_format.value == entry.numeric_format
            and item.layout_id == entry.layout_id
            and entry.offset % item.minimum_alignment == 0
            and entry.byte_count <= item.maximum_tensor_bytes
        ))
        if len(candidates) != 1:
            raise ValueError("fast-load tensor has no unique compatible kernel")
        return candidates[0]


def _copy_source(source: FastLoadSourceTensor, target_fd: int) -> str:
    attributes = source.source_path.lstat()
    if (not stat.S_ISREG(attributes.st_mode) or source.source_path.is_symlink()
            or source.source_offset + source.byte_count > attributes.st_size):
        raise ValueError("fast-load source file is unsafe")
    descriptor = os.open(source.source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, source.source_offset, os.SEEK_SET)
        remaining = source.byte_count
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("fast-load source is truncated")
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise OSError("fast-load artifact copy made no progress")
                view = view[written:]
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_private_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
