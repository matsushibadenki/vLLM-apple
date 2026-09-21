"""Private, signed, environment-bound cache for compiled Core ML artifacts."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

COREML_ARTIFACT_CACHE_SCHEMA_VERSION = 1
MAX_CACHE_ENTRIES = 64
MAX_ARTIFACT_FILES = 8192
MAX_ARTIFACT_BYTES = 32 * 1024**3


@dataclass(frozen=True, slots=True)
class CoreMLArtifactCacheIdentity:
    source_sha256: str
    graph_id: str
    toolchain_version: str
    coreml_version: str
    os_build: str
    compute_units: str

    def __post_init__(self) -> None:
        for name in ("source_sha256", "graph_id"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"invalid Core ML cache {name}")
        for name in ("toolchain_version", "coreml_version", "os_build"):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= 256:
                raise ValueError(f"invalid Core ML cache {name}")
        if self.compute_units not in {
            "all", "cpu_only", "cpu_and_gpu", "cpu_and_neural_engine"
        }:
            raise ValueError("invalid Core ML cache compute units")

    @property
    def cache_id(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class CoreMLArtifactCacheEntry:
    cache_id: str
    artifact_path: Path
    file_count: int
    artifact_bytes: int
    tree_sha256: str


class CoreMLArtifactCache:
    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise ValueError("Core ML cache signing key must contain at least 32 bytes")
        self.root = root.expanduser().resolve(strict=False)
        self._key = signing_key
        self._ensure_private_directory(self.root)
        self._ensure_private_directory(self.root / "entries")
        self._ensure_private_directory(self.root / "quarantine")

    def publish(
        self, identity: CoreMLArtifactCacheIdentity, compiled_model: Path
    ) -> CoreMLArtifactCacheEntry:
        source = compiled_model.expanduser().resolve(strict=True)
        if not source.is_dir() or source.suffix != ".mlmodelc":
            raise ValueError("Core ML cache source must be a compiled model directory")
        destination = self.root / "entries" / identity.cache_id
        if destination.exists():
            return self.load(identity)
        if len(tuple((self.root / "entries").iterdir())) >= MAX_CACHE_ENTRIES:
            raise ValueError("Core ML artifact cache is full")
        temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=self.root / "entries"))
        os.chmod(temporary, 0o700)
        try:
            artifact = temporary / "artifact.mlmodelc"
            artifact.mkdir(mode=0o700)
            file_count, artifact_bytes, tree_digest = _copy_tree(source, artifact)
            manifest = {
                "schema_version": COREML_ARTIFACT_CACHE_SCHEMA_VERSION,
                "cache_id": identity.cache_id,
                "identity": asdict(identity),
                "file_count": file_count,
                "artifact_bytes": artifact_bytes,
                "tree_sha256": tree_digest,
            }
            signature = hmac.new(self._key, _canonical(manifest), hashlib.sha256).hexdigest()
            payload = {**manifest, "signature": signature}
            _write_private_json(temporary / "manifest.json", payload)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return CoreMLArtifactCacheEntry(
            identity.cache_id, destination / "artifact.mlmodelc",
            file_count, artifact_bytes, tree_digest,
        )

    def load(self, identity: CoreMLArtifactCacheIdentity) -> CoreMLArtifactCacheEntry:
        entry = self.root / "entries" / identity.cache_id
        try:
            manifest = _read_manifest(entry / "manifest.json")
            signature = manifest.pop("signature")
            expected = hmac.new(self._key, _canonical(manifest), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("Core ML artifact cache signature mismatch")
            if (manifest.get("schema_version") != COREML_ARTIFACT_CACHE_SCHEMA_VERSION
                    or manifest.get("cache_id") != identity.cache_id
                    or manifest.get("identity") != asdict(identity)):
                raise ValueError("Core ML artifact cache identity mismatch")
            artifact = entry / "artifact.mlmodelc"
            count, size, digest = _tree_digest(artifact)
            if (count != manifest.get("file_count")
                    or size != manifest.get("artifact_bytes")
                    or digest != manifest.get("tree_sha256")):
                raise ValueError("Core ML artifact cache tree mismatch")
            return CoreMLArtifactCacheEntry(identity.cache_id, artifact, count, size, digest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            if entry.exists():
                self._quarantine(entry, "verification_failed")
            raise ValueError("Core ML artifact cache entry was quarantined") from error

    def revoke(self, identity: CoreMLArtifactCacheIdentity, reason: str) -> Path:
        if not reason or len(reason) > 128 or any(ord(value) < 0x20 for value in reason):
            raise ValueError("invalid Core ML artifact revocation reason")
        entry = self.root / "entries" / identity.cache_id
        if not entry.is_dir():
            raise ValueError("Core ML artifact cache entry is unavailable")
        return self._quarantine(entry, reason)

    def _quarantine(self, entry: Path, reason: str) -> Path:
        destination = self.root / "quarantine" / (
            f"{entry.name}-{secrets.token_hex(8)}"
        )
        os.replace(entry, destination)
        _write_private_json(destination / "revocation.json", {
            "schema_version": 1,
            "cache_id": entry.name,
            "reason": reason,
        })
        _fsync_directory(destination.parent)
        return destination

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        attributes = path.lstat()
        if (not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid()
                or attributes.st_mode & 0o077):
            raise ValueError("Core ML artifact cache directory must be private")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_private_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_manifest(path: Path) -> dict[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.fstat(descriptor).st_size > 64 * 1024:
            raise ValueError("Core ML artifact cache manifest is too large")
        value = json.loads(os.read(descriptor, 64 * 1024 + 1))
    finally:
        os.close(descriptor)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "cache_id", "identity", "file_count",
        "artifact_bytes", "tree_sha256", "signature",
    }:
        raise ValueError("Core ML artifact cache manifest is invalid")
    if not isinstance(value["signature"], str) or len(value["signature"]) != 64:
        raise ValueError("Core ML artifact cache signature is invalid")
    return value


def _copy_tree(source: Path, destination: Path) -> tuple[int, int, str]:
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        attributes = path.lstat()
        target = destination / relative
        if stat.S_ISDIR(attributes.st_mode):
            target.mkdir(mode=0o700)
        elif stat.S_ISREG(attributes.st_mode):
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                while chunk := os.read(source_fd, 1024 * 1024):
                    view = memoryview(chunk)
                    while view:
                        written = os.write(target_fd, view)
                        if written <= 0:
                            raise OSError("Core ML artifact cache copy made no progress")
                        view = view[written:]
                os.fsync(target_fd)
            finally:
                os.close(source_fd)
                os.close(target_fd)
        else:
            raise ValueError("Core ML artifact cache source contains an unsafe entry")
    return _tree_digest(destination)


def _tree_digest(root: Path) -> tuple[int, int, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Core ML artifact cache tree is unsafe")
    digest = hashlib.sha256(b"vllm-apple-coreml-artifact-cache-v1\0")
    count = size = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        attributes = path.lstat()
        relative = path.relative_to(root).as_posix().encode()
        if stat.S_ISDIR(attributes.st_mode):
            digest.update(b"d\0" + relative + b"\0")
            continue
        if not stat.S_ISREG(attributes.st_mode):
            raise ValueError("Core ML artifact cache tree contains an unsafe entry")
        count += 1
        size += attributes.st_size
        if count > MAX_ARTIFACT_FILES or size > MAX_ARTIFACT_BYTES:
            raise ValueError("Core ML artifact cache tree exceeds its bounds")
        digest.update(b"f\0" + relative + b"\0" + str(attributes.st_size).encode() + b"\0")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
    if count == 0:
        raise ValueError("Core ML artifact cache tree is empty")
    return count, size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
