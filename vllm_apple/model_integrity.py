from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024
MAXIMUM_FILES = 100_000
MAXIMUM_DEPTH = 32
MAXIMUM_TOTAL_BYTES = 16 * 1024**4


class ModelIntegrityError(RuntimeError):
    pass


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        if depth > MAXIMUM_DEPTH:
            raise ModelIntegrityError("model tree exceeds maximum depth")
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            raise ModelIntegrityError("model tree contains a symbolic link")
                        if entry.is_dir(follow_symlinks=False):
                            pending.append((Path(entry.path), depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            files.append(Path(entry.path))
                            if len(files) > MAXIMUM_FILES:
                                raise ModelIntegrityError("model tree exceeds file limit")
                        else:
                            raise ModelIntegrityError("model tree contains a special file")
                    except OSError as error:
                        raise ModelIntegrityError(
                            "model tree changed during enumeration"
                        ) from error
        except OSError as error:
            raise ModelIntegrityError("model tree cannot be read") from error
    if not files:
        raise ModelIntegrityError("model tree is empty")
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _hash_file(path: Path) -> tuple[int, str, tuple[int, int, int, int, int]]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ModelIntegrityError("model file is not regular")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ModelIntegrityError("model file cannot be opened safely") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ModelIntegrityError("model file changed before hashing")
        while chunk := os.read(descriptor, CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ModelIntegrityError("model file changed while hashing")
    return after.st_size, digest.hexdigest(), identity_after


def build_model_integrity_manifest(model_path: Path) -> dict[str, object]:
    root = model_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ModelIntegrityError("model path must be a directory")
    records: list[dict[str, object]] = []
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    total_bytes = 0
    root_digest = hashlib.sha256(b"vllm-apple-model-integrity-v1\0")
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        size, digest, identity = _hash_file(path)
        total_bytes += size
        if total_bytes > MAXIMUM_TOTAL_BYTES:
            raise ModelIntegrityError("model tree exceeds byte limit")
        encoded = relative.encode("utf-8")
        root_digest.update(len(encoded).to_bytes(4, "big"))
        root_digest.update(encoded)
        root_digest.update(size.to_bytes(8, "big"))
        root_digest.update(bytes.fromhex(digest))
        records.append({"path": relative, "size_bytes": size, "sha256": digest})
        identities[relative] = identity
    final_files = _regular_files(root)
    if [item.relative_to(root).as_posix() for item in final_files] != list(identities):
        raise ModelIntegrityError("model tree changed while hashing")
    for path in final_files:
        relative = path.relative_to(root).as_posix()
        current = path.stat(follow_symlinks=False)
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identities[relative] != identity:
            raise ModelIntegrityError("model tree changed while hashing")
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "file_count": len(records),
        "total_bytes": total_bytes,
        "root_sha256": root_digest.hexdigest(),
        "files": records,
    }


def verify_model_integrity(model_path: Path, manifest_path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(manifest_path, flags)
        opened = os.fstat(descriptor)
    except OSError as error:
        raise ModelIntegrityError("integrity manifest cannot be read") from error
    try:
        if not stat.S_ISREG(opened.st_mode):
            raise ModelIntegrityError("integrity manifest must be a regular file")
        if opened.st_size > 32 * 1024 * 1024:
            raise ModelIntegrityError("integrity manifest exceeds size limit")
        raw = bytearray()
        while chunk := os.read(descriptor, min(CHUNK_BYTES, 32 * 1024 * 1024 + 1 - len(raw))):
            raw.extend(chunk)
            if len(raw) > 32 * 1024 * 1024:
                raise ModelIntegrityError("integrity manifest exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ModelIntegrityError("integrity manifest changed while reading")
    try:
        expected = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelIntegrityError("integrity manifest is not valid JSON") from error
    actual = build_model_integrity_manifest(model_path)
    if not isinstance(expected, dict) or expected != actual:
        raise ModelIntegrityError("model integrity verification failed")
    return actual


def save_model_integrity_manifest(manifest: dict[str, object], output: Path) -> Path:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output
