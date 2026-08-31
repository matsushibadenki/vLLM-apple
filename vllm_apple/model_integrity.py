from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024
MAXIMUM_FILES = 100_000
MAXIMUM_DEPTH = 32
MAXIMUM_TOTAL_BYTES = 16 * 1024**4
MAXIMUM_MANIFEST_BYTES = 32 * 1024 * 1024
MAXIMUM_SIGNATURE_BYTES = 1024 * 1024


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
    raw = _read_bounded_regular_file(
        manifest_path, maximum_bytes=MAXIMUM_MANIFEST_BYTES, description="integrity manifest"
    )
    return _verify_model_integrity_payload(model_path, raw)


def sign_model_integrity_manifest(
    manifest_path: Path,
    certificate_path: Path,
    private_key_path: Path,
    output_path: Path,
) -> Path:
    manifest = _read_bounded_regular_file(
        manifest_path, maximum_bytes=MAXIMUM_MANIFEST_BYTES, description="integrity manifest"
    )
    _decode_manifest(manifest)
    certificate = _validated_regular_path(certificate_path, "signer certificate")
    private_key = _validated_regular_path(
        private_key_path, "signer private key", require_private=True
    )
    destination = output_path.expanduser().resolve(strict=False)
    if destination.exists() or not destination.parent.is_dir():
        raise ModelIntegrityError("signature output must be new with an existing parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    content_descriptor, content_name = tempfile.mkstemp(
        prefix=".manifest-content.", dir=destination.parent
    )
    content = Path(content_name)
    try:
        os.fchmod(content_descriptor, 0o600)
        with os.fdopen(content_descriptor, "wb", closefd=True) as handle:
            content_descriptor = -1
            handle.write(manifest)
            handle.flush()
            os.fsync(handle.fileno())
        completed = subprocess.run(
            [
                "/usr/bin/openssl", "cms", "-sign", "-binary", "-md", "sha256",
                "-in", str(content), "-signer", str(certificate), "-inkey",
                str(private_key), "-outform", "DER", "-out", str(temporary),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise ModelIntegrityError("CMS manifest signing failed")
        attributes = temporary.stat()
        if not 1 <= attributes.st_size <= MAXIMUM_SIGNATURE_BYTES:
            raise ModelIntegrityError("CMS signature exceeds its byte limit")
        os.chmod(temporary, 0o600)
        sync_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(sync_descriptor)
        finally:
            os.close(sync_descriptor)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    except (OSError, subprocess.SubprocessError) as error:
        raise ModelIntegrityError("CMS manifest signing failed") from error
    finally:
        if content_descriptor >= 0:
            os.close(content_descriptor)
        try:
            content.unlink()
        except FileNotFoundError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination.resolve(strict=True)


def verify_signed_model_integrity(
    model_path: Path,
    manifest_path: Path,
    signature_path: Path,
    trusted_ca_path: Path,
    expected_signer_sha256: str,
) -> dict[str, object]:
    verified = verify_detached_cms_json(
        manifest_path,
        signature_path,
        trusted_ca_path,
        expected_signer_sha256,
    )
    manifest = _read_bounded_regular_file(
        manifest_path, maximum_bytes=MAXIMUM_MANIFEST_BYTES, description="integrity manifest"
    )
    if verified != _decode_manifest(manifest):
        raise ModelIntegrityError("CMS manifest verification failed")
    return _verify_model_integrity_payload(model_path, manifest)


def verify_detached_cms_json(
    document_path: Path,
    signature_path: Path,
    trusted_ca_path: Path,
    expected_signer_sha256: str,
) -> dict[str, object]:
    expected_fingerprint = _normalized_fingerprint(expected_signer_sha256)
    document = _read_bounded_regular_file(
        document_path, maximum_bytes=MAXIMUM_MANIFEST_BYTES, description="signed document"
    )
    signature = _validated_regular_path(signature_path, "integrity signature")
    if not 1 <= signature.stat().st_size <= MAXIMUM_SIGNATURE_BYTES:
        raise ModelIntegrityError("integrity signature exceeds its byte limit")
    trusted_ca = _validated_regular_path(trusted_ca_path, "trusted CA certificate")
    with tempfile.TemporaryDirectory(prefix="vllm-apple-cms-") as directory:
        root = Path(directory)
        content = root / "manifest.json"
        signer = root / "signer.pem"
        content.write_bytes(document)
        os.chmod(content, 0o600)
        try:
            completed = subprocess.run(
                [
                    "/usr/bin/openssl", "cms", "-verify", "-binary", "-inform", "DER",
                    "-in", str(signature), "-content", str(content), "-CAfile", str(trusted_ca),
                    "-purpose", "any", "-signer", str(signer), "-out", os.devnull,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ModelIntegrityError("CMS manifest verification failed") from error
        if completed.returncode != 0 or not signer.is_file():
            raise ModelIntegrityError("CMS manifest verification failed")
        actual_fingerprint = _certificate_fingerprint(signer)
    if actual_fingerprint != expected_fingerprint:
        raise ModelIntegrityError("manifest signer identity does not match")
    return _decode_manifest(document)


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
    except OSError as error:
        raise ModelIntegrityError(f"{description} cannot be read") from error
    try:
        if not stat.S_ISREG(opened.st_mode):
            raise ModelIntegrityError(f"{description} must be a regular file")
        if not 1 <= opened.st_size <= maximum_bytes:
            raise ModelIntegrityError(f"{description} exceeds size limit")
        raw = bytearray()
        while chunk := os.read(descriptor, min(CHUNK_BYTES, maximum_bytes + 1 - len(raw))):
            raw.extend(chunk)
            if len(raw) > maximum_bytes:
                raise ModelIntegrityError(f"{description} exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ModelIntegrityError(f"{description} changed while reading")
    return bytes(raw)


def _decode_manifest(raw: bytes) -> dict[str, object]:
    try:
        expected = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelIntegrityError("integrity manifest is not valid JSON") from error
    if not isinstance(expected, dict):
        raise ModelIntegrityError("integrity manifest must be a JSON object")
    return expected


def _verify_model_integrity_payload(model_path: Path, raw: bytes) -> dict[str, object]:
    expected = _decode_manifest(raw)
    actual = build_model_integrity_manifest(model_path)
    if expected != actual:
        raise ModelIntegrityError("model integrity verification failed")
    return actual


def _validated_regular_path(
    path: Path, description: str, *, require_private: bool = False
) -> Path:
    expanded = path.expanduser()
    try:
        original = expanded.lstat()
    except OSError as error:
        raise ModelIntegrityError(f"{description} cannot be read") from error
    if stat.S_ISLNK(original.st_mode):
        raise ModelIntegrityError(f"{description} must not be a symbolic link")
    candidate = expanded.resolve(strict=True)
    attributes = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(attributes.st_mode):
        raise ModelIntegrityError(f"{description} must be a regular file")
    if require_private and (attributes.st_mode & 0o077):
        raise ModelIntegrityError(f"{description} permissions must be private")
    return candidate


def _normalized_fingerprint(value: str) -> str:
    normalized = value.replace(":", "").lower()
    if len(normalized) != 64:
        raise ModelIntegrityError("expected signer fingerprint must be SHA-256")
    try:
        bytes.fromhex(normalized)
    except ValueError as error:
        raise ModelIntegrityError("expected signer fingerprint must be SHA-256") from error
    return normalized


def _certificate_fingerprint(certificate: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/openssl", "x509", "-in", str(certificate), "-noout",
                "-fingerprint", "-sha256",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=True,
            text=True,
        )
        _, value = completed.stdout.strip().split("=", 1)
        return _normalized_fingerprint(value)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ModelIntegrityError("signer certificate fingerprint is unavailable") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
