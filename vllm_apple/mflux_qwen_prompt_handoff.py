"""Private, single-use Qwen-Image-2512 text-embedding process handoff."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

ABI_VERSION = 1
CANDIDATE_ID = "qwen-image-2512-mflux"
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4096
PAYLOAD_NAME = "qwen2512-embeddings.safetensors"
MANIFEST_NAME = "qwen2512-embeddings.json"


def _identity_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _private_directory(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("Qwen-Image-2512 handoff directory must not be a symlink")
    directory = root.resolve(strict=True)
    info = directory.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("Qwen-Image-2512 handoff directory must be private")
    return directory


def _validate_tensors(embeddings: object, mask: object) -> None:
    import numpy as np

    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.dtype != np.float32
        or embeddings.ndim != 3
        or embeddings.shape[0] != 1
        or not 1 <= embeddings.shape[1] <= 1058
        or embeddings.shape[2] != 3584
        or not embeddings.flags.c_contiguous
        or not np.isfinite(embeddings).all()
        or not isinstance(mask, np.ndarray)
        or mask.dtype != np.int32
        or mask.shape != embeddings.shape[:2]
        or not mask.flags.c_contiguous
        or not np.isin(mask, (0, 1)).all()
        or not mask.any()
    ):
        raise ValueError("Qwen-Image-2512 handoff tensor contract is invalid")


def _validate_identity(plan_sha256: str, prompt_sha256: str, sample_index: int) -> None:
    if (
        not _identity_digest(plan_sha256)
        or not _identity_digest(prompt_sha256)
        or type(sample_index) is not int
        or not 0 <= sample_index < 32
    ):
        raise ValueError("Qwen-Image-2512 handoff identity is invalid")


def _write_temp(directory: Path, prefix: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def save_mflux_qwen_prompt_handoff(
    root: Path,
    embeddings: object,
    mask: object,
    *,
    plan_sha256: str,
    prompt_sha256: str,
    sample_index: int,
) -> Path:
    """Atomically publish an exact-F32 copy of BF16-compatible embeddings."""
    from safetensors.numpy import save

    directory = _private_directory(root)
    _validate_identity(plan_sha256, prompt_sha256, sample_index)
    _validate_tensors(embeddings, mask)
    payload = directory / PAYLOAD_NAME
    manifest = directory / MANIFEST_NAME
    if any(path.exists() or path.is_symlink() for path in (payload, manifest)):
        raise ValueError("Qwen-Image-2512 handoff already exists")
    encoded_payload = save({"prompt_embeds": embeddings, "prompt_mask": mask})
    if not 1 <= len(encoded_payload) <= MAX_PAYLOAD_BYTES:
        raise ValueError("Qwen-Image-2512 handoff payload exceeds the limit")
    body = {
        "abi_version": ABI_VERSION,
        "candidate_id": CANDIDATE_ID,
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_sha256,
        "sample_index": sample_index,
        "payload_bytes": len(encoded_payload),
        "payload_sha256": hashlib.sha256(encoded_payload).hexdigest(),
        "embedding_shape": list(embeddings.shape),
        "mask_shape": list(mask.shape),
        "embedding_dtype": "F32",
        "mask_dtype": "I32",
    }
    encoded_manifest = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded_manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("Qwen-Image-2512 handoff manifest exceeds the limit")
    payload_temp = manifest_temp = None
    try:
        payload_temp = _write_temp(directory, ".qwen2512-payload-", encoded_payload)
        manifest_temp = _write_temp(directory, ".qwen2512-manifest-", encoded_manifest)
        os.replace(payload_temp, payload)
        os.replace(manifest_temp, manifest)
        return manifest
    except BaseException:
        payload.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise
    finally:
        if payload_temp is not None:
            payload_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or not 1 <= info.st_size <= maximum_bytes
        ):
            raise ValueError("Qwen-Image-2512 handoff file is unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(maximum_bytes + 1)
        if len(data) != info.st_size:
            raise ValueError("Qwen-Image-2512 handoff file changed during read")
        return data
    finally:
        os.close(descriptor)


def consume_mflux_qwen_prompt_handoff(
    manifest_path: Path,
    *,
    plan_sha256: str,
    prompt_sha256: str,
    sample_index: int,
) -> tuple[object, object]:
    """Validate the bound payload and remove both files on every outcome."""
    from safetensors.numpy import load

    _validate_identity(plan_sha256, prompt_sha256, sample_index)
    unresolved = Path(manifest_path)
    if unresolved.name != MANIFEST_NAME:
        raise ValueError("Qwen-Image-2512 handoff manifest name is invalid")
    directory = _private_directory(unresolved.parent)
    manifest = directory / MANIFEST_NAME
    payload = manifest.with_name(PAYLOAD_NAME)
    try:
        body = json.loads(_read_private_file(manifest, MAX_MANIFEST_BYTES))
        expected_fields = {
            "abi_version", "candidate_id", "plan_sha256", "prompt_sha256",
            "sample_index", "payload_bytes", "payload_sha256", "embedding_shape",
            "mask_shape", "embedding_dtype", "mask_dtype",
        }
        if not isinstance(body, dict) or set(body) != expected_fields:
            raise ValueError("Qwen-Image-2512 handoff manifest schema is invalid")
        if (
            body["abi_version"] != ABI_VERSION
            or body["candidate_id"] != CANDIDATE_ID
            or (body["plan_sha256"], body["prompt_sha256"], body["sample_index"])
            != (plan_sha256, prompt_sha256, sample_index)
            or body["embedding_dtype"] != "F32"
            or body["mask_dtype"] != "I32"
        ):
            raise ValueError("Qwen-Image-2512 handoff identity does not match")
        raw = _read_private_file(payload, MAX_PAYLOAD_BYTES)
        if len(raw) != body["payload_bytes"] or hashlib.sha256(raw).hexdigest() != body["payload_sha256"]:
            raise ValueError("Qwen-Image-2512 handoff payload does not match")
        tensors = load(raw)
        if set(tensors) != {"prompt_embeds", "prompt_mask"}:
            raise ValueError("Qwen-Image-2512 handoff tensor names are invalid")
        embeddings, mask = tensors["prompt_embeds"], tensors["prompt_mask"]
        _validate_tensors(embeddings, mask)
        if list(embeddings.shape) != body["embedding_shape"] or list(mask.shape) != body["mask_shape"]:
            raise ValueError("Qwen-Image-2512 handoff tensor shape does not match")
        return embeddings, mask
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Qwen-Image-2512 handoff is invalid") from error
    finally:
        for path in (payload, manifest):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
