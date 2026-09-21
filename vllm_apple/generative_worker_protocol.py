from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .generative_evaluation import generative_plan_sha256
from .generative_qualification import GenerativeQualificationPlan

GENERATIVE_WORKER_ABI_VERSION = 3
MAX_GENERATIVE_REQUEST_BYTES = 32 * 1024
MAX_PROMPT_BYTES = 16 * 1024
MAX_INPUT_IMAGE_BYTES = 64 * 1024 * 1024


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _private_root(value: str | Path, workspace_root: Path, name: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"generative {name} must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_relative_to(workspace_root):
        raise ValueError(f"generative {name} must be a workspace directory")
    return resolved


def _private_input_image(value: str | Path, workspace_root: Path) -> tuple[Path, str, int]:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ValueError("generative input image must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    info = resolved.stat()
    if (
        not resolved.is_relative_to(workspace_root)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or not 1 <= info.st_size <= MAX_INPUT_IMAGE_BYTES
    ):
        raise ValueError("generative input image must be a private bounded workspace file")
    digest = hashlib.sha256()
    with resolved.open("rb", buffering=0) as handle:
        header = handle.read(16)
        if not (header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")):
            raise ValueError("generative input image must be PNG or JPEG")
        digest.update(header)
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
    ):
        raise ValueError("generative input image changed while hashing")
    return resolved, digest.hexdigest(), info.st_size


def build_generative_worker_request(
    plan: GenerativeQualificationPlan,
    *,
    workspace_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    mode: str,
    prompt: str,
    seed: int,
    sample_index: int,
    input_image_path: str | Path | None = None,
    disk_offload: bool = False,
) -> dict[str, object]:
    if not plan.eligible:
        raise ValueError("cannot build a worker request for an ineligible generation plan")
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("generative workspace root must be a directory")
    model = _private_root(model_root, workspace, "model root")
    output = _private_root(output_root, workspace, "output root")
    if output == model or output.is_relative_to(model):
        raise ValueError("generative output root must be outside the model tree")
    if mode not in plan.candidate.modes:
        raise ValueError("generative mode is not supported by the candidate")
    if (
        not prompt
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
        or any(ord(character) < 0x20 and character not in "\n\t" for character in prompt)
    ):
        raise ValueError("generative prompt is invalid or oversized")
    if not 0 <= seed < 2**64:
        raise ValueError("generative seed is outside the supported range")
    if not 0 <= sample_index < 32:
        raise ValueError("generative sample index is outside the supported range")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    needs_input = mode in {"image-edit", "image-to-video"}
    if needs_input != (input_image_path is not None):
        raise ValueError("generative mode input image requirement is not satisfied")
    input_image = (
        _private_input_image(input_image_path, workspace)
        if input_image_path is not None
        else None
    )
    payload = {
        "abi_version": GENERATIVE_WORKER_ABI_VERSION,
        "operation": "generate_qualification_sample",
        "candidate_id": plan.candidate.candidate_id,
        "model": plan.candidate.model,
        "modality": plan.candidate.modality,
        "plan_sha256": generative_plan_sha256(plan),
        "model_root": str(model),
        "output_root": str(output),
        "mode": mode,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "sample_index": sample_index,
        "width": plan.width,
        "height": plan.height,
        "frames": plan.frames,
        "steps": plan.steps,
        "batch_size": plan.batch_size,
        "memory_hard_ceiling_bytes": plan.artifact_admission.memory_hard_ceiling_bytes,
        "retain_request": False,
        "input_image_path": str(input_image[0]) if input_image else None,
        "input_image_sha256": input_image[1] if input_image else None,
        "input_image_bytes": input_image[2] if input_image else None,
        "disk_offload": disk_offload,
    }
    parse_generative_worker_request(payload, workspace_root=workspace)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_GENERATIVE_REQUEST_BYTES:
        raise ValueError("generative worker request exceeds the bounded limit")
    return payload


def parse_generative_worker_request(
    payload: object, *, workspace_root: str | Path
) -> dict[str, object]:
    expected_v1 = {
        "abi_version",
        "operation",
        "candidate_id",
        "model",
        "modality",
        "plan_sha256",
        "model_root",
        "output_root",
        "mode",
        "prompt",
        "prompt_sha256",
        "seed",
        "sample_index",
        "width",
        "height",
        "frames",
        "steps",
        "batch_size",
        "memory_hard_ceiling_bytes",
        "retain_request",
    }
    expected_v2 = expected_v1 | {
        "input_image_path", "input_image_sha256", "input_image_bytes",
    }
    expected_v3 = expected_v2 | {"disk_offload"}
    if (
        not isinstance(payload, dict)
        or payload.get("abi_version") not in {1, 2, GENERATIVE_WORKER_ABI_VERSION}
        or set(payload) != (
            expected_v1 if payload.get("abi_version") == 1
            else expected_v2 if payload.get("abi_version") == 2
            else expected_v3
        )
    ):
        raise ValueError("generative worker request schema is invalid")
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    prompt = payload["prompt"]
    strings = (payload["candidate_id"], payload["model"], payload["mode"])
    integers = (
        payload["seed"],
        payload["sample_index"],
        payload["width"],
        payload["height"],
        payload["frames"],
        payload["steps"],
        payload["batch_size"],
        payload["memory_hard_ceiling_bytes"],
    )
    if (
        payload["abi_version"] not in {1, 2, GENERATIVE_WORKER_ABI_VERSION}
        or payload["operation"] != "generate_qualification_sample"
        or payload["modality"] not in {"image", "video"}
        or any(not isinstance(value, str) or not value for value in strings)
        or not _is_digest(payload["plan_sha256"])
        or not _is_digest(payload["prompt_sha256"])
        or not isinstance(prompt, str)
        or not prompt
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != payload["prompt_sha256"]
        or any(not isinstance(value, int) or isinstance(value, bool) for value in integers)
        or payload["seed"] < 0
        or payload["seed"] >= 2**64
        or not 0 <= payload["sample_index"] < 32
        or any(payload[name] <= 0 for name in ("width", "height", "frames", "steps", "batch_size"))
        or payload["memory_hard_ceiling_bytes"] <= 0
        or payload["retain_request"] is not False
    ):
        raise ValueError("generative worker request identity is invalid")
    if payload["abi_version"] == GENERATIVE_WORKER_ABI_VERSION and type(
        payload["disk_offload"]
    ) is not bool:
        raise ValueError("generative disk offload flag is invalid")
    model = _private_root(payload["model_root"], workspace, "model root")
    output = _private_root(payload["output_root"], workspace, "output root")
    if output == model or output.is_relative_to(model):
        raise ValueError("generative output root must be outside the model tree")
    if payload["abi_version"] >= 2:
        needs_input = payload["mode"] in {"image-edit", "image-to-video"}
        has_input = payload["input_image_path"] is not None
        if needs_input != has_input:
            raise ValueError("generative mode input image requirement is not satisfied")
        if has_input:
            if (
                not isinstance(payload["input_image_path"], str)
                or not _is_digest(payload["input_image_sha256"])
                or type(payload["input_image_bytes"]) is not int
            ):
                raise ValueError("generative input image identity is invalid")
            path, digest, size = _private_input_image(payload["input_image_path"], workspace)
            if digest != payload["input_image_sha256"] or size != payload["input_image_bytes"]:
                raise ValueError("generative input image digest does not match")
            if path == model or path.is_relative_to(model) or path == output or path.is_relative_to(output):
                raise ValueError("generative input image must be separate from model and output roots")
        elif any(payload[name] is not None for name in (
            "input_image_sha256", "input_image_bytes",
        )):
            raise ValueError("generative request has incomplete input image metadata")
    return payload


def save_private_generative_request(payload: object, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if not 1 <= len(encoded) <= MAX_GENERATIVE_REQUEST_BYTES:
        raise ValueError("generative worker request exceeds the bounded limit")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def consume_private_generative_request(
    path: str | Path, *, workspace_root: str | Path
) -> dict[str, object]:
    request_path = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(request_path, flags)
    info = None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or not 1 <= info.st_size <= MAX_GENERATIVE_REQUEST_BYTES
        ):
            raise ValueError("generative worker request file is not private and bounded")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if remaining or len(raw) != info.st_size:
            raise ValueError("generative worker request changed while being consumed")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("generative worker request is not valid JSON") from error
        return parse_generative_worker_request(payload, workspace_root=workspace_root)
    finally:
        os.close(descriptor)
        try:
            current = request_path.lstat()
            if (
                info is not None
                and current.st_dev == info.st_dev
                and current.st_ino == info.st_ino
            ):
                request_path.unlink()
        except FileNotFoundError:
            pass
