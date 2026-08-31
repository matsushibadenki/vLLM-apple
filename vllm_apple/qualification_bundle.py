from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .model_integrity import (
    ModelIntegrityError,
    sign_model_integrity_manifest,
    verify_detached_cms_json,
)


MAX_REPORT_BYTES = 1024 * 1024


class QualificationBundleError(RuntimeError):
    pass


def _read_report(path: Path) -> tuple[dict[str, object], bytes]:
    candidate = path.expanduser()
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
    except OSError as error:
        raise QualificationBundleError("qualification evidence cannot be read") from error
    try:
        if not stat.S_ISREG(opened.st_mode):
            raise QualificationBundleError("qualification evidence must be a regular file")
        if not 1 <= opened.st_size <= MAX_REPORT_BYTES:
            raise QualificationBundleError("qualification evidence exceeds its byte limit")
        chunks = bytearray()
        while chunk := os.read(descriptor, min(64 * 1024, MAX_REPORT_BYTES + 1 - len(chunks))):
            chunks.extend(chunk)
            if len(chunks) > MAX_REPORT_BYTES:
                raise QualificationBundleError("qualification evidence exceeds its byte limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = bytes(chunks)
    if len(raw) != opened.st_size or (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise QualificationBundleError("qualification evidence changed while reading")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationBundleError("qualification evidence is invalid JSON") from error
    if not isinstance(payload, dict):
        raise QualificationBundleError("qualification evidence must be an object")
    return payload, raw


def build_qualification_bundle(directory: Path) -> dict[str, object]:
    root = directory.expanduser().resolve(strict=True)
    qualification, qualification_raw = _read_report(root / "qualification.json")
    preflight, preflight_raw = _read_report(root / "preflight.json")
    if qualification.get("schema_version") != 1 or preflight.get("schema_version") != 1:
        raise QualificationBundleError("unsupported qualification evidence schema")
    if qualification.get("passed") is not True or qualification.get("shutdown_clean") is not True:
        raise QualificationBundleError("qualification did not pass cleanly")
    if preflight.get("eligible") is not True:
        raise QualificationBundleError("qualification preflight was not eligible")
    backend = qualification.get("backend")
    if backend not in {"vllm_metal", "mlx_lm"} or preflight.get("backend_kind") != backend:
        raise QualificationBundleError("qualification backend evidence does not match")
    if qualification.get("requested_modes") != ["text"]:
        raise QualificationBundleError("qualification is not text-only")
    for key in ("promotion_probe", "soak", "context_reevaluation"):
        value = qualification.get(key)
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise QualificationBundleError(f"qualification {key} evidence is incomplete")
    phase = qualification.get("phase_profile")
    if not isinstance(phase, dict) or phase.get("sample_count", 0) <= 0:
        raise QualificationBundleError("qualification phase evidence is incomplete")
    quality = qualification.get("quality_smoke")
    required_languages = {"english", "japanese", "simplified_chinese"}
    if (
        not isinstance(quality, dict)
        or quality.get("passed") is not True
        or quality.get("stores_generated_text") is not False
        or not isinstance(quality.get("checks"), dict)
        or set(quality["checks"]) != required_languages
        or not all(value is True for value in quality["checks"].values())
    ):
        raise QualificationBundleError("qualification quality evidence is incomplete")
    memory = qualification.get("model_memory_fit")
    if (
        not isinstance(memory, dict)
        or memory.get("fits") is not True
        or not isinstance(memory.get("estimated_resident_bytes"), int)
        or not isinstance(memory.get("hard_ceiling_bytes"), int)
        or memory["estimated_resident_bytes"] > memory["hard_ceiling_bytes"]
    ):
        raise QualificationBundleError("qualification memory evidence is incomplete")
    model = qualification.get("model")
    if not isinstance(model, str) or not model or len(model.encode("utf-8")) > 4096:
        raise QualificationBundleError("qualification model identifier is invalid")

    evidence = {
        "preflight.json": hashlib.sha256(preflight_raw).hexdigest(),
        "qualification.json": hashlib.sha256(qualification_raw).hexdigest(),
    }
    admission_path = root / "artifact-admission.json"
    if admission_path.exists():
        admission, admission_raw = _read_report(admission_path)
        if (
            admission.get("schema_version") != 1
            or admission.get("eligible") is not True
            or admission.get("model") != model
            or admission.get("artifact_bytes") != memory.get("artifact_bytes")
            or admission.get("estimated_resident_bytes")
            != memory.get("estimated_resident_bytes")
        ):
            raise QualificationBundleError("artifact admission evidence does not match")
        evidence["artifact-admission.json"] = hashlib.sha256(admission_raw).hexdigest()

    body: dict[str, object] = {
        "schema_version": 1,
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "backend": backend,
        "requested_modes": ["text"],
        "backend_versions": qualification.get("backend_versions"),
        "evidence_sha256": evidence,
        "passed": True,
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**body, "bundle_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def save_qualification_bundle(bundle: dict[str, object], output: Path) -> Path:
    destination = output.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def verify_qualification_bundle(directory: Path, bundle_path: Path) -> dict[str, object]:
    stored, _ = _read_report(bundle_path)
    expected = build_qualification_bundle(directory)
    if stored != expected:
        raise QualificationBundleError("qualification promotion bundle verification failed")
    return expected


def sign_qualification_bundle(
    directory: Path,
    bundle_path: Path,
    certificate_path: Path,
    private_key_path: Path,
    output_path: Path,
) -> Path:
    verify_qualification_bundle(directory, bundle_path)
    try:
        return sign_model_integrity_manifest(
            bundle_path, certificate_path, private_key_path, output_path
        )
    except ModelIntegrityError as error:
        raise QualificationBundleError(str(error)) from error


def verify_signed_qualification_bundle(
    directory: Path,
    bundle_path: Path,
    signature_path: Path,
    trusted_ca_path: Path,
    expected_signer_sha256: str,
) -> dict[str, object]:
    try:
        signed = verify_detached_cms_json(
            bundle_path,
            signature_path,
            trusted_ca_path,
            expected_signer_sha256,
        )
    except ModelIntegrityError as error:
        raise QualificationBundleError(str(error)) from error
    expected = verify_qualification_bundle(directory, bundle_path)
    if signed != expected:
        raise QualificationBundleError("signed qualification bundle does not match evidence")
    return expected
