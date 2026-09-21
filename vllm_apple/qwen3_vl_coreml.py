"""Qwen3-VL Core ML conversion identity and numerical promotion gate."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .device_capability import ComputeDevice, DeviceCapability
from .execution import ExecutionBackend, WorkloadPhase
from .kernel_probe import KernelMeasurement, KernelProbeConfig, KernelProbeResult, run_kernel_probe
from .model_integrity import ModelIntegrityError, verify_model_integrity
from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec

QWEN3_VL_COREML_CONVERSION_SCHEMA_VERSION = 1
MAX_CONVERSION_MANIFEST_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Qwen3VLCoreMLConversionManifest:
    source_artifact_fingerprint: str
    model_revision: str
    coreml_root_sha256: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    compute_precision: str
    converter_version: str
    schema_version: int = QWEN3_VL_COREML_CONVERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        digests = (self.source_artifact_fingerprint, self.coreml_root_sha256)
        if (
            self.schema_version != QWEN3_VL_COREML_CONVERSION_SCHEMA_VERSION
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or len(self.model_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.model_revision)
            or not _label(self.input_name)
            or not _label(self.output_name)
            or not _shape(self.input_shape)
            or not _shape(self.output_shape)
            or self.compute_precision not in {"fp16", "fp32"}
            or not _label(self.converter_version)
        ):
            raise ValueError("invalid Qwen3-VL Core ML conversion manifest")

    @property
    def conversion_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_shape"] = list(self.output_shape)
        return payload


def load_qwen3_vl_coreml_conversion(
    path: Path,
    *,
    source: Qwen3VLVisionANEAdapterSpec,
    coreml_model_path: Path,
    integrity_manifest_path: Path,
) -> Qwen3VLCoreMLConversionManifest:
    """Load a conversion only when source and compiled tree identities still match."""
    payload = _load_manifest(path)
    expected = {
        "schema_version", "source_artifact_fingerprint", "model_revision",
        "coreml_root_sha256", "input_name", "output_name", "input_shape",
        "output_shape", "compute_precision", "converter_version",
    }
    if set(payload) != expected:
        raise ValueError("invalid Qwen3-VL Core ML conversion manifest fields")
    try:
        manifest = Qwen3VLCoreMLConversionManifest(
            source_artifact_fingerprint=payload["source_artifact_fingerprint"],
            model_revision=payload["model_revision"],
            coreml_root_sha256=payload["coreml_root_sha256"],
            input_name=payload["input_name"],
            output_name=payload["output_name"],
            input_shape=tuple(payload["input_shape"]),
            output_shape=tuple(payload["output_shape"]),
            compute_precision=payload["compute_precision"],
            converter_version=payload["converter_version"],
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid Qwen3-VL Core ML conversion manifest") from error
    if (
        manifest.source_artifact_fingerprint != source.artifact_fingerprint
        or manifest.model_revision != source.model_revision
        or manifest.output_shape[-1] != source.output_hidden_size
    ):
        raise ValueError("Qwen3-VL Core ML conversion source does not match")
    try:
        integrity = verify_model_integrity(coreml_model_path, integrity_manifest_path)
    except ModelIntegrityError as error:
        raise ValueError("Qwen3-VL Core ML conversion integrity does not match") from error
    if integrity.get("root_sha256") != manifest.coreml_root_sha256:
        raise ValueError("Qwen3-VL Core ML conversion integrity does not match")
    return manifest


def save_qwen3_vl_coreml_conversion(
    manifest: Qwen3VLCoreMLConversionManifest, path: Path
) -> Path:
    if not isinstance(manifest, Qwen3VLCoreMLConversionManifest):
        raise ValueError("invalid Qwen3-VL Core ML conversion manifest")
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n").encode()
    if len(payload) > MAX_CONVERSION_MANIFEST_BYTES:
        raise ValueError("Qwen3-VL Core ML conversion manifest exceeds size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def qualify_qwen3_vl_coreml_conversion(
    source: Qwen3VLVisionANEAdapterSpec,
    conversion: Qwen3VLCoreMLConversionManifest,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    baseline: Callable[[], KernelMeasurement],
    candidate: Callable[[], KernelMeasurement],
    samples: int = 3,
    maximum_absolute_error: float = 1e-3,
    maximum_slowdown_ratio: float = 1.0,
) -> tuple[KernelProbeResult, DeviceCapability | None]:
    """Promote only a source-bound conversion passing repeatable numeric evidence."""
    if (
        not isinstance(source, Qwen3VLVisionANEAdapterSpec)
        or not isinstance(conversion, Qwen3VLCoreMLConversionManifest)
        or conversion.source_artifact_fingerprint != source.artifact_fingerprint
        or conversion.model_revision != source.model_revision
        or conversion.output_shape[-1] != source.output_hidden_size
        or not callable(baseline)
        or not callable(candidate)
        or not math.isfinite(maximum_absolute_error)
        or maximum_absolute_error < 0
    ):
        raise ValueError("Qwen3-VL Core ML qualification identity is invalid")
    result = run_kernel_probe(
        KernelProbeConfig(
            hardware_fingerprint,
            environment_fingerprint,
            ExecutionBackend.COREML_DRAFT,
            source.operator,
            samples,
            maximum_slowdown_ratio,
            maximum_absolute_error,
        ),
        baseline,
        candidate,
    )
    capability = None
    if result.passed:
        capability = DeviceCapability(
            ExecutionBackend.COREML_DRAFT,
            ComputeDevice.ANE,
            f"coreml:{conversion.converter_version}",
            hardware_fingerprint,
            environment_fingerprint,
            (source.operator,),
            (WorkloadPhase.AUXILIARY,),
            (conversion.compute_precision,),
            "available",
            "probe_passed",
            (result.probe_id,),
        )
    return result, capability


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        attributes = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(attributes.st_mode)
            or not 1 <= attributes.st_size <= MAX_CONVERSION_MANIFEST_BYTES
        ):
            raise ValueError("Qwen3-VL Core ML conversion manifest is invalid")
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen3-VL Core ML conversion manifest is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("Qwen3-VL Core ML conversion manifest must be an object")
    return value


def _label(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(ord(character) >= 0x20 for character in value)
    )


def _shape(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and 1 <= len(value) <= 8
        and all(type(dimension) is int and 1 <= dimension <= 1_048_576 for dimension in value)
    )
