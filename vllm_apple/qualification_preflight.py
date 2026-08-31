from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .compat import (
    BackendCompatibility,
    MLXBackendCompatibility,
    inspect_backend,
    inspect_mlx_lm_backend,
    assess_candidate_backend,
)
from .hardware import detect_hardware
from .model import ModelInspectionError, assess_model_memory_fit, inspect_model
from .types import HardwareInfo, MemoryPressure

QUALIFICATION_PREFLIGHT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class QualificationPreflight:
    schema_version: int
    eligible: bool
    platform: str
    architecture: str
    soc: str
    gpu_core_count: int | None
    total_memory_bytes: int
    available_memory_bytes: int
    memory_pressure: str
    backend_kind: str
    backend: dict[str, object]
    model: dict[str, object] | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "eligible": self.eligible,
            "platform": self.platform,
            "architecture": self.architecture,
            "soc": self.soc,
            "gpu_core_count": self.gpu_core_count,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_pressure": self.memory_pressure,
            "backend_kind": self.backend_kind,
            "backend": self.backend,
            "model": self.model,
            "issues": list(self.issues),
        }


def run_qualification_preflight(
    backend_executable: str | Path,
    *,
    hardware_detector: Callable[[], HardwareInfo] = detect_hardware,
    backend_kind: str = "vllm_metal",
    backend_inspector: Callable[
        [str | Path | None], BackendCompatibility | MLXBackendCompatibility
    ]
    | None = None,
    candidate_versions: tuple[str, str, str] | None = None,
    model: str | None = None,
    max_model_len: int | None = None,
    requested_modes: tuple[str, ...] = ("text",),
) -> QualificationPreflight:
    if backend_kind not in {"vllm_metal", "mlx_lm"}:
        raise ValueError("unsupported qualification backend")
    hardware = hardware_detector()
    inspector = backend_inspector or (
        inspect_mlx_lm_backend if backend_kind == "mlx_lm" else inspect_backend
    )
    backend = inspector(backend_executable)
    issues: list[str] = []
    if not hardware.is_apple_silicon:
        issues.append("apple_silicon_required")
    if hardware.gpu_core_count is None:
        issues.append("metal_gpu_core_count_unavailable")
    if hardware.memory.pressure is MemoryPressure.CRITICAL:
        issues.append("memory_pressure_critical")
    backend_issues = backend.issues
    if candidate_versions is not None:
        if backend_kind != "vllm_metal" or not isinstance(backend, BackendCompatibility):
            backend_issues = ("candidate_stack_requires_vllm_metal",)
        else:
            backend_issues = assess_candidate_backend(
                backend,
                expected_vllm=candidate_versions[0],
                expected_vllm_metal=candidate_versions[1],
                expected_transformers=candidate_versions[2],
            )
    if backend_issues:
        issues.extend(f"backend:{issue}" for issue in backend_issues)
    model_evidence: dict[str, object] | None = None
    if model is not None:
        try:
            if not requested_modes or len(set(requested_modes)) != len(requested_modes):
                raise ValueError("requested modes are invalid")
            if any(mode not in {"text", "vision", "mtp", "yarn"} for mode in requested_modes):
                raise ValueError("requested modes are invalid")
            if max_model_len is not None and not 1 <= max_model_len <= 16_777_216:
                raise ValueError("maximum model length is outside the supported range")
            inspected = inspect_model(model)
            capability = inspected.architecture_capability
            unsupported_modes = set(requested_modes) - set(capability.modes)
            if unsupported_modes:
                issues.append("model:missing_requested_modes:" + ",".join(sorted(unsupported_modes)))
            optional_features = {
                "vision_encoder": "vision",
                "multi_token_prediction": "mtp",
                "yarn_extended_context": "yarn",
            }
            required_features = tuple(
                feature
                for feature in capability.required_features
                if feature not in optional_features
                or optional_features[feature] in requested_modes
            )
            available_features = frozenset(backend.architecture_features)
            missing_features = tuple(
                feature for feature in required_features if feature not in available_features
            )
            if missing_features:
                issues.append("backend:missing_model_features:" + ",".join(missing_features))
            context_tokens = (
                max_model_len
                or capability.native_context_tokens
                or inspected.memory_spec.model_max_context
                or 4096
            )
            memory_model = inspected
            if "mtp" not in requested_modes and inspected.state_memory_spec is not None:
                memory_model = replace(
                    inspected,
                    state_memory_spec=replace(
                        inspected.state_memory_spec, mtp_working_set_bytes=0
                    ),
                )
            fit = assess_model_memory_fit(
                memory_model, hardware, context_tokens=context_tokens
            )
            if not fit.fits:
                issues.append("model:memory_hard_ceiling_exceeded")
            model_evidence = {
                "identifier": model,
                "architecture": capability.architecture,
                "requested_modes": list(requested_modes),
                "required_features": list(required_features),
                "missing_features": list(missing_features),
                "artifact_bytes": fit.artifact_bytes,
                "estimated_resident_bytes": fit.estimated_resident_bytes,
                "hard_ceiling_bytes": fit.hard_ceiling_bytes,
                "context_tokens": fit.context_tokens,
                "fits_memory": fit.fits,
            }
        except (ModelInspectionError, OSError, ValueError):
            issues.append("model:inspection_failed")
            model_evidence = {"identifier": model, "status": "invalid"}
    return QualificationPreflight(
        QUALIFICATION_PREFLIGHT_SCHEMA_VERSION,
        not issues,
        hardware.platform,
        hardware.architecture,
        hardware.soc,
        hardware.gpu_core_count,
        hardware.memory.total_bytes,
        hardware.memory.available_bytes,
        hardware.memory.pressure.value,
        backend_kind,
        backend.to_dict(),
        model_evidence,
        tuple(issues),
    )
