from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .compat import (
    BackendCompatibility,
    MLXBackendCompatibility,
    inspect_backend,
    inspect_mlx_lm_backend,
)
from .hardware import detect_hardware
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
    if not backend.compatible:
        issues.extend(f"backend:{issue}" for issue in backend.issues)
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
        tuple(issues),
    )
