from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import GIB, HardwareInfo

ARTIFACT_ADMISSION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactAdmission:
    schema_version: int
    model: str
    artifact_bytes: int
    estimated_resident_bytes: int
    memory_hard_ceiling_bytes: int
    disk_free_bytes: int
    disk_required_bytes: int
    fits_memory: bool
    fits_disk: bool
    eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_artifact_admission(
    *,
    model: str,
    artifact_bytes: int,
    estimated_resident_bytes: int,
    hardware: HardwareInfo,
    disk_free_bytes: int,
    staging_factor: float = 1.05,
) -> ArtifactAdmission:
    if (
        not model
        or model != model.strip()
        or len(model.encode("utf-8")) > 4_096
        or any(not character.isprintable() for character in model)
    ):
        raise ValueError("artifact admission model identifier is invalid")
    if artifact_bytes <= 0 or estimated_resident_bytes <= 0 or disk_free_bytes < 0:
        raise ValueError("artifact admission byte counts are invalid")
    if not math.isfinite(staging_factor) or not 1 <= staging_factor <= 3:
        raise ValueError("artifact staging factor must be between 1 and 3")
    emergency_margin = max(GIB, int(hardware.memory.total_bytes * 0.08))
    memory_ceiling = max(0, hardware.memory.available_bytes - emergency_margin)
    disk_required = math.ceil(artifact_bytes * staging_factor)
    fits_memory = estimated_resident_bytes <= memory_ceiling
    fits_disk = disk_required <= disk_free_bytes
    return ArtifactAdmission(
        ARTIFACT_ADMISSION_SCHEMA_VERSION,
        model,
        artifact_bytes,
        estimated_resident_bytes,
        memory_ceiling,
        disk_free_bytes,
        disk_required,
        fits_memory,
        fits_disk,
        fits_memory and fits_disk,
    )


def assess_artifact_admission_for_path(
    *,
    model: str,
    artifact_bytes: int,
    estimated_resident_bytes: int,
    hardware: HardwareInfo,
    target: Path,
    staging_factor: float = 1.05,
) -> ArtifactAdmission:
    candidate = target.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    free = shutil.disk_usage(candidate).free
    return assess_artifact_admission(
        model=model,
        artifact_bytes=artifact_bytes,
        estimated_resident_bytes=estimated_resident_bytes,
        hardware=hardware,
        disk_free_bytes=free,
        staging_factor=staging_factor,
    )
