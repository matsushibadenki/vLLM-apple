from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


OPTIMIZER_SCHEMA_VERSION = 1


class OptimizationObjective(str, Enum):
    BALANCED = "balanced"
    MEMORY = "memory"
    SPEED = "speed"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class QualityBudget:
    maximum_regression: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for domain, value in self.maximum_regression.items():
            if not domain or not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("quality budget domains and values must be valid")
            if not 0 <= float(value) <= 1:
                raise ValueError("quality regression must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return {"maximum_regression": dict(self.maximum_regression)}


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    maximum_memory_bytes: int
    maximum_disk_bytes: int
    maximum_duration_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_memory_bytes <= 0 or self.maximum_disk_bytes <= 0:
            raise ValueError("memory and disk budgets must be positive")
        if self.maximum_duration_seconds is not None and self.maximum_duration_seconds <= 0:
            raise ValueError("duration budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    dataset_id: str
    fingerprint: str
    sample_count: int
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    local_only: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_id or len(self.fingerprint) < 16 or self.sample_count <= 0:
            raise ValueError("invalid calibration manifest")
        if not self.languages or not self.domains:
            raise ValueError("calibration languages and domains cannot be empty")
        if any(not value for value in (*self.languages, *self.domains)):
            raise ValueError("calibration labels cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "sample_count": self.sample_count,
            "languages": list(self.languages),
            "domains": list(self.domains),
            "local_only": self.local_only,
        }


@dataclass(frozen=True, slots=True)
class SourceModel:
    model_id: str
    path: str
    weights_bytes: int
    metadata_fingerprint: str
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.path or self.weights_bytes <= 0:
            raise ValueError("invalid source model")
        if len(self.metadata_fingerprint) < 16:
            raise ValueError("source model fingerprint is too short")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    strategy: str
    target_weight_bits: int
    estimated_output_bytes: int
    required_disk_bytes: int
    estimated_peak_memory_bytes: int
    estimated_duration_seconds: int | None
    within_budget: bool
    executable: bool
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy or self.target_weight_bits not in {4, 8, 16}:
            raise ValueError("invalid optimization candidate")
        if min(
            self.estimated_output_bytes,
            self.required_disk_bytes,
            self.estimated_peak_memory_bytes,
        ) <= 0:
            raise ValueError("candidate resource estimates must be positive")
        if self.estimated_duration_seconds is not None and self.estimated_duration_seconds <= 0:
            raise ValueError("candidate duration must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocking_reasons"] = list(self.blocking_reasons)
        return result


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    plan_id: str
    created_at: str
    objective: OptimizationObjective
    source: SourceModel
    output_path: str
    hardware_fingerprint: str
    quality_budget: QualityBudget
    resource_budget: ResourceBudget
    calibration: CalibrationManifest | None
    candidates: tuple[OptimizationCandidate, ...]
    warnings: tuple[str, ...] = ()
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not self.plan_id or not self.created_at or not self.output_path:
            raise ValueError("invalid optimization plan identity")
        if len(self.hardware_fingerprint) < 16 or not self.candidates:
            raise ValueError("optimization plan requires hardware and candidates")
        if not self.dry_run:
            raise ValueError("O0 plans must remain dry-run only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "objective": self.objective.value,
            "source": self.source.to_dict(),
            "output_path": self.output_path,
            "hardware_fingerprint": self.hardware_fingerprint,
            "quality_budget": self.quality_budget.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    created_at: str
    plan_id: str
    source_hash: str
    output_hash: str
    output_bytes: int
    transforms: tuple[Mapping[str, Any], ...]
    tool_versions: Mapping[str, str]
    calibration_fingerprint: str | None
    evaluation: Mapping[str, float]
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.plan_id or not self.created_at:
            raise ValueError("invalid artifact manifest identity")
        if len(self.source_hash) < 16 or len(self.output_hash) < 16 or self.output_bytes <= 0:
            raise ValueError("invalid artifact hashes or size")
        if not self.transforms or not self.tool_versions:
            raise ValueError("artifact provenance cannot be empty")
        if any(not 0 <= value <= 1 for value in self.evaluation.values()):
            raise ValueError("evaluation scores must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "plan_id": self.plan_id,
            "source_hash": self.source_hash,
            "output_hash": self.output_hash,
            "output_bytes": self.output_bytes,
            "transforms": [dict(value) for value in self.transforms],
            "tool_versions": dict(self.tool_versions),
            "calibration_fingerprint": self.calibration_fingerprint,
            "evaluation": dict(self.evaluation),
            "license": self.license,
        }
