from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..model import InspectedModel
from ..types import HardwareInfo, MIB
from .profiler import OptimizationPerformanceProfile, hardware_fingerprint
from .safety import validate_immutable_output_path
from .types import (
    CalibrationManifest,
    OptimizationCandidate,
    OptimizationObjective,
    OptimizationPlan,
    QualityBudget,
    ResourceBudget,
    SourceModel,
)


def build_dry_run_plan(
    model: InspectedModel,
    hardware: HardwareInfo,
    output_path: Path,
    objective: OptimizationObjective,
    resource_budget: ResourceBudget,
    quality_budget: QualityBudget | None = None,
    calibration: CalibrationManifest | None = None,
    license_name: str | None = None,
    performance_profile: OptimizationPerformanceProfile | None = None,
    *,
    plan_id: str | None = None,
    created_at: str | None = None,
) -> OptimizationPlan:
    safe_output = validate_immutable_output_path(model.path, output_path)
    source = SourceModel(
        model_id=model.model_id,
        path=str(model.path.resolve()),
        weights_bytes=model.memory_spec.weights_bytes,
        metadata_fingerprint=_model_metadata_fingerprint(model),
        license=license_name,
    )
    fingerprint = hardware_fingerprint(hardware)
    if performance_profile and performance_profile.hardware_fingerprint != fingerprint:
        raise ValueError("performance profile does not match current hardware")
    candidates = tuple(
        _candidate(bits, source.weights_bytes, resource_budget, performance_profile)
        for bits in _candidate_order(objective)
    )
    warnings = [] if performance_profile else ["duration_unavailable_without_hardware_profiler"]
    if calibration is None:
        warnings.append("quality_not_evaluated_without_calibration_manifest")
    if license_name is None:
        warnings.append("source_license_not_recorded")
    return OptimizationPlan(
        plan_id=plan_id or uuid.uuid4().hex,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        objective=objective,
        source=source,
        output_path=str(safe_output),
        hardware_fingerprint=fingerprint,
        quality_budget=quality_budget or QualityBudget(),
        resource_budget=resource_budget,
        calibration=calibration,
        candidates=candidates,
        warnings=tuple(warnings),
    )


def _candidate(
    bits: int,
    weights_bytes: int,
    budget: ResourceBudget,
    profile: OptimizationPerformanceProfile | None,
) -> OptimizationCandidate:
    # Packed weights plus conservative 10% metadata/alignment overhead.
    output_bytes = math.ceil(weights_bytes * bits / 16 * 1.10)
    # Exporters are not implemented in O0. Reserve two output-sized work areas
    # plus fixed metadata space so future adapters cannot silently overcommit disk.
    required_disk = output_bytes * 2 + 256 * MIB
    # Until streaming adapters report measured working sets, assume source weights
    # may be resident with an additional conversion workspace.
    peak_memory = weights_bytes + max(512 * MIB, math.ceil(output_bytes * 0.15))
    reasons = ["backend_adapter_not_implemented"]
    if required_disk > budget.maximum_disk_bytes:
        reasons.append("disk_budget_exceeded")
    if peak_memory > budget.maximum_memory_bytes:
        reasons.append("memory_budget_exceeded")
    duration = None
    if profile:
        duration = max(
            1,
            math.ceil(
                weights_bytes / profile.read_bytes_per_second
                + (output_bytes * 2) / profile.write_bytes_per_second
            ),
        )
        if budget.maximum_duration_seconds is not None and duration > budget.maximum_duration_seconds:
            reasons.append("duration_budget_exceeded")
    elif budget.maximum_duration_seconds is not None:
        reasons.append("duration_unavailable_without_hardware_profiler")
    within_budget = not any(reason.endswith("budget_exceeded") for reason in reasons)
    return OptimizationCandidate(
        strategy=f"int{bits}_weight_quantization",
        target_weight_bits=bits,
        estimated_output_bytes=output_bytes,
        required_disk_bytes=required_disk,
        estimated_peak_memory_bytes=peak_memory,
        estimated_duration_seconds=duration,
        within_budget=within_budget,
        executable=False,
        blocking_reasons=tuple(reasons),
    )


def _candidate_order(objective: OptimizationObjective) -> tuple[int, int]:
    if objective == OptimizationObjective.MEMORY:
        return (4, 8)
    return (8, 4)


def _model_metadata_fingerprint(model: InspectedModel) -> str:
    payload = {
        "model_id": model.model_id,
        "path": str(model.path.resolve()),
        "weights_bytes": model.memory_spec.weights_bytes,
        "kv_bytes_per_token": model.memory_spec.kv_bytes_per_token,
        "config": model.config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
