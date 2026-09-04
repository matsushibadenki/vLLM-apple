from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .generative_qualification import GenerativeQualificationPlan


GENERATIVE_EVALUATION_SCHEMA_VERSION = 1
MAX_SAMPLES = 32
MAX_DURATION_MS = 24 * 60 * 60 * 1000
MEMORY_PRESSURES = frozenset({"normal", "warning", "critical", "unknown"})
THERMAL_STATES = frozenset({"nominal", "fair", "serious", "critical", "unknown"})


@dataclass(frozen=True, slots=True)
class GenerativeSampleEvidence:
    sample_index: int
    wall_time_ms: float
    first_output_ms: float
    peak_rss_bytes: int
    memory_pressure: str
    thermal_state: str
    output_width: int
    output_height: int
    output_frames: int
    output_sha256: str
    stores_prompt: bool = False
    stores_output: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.sample_index < MAX_SAMPLES:
            raise ValueError("generative sample index is outside the supported range")
        timings = (self.wall_time_ms, self.first_output_ms)
        if any(not math.isfinite(value) or not 0 < value <= MAX_DURATION_MS for value in timings):
            raise ValueError("generative sample timings are outside the supported range")
        if self.first_output_ms > self.wall_time_ms:
            raise ValueError("first output time must not exceed wall time")
        if not 0 < self.peak_rss_bytes <= 16_384 * 1024**3:
            raise ValueError("generative sample peak RSS is outside the supported range")
        if self.memory_pressure not in MEMORY_PRESSURES:
            raise ValueError("unsupported memory pressure")
        if self.thermal_state not in THERMAL_STATES:
            raise ValueError("unsupported thermal state")
        if self.output_width <= 0 or self.output_height <= 0 or self.output_frames <= 0:
            raise ValueError("generative output dimensions are invalid")
        if len(self.output_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.output_sha256
        ):
            raise ValueError("generative output digest must be lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerativeEvaluationReport:
    schema_version: int
    candidate_id: str
    model: str
    modality: str
    plan_sha256: str
    sample_count: int
    median_wall_time_ms: float
    median_first_output_ms: float
    maximum_peak_rss_bytes: int
    minimum_frames_per_second: float
    samples: tuple[GenerativeSampleEvidence, ...]
    issues: tuple[str, ...]
    stores_prompt: bool
    stores_output: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        payload["issues"] = list(self.issues)
        return payload


def generative_plan_sha256(plan: GenerativeQualificationPlan) -> str:
    encoded = json.dumps(
        plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_generative_qualification(
    plan: GenerativeQualificationPlan,
    samples: tuple[GenerativeSampleEvidence, ...],
) -> GenerativeEvaluationReport:
    if not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("between 1 and 32 generative samples are required")
    indices = tuple(sample.sample_index for sample in samples)
    if indices != tuple(range(len(samples))):
        raise ValueError("generative sample indices must be contiguous from zero")

    issues: list[str] = []
    if not plan.eligible:
        issues.append("qualification_plan_ineligible")
    for sample in samples:
        if sample.peak_rss_bytes > plan.artifact_admission.memory_hard_ceiling_bytes:
            issues.append(f"sample:{sample.sample_index}:peak_rss_exceeds_hard_ceiling")
        if sample.memory_pressure in {"critical", "unknown"}:
            issues.append(f"sample:{sample.sample_index}:unsafe_memory_pressure")
        if sample.thermal_state in {"serious", "critical", "unknown"}:
            issues.append(f"sample:{sample.sample_index}:unsafe_thermal_state")
        if (sample.output_width, sample.output_height, sample.output_frames) != (
            plan.width,
            plan.height,
            plan.frames,
        ):
            issues.append(f"sample:{sample.sample_index}:output_shape_mismatch")
        if sample.stores_prompt or sample.stores_output:
            issues.append(f"sample:{sample.sample_index}:private_content_retained")

    frame_rates = tuple(
        sample.output_frames / (sample.wall_time_ms / 1000.0) for sample in samples
    )
    return GenerativeEvaluationReport(
        GENERATIVE_EVALUATION_SCHEMA_VERSION,
        plan.candidate.candidate_id,
        plan.candidate.model,
        plan.candidate.modality,
        generative_plan_sha256(plan),
        len(samples),
        statistics.median(sample.wall_time_ms for sample in samples),
        statistics.median(sample.first_output_ms for sample in samples),
        max(sample.peak_rss_bytes for sample in samples),
        min(frame_rates),
        samples,
        tuple(issues),
        any(sample.stores_prompt for sample in samples),
        any(sample.stores_output for sample in samples),
        not issues,
    )


def save_generative_evaluation_report(
    report: GenerativeEvaluationReport, path: Path
) -> Path:
    destination = path.expanduser().resolve()
    parent_existed = destination.parent.exists()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        destination.parent.chmod(0o700)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
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
