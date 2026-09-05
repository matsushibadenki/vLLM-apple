from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .generative_qualification import GenerativeQualificationPlan


GENERATIVE_EVALUATION_SCHEMA_VERSION = 1
MAX_SAMPLES = 32
MAX_DURATION_MS = 24 * 60 * 60 * 1000
MAX_REPORT_BYTES = 1024 * 1024
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
class GenerativeEvaluationProvenance:
    platform: str
    architecture: str
    soc: str
    gpu_core_count: int | None
    total_memory_bytes: int
    backend: str
    backend_version: str
    artifact_format: str
    artifact_bytes: int
    quantization: str
    license: str | None
    base_model: str | None

    def __post_init__(self) -> None:
        strings = (
            self.platform,
            self.architecture,
            self.soc,
            self.backend,
            self.backend_version,
            self.artifact_format,
            self.quantization,
        )
        if any(not value or len(value.encode("utf-8")) > 512 for value in strings):
            raise ValueError("generative provenance contains an invalid string")
        if self.gpu_core_count is not None and not 0 < self.gpu_core_count <= 1024:
            raise ValueError("generative provenance GPU count is invalid")
        if not 0 < self.total_memory_bytes <= 16_384 * 1024**3:
            raise ValueError("generative provenance total memory is invalid")
        if not 0 < self.artifact_bytes <= 16_384 * 1024**3:
            raise ValueError("generative provenance artifact bytes are invalid")
        for value in (self.license, self.base_model):
            if value is not None and (not value or len(value.encode("utf-8")) > 4096):
                raise ValueError("generative provenance optional string is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerativeEvaluationReport:
    schema_version: int
    candidate_id: str
    model: str
    modality: str
    plan_sha256: str
    provenance: GenerativeEvaluationProvenance
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
        payload["provenance"] = self.provenance.to_dict()
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
    provenance: GenerativeEvaluationProvenance,
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
    if len(samples) >= 2:
        minimum_peak_rss = min(sample.peak_rss_bytes for sample in samples)
        maximum_peak_rss = max(sample.peak_rss_bytes for sample in samples)
        if maximum_peak_rss > minimum_peak_rss * 1.25:
            issues.append("peak_rss_variance_exceeds_25_percent")
    if plan.promotion_axis == "sample_count_4" and len(samples) != 4:
        issues.append("sample_count_promotion_requires_four_samples")

    frame_rates = tuple(
        sample.output_frames / (sample.wall_time_ms / 1000.0) for sample in samples
    )
    return GenerativeEvaluationReport(
        GENERATIVE_EVALUATION_SCHEMA_VERSION,
        plan.candidate.candidate_id,
        plan.candidate.model,
        plan.candidate.modality,
        generative_plan_sha256(plan),
        provenance,
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


def load_generative_evaluation_report(
    path: Path,
    *,
    expected_provenance: GenerativeEvaluationProvenance | None = None,
    expected_plan_sha256: str | None = None,
) -> GenerativeEvaluationReport:
    source = path.expanduser()
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    opened = None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not 1 <= opened.st_size <= MAX_REPORT_BYTES:
            raise ValueError("generative evaluation report must be a bounded regular file")
        chunks = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened is None
        or remaining
        or len(chunks) != opened.st_size
        or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("generative evaluation report changed while reading")
    try:
        payload = json.loads(chunks)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("generative evaluation report is invalid JSON") from error
    report_keys = {
        "schema_version", "candidate_id", "model", "modality", "plan_sha256",
        "provenance", "sample_count", "median_wall_time_ms", "median_first_output_ms",
        "maximum_peak_rss_bytes", "minimum_frames_per_second", "samples", "issues",
        "stores_prompt", "stores_output", "passed",
    }
    provenance_keys = {
        "platform", "architecture", "soc", "gpu_core_count", "total_memory_bytes",
        "backend", "backend_version", "artifact_format", "artifact_bytes", "quantization",
        "license", "base_model",
    }
    sample_keys = {
        "sample_index", "wall_time_ms", "first_output_ms", "peak_rss_bytes",
        "memory_pressure", "thermal_state", "output_width", "output_height", "output_frames",
        "output_sha256", "stores_prompt", "stores_output",
    }
    if not isinstance(payload, dict) or set(payload) != report_keys or payload["schema_version"] != 1:
        raise ValueError("generative evaluation report schema is invalid")
    provenance_payload = payload["provenance"]
    if not isinstance(provenance_payload, dict) or set(provenance_payload) != provenance_keys:
        raise ValueError("generative evaluation provenance schema is invalid")
    raw_samples = payload["samples"]
    if not isinstance(raw_samples, list) or any(
        not isinstance(sample, dict) or set(sample) != sample_keys for sample in raw_samples
    ):
        raise ValueError("generative evaluation sample schema is invalid")
    try:
        provenance = GenerativeEvaluationProvenance(**provenance_payload)
        samples = tuple(GenerativeSampleEvidence(**sample) for sample in raw_samples)
        report = GenerativeEvaluationReport(
            payload["schema_version"], payload["candidate_id"], payload["model"],
            payload["modality"], payload["plan_sha256"], provenance,
            payload["sample_count"], payload["median_wall_time_ms"],
            payload["median_first_output_ms"], payload["maximum_peak_rss_bytes"],
            payload["minimum_frames_per_second"], samples, tuple(payload["issues"]),
            payload["stores_prompt"], payload["stores_output"], payload["passed"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("generative evaluation report values are invalid") from error
    if report.sample_count != len(samples) or not samples:
        raise ValueError("generative evaluation sample count does not match")
    expected_aggregates = (
        statistics.median(sample.wall_time_ms for sample in samples),
        statistics.median(sample.first_output_ms for sample in samples),
        max(sample.peak_rss_bytes for sample in samples),
        min(sample.output_frames / (sample.wall_time_ms / 1000.0) for sample in samples),
    )
    actual_aggregates = (
        report.median_wall_time_ms, report.median_first_output_ms,
        report.maximum_peak_rss_bytes, report.minimum_frames_per_second,
    )
    if actual_aggregates != expected_aggregates:
        raise ValueError("generative evaluation aggregates do not match samples")
    if report.stores_prompt != any(sample.stores_prompt for sample in samples) or (
        report.stores_output != any(sample.stores_output for sample in samples)
    ):
        raise ValueError("generative evaluation privacy aggregate does not match samples")
    if report.passed != (not report.issues):
        raise ValueError("generative evaluation pass state does not match issues")
    if expected_provenance is not None and report.provenance != expected_provenance:
        raise ValueError("generative evaluation provenance does not match")
    if expected_plan_sha256 is not None and report.plan_sha256 != expected_plan_sha256:
        raise ValueError("generative evaluation plan does not match")
    return report
