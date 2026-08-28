from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .types import OPTIMIZER_SCHEMA_VERSION


MAX_EVALUATION_SLICES = 64
MAX_EVALUATION_REPORT_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PerplexitySlice:
    domain: str
    language: str
    sample_count: int
    token_count: int
    mean_negative_log_likelihood: float
    perplexity: float

    def __post_init__(self) -> None:
        if (
            not self.domain
            or not self.language
            or self.sample_count <= 0
            or self.token_count <= 0
            or not math.isfinite(self.mean_negative_log_likelihood)
            or self.mean_negative_log_likelihood < 0
            or not math.isfinite(self.perplexity)
            or self.perplexity < 1
        ):
            raise ValueError("invalid perplexity slice")

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "language": self.language,
            "sample_count": self.sample_count,
            "token_count": self.token_count,
            "mean_negative_log_likelihood": self.mean_negative_log_likelihood,
            "perplexity": self.perplexity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PerplexitySlice":
        required = {
            "domain",
            "language",
            "sample_count",
            "token_count",
            "mean_negative_log_likelihood",
            "perplexity",
        }
        if set(payload) != required:
            raise ValueError("malformed perplexity slice")
        _require_strings(payload, ("domain", "language"))
        _require_integers(payload, ("sample_count", "token_count"))
        _require_numbers(payload, ("mean_negative_log_likelihood", "perplexity"))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PerplexityEvaluationReport:
    model_path: str
    model_hash: str
    dataset_path: str
    dataset_fingerprint: str
    sample_count: int
    token_count: int
    mean_negative_log_likelihood: float
    perplexity: float
    elapsed_milliseconds: int
    peak_rss_bytes: int
    slices: tuple[PerplexitySlice, ...]

    def __post_init__(self) -> None:
        if not Path(self.model_path).is_absolute() or not Path(self.dataset_path).is_absolute():
            raise ValueError("evaluation paths must be absolute")
        if len(self.model_hash) != 64 or len(self.dataset_fingerprint) != 64:
            raise ValueError("evaluation fingerprints must be SHA-256")
        if (
            self.sample_count <= 0
            or self.token_count <= 0
            or self.elapsed_milliseconds < 0
            or self.peak_rss_bytes <= 0
            or not math.isfinite(self.mean_negative_log_likelihood)
            or self.mean_negative_log_likelihood < 0
            or not math.isfinite(self.perplexity)
            or self.perplexity < 1
        ):
            raise ValueError("invalid evaluation measurements")
        if not self.slices or len(self.slices) > MAX_EVALUATION_SLICES:
            raise ValueError("evaluation slices must be bounded and non-empty")
        if sum(value.sample_count for value in self.slices) != self.sample_count:
            raise ValueError("evaluation slice sample counts do not match")
        if sum(value.token_count for value in self.slices) != self.token_count:
            raise ValueError("evaluation slice token counts do not match")
        identities = {(value.domain, value.language) for value in self.slices}
        if len(identities) != len(self.slices):
            raise ValueError("evaluation slices must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "dataset_path": self.dataset_path,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_count": self.sample_count,
            "token_count": self.token_count,
            "mean_negative_log_likelihood": self.mean_negative_log_likelihood,
            "perplexity": self.perplexity,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "slices": [value.to_dict() for value in self.slices],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PerplexityEvaluationReport":
        required = {
            "schema_version",
            "model_path",
            "model_hash",
            "dataset_path",
            "dataset_fingerprint",
            "sample_count",
            "token_count",
            "mean_negative_log_likelihood",
            "perplexity",
            "elapsed_milliseconds",
            "peak_rss_bytes",
            "slices",
        }
        if set(payload) != required or payload.get("schema_version") != OPTIMIZER_SCHEMA_VERSION:
            raise ValueError("unsupported or malformed evaluation report")
        _require_strings(payload, ("model_path", "model_hash", "dataset_path", "dataset_fingerprint"))
        _require_integers(
            payload,
            ("sample_count", "token_count", "elapsed_milliseconds", "peak_rss_bytes"),
        )
        _require_numbers(payload, ("mean_negative_log_likelihood", "perplexity"))
        raw_slices = payload["slices"]
        if not isinstance(raw_slices, list) or any(not isinstance(value, dict) for value in raw_slices):
            raise ValueError("evaluation slices must be an array of objects")
        return cls(
            model_path=payload["model_path"],
            model_hash=payload["model_hash"],
            dataset_path=payload["dataset_path"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            sample_count=payload["sample_count"],
            token_count=payload["token_count"],
            mean_negative_log_likelihood=float(payload["mean_negative_log_likelihood"]),
            perplexity=float(payload["perplexity"]),
            elapsed_milliseconds=payload["elapsed_milliseconds"],
            peak_rss_bytes=payload["peak_rss_bytes"],
            slices=tuple(PerplexitySlice.from_dict(value) for value in raw_slices),
        )


@dataclass(frozen=True, slots=True)
class QualityGateSlice:
    domain: str
    language: str
    baseline_perplexity: float
    candidate_perplexity: float
    relative_regression: float
    maximum_regression: float
    passed: bool

    def __post_init__(self) -> None:
        values = (
            self.baseline_perplexity,
            self.candidate_perplexity,
            self.relative_regression,
            self.maximum_regression,
        )
        if (
            not self.domain
            or not self.language
            or any(not math.isfinite(value) for value in values)
            or self.baseline_perplexity < 1
            or self.candidate_perplexity < 1
            or self.relative_regression < -1
            or not 0 <= self.maximum_regression <= 1
            or self.passed != (self.relative_regression <= self.maximum_regression)
        ):
            raise ValueError("invalid quality gate slice")

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "language": self.language,
            "baseline_perplexity": self.baseline_perplexity,
            "candidate_perplexity": self.candidate_perplexity,
            "relative_regression": self.relative_regression,
            "maximum_regression": self.maximum_regression,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    created_at: str
    dataset_fingerprint: str
    baseline_model_hash: str
    candidate_model_hash: str
    approved: bool
    slices: tuple[QualityGateSlice, ...]
    untested_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.created_at
            or len(self.dataset_fingerprint) != 64
            or len(self.baseline_model_hash) != 64
            or len(self.candidate_model_hash) != 64
            or not self.slices
            or len(self.slices) > MAX_EVALUATION_SLICES
            or self.approved != all(value.passed for value in self.slices)
            or not self.untested_capabilities
            or any(not value for value in self.untested_capabilities)
        ):
            raise ValueError("invalid quality gate report")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "created_at": self.created_at,
            "dataset_fingerprint": self.dataset_fingerprint,
            "baseline_model_hash": self.baseline_model_hash,
            "candidate_model_hash": self.candidate_model_hash,
            "approved": self.approved,
            "slices": [value.to_dict() for value in self.slices],
            "untested_capabilities": list(self.untested_capabilities),
        }


def compare_perplexity_reports(
    baseline: PerplexityEvaluationReport,
    candidate: PerplexityEvaluationReport,
    maximum_regression: float,
) -> QualityGateReport:
    if not isinstance(maximum_regression, (int, float)) or isinstance(maximum_regression, bool):
        raise ValueError("maximum regression must be numeric")
    maximum = float(maximum_regression)
    if not math.isfinite(maximum) or not 0 <= maximum <= 1:
        raise ValueError("maximum regression must be between zero and one")
    if baseline.dataset_fingerprint != candidate.dataset_fingerprint:
        raise ValueError("quality reports must use the same dataset fingerprint")
    baseline_slices = {(value.domain, value.language): value for value in baseline.slices}
    candidate_slices = {(value.domain, value.language): value for value in candidate.slices}
    if baseline_slices.keys() != candidate_slices.keys():
        raise ValueError("quality reports must contain identical evaluation slices")
    comparisons: list[QualityGateSlice] = []
    for identity in sorted(baseline_slices):
        baseline_slice = baseline_slices[identity]
        candidate_slice = candidate_slices[identity]
        if (
            baseline_slice.sample_count != candidate_slice.sample_count
            or baseline_slice.token_count != candidate_slice.token_count
        ):
            raise ValueError("quality report slice counts do not match")
        regression = (
            candidate_slice.perplexity - baseline_slice.perplexity
        ) / baseline_slice.perplexity
        comparisons.append(
            QualityGateSlice(
                domain=identity[0],
                language=identity[1],
                baseline_perplexity=baseline_slice.perplexity,
                candidate_perplexity=candidate_slice.perplexity,
                relative_regression=regression,
                maximum_regression=maximum,
                passed=regression <= maximum,
            )
        )
    return QualityGateReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=baseline.dataset_fingerprint,
        baseline_model_hash=baseline.model_hash,
        candidate_model_hash=candidate.model_hash,
        approved=all(value.passed for value in comparisons),
        slices=tuple(comparisons),
        untested_capabilities=(
            "generation_quality",
            "long_context",
            "code",
            "mathematics",
            "safety_alignment",
        ),
    )


def load_perplexity_report(path: Path) -> PerplexityEvaluationReport:
    candidate = path.expanduser()
    info = candidate.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not 0 < info.st_size <= MAX_EVALUATION_REPORT_BYTES
    ):
        raise ValueError("evaluation report must be a bounded owned regular file")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation report must be a JSON object")
    return PerplexityEvaluationReport.from_dict(payload)


def persist_evaluation_report(payload: dict[str, object], path: Path) -> Path:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists() or destination == Path(destination.anchor):
        raise ValueError("immutable evaluation report output already exists or is unsafe")
    if not destination.parent.is_dir():
        raise ValueError("evaluation report parent must already exist")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not 0 < len(encoded) <= MAX_EVALUATION_REPORT_BYTES:
        raise ValueError("evaluation report exceeds its byte limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination.resolve(strict=True)


def _require_strings(payload: dict[str, object], names: tuple[str, ...]) -> None:
    if any(not isinstance(payload[name], str) for name in names):
        raise ValueError("report string field is invalid")


def _require_integers(payload: dict[str, object], names: tuple[str, ...]) -> None:
    if any(not isinstance(payload[name], int) or isinstance(payload[name], bool) for name in names):
        raise ValueError("report integer field is invalid")


def _require_numbers(payload: dict[str, object], names: tuple[str, ...]) -> None:
    if any(
        not isinstance(payload[name], (int, float)) or isinstance(payload[name], bool)
        for name in names
    ):
        raise ValueError("report numeric field is invalid")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
