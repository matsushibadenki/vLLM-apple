from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import MAX_EVALUATION_REPORT_BYTES, MAX_EVALUATION_SLICES
from .types import OPTIMIZER_SCHEMA_VERSION


MAX_GENERATION_TOKENS = 256
MAX_PROMPT_TOKENS = 65_536


@dataclass(frozen=True, slots=True)
class GenerationSampleResult:
    sample_id: str
    domain: str
    language: str
    prompt_token_count: int
    token_ids: tuple[int, ...]
    output_fingerprint: str
    expectation_score: float

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or not self.domain
            or not self.language
            or not 1 <= self.prompt_token_count <= MAX_PROMPT_TOKENS
            or not self.token_ids
            or len(self.token_ids) > MAX_GENERATION_TOKENS
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.token_ids)
            or len(self.output_fingerprint) != 64
            or not math.isfinite(self.expectation_score)
            or not 0 <= self.expectation_score <= 1
        ):
            raise ValueError("invalid generation sample result")
        if self.output_fingerprint != generation_token_fingerprint(self.token_ids):
            raise ValueError("generation token fingerprint does not match")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "language": self.language,
            "prompt_token_count": self.prompt_token_count,
            "token_ids": list(self.token_ids),
            "output_fingerprint": self.output_fingerprint,
            "expectation_score": self.expectation_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GenerationSampleResult":
        required = {
            "sample_id",
            "domain",
            "language",
            "prompt_token_count",
            "token_ids",
            "output_fingerprint",
            "expectation_score",
        }
        if set(payload) != required:
            raise ValueError("malformed generation sample result")
        if any(not isinstance(payload[name], str) for name in ("sample_id", "domain", "language", "output_fingerprint")):
            raise ValueError("generation sample strings are invalid")
        raw_tokens = payload["token_ids"]
        prompt_token_count = payload["prompt_token_count"]
        if not isinstance(prompt_token_count, int) or isinstance(prompt_token_count, bool):
            raise ValueError("generation prompt token count is invalid")
        if not isinstance(raw_tokens, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in raw_tokens
        ):
            raise ValueError("generation sample tokens are invalid")
        score = payload["expectation_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("generation expectation score is invalid")
        return cls(
            sample_id=payload["sample_id"],
            domain=payload["domain"],
            language=payload["language"],
            prompt_token_count=prompt_token_count,
            token_ids=tuple(raw_tokens),
            output_fingerprint=payload["output_fingerprint"],
            expectation_score=float(score),
        )


@dataclass(frozen=True, slots=True)
class GenerationEvaluationReport:
    model_path: str
    model_hash: str
    dataset_path: str
    dataset_fingerprint: str
    prompt_format: str
    maximum_prompt_tokens: int
    maximum_new_tokens: int
    elapsed_milliseconds: int
    peak_rss_bytes: int
    samples: tuple[GenerationSampleResult, ...]

    def __post_init__(self) -> None:
        if not Path(self.model_path).is_absolute() or not Path(self.dataset_path).is_absolute():
            raise ValueError("generation evaluation paths must be absolute")
        if len(self.model_hash) != 64 or len(self.dataset_fingerprint) != 64:
            raise ValueError("generation evaluation fingerprints must be SHA-256")
        if self.prompt_format not in {"raw", "chat_template"}:
            raise ValueError("generation prompt format is invalid")
        if (
            not 1 <= self.maximum_prompt_tokens <= MAX_PROMPT_TOKENS
            or not 1 <= self.maximum_new_tokens <= MAX_GENERATION_TOKENS
            or self.elapsed_milliseconds < 0
            or self.peak_rss_bytes <= 0
            or not self.samples
            or len(self.samples) > MAX_EVALUATION_SLICES
        ):
            raise ValueError("invalid generation evaluation measurements")
        if len({value.sample_id for value in self.samples}) != len(self.samples):
            raise ValueError("generation sample IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "dataset_path": self.dataset_path,
            "dataset_fingerprint": self.dataset_fingerprint,
            "prompt_format": self.prompt_format,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_new_tokens": self.maximum_new_tokens,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "samples": [value.to_dict() for value in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GenerationEvaluationReport":
        required = {
            "schema_version",
            "model_path",
            "model_hash",
            "dataset_path",
            "dataset_fingerprint",
            "prompt_format",
            "maximum_prompt_tokens",
            "maximum_new_tokens",
            "elapsed_milliseconds",
            "peak_rss_bytes",
            "samples",
        }
        if set(payload) != required or payload.get("schema_version") != OPTIMIZER_SCHEMA_VERSION:
            raise ValueError("unsupported or malformed generation evaluation report")
        string_fields = (
            "model_path",
            "model_hash",
            "dataset_path",
            "dataset_fingerprint",
            "prompt_format",
        )
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise ValueError("generation evaluation string fields are invalid")
        integer_fields = (
            "maximum_prompt_tokens",
            "maximum_new_tokens",
            "elapsed_milliseconds",
            "peak_rss_bytes",
        )
        if any(
            not isinstance(payload[name], int) or isinstance(payload[name], bool)
            for name in integer_fields
        ):
            raise ValueError("generation evaluation integer fields are invalid")
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list) or any(not isinstance(value, dict) for value in raw_samples):
            raise ValueError("generation evaluation samples are invalid")
        return cls(
            model_path=payload["model_path"],
            model_hash=payload["model_hash"],
            dataset_path=payload["dataset_path"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            prompt_format=payload["prompt_format"],
            maximum_prompt_tokens=payload["maximum_prompt_tokens"],
            maximum_new_tokens=payload["maximum_new_tokens"],
            elapsed_milliseconds=payload["elapsed_milliseconds"],
            peak_rss_bytes=payload["peak_rss_bytes"],
            samples=tuple(GenerationSampleResult.from_dict(value) for value in raw_samples),
        )


@dataclass(frozen=True, slots=True)
class GenerationGateSample:
    sample_id: str
    domain: str
    language: str
    token_agreement: float
    minimum_token_agreement: float
    baseline_expectation_score: float
    candidate_expectation_score: float
    expectation_regression: float
    maximum_expectation_regression: float
    passed: bool

    def __post_init__(self) -> None:
        scores = (
            self.token_agreement,
            self.minimum_token_agreement,
            self.baseline_expectation_score,
            self.candidate_expectation_score,
            self.maximum_expectation_regression,
        )
        if (
            not self.sample_id
            or not self.domain
            or not self.language
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)
            or not math.isfinite(self.expectation_regression)
            or not -1 <= self.expectation_regression <= 1
            or self.passed
            != (
                self.token_agreement >= self.minimum_token_agreement
                and self.expectation_regression <= self.maximum_expectation_regression
            )
        ):
            raise ValueError("invalid generation gate sample")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "language": self.language,
            "token_agreement": self.token_agreement,
            "minimum_token_agreement": self.minimum_token_agreement,
            "baseline_expectation_score": self.baseline_expectation_score,
            "candidate_expectation_score": self.candidate_expectation_score,
            "expectation_regression": self.expectation_regression,
            "maximum_expectation_regression": self.maximum_expectation_regression,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class GenerationQualityGateReport:
    created_at: str
    dataset_fingerprint: str
    baseline_model_hash: str
    candidate_model_hash: str
    approved: bool
    samples: tuple[GenerationGateSample, ...]
    untested_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.created_at
            or len(self.dataset_fingerprint) != 64
            or len(self.baseline_model_hash) != 64
            or len(self.candidate_model_hash) != 64
            or not self.samples
            or len(self.samples) > MAX_EVALUATION_SLICES
            or self.approved != all(value.passed for value in self.samples)
            or not self.untested_capabilities
        ):
            raise ValueError("invalid generation quality gate report")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "created_at": self.created_at,
            "dataset_fingerprint": self.dataset_fingerprint,
            "baseline_model_hash": self.baseline_model_hash,
            "candidate_model_hash": self.candidate_model_hash,
            "approved": self.approved,
            "samples": [value.to_dict() for value in self.samples],
            "untested_capabilities": list(self.untested_capabilities),
        }


def compare_generation_reports(
    baseline: GenerationEvaluationReport,
    candidate: GenerationEvaluationReport,
    *,
    minimum_token_agreement: float,
    maximum_expectation_regression: float,
) -> GenerationQualityGateReport:
    minimum = _bounded_fraction("minimum token agreement", minimum_token_agreement)
    maximum = _bounded_fraction(
        "maximum expectation regression",
        maximum_expectation_regression,
    )
    if baseline.dataset_fingerprint != candidate.dataset_fingerprint:
        raise ValueError("generation reports must use the same dataset fingerprint")
    if baseline.maximum_new_tokens != candidate.maximum_new_tokens:
        raise ValueError("generation reports must use the same token limit")
    if baseline.prompt_format != candidate.prompt_format:
        raise ValueError("generation reports must use the same prompt format")
    if baseline.maximum_prompt_tokens != candidate.maximum_prompt_tokens:
        raise ValueError("generation reports must use the same prompt token limit")
    baseline_samples = {value.sample_id: value for value in baseline.samples}
    candidate_samples = {value.sample_id: value for value in candidate.samples}
    if baseline_samples.keys() != candidate_samples.keys():
        raise ValueError("generation reports must contain identical samples")
    comparisons: list[GenerationGateSample] = []
    for sample_id in sorted(baseline_samples):
        reference = baseline_samples[sample_id]
        optimized = candidate_samples[sample_id]
        if (reference.domain, reference.language) != (optimized.domain, optimized.language):
            raise ValueError("generation sample labels do not match")
        if reference.prompt_token_count != optimized.prompt_token_count:
            raise ValueError("generation sample prompt token counts do not match")
        denominator = max(len(reference.token_ids), len(optimized.token_ids))
        matching = sum(
            left == right for left, right in zip(reference.token_ids, optimized.token_ids)
        )
        agreement = matching / denominator
        expectation_regression = reference.expectation_score - optimized.expectation_score
        comparisons.append(
            GenerationGateSample(
                sample_id=sample_id,
                domain=reference.domain,
                language=reference.language,
                token_agreement=agreement,
                minimum_token_agreement=minimum,
                baseline_expectation_score=reference.expectation_score,
                candidate_expectation_score=optimized.expectation_score,
                expectation_regression=expectation_regression,
                maximum_expectation_regression=maximum,
                passed=agreement >= minimum and expectation_regression <= maximum,
            )
        )
    return GenerationQualityGateReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=baseline.dataset_fingerprint,
        baseline_model_hash=baseline.model_hash,
        candidate_model_hash=candidate.model_hash,
        approved=all(value.passed for value in comparisons),
        samples=tuple(comparisons),
        untested_capabilities=_untested_capabilities(baseline),
    )


def load_generation_report(path: Path) -> GenerationEvaluationReport:
    candidate = path.expanduser()
    info = candidate.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not 0 < info.st_size <= MAX_EVALUATION_REPORT_BYTES
    ):
        raise ValueError("generation report must be a bounded owned regular file")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation report must be a JSON object")
    return GenerationEvaluationReport.from_dict(payload)


def generation_token_fingerprint(token_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256(b"vllm-apple-generation-tokens-v1\0")
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _untested_capabilities(report: GenerationEvaluationReport) -> tuple[str, ...]:
    tested = {
        "long_context" if sample.domain == "long-context" else sample.domain
        for sample in report.samples
    }
    capabilities = ("long_context", "code", "mathematics", "safety_alignment")
    return tuple(value for value in capabilities if value not in tested)


def _bounded_fraction(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return result
