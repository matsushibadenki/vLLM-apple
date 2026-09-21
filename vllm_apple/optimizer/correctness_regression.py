"""Fail-closed multi-run correctness regression gates for real models."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from .generation_evaluation import (
    GenerationEvaluationReport,
    compare_generation_reports,
)
from .types import OPTIMIZER_SCHEMA_VERSION

MAX_REGRESSION_RUNS = 16
REQUIRED_LANGUAGES = frozenset({"en", "ja", "zh-Hans"})


@dataclass(frozen=True, slots=True)
class CorrectnessRegressionRun:
    run_index: int
    candidate_model_hash: str
    elapsed_milliseconds: int
    peak_rss_bytes: int
    minimum_token_agreement: float
    maximum_expectation_regression: float
    deterministic_with_prior_runs: bool
    passed: bool

    def __post_init__(self) -> None:
        fractions = (self.minimum_token_agreement, self.maximum_expectation_regression)
        if (
            not 1 <= self.run_index <= MAX_REGRESSION_RUNS
            or len(self.candidate_model_hash) != 64
            or self.elapsed_milliseconds < 0
            or self.peak_rss_bytes <= 0
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions)
            or not isinstance(self.passed, bool)
        ):
            raise ValueError("invalid correctness regression run")


@dataclass(frozen=True, slots=True)
class RealModelCorrectnessRegressionReport:
    suite_version: int
    baseline_model_hash: str
    candidate_model_hash: str
    dataset_fingerprint: str
    sample_count: int
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    required_runs: int
    runs: tuple[CorrectnessRegressionRun, ...]
    maximum_latency_regression: float
    maximum_rss_regression: float
    latency_regression: float
    rss_regression: float
    approved: bool
    report_id: str

    def __post_init__(self) -> None:
        if (
            self.suite_version != 1
            or len(self.baseline_model_hash) != 64
            or len(self.candidate_model_hash) != 64
            or len(self.dataset_fingerprint) != 64
            or not 1 <= self.sample_count <= 64
            or not 2 <= self.required_runs <= MAX_REGRESSION_RUNS
            or len(self.runs) != self.required_runs
            or tuple(run.run_index for run in self.runs) != tuple(range(1, self.required_runs + 1))
            or not REQUIRED_LANGUAGES.issubset(self.languages)
            or not self.domains
            or len(self.report_id) != 64
        ):
            raise ValueError("invalid real-model correctness regression report")
        measurements = (
            self.maximum_latency_regression,
            self.maximum_rss_regression,
            self.latency_regression,
            self.rss_regression,
        )
        if any(not math.isfinite(value) or value < -1 for value in measurements):
            raise ValueError("invalid real-model regression measurement")
        expected = (
            all(run.passed for run in self.runs)
            and self.latency_regression <= self.maximum_latency_regression
            and self.rss_regression <= self.maximum_rss_regression
        )
        if self.approved != expected or self.report_id != _report_id(asdict(self)):
            raise ValueError("real-model regression decision or ID does not match")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = OPTIMIZER_SCHEMA_VERSION
        payload["languages"] = list(self.languages)
        payload["domains"] = list(self.domains)
        payload["runs"] = [asdict(run) for run in self.runs]
        return payload


def evaluate_real_model_regression(
    baseline: GenerationEvaluationReport,
    candidates: tuple[GenerationEvaluationReport, ...],
    *,
    minimum_token_agreement: float,
    maximum_expectation_regression: float,
    maximum_latency_regression: float,
    maximum_rss_regression: float,
) -> RealModelCorrectnessRegressionReport:
    if not 2 <= len(candidates) <= MAX_REGRESSION_RUNS:
        raise ValueError("correctness regression requires 2-16 candidate runs")
    if baseline.elapsed_milliseconds <= 0:
        raise ValueError("correctness regression baseline latency must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (
        maximum_latency_regression, maximum_rss_regression
    )):
        raise ValueError("performance regression limits must be finite and nonnegative")
    languages = tuple(sorted({sample.language for sample in baseline.samples}))
    domains = tuple(sorted({sample.domain for sample in baseline.samples}))
    if not REQUIRED_LANGUAGES.issubset(languages):
        raise ValueError("correctness regression requires en, ja, and zh-Hans samples")
    model_hashes = {candidate.model_hash for candidate in candidates}
    if len(model_hashes) != 1:
        raise ValueError("candidate runs must use one model hash")
    reference_fingerprints: tuple[str, ...] | None = None
    runs: list[CorrectnessRegressionRun] = []
    for index, candidate in enumerate(candidates, 1):
        gate = compare_generation_reports(
            baseline,
            candidate,
            minimum_token_agreement=minimum_token_agreement,
            maximum_expectation_regression=maximum_expectation_regression,
        )
        fingerprints = tuple(
            sample.output_fingerprint for sample in sorted(candidate.samples, key=lambda item: item.sample_id)
        )
        deterministic = reference_fingerprints in {None, fingerprints}
        reference_fingerprints = fingerprints if reference_fingerprints is None else reference_fingerprints
        runs.append(CorrectnessRegressionRun(
            run_index=index,
            candidate_model_hash=candidate.model_hash,
            elapsed_milliseconds=candidate.elapsed_milliseconds,
            peak_rss_bytes=candidate.peak_rss_bytes,
            minimum_token_agreement=min(sample.token_agreement for sample in gate.samples),
            maximum_expectation_regression=max(sample.expectation_regression for sample in gate.samples),
            deterministic_with_prior_runs=deterministic,
            passed=gate.approved and deterministic,
        ))
    latency = max(candidate.elapsed_milliseconds for candidate in candidates)
    rss = max(candidate.peak_rss_bytes for candidate in candidates)
    latency_regression = (latency - baseline.elapsed_milliseconds) / baseline.elapsed_milliseconds
    rss_regression = (rss - baseline.peak_rss_bytes) / baseline.peak_rss_bytes
    values = dict(
        suite_version=1,
        baseline_model_hash=baseline.model_hash,
        candidate_model_hash=next(iter(model_hashes)),
        dataset_fingerprint=baseline.dataset_fingerprint,
        sample_count=len(baseline.samples),
        languages=languages,
        domains=domains,
        required_runs=len(candidates),
        runs=tuple(runs),
        maximum_latency_regression=float(maximum_latency_regression),
        maximum_rss_regression=float(maximum_rss_regression),
        latency_regression=latency_regression,
        rss_regression=rss_regression,
    )
    approved = (
        all(run.passed for run in runs)
        and latency_regression <= maximum_latency_regression
        and rss_regression <= maximum_rss_regression
    )
    identity = dict(values, approved=approved)
    return RealModelCorrectnessRegressionReport(
        **identity, report_id=_report_id(identity)
    )


def _report_id(payload: dict[str, object]) -> str:
    payload = dict(payload)
    payload.pop("report_id", None)
    runs = payload.get("runs")
    if isinstance(runs, (tuple, list)):
        payload["runs"] = [asdict(run) if isinstance(run, CorrectnessRegressionRun) else run for run in runs]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"vllm-apple-real-model-regression-v1\0" + encoded).hexdigest()
