"""Scope-bound E2E selection shared by kernel, quantization and speculative trials.

Evidence is supplied by a collector. Unique process IDs are necessary provenance,
not proof that measurement acquisition was independent or interference-free.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SelectionPolicy:
    scope_sha256: str
    quality_slices: tuple[str, ...]
    maximum_memory_bytes: int
    minimum_improvement: float = 0.05
    maximum_p95_regression: float = 0.05
    maximum_spread: float = 0.05

    def __post_init__(self) -> None:
        if (len(self.scope_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.scope_sha256)
                or not 1 <= len(self.quality_slices) <= 64
                or len(set(self.quality_slices)) != len(self.quality_slices)
                or any(not s or len(s) > 128 for s in self.quality_slices)
                or type(self.maximum_memory_bytes) is not int
                or self.maximum_memory_bytes <= 0
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v < 1 for v in (
                           self.minimum_improvement, self.maximum_p95_regression,
                           self.maximum_spread))
                or self.minimum_improvement < 0.05
                or self.maximum_p95_regression > 0.05
                or self.maximum_spread > 0.05):
            raise ValueError("invalid P3 selection policy")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class Trial:
    acquisition_id: str
    scope_sha256: str
    policy_sha256: str
    e2e_seconds: float
    ttft_p95_seconds: float
    tpot_p95_seconds: float
    peak_memory_bytes: int
    quality: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (not self.acquisition_id or len(self.acquisition_id) > 256
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (self.e2e_seconds, self.ttft_p95_seconds,
                                 self.tpot_p95_seconds))
                or type(self.peak_memory_bytes) is not int or self.peak_memory_bytes <= 0
                or not 1 <= len(self.quality) <= 64
                or len({name for name, _ in self.quality}) != len(self.quality)
                or any(not name or type(passed) is not bool for name, passed in self.quality)):
            raise ValueError("invalid P3 trial")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    trials: tuple[Trial, ...]

    def __post_init__(self) -> None:
        if (not self.candidate_id or len(self.candidate_id) > 128
                or self.kind not in ("baseline", "kernel", "quantization", "speculative")
                or not 1 <= len(self.trials) <= 64):
            raise ValueError("invalid P3 candidate")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_candidate(
    policy: SelectionPolicy,
    baseline: Candidate,
    candidates: tuple[Candidate, ...],
    *,
    independent_acquisition_verified: bool = False,
    prerequisites_verified: bool = False,
) -> dict:
    """Retain baseline on any missing gate; never apply a configuration automatically."""
    if (baseline.kind != "baseline" or not 1 <= len(candidates) <= 64
            or len({c.candidate_id for c in (baseline, *candidates)}) != len(candidates) + 1
            or any(c.kind == "baseline" for c in candidates)
            or type(independent_acquisition_verified) is not bool
            or type(prerequisites_verified) is not bool):
        raise ValueError("invalid P3 comparison")
    all_trials = [t for c in (baseline, *candidates) for t in c.trials]
    acquisition_ids = [t.acquisition_id for t in all_trials]
    duplicate_acquisition = len(set(acquisition_ids)) != len(acquisition_ids)

    def failures(candidate: Candidate) -> list[str]:
        reasons = []
        if len(candidate.trials) < 3:
            reasons.append("insufficient_independent_runs")
        if duplicate_acquisition or not independent_acquisition_verified:
            reasons.append("independent_acquisition_unverified")
        if any(t.scope_sha256 != policy.scope_sha256 or t.policy_sha256 != policy.digest
               for t in candidate.trials):
            reasons.append("scope_or_policy_mismatch")
        if any(set(dict(t.quality)) != set(policy.quality_slices)
               or not all(dict(t.quality).values()) for t in candidate.trials):
            reasons.append("quality_slice_failed_or_missing")
        if any(t.peak_memory_bytes > policy.maximum_memory_bytes for t in candidate.trials):
            reasons.append("memory_budget_exceeded")
        times = [t.e2e_seconds for t in candidate.trials]
        if (max(times) - min(times)) / statistics.median(times) > policy.maximum_spread:
            reasons.append("measurement_spread_exceeded")
        return reasons

    baseline_failures = failures(baseline)
    base_times = [t.e2e_seconds for t in baseline.trials]
    base_median = statistics.median(base_times)
    ranking = []
    for candidate in candidates:
        reasons = failures(candidate)
        if baseline_failures:
            reasons.append("baseline_unqualified")
        times = [t.e2e_seconds for t in candidate.trials]
        median = statistics.median(times)
        improvement = 1 - median / base_median
        if improvement + 1e-12 < policy.minimum_improvement:
            reasons.append("e2e_improvement_below_threshold")
        # Require separation of entire run ranges, in addition to median and spread gates.
        if max(times) >= min(base_times):
            reasons.append("improvement_within_measurement_noise")
        for field in ("ttft_p95_seconds", "tpot_p95_seconds"):
            if max(getattr(t, field) for t in candidate.trials) > (
                    min(getattr(t, field) for t in baseline.trials)
                    * (1 + policy.maximum_p95_regression) + 1e-12):
                reasons.append(field + "_regressed")
        ranking.append(dict(candidate_id=candidate.candidate_id, kind=candidate.kind,
                            eligible=not reasons, rejection_reasons=reasons,
                            median_e2e_seconds=median, relative_improvement=improvement))
    eligible = sorted((r for r in ranking if r["eligible"]),
                      key=lambda r: (r["median_e2e_seconds"], r["candidate_id"]))
    payload = dict(schema_version=1, policy=asdict(policy), policy_sha256=policy.digest,
                   evidence_sha256=_digest([asdict(c) for c in (baseline, *candidates)]),
                   selected_candidate_id=eligible[0]["candidate_id"] if eligible
                   else baseline.candidate_id, baseline_retained=not eligible,
                   baseline_rejection_reasons=baseline_failures, candidates=ranking,
                   independent_acquisition_verified=independent_acquisition_verified,
                   standard_adoption_eligible=bool(eligible) and prerequisites_verified,
                   automatic_application=False)
    payload["report_id"] = _digest(payload)
    return payload
