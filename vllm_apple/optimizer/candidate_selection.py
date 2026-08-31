from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .types import OPTIMIZER_SCHEMA_VERSION


MAX_SELECTION_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    quality_approved: bool
    task_score: float
    artifact_bytes: int
    throughput_tokens_per_second: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        numeric = (self.task_score, self.throughput_tokens_per_second)
        if (
            not self.candidate_id
            or len(self.candidate_id) > 128
            or any(not math.isfinite(value) for value in numeric)
            or not 0 <= self.task_score <= 1
            or self.artifact_bytes <= 0
            or self.throughput_tokens_per_second <= 0
            or self.peak_rss_bytes <= 0
        ):
            raise ValueError("invalid optimization candidate evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "quality_approved": self.quality_approved,
            "task_score": self.task_score,
            "artifact_bytes": self.artifact_bytes,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    rank: int | None
    evidence: CandidateEvidence
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.eligible:
            if self.rank is None or self.rank <= 0 or self.rejection_reasons:
                raise ValueError("eligible candidate ranking is inconsistent")
        elif self.rank is not None or not self.rejection_reasons:
            raise ValueError("rejected candidate ranking is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            **self.evidence.to_dict(),
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class CandidateComparisonReport:
    created_at: str
    winner_candidate_id: str
    ranking_policy: tuple[str, ...]
    candidates: tuple[CandidateRanking, ...]

    def __post_init__(self) -> None:
        eligible = tuple(value for value in self.candidates if value.eligible)
        if (
            not self.created_at
            or not self.winner_candidate_id
            or not self.ranking_policy
            or not self.candidates
            or len(self.candidates) > MAX_SELECTION_CANDIDATES
            or not eligible
            or eligible[0].evidence.candidate_id != self.winner_candidate_id
            or tuple(value.rank for value in eligible) != tuple(range(1, len(eligible) + 1))
        ):
            raise ValueError("invalid candidate comparison report")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPTIMIZER_SCHEMA_VERSION,
            "created_at": self.created_at,
            "winner_candidate_id": self.winner_candidate_id,
            "ranking_policy": list(self.ranking_policy),
            "candidates": [value.to_dict() for value in self.candidates],
        }


def compare_candidates(
    candidates: tuple[CandidateEvidence, ...],
    *,
    created_at: str | None = None,
) -> CandidateComparisonReport:
    if not candidates or len(candidates) > MAX_SELECTION_CANDIDATES:
        raise ValueError("candidate comparison must be bounded and non-empty")
    identities = {value.candidate_id for value in candidates}
    if len(identities) != len(candidates):
        raise ValueError("candidate identifiers must be unique")

    approved = sorted(
        (value for value in candidates if value.quality_approved),
        key=lambda value: (
            -value.task_score,
            value.artifact_bytes,
            -value.throughput_tokens_per_second,
            value.peak_rss_bytes,
            value.candidate_id,
        ),
    )
    if not approved:
        raise ValueError("no candidate passed the quality gate")

    rankings = [
        CandidateRanking(rank=index, evidence=value, eligible=True, rejection_reasons=())
        for index, value in enumerate(approved, start=1)
    ]
    rankings.extend(
        CandidateRanking(
            rank=None,
            evidence=value,
            eligible=False,
            rejection_reasons=("quality_gate_failed",),
        )
        for value in sorted(
            (value for value in candidates if not value.quality_approved),
            key=lambda value: value.candidate_id,
        )
    )
    return CandidateComparisonReport(
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        winner_candidate_id=approved[0].candidate_id,
        ranking_policy=(
            "quality_gate_required",
            "task_score_descending",
            "artifact_bytes_ascending",
            "throughput_tokens_per_second_descending",
            "peak_rss_bytes_ascending",
            "candidate_id_ascending",
        ),
        candidates=tuple(rankings),
    )
