"""Deterministic structural optimization candidate generation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .importance_analysis import ImportanceKey, ImportanceKind
from .structural_optimization import FunctionalSimilarity, StructuralComponentKind

MAX_STRUCTURAL_CANDIDATES = 4096


class StructuralCandidateKind(str, Enum):
    LAYER_BYPASS = "layer_bypass"
    HEAD_MERGE = "head_merge"
    LAYER_MERGE = "layer_merge"


@dataclass(frozen=True, slots=True)
class SimilarityObservation:
    left: ImportanceKey
    right: ImportanceKey
    similarity: FunctionalSimilarity

    def __post_init__(self) -> None:
        expected = {
            ImportanceKind.LAYER: StructuralComponentKind.LAYER,
            ImportanceKind.ATTENTION_HEAD: StructuralComponentKind.ATTENTION_HEAD,
        }
        if (not isinstance(self.left, ImportanceKey)
                or not isinstance(self.right, ImportanceKey)
                or self.left == self.right or self.left.kind is not self.right.kind
                or self.left.kind not in expected
                or not isinstance(self.similarity, FunctionalSimilarity)
                or self.similarity.kind is not expected[self.left.kind]):
            raise ValueError("invalid structural similarity observation")


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    kind: StructuralCandidateKind
    components: tuple[ImportanceKey, ...]
    score: float
    evidence_id: str
    candidate_id: str


def generate_structural_candidates(
    importance_report: dict[str, object],
    similarities: Sequence[SimilarityObservation],
    *,
    bypass_importance_ceiling: float = .01,
    merge_similarity_floor: float = .99,
    maximum_candidates: int = 256,
) -> tuple[StructuralCandidate, ...]:
    if (not _unit_interval(bypass_importance_ceiling)
            or not _unit_interval(merge_similarity_floor)
            or type(maximum_candidates) is not int
            or not 1 <= maximum_candidates <= MAX_STRUCTURAL_CANDIDATES
            or len(similarities) > MAX_STRUCTURAL_CANDIDATES):
        raise ValueError("invalid structural candidate configuration")
    components = importance_report.get("components")
    report_id = importance_report.get("report_id")
    if (not isinstance(components, (tuple, list)) or not isinstance(report_id, str)
            or len(report_id) != 64):
        raise ValueError("invalid importance report")
    candidates: list[StructuralCandidate] = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("invalid importance component")
        try:
            kind = ImportanceKind(component["kind"])
            layer = component["layer"]
            index = component["index"]
            score = component["normalized_score"]
            key = ImportanceKey(kind, layer, index)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid importance component") from error
        if not _unit_interval(score):
            raise ValueError("invalid normalized importance score")
        if kind is ImportanceKind.LAYER and score <= bypass_importance_ceiling:
            candidates.append(_candidate(
                StructuralCandidateKind.LAYER_BYPASS, (key,), 1.0 - score, report_id
            ))
    for observation in similarities:
        if not isinstance(observation, SimilarityObservation):
            raise ValueError("invalid similarity observation")
        similarity = observation.similarity.cosine_similarity
        if similarity < merge_similarity_floor:
            continue
        if observation.left.kind is ImportanceKind.ATTENTION_HEAD:
            if observation.left.layer != observation.right.layer:
                continue
            candidate_kind = StructuralCandidateKind.HEAD_MERGE
        else:
            if abs(observation.left.layer - observation.right.layer) != 1:
                continue
            candidate_kind = StructuralCandidateKind.LAYER_MERGE
        keys = tuple(sorted((observation.left, observation.right)))
        candidates.append(_candidate(
            candidate_kind, keys, similarity, observation.similarity.comparison_id
        ))
    candidates.sort(key=lambda item: (-item.score, item.kind.value, item.candidate_id))
    return tuple(candidates[:maximum_candidates])


def _candidate(
    kind: StructuralCandidateKind,
    components: tuple[ImportanceKey, ...],
    score: float,
    evidence_id: str,
) -> StructuralCandidate:
    canonical = json.dumps({
        "kind": kind.value,
        "components": [(key.kind.value, key.layer, key.index) for key in components],
        "score": score,
        "evidence_id": evidence_id,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return StructuralCandidate(
        kind, components, score, evidence_id, hashlib.sha256(canonical).hexdigest()
    )


def _unit_interval(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)
