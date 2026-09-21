"""Optional bounded LoRA/SFT repair adapter and mandatory before/after gate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from .evaluation import (
    PerplexityEvaluationReport,
    QualityGateReport,
    compare_perplexity_reports,
)


class RepairMethod(str, Enum):
    LORA = "lora"
    SFT = "sft"


@dataclass(frozen=True, slots=True)
class RepairRequest:
    method: RepairMethod
    candidate_model_hash: str
    dataset_fingerprint: str
    maximum_samples: int
    epochs: int
    learning_rate: float
    lora_rank: int | None
    seed: int

    def __post_init__(self) -> None:
        if (not isinstance(self.method, RepairMethod)
                or any(len(value) != 64 or any(character not in "0123456789abcdef"
                                               for character in value)
                       for value in (self.candidate_model_hash, self.dataset_fingerprint))
                or type(self.maximum_samples) is not int
                or not 1 <= self.maximum_samples <= 1_000_000
                or type(self.epochs) is not int or not 1 <= self.epochs <= 100
                or not isinstance(self.learning_rate, (int, float))
                or isinstance(self.learning_rate, bool)
                or not math.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= 1
                or (self.method is RepairMethod.LORA
                    and (type(self.lora_rank) is not int or not 1 <= self.lora_rank <= 1024))
                or (self.method is RepairMethod.SFT and self.lora_rank is not None)
                or type(self.seed) is not int or not 0 <= self.seed < (1 << 63)):
            raise ValueError("invalid repair request")


@dataclass(frozen=True, slots=True)
class RepairArtifact:
    path: Path
    model_hash: str
    source_model_hash: str
    method: RepairMethod

    def __post_init__(self) -> None:
        if (not self.path.is_absolute() or not self.path.exists()
                or any(len(value) != 64 for value in (self.model_hash, self.source_model_hash))
                or not isinstance(self.method, RepairMethod)):
            raise ValueError("invalid repair artifact")


class RepairAdapter(Protocol):
    def repair(self, request: RepairRequest) -> RepairArtifact: ...


@dataclass(frozen=True, slots=True)
class RepairEvaluation:
    method: RepairMethod
    source_model_hash: str
    repaired_model_hash: str
    before_perplexity: float
    after_perplexity: float
    relative_change: float
    quality_gate: QualityGateReport
    approved: bool


def run_repair_and_evaluate(
    adapter: RepairAdapter,
    request: RepairRequest,
    before: PerplexityEvaluationReport,
    evaluate: Callable[[Path], PerplexityEvaluationReport],
    *,
    maximum_regression: float = 0.0,
) -> RepairEvaluation:
    if (not hasattr(adapter, "repair") or not isinstance(request, RepairRequest)
            or not isinstance(before, PerplexityEvaluationReport) or not callable(evaluate)):
        raise ValueError("invalid repair evaluation input")
    if (before.model_hash != request.candidate_model_hash
            or before.dataset_fingerprint != request.dataset_fingerprint):
        raise ValueError("repair request does not match before evaluation")
    artifact = adapter.repair(request)
    if (not isinstance(artifact, RepairArtifact)
            or artifact.source_model_hash != request.candidate_model_hash
            or artifact.method is not request.method):
        raise ValueError("repair adapter returned mismatched artifact")
    after = evaluate(artifact.path)
    if not isinstance(after, PerplexityEvaluationReport) or after.model_hash != artifact.model_hash:
        raise ValueError("repair evaluation does not match artifact")
    gate = compare_perplexity_reports(before, after, maximum_regression)
    relative_change = (after.perplexity - before.perplexity) / before.perplexity
    return RepairEvaluation(
        request.method, before.model_hash, after.model_hash, before.perplexity,
        after.perplexity, relative_change, gate, gate.approved,
    )
