"""Combined correctness and end-to-end promotion gate for numeric routes."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .workload_performance import (
    EndToEndPerformanceProfile,
    evaluate_end_to_end_promotion,
)


@dataclass(frozen=True, slots=True)
class NumericPromotionThresholds:
    maximum_absolute_error: float
    maximum_rmse: float
    maximum_quality_regression: float
    minimum_latency_improvement_ratio: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.maximum_absolute_error,
            self.maximum_rmse,
            self.maximum_quality_regression,
            self.minimum_latency_improvement_ratio,
        )
        if (any(not math.isfinite(value) or value < 0 for value in values)
                or self.minimum_latency_improvement_ratio >= 1):
            raise ValueError("invalid numeric promotion thresholds")


@dataclass(frozen=True, slots=True)
class NumericPromotionEvidence:
    scalar_maximum_absolute_error: float
    scalar_rmse: float
    operator_output_matches: bool
    baseline_quality_score: float
    candidate_quality_score: float
    baseline_performance: EndToEndPerformanceProfile
    candidate_performance: EndToEndPerformanceProfile

    def __post_init__(self) -> None:
        values = (
            self.scalar_maximum_absolute_error,
            self.scalar_rmse,
            self.baseline_quality_score,
            self.candidate_quality_score,
        )
        if (any(not math.isfinite(value) for value in values)
                or self.scalar_maximum_absolute_error < 0
                or self.scalar_rmse < 0
                or type(self.operator_output_matches) is not bool):
            raise ValueError("invalid numeric promotion evidence")


@dataclass(frozen=True, slots=True)
class NumericPromotionDecision:
    promoted: bool
    reason: str
    quality_regression: float
    latency_improvement_ratio: float


def evaluate_numeric_promotion(
    evidence: NumericPromotionEvidence,
    thresholds: NumericPromotionThresholds,
) -> NumericPromotionDecision:
    if not isinstance(evidence, NumericPromotionEvidence) or not isinstance(
        thresholds, NumericPromotionThresholds
    ):
        raise ValueError("invalid numeric promotion input")
    if evidence.scalar_maximum_absolute_error > thresholds.maximum_absolute_error:
        return NumericPromotionDecision(False, "scalar_absolute_error", 0, 0)
    if evidence.scalar_rmse > thresholds.maximum_rmse:
        return NumericPromotionDecision(False, "scalar_rmse", 0, 0)
    if not evidence.operator_output_matches:
        return NumericPromotionDecision(False, "operator_output_mismatch", 0, 0)
    quality_regression = max(
        0.0, evidence.baseline_quality_score - evidence.candidate_quality_score
    )
    if quality_regression > thresholds.maximum_quality_regression:
        return NumericPromotionDecision(False, "model_quality_regression", quality_regression, 0)
    performance = evaluate_end_to_end_promotion(
        evidence.baseline_performance,
        evidence.candidate_performance,
        minimum_improvement_ratio=thresholds.minimum_latency_improvement_ratio,
    )
    if not performance.promoted:
        return NumericPromotionDecision(
            False, f"performance_{performance.reason}", quality_regression,
            performance.latency_improvement_ratio,
        )
    return NumericPromotionDecision(
        True, "promoted", quality_regression, performance.latency_improvement_ratio
    )
