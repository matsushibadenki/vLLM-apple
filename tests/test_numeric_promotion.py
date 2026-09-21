import unittest

from vllm_apple.numeric_promotion import (
    NumericPromotionEvidence,
    NumericPromotionThresholds,
    evaluate_numeric_promotion,
)
from vllm_apple.workload_performance import (
    EndToEndPerformanceSample,
    EndToEndPhase,
    build_end_to_end_performance_profile,
)


def profile(backend, latency, *, memory=100, energy=1.0, digest="a"):
    samples = tuple(
        EndToEndPerformanceSample(latency, 10, memory, digest * 64, energy)
        for _ in range(3)
    )
    return build_end_to_end_performance_profile(
        hardware_fingerprint="m4", model_id="model", backend=backend,
        phase=EndToEndPhase.DECODE, workload_identity="shape-1", samples=samples,
    )


class NumericPromotionTests(unittest.TestCase):
    def evidence(self, **changes):
        values = dict(
            scalar_maximum_absolute_error=0.01,
            scalar_rmse=0.005,
            operator_output_matches=True,
            baseline_quality_score=0.90,
            candidate_quality_score=0.899,
            baseline_performance=profile("fp16", 100),
            candidate_performance=profile("int8", 80, memory=90, energy=0.9),
        )
        values.update(changes)
        return NumericPromotionEvidence(**values)

    def thresholds(self):
        return NumericPromotionThresholds(0.02, 0.01, 0.005)

    def test_all_correctness_and_performance_gates_promote(self):
        decision = evaluate_numeric_promotion(self.evidence(), self.thresholds())
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.reason, "promoted")

    def test_each_correctness_layer_fails_closed(self):
        cases = (
            (dict(scalar_maximum_absolute_error=0.03), "scalar_absolute_error"),
            (dict(scalar_rmse=0.02), "scalar_rmse"),
            (dict(operator_output_matches=False), "operator_output_mismatch"),
            (dict(candidate_quality_score=0.8), "model_quality_regression"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    evaluate_numeric_promotion(
                        self.evidence(**changes), self.thresholds()
                    ).reason,
                    reason,
                )

    def test_memory_energy_and_latency_regressions_do_not_promote(self):
        candidates = (
            profile("int8", 80, memory=101, energy=0.9),
            profile("int8", 80, memory=90, energy=1.1),
            profile("int8", 99, memory=90, energy=0.9),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate.profile_id):
                decision = evaluate_numeric_promotion(
                    self.evidence(candidate_performance=candidate), self.thresholds()
                )
                self.assertFalse(decision.promoted)
                self.assertTrue(decision.reason.startswith("performance_"))


if __name__ == "__main__":
    unittest.main()
