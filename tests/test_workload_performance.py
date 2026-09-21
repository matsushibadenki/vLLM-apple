import unittest

from vllm_apple.workload_performance import (
    EndToEndPerformanceSample,
    EndToEndPhase,
    build_end_to_end_performance_profile,
    evaluate_end_to_end_promotion,
)


DIGEST = "a" * 64


def profile(backend, latency, *, peak=100, digest=DIGEST, phase=EndToEndPhase.DECODE):
    return build_end_to_end_performance_profile(
        hardware_fingerprint="hardware",
        model_id="model",
        backend=backend,
        phase=phase,
        workload_identity="shape-batch-context",
        samples=tuple(
            EndToEndPerformanceSample(latency + index, 10, peak, digest, 1.0)
            for index in range(3)
        ),
    )


class WorkloadPerformanceTests(unittest.TestCase):
    def test_all_required_phases_are_explicit(self):
        self.assertEqual(
            {phase.value for phase in EndToEndPhase},
            {"prefill", "decode", "vision_encoder", "audio_encoder", "sampling", "draft", "verify"},
        )

    def test_profile_is_bounded_deterministic_and_exposes_throughput(self):
        first = profile("cpu", 100)
        second = profile("cpu", 100)
        self.assertEqual(first.profile_id, second.profile_id)
        self.assertEqual(first.sample_count, 3)
        self.assertGreater(first.throughput_units_per_second, 0)
        self.assertEqual(first.to_dict()["phase"], "decode")

    def test_promotion_requires_correctness_speed_memory_and_energy(self):
        baseline = profile("cpu", 100, peak=100)
        candidate = profile("ane", 80, peak=90)
        self.assertTrue(evaluate_end_to_end_promotion(baseline, candidate).promoted)
        self.assertEqual(
            evaluate_end_to_end_promotion(baseline, profile("ane", 80, peak=101)).reason,
            "peak_memory_regression",
        )
        self.assertEqual(
            evaluate_end_to_end_promotion(
                baseline, profile("ane", 80, digest="b" * 64)
            ).reason,
            "output_mismatch",
        )

    def test_identity_mismatch_is_not_compared(self):
        baseline = profile("cpu", 100)
        candidate = profile("ane", 80, phase=EndToEndPhase.PREFILL)
        self.assertEqual(
            evaluate_end_to_end_promotion(baseline, candidate).reason,
            "identity_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
