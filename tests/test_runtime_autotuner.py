import unittest

from vllm_apple.runtime_autotuner import (
    RuntimeTuningConfiguration,
    RuntimeTuningMeasurement,
    tune_runtime_configuration,
)


class RuntimeAutotunerTests(unittest.TestCase):
    def test_selects_fastest_correct_candidate_within_memory_ceiling(self):
        digest = "d" * 64
        slow = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(1, 64, 16, 256, "safe"),
            (100, 110, 90), (50, 55, 45), 1000, digest,
        )
        fast = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(2, 128, 16, 512, "fast"),
            (50, 55, 45), (25, 30, 20), 2000, digest,
        )
        wrong = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(4, 256, 32, 1024, "wrong"),
            (1, 1, 1), (1, 1, 1), 1000, "e" * 64,
        )
        report = tune_runtime_configuration(
            "a" * 24, "b" * 64, (slow, fast, wrong),
            baseline_output_digest=digest, maximum_peak_memory_bytes=2500,
        )
        self.assertEqual(report.winner.kernel, "fast")
        self.assertEqual(report.qualified_candidates, 2)
        self.assertEqual(report.rejected_candidates, 1)

    def test_all_rejected_fails_closed(self):
        item = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(1, 64, 16, 256, "safe"),
            (1, 1, 1), (1, 1, 1), 100, "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "no runtime"):
            tune_runtime_configuration(
                "a" * 24, "b" * 64, (item,), baseline_output_digest="b" * 64,
                maximum_peak_memory_bytes=100,
            )


if __name__ == "__main__":
    unittest.main()
