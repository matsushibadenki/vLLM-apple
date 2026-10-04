import unittest

from vllm_apple.runtime_autotuner import (
    RuntimeTuningConfiguration,
    RuntimeTuningMeasurement,
    confirm_runtime_tuning,
    tune_runtime_configuration,
)


class RuntimeAutotunerTests(unittest.TestCase):
    def test_confirmation_binds_policy_and_separate_rounds(self):
        safe = RuntimeTuningMeasurement(RuntimeTuningConfiguration(1, 64, 16, 256, 'safe'),
                                        (100, 100, 100), (100, 100, 100), 100, 'd'*64)
        fast = RuntimeTuningMeasurement(RuntimeTuningConfiguration(1, 64, 16, 256, 'fast'),
                                        (80, 80, 80), (80, 80, 80), 100, 'd'*64)
        def run(round_id, candidates, margin=.02):
            return tune_runtime_configuration('a'*24, round_id*64, candidates,
                baseline_output_digest='d'*64, maximum_peak_memory_bytes=100,
                baseline_configuration=safe.configuration, minimum_relative_improvement=margin)
        proposal = run('b', (safe, fast))
        confirmed = confirm_runtime_tuning(proposal, run('c', (safe, fast)))
        self.assertTrue(confirmed['confirmed'])
        self.assertFalse(confirmed['automatic_application'])
        self.assertFalse(confirm_runtime_tuning(proposal, run('c', (safe,)))['confirmed'])
        with self.assertRaises(ValueError):
            confirm_runtime_tuning(proposal, proposal)
        with self.assertRaises(ValueError):
            confirm_runtime_tuning(proposal, run('c', (safe, fast), margin=.1))

    def test_rejects_noisy_fast_candidate_and_retains_baseline_for_small_gain(self):
        def measurement(kernel, samples):
            return RuntimeTuningMeasurement(
                RuntimeTuningConfiguration(1, 64, 16, 256, kernel),
                samples, samples, 100, "d" * 64,
            )
        baseline = measurement("safe", (100, 100, 100))
        small_gain = measurement("small", (99, 99, 99))
        noisy = measurement("noisy", (1, 50, 90))
        result = tune_runtime_configuration(
            "a" * 24, "b" * 64, (baseline, small_gain, noisy),
            baseline_output_digest="d" * 64, maximum_peak_memory_bytes=100,
            maximum_relative_spread=.05, baseline_configuration=baseline.configuration,
        )
        self.assertEqual(result.winner, baseline.configuration)
        self.assertTrue(result.baseline_retained)
        self.assertEqual(result.rejected_candidates, 1)
        with self.assertRaisesRegex(ValueError, "baseline configuration"):
            tune_runtime_configuration(
                "a" * 24, "b" * 64, (baseline, noisy),
                baseline_output_digest="d" * 64, maximum_peak_memory_bytes=100,
                maximum_relative_spread=.05, baseline_configuration=noisy.configuration,
            )

    def test_report_identity_binds_sample_evidence(self):
        configuration = RuntimeTuningConfiguration(1, 64, 16, 256, "safe")
        reports = [tune_runtime_configuration(
            "a" * 24, "b" * 64,
            (RuntimeTuningMeasurement(configuration, samples, samples, 100, "d" * 64),),
            baseline_output_digest="d" * 64, maximum_peak_memory_bytes=100,
        ) for samples in ((99, 100, 101), (98, 100, 102))]
        self.assertNotEqual(reports[0].report_id, reports[1].report_id)

    def test_stable_large_gain_is_selected_and_invalid_policy_rejected(self):
        baseline = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(1, 64, 16, 256, "safe"),
            (100, 100, 100), (100, 100, 100), 100, "d" * 64,
        )
        fast = RuntimeTuningMeasurement(
            RuntimeTuningConfiguration(1, 64, 16, 256, "fast"),
            (80, 80, 80), (80, 80, 80), 100, "d" * 64,
        )
        kwargs = dict(baseline_output_digest="d" * 64, maximum_peak_memory_bytes=100,
                      baseline_configuration=baseline.configuration)
        result = tune_runtime_configuration(
            "a" * 24, "b" * 64, (baseline, fast), maximum_relative_spread=.05, **kwargs,
        )
        self.assertEqual(result.winner, fast.configuration)
        self.assertFalse(result.baseline_retained)
        for bad in (True, float("nan"), -1, 2):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "stability policy"):
                tune_runtime_configuration(
                    "a" * 24, "b" * 64, (baseline,), maximum_relative_spread=bad, **kwargs,
                )

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
