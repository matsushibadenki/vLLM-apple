import unittest

from vllm_apple.phase_probe import PhaseMeasurement, PhaseProbeConfig, StreamProbeResult
from vllm_apple.quality_smoke import run_serving_quality_smoke


class QualitySmokeTests(unittest.TestCase):
    def test_multilingual_checks_store_no_generated_text(self) -> None:
        expected_values = []

        def measure(config, *, expected_text=None, expected_match_mode="contains"):
            expected_values.append((config.prompt, expected_text, expected_match_mode))
            return StreamProbeResult(
                PhaseMeasurement(1, 2, 3, 4, 1, 1024),
                True,
                1024,
            )

        report = run_serving_quality_smoke(
            PhaseProbeConfig(
                base_url="http://127.0.0.1:8001",
                model="model",
                hardware_fingerprint="hardware",
            ),
            measure=measure,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(set(report["checks"]), {"english", "japanese", "simplified_chinese"})
        self.assertFalse(report["stores_generated_text"])
        self.assertEqual(len(expected_values), 3)
        self.assertTrue(all(value[2] == "exact" for value in expected_values))

    def test_any_language_failure_fails_closed(self) -> None:
        calls = 0

        def measure(config, *, expected_text=None, expected_match_mode="contains"):
            nonlocal calls
            calls += 1
            return StreamProbeResult(
                PhaseMeasurement(1, 2, 3, 4, 1, 1024),
                calls != 2,
                1024,
            )

        report = run_serving_quality_smoke(
            PhaseProbeConfig(
                base_url="http://127.0.0.1:8001",
                model="model",
                hardware_fingerprint="hardware",
            ),
            measure=measure,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["japanese"])


if __name__ == "__main__":
    unittest.main()
