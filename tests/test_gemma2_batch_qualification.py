import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_gemma2_batch_mask.py"
SPEC = importlib.util.spec_from_file_location("gemma2_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class Gemma2QualificationTests(unittest.TestCase):
    def test_duration_validation(self):
        qualification._validate_duration(0, False)
        qualification._validate_duration(1800, True)
        for value in (-1, float("inf"), float("nan"), 28_801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qualification._validate_duration(value, False)
        with self.assertRaises(ValueError):
            qualification._validate_duration(1799.9, True)

    def test_window_aggregation_is_constant_shape_and_requires_all_results(self):
        summary = qualification._new_stability_summary(10, 100)
        first = {"requests": 2, "completed": 2, "failed": 0,
                 "quality_passed": 2, "slo_quality_passed": 2, "errors": []}
        last = {"requests": 1, "completed": 1, "failed": 0,
                "quality_passed": 1, "slo_quality_passed": 1, "errors": []}
        qualification._accumulate_window(summary, first)
        qualification._accumulate_window(summary, last)
        self.assertEqual(summary["requests"], 3)
        self.assertIs(summary["first_window"], first)
        self.assertIs(summary["last_window"], last)
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=True))
        summary.update(cancel_attempts=1, cancel_passed=1,
                       slow_consumer_attempts=1, slow_consumer_passed=1)
        self.assertTrue(qualification._stability_passed(
            summary, require_fault_checks=True))

    def test_failed_window_rejects_stability(self):
        summary = qualification._new_stability_summary(10, 100)
        qualification._accumulate_window(summary, {
            "requests": 2, "completed": 1, "failed": 1,
            "quality_passed": 1, "slo_quality_passed": 1,
            "errors": [{"code": "timeout"}],
        })
        self.assertEqual(summary["errors"], {"timeout": 1})
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False))


if __name__ == "__main__":
    unittest.main()
