import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_gemma2_batch_mask.py"
SPEC = importlib.util.spec_from_file_location("gemma2_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class Gemma2QualificationTests(unittest.TestCase):
    def test_resource_growth_and_undrained_registry_fail(self):
        samples = [dict(pid=1, threads=4, open_fds=8, allocator_active_bytes=100,
                        allocator_cache_bytes=100, registry_active=0, registry_queued=0)
                   for _ in range(8)]
        self.assertTrue(qualification._resource_plateau(samples)['plateau_observed'])
        samples[-1]['open_fds'] = 20
        self.assertFalse(qualification._resource_plateau(samples)['plateau_observed'])
        samples[-1]['open_fds'] = 8
        samples[-1]['registry_queued'] = 1
        self.assertFalse(qualification._resource_plateau(samples)['plateau_observed'])
        self.assertFalse(qualification._resource_plateau(samples[:2])['plateau_observed'])
    def test_duration_validation(self):
        qualification._validate_duration(0, False)
        qualification._validate_duration(1800, True)
        qualification._validate_duration(28_800, False, True)
        for value in (-1, float("inf"), float("nan"), 28_801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qualification._validate_duration(value, False)
        with self.assertRaises(ValueError):
            qualification._validate_duration(1799.9, True)
        with self.assertRaises(ValueError):
            qualification._validate_duration(28_799.9, False, True)

    def test_rss_trend_requires_recent_plateau(self):
        plateau = qualification._rss_trend([
            {"elapsed_seconds": index * 3600, "rss_bytes": 100_000_000 + index * 1024}
            for index in range(8)
        ])
        self.assertTrue(plateau["plateau_observed"])
        growing = qualification._rss_trend([
            {"elapsed_seconds": index * 3600,
             "rss_bytes": 100_000_000 + index * 32 * 1024 * 1024}
            for index in range(8)
        ])
        self.assertFalse(growing["plateau_observed"])
        restarted = qualification._rss_trend([
            {"elapsed_seconds": 0, "rss_bytes": 900_000_000, "pid": 1},
            {"elapsed_seconds": 1, "rss_bytes": 950_000_000, "pid": 1},
            {"elapsed_seconds": 2, "rss_bytes": 100_000_000, "pid": 2},
            {"elapsed_seconds": 3, "rss_bytes": 101_000_000, "pid": 2},
        ])
        self.assertFalse(restarted["plateau_observed"])
        self.assertEqual(restarted["current_pid_sample_count"], 2)
        insufficient = qualification._rss_trend([
            {"elapsed_seconds": 0, "rss_bytes": 1},
        ])
        self.assertIsNone(insufficient["slope_bytes_per_hour"])
        self.assertFalse(insufficient["plateau_observed"])

    def test_window_aggregation_is_constant_shape_and_requires_all_results(self):
        with patch.object(
            qualification, "detect_thermal_state",
            return_value=Mock(value="nominal")
        ):
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
        summary.update(
            queued_cancel_attempts=1, queued_cancel_passed=1,
            timeout_attempts=1, timeout_passed=1,
            rss_trend={"plateau_observed": True},
            worker_crash_attempts=1, worker_crash_passed=1,
            all_epoch_resources_passed=True,
        )
        self.assertTrue(qualification._stability_passed(
            summary, require_fault_checks=True, require_long_window_checks=True))
        summary['all_epoch_resources_passed'] = False
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=True, require_long_window_checks=True))
        summary['all_epoch_resources_passed'] = True
        summary["rss_trend"] = {"plateau_observed": False}
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=True, require_long_window_checks=True))

    def test_failed_window_rejects_stability(self):
        with patch.object(
            qualification, "detect_thermal_state",
            return_value=Mock(value="nominal")
        ):
            summary = qualification._new_stability_summary(10, 100)
        qualification._accumulate_window(summary, {
            "requests": 2, "completed": 1, "failed": 1,
            "quality_passed": 1, "slo_quality_passed": 1,
            "errors": [{"code": "timeout"}],
        })
        self.assertEqual(summary["errors"], {"timeout": 1})
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False))

    def test_unsafe_thermal_sample_rejects_stability(self):
        with patch.object(
            qualification, "detect_thermal_state",
            return_value=Mock(value="nominal")
        ):
            summary = qualification._new_stability_summary(10, 100)
        qualification._accumulate_window(summary, {
            "requests": 1, "completed": 1, "failed": 0,
            "quality_passed": 1, "slo_quality_passed": 1, "errors": [],
        })
        summary["unsafe_thermal_samples"] = 1
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False))


if __name__ == "__main__":
    unittest.main()
