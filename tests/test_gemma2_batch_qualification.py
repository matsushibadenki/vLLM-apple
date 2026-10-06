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
    def test_awake_gate_rejects_suspend_unknown_and_transient_power_change(self):
        summary = dict(requests=1, completed=1, quality_passed=1, slo_quality_passed=1,
                       failed=0, sleep_wake_observations=0)
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False, require_awake_conditions=True))
        ac = dict(power_source='AC Power', power_mode='automatic')
        qualification._record_operating_conditions(summary, ac)
        self.assertTrue(qualification._stability_passed(
            summary, require_fault_checks=False, require_awake_conditions=True))
        summary['sleep_wake_observations'] = 1
        self.assertIn('suspend_gap_observed', qualification._awake_condition_rejections(summary))
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False, require_awake_conditions=True))
        summary['sleep_wake_observations'] = 0
        qualification._record_operating_conditions(summary, dict(ac, power_source='Battery Power'))
        qualification._record_operating_conditions(summary, ac)
        self.assertFalse(qualification._stability_passed(
            summary, require_fault_checks=False, require_awake_conditions=True))
        self.assertTrue(summary['operating_conditions']['changed'])

    def test_unknown_observation_does_not_become_known_later(self):
        summary = {}
        qualification._record_operating_conditions(summary, dict(power_source='unknown'))
        qualification._record_operating_conditions(summary, dict(
            power_source='AC Power', power_mode='automatic'))
        self.assertIn('operating_conditions_unverified',
                      qualification._awake_condition_rejections(summary))
    def test_failed_windows_survive_later_success_and_remain_bounded(self):
        with patch.object(qualification, "detect_thermal_state") as thermal:
            thermal.return_value.value = "fair"
            summary = qualification._new_stability_summary(90, 100)
        failed = dict(requests=3, completed=3, failed=0, quality_passed=3,
                      slo_quality_passed=2, errors={}, started_at="failure",
                      failure_diagnostics={"observed": 1})
        for _ in range(70):
            qualification._accumulate_window(summary, failed)
        qualification._accumulate_window(summary, dict(failed, slo_quality_passed=3,
                                                       started_at="success"))
        self.assertEqual(summary["failed_windows_observed"], 70)
        self.assertEqual(len(summary["failed_windows"]), 64)
        self.assertEqual(summary["last_window"]["started_at"], "success")
        self.assertEqual(summary["failed_windows"][0]["started_at"], "failure")
    def test_workload_diagnostic_separates_phases_without_relaxing_gate(self):
        samples = [dict(pid=1, workload_sha256=label * 64, threads=3, open_fds=12,
                        allocator_active_bytes=size, allocator_cache_bytes=0,
                        registry_active=0, registry_queued=0)
                   for _ in range(8) for label, size in (('a', 100), ('b', 200 * 1024 * 1024))]
        self.assertFalse(qualification._resource_plateau(samples)['plateau_observed'])
        diagnostic = qualification._resource_by_workload(samples)
        self.assertFalse(diagnostic['qualification'])
        self.assertTrue(all(r['plateau_observed'] for r in diagnostic['workloads'].values()))
        samples[-1]['allocator_active_bytes'] += 100 * 1024 * 1024
        self.assertFalse(qualification._resource_by_workload(samples)['workloads']['b'*64]['plateau_observed'])
        samples[-1].pop('workload_sha256')
        self.assertEqual(qualification._resource_by_workload(samples)['unlabelled_samples'], 1)

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

    def test_rss_plateau_rejects_invalid_sample_time_and_value(self):
        samples = [dict(elapsed_seconds=i, rss_bytes=100, pid=1) for i in range(4)]
        for field, value in (('elapsed_seconds', 2), ('elapsed_seconds', 1),
                             ('elapsed_seconds', float('nan')), ('elapsed_seconds', float('inf')),
                             ('elapsed_seconds', -1), ('rss_bytes', -1), ('rss_bytes', None)):
            changed = [dict(row) for row in samples]
            changed[-1][field] = value
            result = qualification._rss_trend(changed)
            self.assertFalse(result['plateau_observed'])
            self.assertIsNone(result['slope_bytes_per_hour'])
            self.assertEqual(result['rejection_reason'], 'invalid_epoch_samples')
        self.assertTrue(qualification._rss_trend(samples)['plateau_observed'])

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
