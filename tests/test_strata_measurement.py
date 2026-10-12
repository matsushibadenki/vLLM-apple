import unittest
from unittest.mock import Mock, patch

from scripts.strata_measurement import HostLoadMonitor, cpu_delta, quiet


class HostLoadTests(unittest.TestCase):
    def test_short_tail_merges_without_discarding_cpu_delta(self):
        monitor = HostLoadMonitor(1, interval=1)
        monitor.stop = Mock()
        monitor.thread = Mock()
        monitor.observations = [dict(total_percent=0)]
        monitor.before = {(-1, 0): (10.1, "__system__")}
        monitor.started = 0.95
        monitor.last_before = {(-1, 0): (10, "__system__")}
        monitor.last_started = 0
        with patch("scripts.strata_measurement.time.monotonic", return_value=1), patch(
            "scripts.strata_measurement.cpu_sample",
            return_value={(-1, 0): (10.2, "__system__")},
        ):
            monitor.__exit__()
        self.assertEqual(len(monitor.observations), 1)
        self.assertEqual(monitor.observations[0]["duration_seconds"], 1)
        self.assertAlmostEqual(monitor.observations[0]["total_percent"], 20)

    def test_counter_delta_detects_multi_core_compile(self):
        before = {(1, 1): (10, "rustc"), (2, 1): (2, "idle")}
        after = {(1, 1): (15, "rustc"), (2, 1): (2.1, "idle")}
        observed = cpu_delta(before, after, 1)
        self.assertAlmostEqual(observed["total_percent"], 510)
        self.assertFalse(quiet(observed))

    def test_new_pid_and_reused_pid_fail_closed(self):
        observed = cpu_delta({(1, 1): (10, "old")}, {(1, 2): (0.1, "new")}, 1)
        self.assertEqual(observed["new_or_exited_processes"], 2)
        self.assertFalse(quiet(observed))

    def test_low_load_passes_without_negative_counter_noise(self):
        observed = cpu_delta({(1, 1): (1, "idle")}, {(1, 1): (0.9, "idle")}, 1)
        self.assertTrue(quiet(observed))

    def test_aggregate_rejects_load_from_inaccessible_processes(self):
        before = {(-1, 0): (10, "__system__"), (1, 1): (1, "idle")}
        after = {(-1, 0): (15, "__system__"), (1, 1): (1, "idle")}
        observed = cpu_delta(before, after, 1)
        self.assertEqual(observed["total_percent"], 500)
        self.assertEqual(observed["maximum_process_percent"], 0)
        self.assertFalse(quiet(observed))


if __name__ == "__main__":
    unittest.main()
