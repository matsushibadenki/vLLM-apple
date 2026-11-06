import unittest
from unittest.mock import Mock, patch

from scripts.strata_measurement import CPUSnapshot, HostLoadMonitor, cpu_delta, cpu_sample, quiet


class HostLoadTests(unittest.TestCase):
    def test_diagnostic_snapshot_preserves_existing_descendant_exclusion(self):
        def process(pid, ppid, created):
            return Mock(pid=pid, info=dict(pid=pid, ppid=ppid, name=str(pid),
                        create_time=created, cpu_times=Mock(user=1, system=2)))
        processes = [process(1, 0, 10), process(2, 1, 11), process(3, 2, 12),
                     process(4, 1, 9), process(5, 0, 1)]
        psutil = Mock()
        psutil.cpu_times.return_value = Mock(user=100, system=10)
        psutil.process_iter.return_value = processes
        psutil.Process.return_value.children.return_value = [Mock(pid=2), Mock(pid=3)]
        psutil.Error = RuntimeError
        with patch.dict("sys.modules", psutil=psutil):
            snapshot = cpu_sample(1)
        psutil.process_iter.assert_called_once()
        psutil.Process.return_value.children.assert_called_once_with(recursive=True)
        self.assertEqual(set(snapshot), {(-1, 0), (4, 9), (5, 1)})
        self.assertEqual(set(snapshot.excluded), {(1, 10), (2, 11), (3, 12)})
        self.assertEqual(snapshot[(-1, 0)][0], 110)

    def test_diagnostic_cpu_is_never_subtracted_from_total_or_thresholds(self):
        before = CPUSnapshot({(-1, 0): (0, "system")})
        after = CPUSnapshot({(-1, 0): (2.1, "system")})
        before.excluded = {(1, 0): (0, "observer"), (2, 0): (0, "worker")}
        after.excluded = {(1, 0): (0.2, "observer"), (2, 0): (1, "worker")}
        after.observer_pid = 1
        after.collection_seconds = 0.01
        observation = cpu_delta(before, after, 1)
        self.assertEqual(observation["total_percent"], 210)
        self.assertEqual(observation["observer_cpu_percent"], 20)
        self.assertEqual(observation["worker_tree_cpu_percent"], 100)
        self.assertFalse(quiet(observation))
        self.assertFalse(quiet(dict(total_percent=0, maximum_process_percent=76,
                                    new_or_exited_processes=0)))
        self.assertFalse(quiet(dict(total_percent=0, maximum_process_percent=0,
                                    new_or_exited_processes=1)))

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
