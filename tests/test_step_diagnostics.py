import unittest
from unittest.mock import patch

from vllm_apple.step_diagnostics import StepDiagnostics


class StepDiagnosticTests(unittest.TestCase):
    def test_preserves_results_and_keeps_latest_slow_steps_bounded(self):
        diagnostics = StepDiagnostics()
        output = object()
        wrapped = diagnostics.wrap(lambda value: value)
        with patch("vllm_apple.step_diagnostics.time.monotonic_ns",
                   side_effect=[value for _ in range(70) for value in (0, 300_000_000)]):
            for _ in range(70):
                self.assertIs(wrapped(output), output)
        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot["sample_count"], 70)
        self.assertEqual(snapshot["slow_steps_observed"], 70)
        self.assertEqual(len(snapshot["recent_slow_steps"]), 64)
        self.assertNotIn(str(output), str(snapshot))

    def test_exception_propagates_without_private_message(self):
        diagnostics = StepDiagnostics()

        def failure():
            raise ValueError("private model detail")

        with self.assertRaisesRegex(ValueError, "private model detail"):
            diagnostics.wrap(failure)()
        self.assertEqual(diagnostics.snapshot()["exceptions"], 1)
        self.assertNotIn("private model detail", str(diagnostics.snapshot()))

    def test_thread_cpu_is_separate_from_wall_and_does_not_change_result(self):
        diagnostics = StepDiagnostics()
        result = object()
        with patch('vllm_apple.step_diagnostics.time.monotonic_ns', side_effect=[0, 300_000_000]), \
             patch('vllm_apple.step_diagnostics.time.thread_time_ns', side_effect=[0, 10_000_000]):
            self.assertIs(diagnostics.wrap(lambda: result)(), result)
        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot['statistics']['mean_ms'], 300)
        self.assertEqual(snapshot['thread_cpu_statistics']['mean_ms'], 10)
        self.assertEqual(snapshot['recent_slow_steps'][0]['thread_cpu_ms'], 10)
        self.assertEqual(snapshot['sample_count'], 1)
