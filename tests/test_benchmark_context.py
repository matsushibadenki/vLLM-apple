import subprocess
import unittest
from unittest.mock import patch

from vllm_apple.benchmark_context import _process_age_seconds, observe_benchmark_context
from vllm_apple.types import ThermalState


class BenchmarkContextTests(unittest.TestCase):
    def test_process_age_formats_and_failures(self):
        for value, expected in (("02:03", 123), ("01:02:03", 3723),
                                ("2-01:02:03", 176523), ("garbage", None),
                                ("01:99", None)):
            with self.subTest(value=value), patch(
                "vllm_apple.benchmark_context.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, value),
            ):
                self.assertEqual(_process_age_seconds(123), expected)
        with patch("vllm_apple.benchmark_context.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ps", 1)):
            self.assertIsNone(_process_age_seconds(123))
        self.assertIsNone(_process_age_seconds(None))

    def test_known_and_unavailable_observations(self):
        with patch("vllm_apple.benchmark_context._pmset", side_effect=[
            "Now drawing from 'AC Power'", "AC Power:\n lowpowermode 0\n",
        ]), patch("vllm_apple.benchmark_context.detect_thermal_state",
                  return_value=ThermalState.NOMINAL):
            result = observe_benchmark_context()
        self.assertEqual(result["power_source"], "AC Power")
        self.assertEqual(result["power_mode"], "automatic")
        self.assertIsNone(result["target_process_age_seconds"])
        with patch("vllm_apple.benchmark_context._pmset", return_value=None), patch(
            "vllm_apple.benchmark_context.detect_thermal_state", return_value=ThermalState.UNKNOWN,
        ):
            result = observe_benchmark_context()
        self.assertEqual(result["power_source"], "unknown")
        self.assertEqual(result["power_mode"], "unknown")
