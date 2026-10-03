import subprocess
import unittest
from unittest.mock import patch

from vllm_apple.benchmark_context import (
    _process_age_seconds,
    _system_context,
    observe_benchmark_context,
)
from vllm_apple.types import MemoryInfo, MemoryPressure, ThermalState


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

    @patch("vllm_apple.benchmark_context._system_context", return_value={})
    def test_known_and_unavailable_observations(self, _system):
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

    def test_system_observations_preserve_estimate_provenance(self):
        for source, fallback in (("sysctl", False),
                                 ("sysctl+available-conservative-estimate", True)):
            with self.subTest(source=source), patch(
                "vllm_apple.benchmark_context.os.getloadavg", return_value=(1., 2., 3.),
            ), patch("vllm_apple.benchmark_context.os.cpu_count", return_value=8), patch(
                "vllm_apple.benchmark_context.detect_memory",
                return_value=MemoryInfo(1000, 100, pressure=MemoryPressure.WARNING, source=source),
            ):
                result = _system_context()
            self.assertEqual(result["load_average_1_5_15_minutes"], [1., 2., 3.])
            self.assertEqual(result["logical_cpu_count"], 8)
            self.assertEqual(result["memory"]["available_is_fallback"], fallback)
            self.assertTrue(result["memory"]["available_is_estimate"])
            self.assertEqual(result["memory"]["pressure_estimate"], "warning")

    def test_system_missing_and_invalid_values_are_not_idle(self):
        for load in (OSError("unavailable"), (float("nan"), 0., 0.), (-1., 0., 0.)):
            kwargs = {"side_effect": load} if isinstance(load, OSError) else {"return_value": load}
            with self.subTest(load=load), patch(
                "vllm_apple.benchmark_context.os.getloadavg", **kwargs,
            ), patch("vllm_apple.benchmark_context.os.cpu_count", return_value=None), patch(
                "vllm_apple.benchmark_context.detect_memory", side_effect=RuntimeError("unavailable"),
            ):
                result = _system_context()
            self.assertIsNone(result["load_average_1_5_15_minutes"])
            self.assertIsNone(result["logical_cpu_count"])
            self.assertIsNone(result["memory"])
