from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from vllm_apple.hardware import (
    _ioreg_gpu_core_count,
    _parse_power_mode,
    _system_profiler_chip,
    detect_hardware,
    detect_thermal_state,
)
from vllm_apple.types import PowerMode, ThermalState


class HardwareTests(unittest.TestCase):
    def test_thermal_state_maps_nsprocessinfo_and_fails_soft(self) -> None:
        for value, expected in (
            (0, ThermalState.NOMINAL),
            (1, ThermalState.FAIR),
            (2, ThermalState.SERIOUS),
            (3, ThermalState.CRITICAL),
            (None, ThermalState.UNKNOWN),
            (99, ThermalState.UNKNOWN),
        ):
            with patch("vllm_apple.hardware._ns_process_info_integer", return_value=value):
                self.assertEqual(detect_thermal_state(), expected)

    def test_power_mode_uses_only_the_active_power_source(self) -> None:
        settings = (
            "Battery Power:\n lowpowermode 1\n powermode 1\n"
            "AC Power:\n lowpowermode 0\n powermode 2\n"
        )
        self.assertEqual(
            _parse_power_mode("Now drawing from 'AC Power'\n", settings),
            PowerMode.HIGH_POWER,
        )
        self.assertEqual(
            _parse_power_mode("Now drawing from 'Battery Power'\n", settings),
            PowerMode.LOW_POWER,
        )
        self.assertEqual(_parse_power_mode("unknown", settings), PowerMode.UNKNOWN)

    def test_power_mode_treats_disabled_low_power_as_automatic(self) -> None:
        self.assertEqual(
            _parse_power_mode(
                "Now drawing from 'AC Power'\n", "AC Power:\n lowpowermode 0\n"
            ),
            PowerMode.AUTOMATIC,
        )

    def test_ioreg_gpu_core_count(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='    "gpu-core-count" = 10\n',
            stderr="",
        )
        with patch("vllm_apple.hardware.subprocess.run", return_value=completed):
            self.assertEqual(_ioreg_gpu_core_count(), 10)

    def test_ioreg_gpu_core_count_fails_closed(self) -> None:
        with patch(
            "vllm_apple.hardware.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ioreg", 1.0),
        ):
            self.assertIsNone(_ioreg_gpu_core_count())

    def test_ioreg_gpu_core_count_decodes_little_endian_data(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='    "gpu-core-count" = <28000000>\n',
            stderr="",
        )
        with patch("vllm_apple.hardware.subprocess.run", return_value=completed):
            self.assertEqual(_ioreg_gpu_core_count(), 40)

    def test_system_profiler_chip_supports_localized_labels(self) -> None:
        for label in ("Chip", "チップ", "芯片"):
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"Hardware:\n    {label}: Apple M4 Max\n",
                stderr="",
            )
            with patch("vllm_apple.hardware.subprocess.run", return_value=completed):
                self.assertEqual(_system_profiler_chip(), "Apple M4 Max")

    def test_system_profiler_chip_rejects_oversized_output(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="x" * (64 * 1024 + 1), stderr=""
        )
        with patch("vllm_apple.hardware.subprocess.run", return_value=completed):
            self.assertIsNone(_system_profiler_chip())

    def test_detect_hardware_replaces_generic_arm_processor_with_chip(self) -> None:
        with (
            patch("vllm_apple.hardware.platform.system", return_value="Darwin"),
            patch("vllm_apple.hardware.platform.machine", return_value="arm64"),
            patch("vllm_apple.hardware.platform.processor", return_value="arm"),
            patch("vllm_apple.hardware._sysctl", return_value=None),
            patch("vllm_apple.hardware._system_profiler_chip", return_value="Apple M4"),
            patch("vllm_apple.hardware._ioreg_gpu_core_count", return_value=10),
            patch("vllm_apple.hardware.detect_memory"),
            patch(
                "vllm_apple.hardware.detect_thermal_state", return_value=ThermalState.FAIR
            ),
            patch(
                "vllm_apple.hardware.detect_power_mode", return_value=PowerMode.LOW_POWER
            ),
        ):
            hardware = detect_hardware()
            self.assertEqual(hardware.soc, "Apple M4")
            self.assertEqual(hardware.thermal_state, ThermalState.FAIR)
            self.assertEqual(hardware.power_mode, PowerMode.LOW_POWER)


if __name__ == "__main__":
    unittest.main()
