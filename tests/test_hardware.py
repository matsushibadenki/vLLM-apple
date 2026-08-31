from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from vllm_apple.hardware import _ioreg_gpu_core_count, _system_profiler_chip, detect_hardware


class HardwareTests(unittest.TestCase):
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
        ):
            self.assertEqual(detect_hardware().soc, "Apple M4")


if __name__ == "__main__":
    unittest.main()
