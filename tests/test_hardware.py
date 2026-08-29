from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from vllm_apple.hardware import _ioreg_gpu_core_count


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


if __name__ == "__main__":
    unittest.main()
