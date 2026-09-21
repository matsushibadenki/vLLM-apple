import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.qwen_image_21_torchao_readiness import (
    inspect_qwen_image_21_torchao_readiness,
)


class QwenImage21TorchAOReadinessTests(unittest.TestCase):
    def test_conversion_and_mps_readiness_are_separate(self) -> None:
        payload = {
            "torch_version": "2.14.0",
            "torchao_version": "0.18.0",
            "diffusers_version": "0.41.0.dev0",
            "diffusers_pipeline_quantization_api": True,
            "int8_weight_only_api": True,
            "cpu_int8_probe": True,
            "mps_built": True,
            "mps_available": False,
            "mps_int8_weight_only_probe": False,
            "mps_error": None,
            "int4_weight_only_api": True,
            "cpu_int4_weight_only_probe": False,
            "cpu_int4_error": "ImportError: Requires mslk >= 1.0.0",
            "mps_int4_weight_only_probe": False,
            "mps_int4_error": None,
        }
        completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        with patch("pathlib.Path.is_file", return_value=True), patch(
            "os.access", return_value=True
        ), patch("subprocess.run", return_value=completed):
            report = inspect_qwen_image_21_torchao_readiness(Path("python"))
        self.assertTrue(report["conversion_ready"])
        self.assertFalse(report["mps_runtime_ready"])
        self.assertFalse(report["int4_conversion_ready"])
        self.assertFalse(report["int4_mps_runtime_ready"])
        self.assertFalse(report["loads_model_weights"])

    def test_probe_rejects_unexpected_schema(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "{}", "")
        with patch("pathlib.Path.is_file", return_value=True), patch(
            "os.access", return_value=True
        ), patch("subprocess.run", return_value=completed), self.assertRaisesRegex(
            ValueError, "invalid schema"
        ):
            inspect_qwen_image_21_torchao_readiness(Path("python"))
