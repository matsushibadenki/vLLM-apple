import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class QwenImage21ResidencyCLITests(unittest.TestCase):
    def test_cli_emits_load_free_int8_plan(self) -> None:
        artifact = {
            "pipeline_class": "QwenImage21Pipeline",
            "components": [
                {"name": "transformer", "role": "denoiser", "artifact_bytes": 14 * GIB},
                {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 17 * GIB},
                {"name": "vae", "role": "vae", "artifact_bytes": GIB},
            ],
        }
        hardware = HardwareInfo(
            "Darwin", "arm64", "Apple M4", 10, 10, 10,
            MemoryInfo(32 * GIB, 28 * GIB), True, "test",
        )
        output = io.StringIO()
        with patch(
            "vllm_apple.cli.inspect_generative_artifact", return_value=artifact
        ), patch(
            "vllm_apple.cli.detect_hardware", return_value=hardware
        ), redirect_stdout(output):
            code = main(["qwen-image-2.1-residency-plan", "model", "--target-bits", "8"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["target_quantization"], "int8")
        self.assertFalse(report["eligible_for_generation"])
