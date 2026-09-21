import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from vllm_apple.cli import main


class QwenImage21ConvertCLITests(unittest.TestCase):
    def test_ineligible_plan_never_starts_worker(self) -> None:
        plan = {"eligible": False, "issues": ["memory"]}
        with patch("vllm_apple.cli.inspect_generative_artifact", return_value={}), patch(
            "vllm_apple.cli.inspect_qwen_image_21_torchao_readiness",
            return_value={"conversion_ready": True},
        ), patch("vllm_apple.cli.detect_hardware"), patch(
            "vllm_apple.cli.build_qwen_image_21_streaming_conversion_plan", return_value=plan
        ), patch("vllm_apple.cli.run_qwen_image_21_conversion_worker") as worker:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "qwen-image-2.1-convert", "source", "output",
                    "--python", "/runtime/python",
                ])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["started"])
        worker.assert_not_called()

    def test_eligible_plan_runs_isolated_worker(self) -> None:
        plan = {"eligible": True, "issues": []}
        conversion = {"passed": True, "artifact_root_sha256": "a" * 64}
        with patch("vllm_apple.cli.inspect_generative_artifact", return_value={}), patch(
            "vllm_apple.cli.inspect_qwen_image_21_torchao_readiness",
            return_value={"conversion_ready": True},
        ), patch("vllm_apple.cli.detect_hardware"), patch(
            "vllm_apple.cli.build_qwen_image_21_streaming_conversion_plan", return_value=plan
        ), patch(
            "vllm_apple.cli.run_qwen_image_21_conversion_worker", return_value=conversion
        ) as worker:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "qwen-image-2.1-convert", "source", "output",
                    "--python", "/runtime/python", "--timeout", "60",
                ])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["started"])
        worker.assert_called_once()
