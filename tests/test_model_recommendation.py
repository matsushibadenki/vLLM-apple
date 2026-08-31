import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.schema_validator import validate_instance
from vllm_apple.cli import main
from vllm_apple.model import inspect_model
from vllm_apple.model_recommendation import build_model_recommendation
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def hardware() -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Apple M4",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=24 * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


def model(directory: str) -> Path:
    root = Path(directory) / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma",
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 32,
                "max_position_embeddings": 4096,
                "torch_dtype": "float16",
            }
        )
    )
    (root / "weights.safetensors").write_bytes(b"x" * 1024)
    return root


class ModelRecommendationTests(unittest.TestCase):
    def test_report_is_schema_valid_and_does_not_expose_config_or_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            report = build_model_recommendation(
                inspect_model(root), hardware(), backend="mlx_lm"
            ).to_dict()
        schema = json.loads(
            Path("schemas/runtime/model-recommendation-v1.schema.json").read_text()
        )
        validate_instance(report, schema)
        self.assertTrue(report["runnable"])
        self.assertEqual(report["recommended_tier"], "balanced")
        self.assertEqual(report["recommended_context_tokens"], 4096)
        self.assertNotIn("config", report)
        self.assertNotIn("path", report)

    def test_cli_returns_machine_readable_recommendation_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            output = StringIO()
            with patch("vllm_apple.cli.detect_hardware", return_value=hardware()), redirect_stdout(
                output
            ):
                exit_code = main(["inspect-model", str(root), "--backend", "mlx_lm"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["backend_compatible"])
        self.assertTrue(payload["fits_memory"])


if __name__ == "__main__":
    unittest.main()
