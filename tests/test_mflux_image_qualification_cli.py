import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class MFluxImageQualificationCLITests(unittest.TestCase):
    def test_qwen_2512_cli_connects_readiness_admission_and_worker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"weight")
            artifact = {
                "artifact_format": "mflux",
                "artifact_bytes": 80,
                "quantization": {"bits": 4},
                "license": "apache-2.0",
                "base_model": "Qwen/Qwen-Image-2512",
                "components": [
                    {"name": "transformer", "role": "denoiser", "artifact_bytes": 50},
                    {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 20},
                    {"name": "vae", "role": "vae", "artifact_bytes": 10},
                ],
            }
            readiness = {
                "mflux_version": "mlx-gen-bundled-0.33.1",
                "artifact": artifact,
                "candidates": {"qwen-image-2512": {"ready": True}},
            }
            hardware = HardwareInfo(
                "Darwin", "arm64", "Apple M4", 10, 10, 10,
                MemoryInfo(32 * GIB, 28 * GIB), True, "test",
            )
            captured = {}

            def run(plan, **kwargs):
                captured["plan"] = plan
                captured["kwargs"] = kwargs
                return SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})

            output = io.StringIO()
            with patch(
                "vllm_apple.cli.inspect_mflux_generative_readiness",
                return_value=readiness,
            ), patch(
                "vllm_apple.cli.detect_hardware", return_value=hardware
            ), patch(
                "vllm_apple.cli.run_generative_qualification", side_effect=run
            ), redirect_stdout(output):
                code = main([
                    "mflux-image-qualification", str(model),
                    "--python", "/test/python", "--resident-gib", "8",
                    "--workspace-root", str(root),
                    "--private-root", str(root / "private"),
                    "--report", str(root / "report.json"),
                ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"passed": True})
        self.assertEqual(captured["plan"].candidate.candidate_id, "qwen-image-2512")
        self.assertEqual(captured["plan"].quantization, "int4")
        self.assertEqual(captured["kwargs"]["mode"], "text-to-image")
        self.assertEqual(
            captured["kwargs"]["worker_command"][-1],
            "vllm_apple.mflux_generation_worker",
        )

    def test_non_4bit_artifact_is_rejected_before_worker(self) -> None:
        readiness = {
            "mflux_version": "test",
            "artifact": {"quantization": {"bits": 8}},
            "candidates": {"qwen-image-2512": {"ready": True}},
        }
        with patch(
            "vllm_apple.cli.inspect_mflux_generative_readiness",
            return_value=readiness,
        ), patch("vllm_apple.cli.run_generative_qualification") as run, redirect_stdout(
            io.StringIO()
        ):
            code = main([
                "mflux-image-qualification", "model",
                "--python", "/test/python", "--resident-gib", "8",
            ])
        self.assertEqual(code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
