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


class DiffusersImageQualificationCLITests(unittest.TestCase):
    def test_qwen_cli_connects_phase_admission_digest_and_private_worker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"test weight")
            artifact = {
                "artifact_format": "diffusers",
                "pipeline_class": "QwenImage21Pipeline",
                "artifact_bytes": 80,
                "quantization": {},
                "license": "qwen-research",
                "base_model": None,
                "components": [
                    {"name": "transformer", "role": "denoiser", "artifact_bytes": 50},
                    {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 20},
                    {"name": "vae", "role": "vae", "artifact_bytes": 10},
                ],
                "inspectable": True,
            }
            readiness = {
                "diffusers_version": "0.41.0.dev0",
                "candidates": {"qwen-image-2.1": {"ready": True}},
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
                "vllm_apple.cli.inspect_diffusers_generative_readiness",
                return_value=readiness,
            ), patch(
                "vllm_apple.cli.inspect_generative_artifact", return_value=artifact
            ), patch(
                "vllm_apple.cli.estimate_qwen_image_21_resident_bytes",
                return_value=2 * GIB,
            ), patch(
                "vllm_apple.cli.detect_hardware", return_value=hardware
            ), patch(
                "vllm_apple.cli.run_generative_qualification", side_effect=run
            ), redirect_stdout(output):
                code = main([
                    "diffusers-image-qualification", str(model),
                    "--python", "/test/python", "--resident-auto",
                    "--workspace-root", str(root),
                    "--private-root", str(root / "private"),
                    "--report", str(root / "report.json"),
                ])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"passed": True})
        self.assertEqual(captured["plan"].candidate.candidate_id, "qwen-image-2.1")
        self.assertEqual(captured["plan"].quantization, "none")
        self.assertEqual(captured["plan"].steps, 20)
        self.assertEqual(captured["kwargs"]["mode"], "text-to-image")
        self.assertEqual(
            captured["kwargs"]["worker_command"][-1],
            "vllm_apple.diffusers_generation_worker",
        )
        self.assertEqual(len(captured["kwargs"]["provenance"].artifact_root_sha256), 64)

    def test_qwen_cli_rejects_wrong_pipeline_before_generation(self) -> None:
        readiness = {
            "diffusers_version": "0.41.0.dev0",
            "candidates": {"qwen-image-2.1": {"ready": True}},
        }
        artifact = {
            "artifact_format": "diffusers",
            "pipeline_class": "QwenImagePipeline",
        }
        with patch(
            "vllm_apple.cli.inspect_diffusers_generative_readiness",
            return_value=readiness,
        ), patch(
            "vllm_apple.cli.inspect_generative_artifact", return_value=artifact
        ), patch(
            "vllm_apple.cli.run_generative_qualification"
        ) as run, redirect_stdout(io.StringIO()):
            code = main([
                "diffusers-image-qualification", "model",
                "--python", "/test/python", "--resident-auto",
            ])
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_resolution_promotion_binds_an_all_normal_baseline(self) -> None:
        artifact = {
            "artifact_format": "diffusers",
            "pipeline_class": "QwenImage21Pipeline",
            "artifact_bytes": 80,
            "quantization": {"bits": 8},
            "license": "other",
            "base_model": None,
            "components": [
                {"name": "transformer", "role": "denoiser", "artifact_bytes": 50},
                {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 20},
                {"name": "vae", "role": "vae", "artifact_bytes": 10},
            ],
            "inspectable": True,
        }
        readiness = {
            "diffusers_version": "0.41.0.dev0",
            "candidates": {"qwen-image-2.1": {"ready": True}},
        }
        hardware = HardwareInfo(
            "Darwin", "arm64", "Apple M4", 10, 10, 10,
            MemoryInfo(32 * GIB, 28 * GIB), True, "test",
        )
        sample = SimpleNamespace(
            output_width=512, output_height=512, output_frames=1,
            memory_pressure="normal",
        )
        baseline = SimpleNamespace(
            passed=True, candidate_id="qwen-image-2.1", plan_sha256="a" * 64,
            sample_count=4, samples=(sample, sample, sample, sample),
        )
        captured = {}

        def run(plan, **kwargs):
            captured["plan"] = plan
            return SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})

        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"weight")
            with patch(
                "vllm_apple.cli.inspect_diffusers_generative_readiness",
                return_value=readiness,
            ), patch(
                "vllm_apple.cli.inspect_generative_artifact", return_value=artifact
            ), patch(
                "vllm_apple.cli.estimate_qwen_image_21_resident_bytes",
                return_value=2 * GIB,
            ), patch(
                "vllm_apple.cli.detect_hardware", return_value=hardware
            ), patch(
                "vllm_apple.cli.load_generative_evaluation_report", return_value=baseline
            ), patch(
                "vllm_apple.cli.run_generative_qualification", side_effect=run
            ), redirect_stdout(io.StringIO()):
                code = main([
                    "diffusers-image-qualification", str(model),
                    "--python", "/test/python", "--resident-auto",
                    "--width", "768", "--height", "768",
                    "--baseline-report", str(root / "baseline.json"),
                    "--workspace-root", str(root),
                    "--private-root", str(root / "private"),
                    "--report", str(root / "report.json"),
                ])
        self.assertEqual(code, 0)
        self.assertEqual(captured["plan"].promotion_axis, "resolution")
        self.assertEqual(captured["plan"].baseline_plan_sha256, "a" * 64)
        self.assertEqual(captured["plan"].issues, ())
