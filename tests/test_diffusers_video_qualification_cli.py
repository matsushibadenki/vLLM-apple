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


class DiffusersVideoQualificationCLITests(unittest.TestCase):
    def test_formal_cli_connects_readiness_admission_and_video_worker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"test weight")
            artifact = {
                "artifact_format": "diffusers",
                "artifact_bytes": 80,
                "quantization": {"bits": 4},
                "license": "apache-2.0",
                "base_model": "Wan-AI/Wan2.2-TI2V-5B",
                "components": [
                    {"name": "transformer", "role": "denoiser", "artifact_bytes": 60},
                    {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 10},
                    {"name": "vae", "role": "vae", "artifact_bytes": 10},
                ],
            }
            readiness = {
                "ready": True,
                "candidate_id": "wan2.2-ti2v-5b",
                "diffusers_version": "0.34.0",
                "artifact": artifact,
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
                "vllm_apple.cli.inspect_diffusers_video_readiness",
                return_value=readiness,
            ), patch("vllm_apple.cli.detect_hardware", return_value=hardware), patch(
                "vllm_apple.cli.run_generative_qualification", side_effect=run
            ), redirect_stdout(output):
                code = main(
                    [
                        "diffusers-video-qualification",
                        str(model),
                        "--python",
                        "/test/python",
                        "--resident-gib",
                        "18",
                        "--workspace-root",
                        str(root),
                        "--private-root",
                        str(root / "private"),
                        "--report",
                        str(root / "report.json"),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"passed": True})
        self.assertEqual(captured["plan"].candidate.candidate_id, "wan2.2-ti2v-5b")
        self.assertEqual(captured["plan"].quantization, "int4")
        self.assertEqual(len(captured["kwargs"]["provenance"].artifact_root_sha256), 64)
        self.assertEqual(captured["kwargs"]["mode"], "text-to-video")
        self.assertEqual(
            captured["kwargs"]["worker_command"][-1],
            "vllm_apple.diffusers_video_generation_worker",
        )

    def test_mlx_gen_formal_cli_selects_the_video_worker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"test weight")
            artifact = {
                "artifact_format": "mlx-gen",
                "artifact_bytes": 80,
                "quantization": {"bits": 8},
                "license": "apache-2.0",
                "base_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "components": [
                    {"name": "transformer", "role": "denoiser", "artifact_bytes": 60},
                    {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 10},
                    {"name": "vae", "role": "vae", "artifact_bytes": 10},
                ],
            }
            readiness = {
                "ready": True,
                "candidate_id": "wan2.2-ti2v-5b",
                "mlx_gen_version": "0.33.1",
                "artifact": artifact,
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

            with patch(
                "vllm_apple.cli.inspect_mlx_gen_video_readiness", return_value=readiness
            ), patch("vllm_apple.cli.detect_hardware", return_value=hardware), patch(
                "vllm_apple.cli.run_generative_qualification", side_effect=run
            ), redirect_stdout(io.StringIO()):
                code = main([
                    "mlx-gen-video-qualification", str(model),
                    "--python", "/test/python", "--resident-gib", "18",
                    "--workspace-root", str(root), "--private-root", str(root / "private"),
                    "--report", str(root / "report.json"),
                ])
        self.assertEqual(code, 0)
        self.assertEqual(captured["plan"].quantization, "int8")
        self.assertEqual(captured["kwargs"]["mode"], "text-to-video")
        self.assertEqual(
            captured["kwargs"]["worker_command"][-1],
            "vllm_apple.mlx_gen_video_generation_worker",
        )

    def test_mlx_gen_frame_promotion_requires_a_baseline_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model.joinpath("weight.safetensors").write_bytes(b"test weight")
            artifact = {
                "artifact_format": "mlx-gen", "artifact_bytes": 80,
                "quantization": {"bits": 8}, "license": "apache-2.0",
                "base_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "components": [
                    {"name": "transformer", "role": "denoiser", "artifact_bytes": 60},
                    {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 10},
                    {"name": "vae", "role": "vae", "artifact_bytes": 10},
                ],
            }
            readiness = {
                "ready": True, "candidate_id": "wan2.2-ti2v-5b",
                "mlx_gen_version": "0.33.1", "artifact": artifact,
            }
            machine = HardwareInfo(
                "Darwin", "arm64", "Apple M4", 10, 10, 10,
                MemoryInfo(32 * GIB, 28 * GIB), True, "test",
            )
            output = io.StringIO()
            with patch(
                "vllm_apple.cli.inspect_mlx_gen_video_readiness", return_value=readiness
            ), patch("vllm_apple.cli.detect_hardware", return_value=machine), redirect_stdout(output):
                code = main([
                    "mlx-gen-video-qualification", str(model), "--python", "/test/python",
                    "--resident-gib", "12", "--frames", "49",
                    "--workspace-root", str(root), "--private-root", str(root / "private"),
                ])
        self.assertEqual(code, 2)
        self.assertIn("baseline-report", json.loads(output.getvalue())["detail"])


if __name__ == "__main__":
    unittest.main()
