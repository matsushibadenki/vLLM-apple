import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mlx_gen_video_readiness import assess_mlx_gen_video_readiness


def write_artifact(root: Path, *, bits=8, pipeline="WanPipeline") -> None:
    root.joinpath("README.md").write_text(
        "---\nlibrary_name: mlx-gen\nlicense: apache-2.0\n"
        "base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers\n---\n"
    )
    root.joinpath("model_index.json").write_text(json.dumps({"_class_name": pipeline}))
    for component in ("transformer", "text_encoder", "vae"):
        path = root / component
        path.mkdir()
        path.joinpath("model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"mflux_version": "0.33.1", "quantization_level": str(bits)}})
        )
        path.joinpath("0.safetensors").write_bytes(b"weight")


class MLXGenVideoReadinessTests(unittest.TestCase):
    def test_q8_wan_artifact_passes_without_backend_import(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root)
            report = assess_mlx_gen_video_readiness(
                executable="/test/python",
                version="0.33.1",
                cli_registered=True,
                model=root,
            )
        self.assertTrue(report["ready"])
        self.assertFalse(report["imports_backend"])
        self.assertFalse(report["loads_weights"])
        self.assertFalse(report["allocates_model_or_metal"])

    def test_version_quantization_and_pipeline_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, bits=4, pipeline="OtherPipeline")
            report = assess_mlx_gen_video_readiness(
                executable="/test/python",
                version="0.33.0",
                cli_registered=False,
                model=root,
            )
        self.assertFalse(report["ready"])
        self.assertIn("mlx_gen_version_below_0.33.1", report["issues"])
        self.assertIn("mlxgen_console_script_missing", report["issues"])
        self.assertIn("unexpected_pipeline_class", report["issues"])
        self.assertIn("expected_mixed_q8_bf16_quantization", report["issues"])


if __name__ == "__main__":
    unittest.main()
