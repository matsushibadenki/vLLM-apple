import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mflux_generative_readiness import inspect_mflux_generative_sources


class MFluxGenerativeReadinessTests(unittest.TestCase):
    def test_static_scan_finds_supported_model_classes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.py").write_text(
                "class ModelConfig: pass\nclass ZImageTurbo: pass\nclass QwenImage: pass\n"
            )
            report = inspect_mflux_generative_sources(
                root, version="test", executable="/test/python"
            )
        self.assertTrue(report["ready"])
        self.assertEqual(
            report["ready_candidates"],
            ["qwen-image-2512", "z-image-turbo-mlx-4bit"],
        )
        self.assertFalse(report["imports_backend"])
        self.assertFalse(report["allocates_model_or_metal"])

    def test_incompatible_mlx_diffusers_layout_is_not_promoted(self) -> None:
        with TemporaryDirectory() as directory, TemporaryDirectory() as model_directory:
            root = Path(directory)
            root.joinpath("models.py").write_text(
                "class ModelConfig: pass\nclass ZImageTurbo: pass\nclass QwenImage: pass\n"
            )
            model = Path(model_directory)
            model.joinpath("model_index.json").write_text(
                json.dumps({"_class_name": "ZImagePipeline"})
            )
            model.joinpath("quantize_config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 64}})
            )
            for component in ("transformer", "text_encoder", "vae"):
                path = model / component
                path.mkdir()
                path.joinpath("weights.safetensors").write_bytes(b"x")
            report = inspect_mflux_generative_sources(
                root,
                version="test",
                executable="/test/python",
                model=model,
            )
        self.assertFalse(report["ready"])
        candidate = report["candidates"]["z-image-turbo-mlx-4bit"]
        self.assertFalse(candidate["artifact_format_compatible"])
        self.assertIn(
            "unsupported_artifact_format:mlx-diffusers-conversion", candidate["issues"]
        )


if __name__ == "__main__":
    unittest.main()
