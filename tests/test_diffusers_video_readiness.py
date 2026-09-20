import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.diffusers_video_readiness import assess_diffusers_video_readiness
from vllm_apple.generative_artifact_inspection import inspect_generative_artifact


def artifact(root: Path, *, bits: int = 4, pipeline: str = "WanPipeline"):
    root.joinpath("model_index.json").write_text(json.dumps({"_class_name": pipeline}))
    for component in ("transformer", "text_encoder", "vae"):
        component_root = root / component
        component_root.mkdir()
        component_root.joinpath("weights.safetensors").write_bytes(b"weight")
    root.joinpath("transformer", "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "test", "load_in_4bit": bits == 4,
                                      "load_in_8bit": bits == 8}}
        )
    )
    return inspect_generative_artifact(root)


def backend(*, missing=()):
    return {
        "diffusers_version": "0.34.0",
        "candidates": {
            "wan2.2-ti2v-5b": {
                "required_pipeline_classes": ["WanPipeline", "WanImageToVideoPipeline"],
                "missing_pipeline_classes": list(missing),
            }
        },
    }


class DiffusersVideoReadinessTests(unittest.TestCase):
    def test_quantized_wan_artifact_passes_without_loading_weights(self) -> None:
        with TemporaryDirectory() as directory:
            report = assess_diffusers_video_readiness(
                executable="/test/python",
                backend=backend(),
                artifact=artifact(Path(directory)),
            )
        self.assertTrue(report["ready"])
        self.assertEqual(report["artifact"]["quantization"], {"bits": 4, "method": "test"})
        self.assertFalse(report["imports_backend"])
        self.assertFalse(report["loads_weights"])
        self.assertFalse(report["allocates_model_or_metal"])

    def test_unquantized_or_wrong_pipeline_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = assess_diffusers_video_readiness(
                executable="/test/python",
                backend=backend(),
                artifact=artifact(root, bits=0, pipeline="WanImageToVideoPipeline"),
            )
        self.assertFalse(report["ready"])
        self.assertIn("unexpected_pipeline_class", report["issues"])
        self.assertIn("expected_4bit_or_8bit_quantization", report["issues"])

    def test_missing_wan_pipeline_fails_even_when_i2v_exists(self) -> None:
        with TemporaryDirectory() as directory:
            report = assess_diffusers_video_readiness(
                executable="/test/python",
                backend=backend(missing=("WanPipeline",)),
                artifact=artifact(Path(directory), bits=8),
            )
        self.assertFalse(report["ready"])
        self.assertIn("wan_pipeline_unavailable", report["issues"])


if __name__ == "__main__":
    unittest.main()
