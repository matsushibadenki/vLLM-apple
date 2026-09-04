import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.generative_artifact_inspection import (
    GenerativeArtifactInspectionError,
    inspect_generative_artifact,
)


def write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class GenerativeArtifactInspectionTests(unittest.TestCase):
    def test_detects_mlx_diffusers_conversion_without_loading_weights(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_index.json").write_text(
                json.dumps({"_class_name": "ZImagePipeline"})
            )
            (root / "quantize_config.json").write_text(
                json.dumps({"quantization_config": {"bits": 4, "group_size": 64}})
            )
            write(root / "transformer" / "weights.safetensors", 11)
            write(root / "text_encoder" / "weights.safetensors", 7)
            write(root / "tokenizer" / "tokenizer.json", 3)
            write(root / "vae" / "weights.safetensors", 5)

            report = inspect_generative_artifact(root)

        self.assertEqual(report["artifact_format"], "mlx-diffusers-conversion")
        self.assertEqual(report["backend_kind"], "mlx-diffusers")
        self.assertEqual(report["pipeline_class"], "ZImagePipeline")
        self.assertEqual(report["quantization"], {"bits": 4, "group_size": 64})
        self.assertEqual(report["artifact_bytes"], 113)
        self.assertTrue(report["inspectable"])
        self.assertFalse(report["weights_loaded"])

    def test_detects_mflux_shards(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "transformer").mkdir()
            (root / "transformer" / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"mflux_version": "0.14.0", "quantization_level": "4"}})
            )
            write(root / "transformer" / "weights.safetensors", 11)
            write(root / "text_encoder" / "weights.safetensors", 7)
            write(root / "vae" / "weights.safetensors", 5)

            report = inspect_generative_artifact(root)

        self.assertEqual(report["artifact_format"], "mflux")
        self.assertEqual(report["backend_kind"], "mflux")
        self.assertEqual(report["quantization"], {"bits": 4})
        self.assertTrue(report["inspectable"])

    def test_model_card_distinguishes_mlx_gen_from_shared_mflux_layout(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text(
                "---\nlibrary_name: mlx-gen\nlicense: other\n"
                "base_model: black-forest-labs/FLUX.2-klein-base-9B\n---\n"
            )
            for component in ("transformer", "text_encoder", "vae"):
                component_root = root / component
                component_root.mkdir()
                component_root.joinpath("model.safetensors.index.json").write_text(
                    json.dumps(
                        {"metadata": {"mflux_version": "0.18.2", "quantization_level": "4"}}
                    )
                )
                component_root.joinpath("0.safetensors").write_bytes(b"x")
            report = inspect_generative_artifact(root)
        self.assertEqual(report["artifact_format"], "mlx-gen")
        self.assertEqual(report["backend_kind"], "mlx-gen")
        self.assertEqual(report["license"], "other")
        self.assertEqual(
            report["base_model"], "black-forest-labs/FLUX.2-klein-base-9B"
        )

    def test_rejects_symlinked_artifact_content(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("x")
            (root / "linked").symlink_to(target)
            with self.assertRaisesRegex(GenerativeArtifactInspectionError, "symlinks"):
                inspect_generative_artifact(root)


if __name__ == "__main__":
    unittest.main()
