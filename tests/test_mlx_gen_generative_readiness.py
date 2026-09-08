import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mlx_gen_generative_readiness import (
    assess_mlx_gen_generative_readiness,
    select_mlx_gen_qualification_candidate,
)


def model_fixture(root: Path) -> None:
    root.joinpath("README.md").write_text(
        "---\nlibrary_name: mlx-gen\nlicense: other\n"
        "base_model: black-forest-labs/FLUX.2-klein-base-9B\n---\n"
    )
    for component in ("transformer", "text_encoder", "vae"):
        directory = root / component
        directory.mkdir()
        directory.joinpath("model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"mflux_version": "0.18.2", "quantization_level": "4"}})
        )
        directory.joinpath("0.safetensors").write_bytes(b"x")


class MLXGenGenerativeReadinessTests(unittest.TestCase):
    def test_low_cache_profile_selection_is_exact_and_flux_only(self) -> None:
        self.assertEqual(
            select_mlx_gen_qualification_candidate("flux2-klein-9b-base", 0.25),
            "flux2-klein-9b-base-low-cache",
        )
        with self.assertRaisesRegex(ValueError, "only FLUX"):
            select_mlx_gen_qualification_candidate("flux2-klein-9b-base", 0.5)
        with self.assertRaisesRegex(ValueError, "only FLUX"):
            select_mlx_gen_qualification_candidate("z-image-turbo-mlx-4bit", 0.25)

    def test_matching_backend_and_artifact_are_ready_without_loading(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_fixture(root)
            report = assess_mlx_gen_generative_readiness(
                executable="/test/python",
                version="0.18.2",
                cli_registered=True,
                model=root,
            )
        self.assertTrue(report["ready"])
        self.assertFalse(report["imports_backend"])
        self.assertFalse(report["allocates_model_or_metal"])

    def test_old_backend_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_fixture(root)
            report = assess_mlx_gen_generative_readiness(
                executable="/test/python",
                version="0.18.1",
                cli_registered=True,
                model=root,
            )
        self.assertFalse(report["ready"])
        self.assertIn("mlx_gen_version_below_0.18.2", report["issues"])

    def test_native_z_image_package_is_ready_with_the_matching_cli(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text(
                "---\nlibrary_name: mlx-gen\nlicense: apache-2.0\n"
                "base_model: Tongyi-MAI/Z-Image-Turbo\n---\n"
            )
            for component in ("transformer", "text_encoder", "vae"):
                path = root / component
                path.mkdir()
                path.joinpath("model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "mflux_version": "0.1.0",
                                "quantization_level": "4",
                            }
                        }
                    )
                )
                path.joinpath("weights.safetensors").write_bytes(b"x")
            report = assess_mlx_gen_generative_readiness(
                executable="/test/python",
                version="0.33.1",
                cli_registered=True,
                z_image_cli_registered=True,
                model=root,
            )
        self.assertTrue(report["ready"])
        self.assertEqual(report["candidate_id"], "z-image-turbo-mlx-4bit")
        self.assertEqual(report["minimum_version"], "0.33.1")

    def test_z_image_requires_the_candidate_specific_cli(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text(
                "---\nlibrary_name: mlx\nlicense: apache-2.0\n"
                "base_model: Tongyi-MAI/Z-Image-Turbo\n---\n"
            )
            root.joinpath("model_index.json").write_text(
                json.dumps({"_class_name": "ZImagePipeline"})
            )
            root.joinpath("quantize_config.json").write_text(
                json.dumps({"quantization": {"bits": 4}})
            )
            for component in ("transformer", "text_encoder", "vae"):
                path = root / component
                path.mkdir()
                path.joinpath("weights.safetensors").write_bytes(b"x")
            report = assess_mlx_gen_generative_readiness(
                executable="/test/python",
                version="0.33.1",
                cli_registered=True,
                z_image_cli_registered=False,
                model=root,
            )
        self.assertFalse(report["ready"])
        self.assertIn("z_image_turbo_console_script_missing", report["issues"])
        self.assertIn(
            "unsupported_artifact_format:mlx-diffusers-conversion", report["issues"]
        )


if __name__ == "__main__":
    unittest.main()
