import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mlx_gen_generative_readiness import (
    assess_mlx_gen_generative_readiness,
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


if __name__ == "__main__":
    unittest.main()
