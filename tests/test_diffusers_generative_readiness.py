import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.diffusers_generative_readiness import inspect_diffusers_generative_sources


class DiffusersGenerativeReadinessTests(unittest.TestCase):
    def test_static_source_scan_maps_pipeline_classes_to_candidates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pipelines.py").write_text(
                "\n".join(
                    f"class {name}: pass"
                    for name in (
                        "Flux2Pipeline",
                        "Flux2KleinPipeline",
                        "QwenImagePipeline",
                        "WanPipeline",
                        "WanImageToVideoPipeline",
                        "HunyuanVideo15Pipeline",
                        "HunyuanVideo15ImageToVideoPipeline",
                    )
                ),
                encoding="utf-8",
            )
            report = inspect_diffusers_generative_sources(
                root, version="test", executable="/test/python"
            )
        self.assertTrue(report["ready"])
        self.assertEqual(len(report["ready_candidates"]), 6)
        self.assertFalse(report["imports_backend"])
        self.assertFalse(report["allocates_model_or_metal"])

    def test_candidate_is_not_promoted_when_one_mode_pipeline_is_missing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wan.py").write_text("class WanPipeline: pass\n", encoding="utf-8")
            report = inspect_diffusers_generative_sources(
                root, version="test", executable="/test/python"
            )
        wan = report["candidates"]["wan2.2-ti2v-5b"]
        self.assertFalse(wan["ready"])
        self.assertEqual(wan["missing_pipeline_classes"], ["WanImageToVideoPipeline"])

    def test_invalid_or_oversized_sources_do_not_supply_symbols(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "invalid.py").write_text("class Broken(", encoding="utf-8")
            (root / "empty.py").write_text("", encoding="utf-8")
            report = inspect_diffusers_generative_sources(
                root, version="test", executable="/test/python"
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["ready_candidates"], [])

    def test_generic_flux2_pipeline_does_not_promote_klein_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "flux2.py").write_text("class Flux2Pipeline: pass\n", encoding="utf-8")
            report = inspect_diffusers_generative_sources(
                root, version="test", executable="/test/python"
            )
        klein = report["candidates"]["flux2-klein-9b-base"]
        self.assertFalse(klein["ready"])
        self.assertEqual(klein["missing_pipeline_classes"], ["Flux2KleinPipeline"])


if __name__ == "__main__":
    unittest.main()
