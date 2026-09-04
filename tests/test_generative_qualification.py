import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.schema_validator import validate_instance
from vllm_apple.generative_qualification import (
    GENERATIVE_CANDIDATES,
    GenerativeArtifactComponent,
    build_generative_qualification_plan,
    parse_generative_component,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def hardware() -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Apple M4",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=10,
        memory=MemoryInfo(32 * GIB, 28 * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


def components(artifact_gib: int, resident_gib: int) -> tuple[GenerativeArtifactComponent, ...]:
    return (
        GenerativeArtifactComponent(
            "transformer", "denoiser", (artifact_gib - 2) * GIB, (resident_gib - 4) * GIB
        ),
        GenerativeArtifactComponent("encoder", "text_encoder", GIB, 2 * GIB),
        GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
    )


class GenerativeQualificationTests(unittest.TestCase):
    def test_catalog_contains_requested_image_and_video_candidates(self) -> None:
        self.assertEqual(len(GENERATIVE_CANDIDATES), 7)
        self.assertIn("z-image-turbo-mlx-4bit", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base", GENERATIVE_CANDIDATES)
        self.assertIn("qwen-image-2512", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-dev", GENERATIVE_CANDIDATES)
        self.assertIn("wan2.2-ti2v-5b", GENERATIVE_CANDIDATES)
        self.assertIn("hunyuanvideo-1.5-8.3b", GENERATIVE_CANDIDATES)
        self.assertIn("wan2.2-a14b-quantized", GENERATIVE_CANDIDATES)

    def test_bounded_quantized_plan_passes_load_before_admission(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="wan2.2-ti2v-5b",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="int4",
                components=components(8, 18),
            )
        self.assertTrue(plan.initial_profile)
        self.assertTrue(plan.eligible)
        schema = json.loads(
            Path("schemas/runtime/generative-qualification-plan-v1.schema.json").read_text()
        )
        validate_instance(plan.to_dict(), schema)

    def test_unquantized_stretch_model_is_rejected_without_loading(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="flux2-dev",
                artifact_bytes=20 * GIB,
                estimated_resident_bytes=20 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="none",
                components=components(20, 20),
            )
        self.assertFalse(plan.eligible)
        self.assertIn("candidate_requires_quantization_on_m4_32gb", plan.issues)

    def test_larger_profile_requires_initial_profile_qualification_first(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="hunyuanvideo-1.5-8.3b",
                artifact_bytes=12 * GIB,
                estimated_resident_bytes=20 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="none",
                components=components(12, 20),
                width=1280,
                height=720,
                frames=97,
            )
        self.assertFalse(plan.initial_profile)
        self.assertFalse(plan.eligible)
        self.assertIn("initial_profile_limits_exceeded", plan.issues)

    def test_component_totals_must_match_aggregate_admission(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="qwen-image-2512",
                artifact_bytes=9 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="int4",
                components=components(8, 18),
            )
        self.assertFalse(plan.component_totals_verified)
        self.assertFalse(plan.eligible)
        self.assertIn("component_totals_mismatch", plan.issues)

    def test_component_parser_is_bounded_and_typed(self) -> None:
        component = parse_generative_component("vae:vae:1024:2048")
        self.assertEqual(component.role, "vae")
        self.assertEqual(component.estimated_resident_bytes, 2048)
        with self.assertRaisesRegex(ValueError, "component must use"):
            parse_generative_component("vae:1024")


if __name__ == "__main__":
    unittest.main()
