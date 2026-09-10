import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.schema_validator import validate_instance
from vllm_apple.generative_qualification import (
    GENERATIVE_CANDIDATES,
    GenerativeArtifactComponent,
    GenerativeBaselineEvidence,
    build_generative_qualification_plan,
    generative_plan_sha256,
    generative_promotion_chain_sha256,
    parse_generative_component,
    promote_generative_chained_resolution_plan,
    promote_generative_resolution_plan,
    promote_generative_sample_count_plan,
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
        self.assertEqual(len(GENERATIVE_CANDIDATES), 11)
        self.assertIn("z-image-turbo-mlx-4bit", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base-low-cache", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base-blockwise", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base-attention-chunked", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-klein-9b-base-mlp-chunked", GENERATIVE_CANDIDATES)
        self.assertIn("qwen-image-2512", GENERATIVE_CANDIDATES)
        self.assertIn("flux2-dev", GENERATIVE_CANDIDATES)
        self.assertIn("wan2.2-ti2v-5b", GENERATIVE_CANDIDATES)
        self.assertIn("hunyuanvideo-1.5-8.3b", GENERATIVE_CANDIDATES)
        self.assertIn("wan2.2-a14b-quantized", GENERATIVE_CANDIDATES)

    def test_low_cache_profile_has_a_distinct_plan_identity(self) -> None:
        with TemporaryDirectory() as directory:
            arguments = {
                "artifact_bytes": 8 * GIB,
                "estimated_resident_bytes": 18 * GIB,
                "hardware": hardware(),
                "target": Path(directory),
                "quantization": "int4",
                "components": components(8, 18),
                "steps": 20,
            }
            normal = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base", **arguments
            )
            low_cache = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base-low-cache", **arguments
            )
        self.assertNotEqual(generative_plan_sha256(normal), generative_plan_sha256(low_cache))
        self.assertIn("mlx-cache-limit-250mb", low_cache.candidate.required_strategies)

    def test_blockwise_profile_has_a_distinct_plan_identity(self) -> None:
        with TemporaryDirectory() as directory:
            arguments = {
                "artifact_bytes": 8 * GIB,
                "estimated_resident_bytes": 18 * GIB,
                "hardware": hardware(),
                "target": Path(directory),
                "quantization": "int4",
                "components": components(8, 18),
                "steps": 20,
            }
            normal = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base", **arguments
            )
            blockwise = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base-blockwise", **arguments
            )
        self.assertNotEqual(generative_plan_sha256(normal), generative_plan_sha256(blockwise))
        self.assertIn(
            "transformer-block-materialization", blockwise.candidate.required_strategies
        )

    def test_attention_chunk_profile_has_a_distinct_plan_identity(self) -> None:
        with TemporaryDirectory() as directory:
            arguments = {
                "artifact_bytes": 8 * GIB,
                "estimated_resident_bytes": 18 * GIB,
                "hardware": hardware(),
                "target": Path(directory),
                "quantization": "int4",
                "components": components(8, 18),
                "steps": 20,
            }
            normal = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base", **arguments
            )
            chunked = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base-attention-chunked", **arguments
            )
        self.assertNotEqual(generative_plan_sha256(normal), generative_plan_sha256(chunked))
        self.assertIn("attention-query-chunk-512", chunked.candidate.required_strategies)

    def test_mlp_chunk_profile_has_a_distinct_plan_identity(self) -> None:
        with TemporaryDirectory() as directory:
            arguments = {
                "artifact_bytes": 8 * GIB,
                "estimated_resident_bytes": 18 * GIB,
                "hardware": hardware(),
                "target": Path(directory),
                "quantization": "int4",
                "components": components(8, 18),
                "steps": 20,
            }
            normal = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base", **arguments
            )
            chunked = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base-mlp-chunked", **arguments
            )
        self.assertNotEqual(generative_plan_sha256(normal), generative_plan_sha256(chunked))
        self.assertIn("qkv-mlp-sequence-chunk-512", chunked.candidate.required_strategies)

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

        with self.assertRaisesRegex(ValueError, "single-axis"):
            promote_generative_resolution_plan(
                plan,
                baseline_candidate_id="hunyuanvideo-1.5-8.3b",
                baseline_plan_sha256="a" * 64,
                baseline_sample_count=2,
                baseline_width=640,
                baseline_height=360,
                baseline_frames=33,
                baseline_memory_pressures=("normal", "normal"),
            )

    def test_resolution_only_plan_can_be_promoted_by_a_bound_baseline(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="int4",
                components=components(8, 18),
                width=768,
                height=768,
                steps=20,
            )
        promoted = promote_generative_resolution_plan(
            plan,
            baseline_candidate_id="flux2-klein-9b-base",
            baseline_plan_sha256="a" * 64,
            baseline_sample_count=2,
            baseline_width=512,
            baseline_height=512,
            baseline_frames=1,
            baseline_memory_pressures=("normal", "normal"),
        )
        self.assertTrue(promoted.eligible)
        self.assertEqual(promoted.promotion_axis, "resolution")
        self.assertEqual(promoted.baseline_plan_sha256, "a" * 64)

        with self.assertRaisesRegex(ValueError, "all-normal"):
            promote_generative_resolution_plan(
                plan,
                baseline_candidate_id="flux2-klein-9b-base",
                baseline_plan_sha256="a" * 64,
                baseline_sample_count=2,
                baseline_width=512,
                baseline_height=512,
                baseline_frames=1,
                baseline_memory_pressures=("normal", "warning"),
            )

    def test_same_workload_can_promote_from_two_to_four_samples(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="int4",
                components=components(8, 18),
                width=768,
                height=768,
                steps=20,
            )
        promoted = promote_generative_sample_count_plan(
            plan,
            baseline_candidate_id="flux2-klein-9b-base",
            baseline_plan_sha256="b" * 64,
            baseline_sample_count=2,
            baseline_width=768,
            baseline_height=768,
            baseline_frames=1,
            baseline_memory_pressures=("normal", "normal"),
            target_sample_count=4,
        )
        self.assertTrue(promoted.eligible)
        self.assertEqual(promoted.promotion_axis, "sample_count_4")

        with self.assertRaisesRegex(ValueError, "identity"):
            promote_generative_sample_count_plan(
                plan,
                baseline_candidate_id="flux2-klein-9b-base",
                baseline_plan_sha256="not-a-hash",
                baseline_sample_count=2,
                baseline_width=768,
                baseline_height=768,
                baseline_frames=1,
                baseline_memory_pressures=("normal", "normal"),
                target_sample_count=4,
            )

        with self.assertRaisesRegex(ValueError, "all-normal"):
            promote_generative_sample_count_plan(
                plan,
                baseline_candidate_id="flux2-klein-9b-base",
                baseline_plan_sha256="b" * 64,
                baseline_sample_count=2,
                baseline_width=768,
                baseline_height=768,
                baseline_frames=1,
                baseline_memory_pressures=("normal", "warning"),
                target_sample_count=4,
            )

    def test_second_resolution_promotion_verifies_the_complete_plan_chain(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "candidate_id": "flux2-klein-9b-base",
                "artifact_bytes": 8 * GIB,
                "estimated_resident_bytes": 18 * GIB,
                "hardware": hardware(),
                "target": root,
                "quantization": "int4",
                "components": components(8, 18),
                "steps": 20,
            }
            initial = build_generative_qualification_plan(**common, width=512, height=512)
            intermediate = build_generative_qualification_plan(
                **common, width=768, height=768
            )
            target = build_generative_qualification_plan(**common, width=1024, height=1024)

        initial_evidence = GenerativeBaselineEvidence(
            "flux2-klein-9b-base",
            generative_plan_sha256(initial),
            2,
            512,
            512,
            1,
            ("normal", "normal"),
        )
        resolution_plan = promote_generative_resolution_plan(
            intermediate,
            baseline_candidate_id=initial_evidence.candidate_id,
            baseline_plan_sha256=initial_evidence.plan_sha256,
            baseline_sample_count=initial_evidence.sample_count,
            baseline_width=initial_evidence.width,
            baseline_height=initial_evidence.height,
            baseline_frames=initial_evidence.frames,
            baseline_memory_pressures=initial_evidence.memory_pressures,
        )
        resolution_evidence = GenerativeBaselineEvidence(
            "flux2-klein-9b-base",
            generative_plan_sha256(resolution_plan),
            2,
            768,
            768,
            1,
            ("normal", "normal"),
        )
        stability_plan = promote_generative_sample_count_plan(
            intermediate,
            baseline_candidate_id=resolution_evidence.candidate_id,
            baseline_plan_sha256=resolution_evidence.plan_sha256,
            baseline_sample_count=resolution_evidence.sample_count,
            baseline_width=resolution_evidence.width,
            baseline_height=resolution_evidence.height,
            baseline_frames=resolution_evidence.frames,
            baseline_memory_pressures=resolution_evidence.memory_pressures,
            target_sample_count=4,
        )
        stability_evidence = GenerativeBaselineEvidence(
            "flux2-klein-9b-base",
            generative_plan_sha256(stability_plan),
            4,
            768,
            768,
            1,
            ("normal",) * 4,
        )

        promoted = promote_generative_chained_resolution_plan(
            target,
            stability_baseline=stability_evidence,
            initial_baseline=initial_evidence,
        )
        self.assertTrue(promoted.eligible)
        self.assertEqual(promoted.promotion_axis, "resolution")
        self.assertEqual(
            promoted.baseline_plan_sha256,
            generative_promotion_chain_sha256(stability_evidence, initial_evidence),
        )

        substituted_root = GenerativeBaselineEvidence(
            initial_evidence.candidate_id,
            "f" * 64,
            initial_evidence.sample_count,
            initial_evidence.width,
            initial_evidence.height,
            initial_evidence.frames,
            initial_evidence.memory_pressures,
        )
        substituted = promote_generative_chained_resolution_plan(
            target,
            stability_baseline=stability_evidence,
            initial_baseline=substituted_root,
        )
        self.assertNotEqual(
            substituted.baseline_plan_sha256, promoted.baseline_plan_sha256
        )

    def test_second_resolution_promotion_requires_all_normal_stability(self) -> None:
        with TemporaryDirectory() as directory:
            target = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware(),
                target=Path(directory),
                quantization="int4",
                components=components(8, 18),
                width=1024,
                height=1024,
                steps=20,
            )

        def evidence(digest, count, size, pressures):
            return GenerativeBaselineEvidence(
                "flux2-klein-9b-base", digest, count, size, size, 1, pressures
            )

        with self.assertRaisesRegex(ValueError, "all-normal"):
            promote_generative_chained_resolution_plan(
                target,
                stability_baseline=evidence(
                    "a" * 64, 4, 768, ("normal", "normal", "normal", "warning")
                ),
                initial_baseline=evidence("c" * 64, 2, 512, ("normal", "normal")),
            )

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
