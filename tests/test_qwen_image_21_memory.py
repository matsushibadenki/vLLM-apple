import unittest

from vllm_apple.qwen_image_21_memory import (
    QWEN_IMAGE_21_PHASE_MARGIN_BYTES,
    estimate_qwen_image_21_resident_bytes,
)


class QwenImage21MemoryTests(unittest.TestCase):
    def test_estimate_preserves_complete_weight_floor_in_unified_memory(self) -> None:
        artifact = {
            "artifact_bytes": 32_000,
            "components": [
                {"role": "denoiser", "artifact_bytes": 14_000},
                {"role": "text_encoder", "artifact_bytes": 17_000},
                {"role": "vae", "artifact_bytes": 1_000},
            ]
        }
        self.assertEqual(
            estimate_qwen_image_21_resident_bytes(artifact, width=512, height=512),
            32_000 + QWEN_IMAGE_21_PHASE_MARGIN_BYTES + 512 * 512 * 16,
        )

    def test_estimate_rejects_missing_phase(self) -> None:
        artifact = {
            "artifact_bytes": 15_000,
            "components": [
                {"role": "denoiser", "artifact_bytes": 14_000},
                {"role": "vae", "artifact_bytes": 1_000},
            ]
        }
        with self.assertRaisesRegex(ValueError, "text_encoder"):
            estimate_qwen_image_21_resident_bytes(artifact, width=512, height=512)

    def test_component_staged_estimate_uses_larger_disjoint_load_phase(self) -> None:
        artifact = {
            "artifact_bytes": 32_000,
            "components": [
                {"role": "denoiser", "artifact_bytes": 14_000},
                {"role": "text_encoder", "artifact_bytes": 15_000},
                {"role": "vae", "artifact_bytes": 1_000},
            ],
        }
        self.assertEqual(
            estimate_qwen_image_21_resident_bytes(
                artifact, width=512, height=512, component_staged=True
            ),
            25_500 + QWEN_IMAGE_21_PHASE_MARGIN_BYTES + 512 * 512 * 16,
        )

    def test_component_staged_estimate_rejects_overlapping_components(self) -> None:
        artifact = {
            "artifact_bytes": 10,
            "components": [
                {"role": "denoiser", "artifact_bytes": 5},
                {"role": "text_encoder", "artifact_bytes": 5},
                {"role": "vae", "artifact_bytes": 5},
            ],
        }
        with self.assertRaisesRegex(ValueError, "exceed"):
            estimate_qwen_image_21_resident_bytes(
                artifact, width=1, height=1, component_staged=True
            )

    def test_estimate_rejects_non_unit_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            estimate_qwen_image_21_resident_bytes(
                {"artifact_bytes": 1, "components": []},
                width=512,
                height=512,
                batch_size=2,
            )
