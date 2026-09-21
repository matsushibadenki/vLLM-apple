import unittest

from vllm_apple.qwen_image_21_residency import build_qwen_image_21_residency_plan
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def hardware(available_gib: int = 28) -> HardwareInfo:
    return HardwareInfo(
        "Darwin",
        "arm64",
        "Apple M4",
        10,
        10,
        10,
        MemoryInfo(32 * GIB, available_gib * GIB),
        True,
        "test",
    )


def artifact() -> dict[str, object]:
    return {
        "pipeline_class": "QwenImage21Pipeline",
        "components": [
            {"name": "transformer", "role": "denoiser", "artifact_bytes": 14 * GIB},
            {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 17 * GIB},
            {"name": "vae", "role": "vae", "artifact_bytes": GIB},
        ],
    }


class QwenImage21ResidencyTests(unittest.TestCase):
    def test_int8_plan_scales_weight_components_and_stays_fail_closed(self) -> None:
        plan = build_qwen_image_21_residency_plan(artifact(), hardware(), target_bits=8)
        self.assertEqual(plan["projected_weight_bytes"], 31 * GIB // 2 + GIB)
        self.assertTrue(plan["fits_physical_memory"])
        self.assertTrue(plan["fits_current_available_memory"])
        self.assertFalse(plan["runtime_support_verified"])
        self.assertFalse(plan["eligible_for_generation"])
        self.assertEqual(plan["next_gate"], "produce-and-inspect-quantized-artifact")

    def test_int4_plan_does_not_quantize_vae(self) -> None:
        plan = build_qwen_image_21_residency_plan(artifact(), hardware(), target_bits=4)
        self.assertEqual(plan["projected_weight_bytes"], 31 * GIB // 4 + GIB)
        vae = next(item for item in plan["components"] if item["role"] == "vae")
        self.assertEqual(vae["target_bits"], 16)

    def test_wrong_pipeline_is_rejected(self) -> None:
        value = artifact()
        value["pipeline_class"] = "QwenImagePipeline"
        with self.assertRaisesRegex(ValueError, "QwenImage21Pipeline"):
            build_qwen_image_21_residency_plan(value, hardware(), target_bits=8)
