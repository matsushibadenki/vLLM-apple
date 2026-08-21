import unittest

from vllm_apple.context import ContextPolicy, recommend_context
from vllm_apple.types import GIB, MemoryInfo, MemoryPressure, ModelMemorySpec


class ContextRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ContextPolicy(
            os_reserve_ratio=0.10,
            minimum_os_reserve_bytes=GIB,
            safety_headroom_ratio=0.05,
            minimum_safety_headroom_bytes=GIB,
            default_workspace_ratio=0.0,
            token_block_size=256,
        )

    def test_uses_current_availability_as_hard_limit(self) -> None:
        memory = MemoryInfo(
            total_bytes=32 * GIB,
            available_bytes=10 * GIB,
            pressure=MemoryPressure.WARNING,
        )
        model = ModelMemorySpec("model", 4 * GIB, GIB // 4096)
        result = recommend_context(memory, model, self.policy)
        expected_safety = int(32 * GIB * 0.05)
        self.assertEqual(result.allocatable_bytes, 10 * GIB - expected_safety)
        self.assertEqual(result.limiting_factor, "current_memory_availability")
        self.assertTrue(all(tier.max_tokens % 256 == 0 for tier in result.tiers))
        self.assertLess(result.tiers[0].max_tokens, result.tiers[1].max_tokens)
        self.assertLess(result.tiers[1].max_tokens, result.tiers[2].max_tokens)

    def test_insufficient_memory_returns_zero_without_underflow(self) -> None:
        memory = MemoryInfo(total_bytes=8 * GIB, available_bytes=2 * GIB)
        model = ModelMemorySpec("large", 7 * GIB, 1024)
        result = recommend_context(memory, model, self.policy)
        self.assertEqual(result.limiting_factor, "insufficient_memory")
        self.assertEqual([tier.max_tokens for tier in result.tiers], [0, 0, 0])

    def test_model_limit_is_respected_and_aligned_down(self) -> None:
        memory = MemoryInfo(total_bytes=64 * GIB, available_bytes=64 * GIB)
        model = ModelMemorySpec("limited", GIB, 1024, model_max_context=10_001)
        result = recommend_context(memory, model, self.policy)
        self.assertEqual(result.tiers[-1].max_tokens, 9_984)
        self.assertEqual(result.limiting_factor, "model_max_context")


if __name__ == "__main__":
    unittest.main()
