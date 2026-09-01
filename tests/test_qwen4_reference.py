import math
import unittest

from vllm_apple.qwen4_reference import (
    qwen4_gated_residual_reference,
    qwen4_qsa_select_tokens_reference,
)


class Qwen4ReferenceTests(unittest.TestCase):
    def test_gated_residual_matches_zero_projection_fixture(self) -> None:
        result = qwen4_gated_residual_reference(
            [3.0, 4.0, 0.0, 5.0],
            hc_count=2,
            hidden_size=2,
            norm_weight=[0.0] * 4,
            mix_down_weight=[[0.0] * 4],
            mix_up_weight=[[0.0], [0.0], [0.0], [0.0]],
            inject_weight=[[0.0] * 4, [0.0] * 4],
            eps=1e-12,
        )
        mixed = result["mixed_input"]
        assert mixed is not None
        self.assertTrue(math.isclose(mixed[0], 3.0 / math.sqrt(12.5) / 4.0, abs_tol=1e-10))
        self.assertTrue(
            math.isclose(
                mixed[1],
                (4.0 / math.sqrt(12.5) + 5.0 / math.sqrt(12.5)) / 4.0,
                abs_tol=1e-10,
            )
        )
        self.assertEqual(result["injection_weights"], [1.0, 1.0])
        self.assertEqual(result["hyper_input"], [3.0, 4.0, 0.0, 5.0])

    def test_qsa_selects_highest_complete_block_and_preserves_tail(self) -> None:
        selected = qwen4_qsa_select_tokens_reference(
            [[0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 4.0], [8.0, 0.0]],
            [0, 1, 2, 3, 4],
            compress_ratio=2,
            token_budget=2,
        )
        self.assertEqual(selected, [2, 3, 4])

    def test_qsa_rejects_duplicate_visible_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "visible indices"):
            qwen4_qsa_select_tokens_reference(
                [[1.0]], [[1.0], [2.0]], [0, 0], compress_ratio=1, token_budget=1
            )


if __name__ == "__main__":
    unittest.main()
