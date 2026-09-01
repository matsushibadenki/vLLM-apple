import unittest
from dataclasses import replace

from vllm_apple.qwen4_cache_contract import (
    advance_qwen4_cache,
    empty_qwen4_cache,
    qwen4_cache_layout,
    run_qwen4_cache_fixture,
    verify_qwen4_cache_state,
)


class Qwen4CacheContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = {
            "model_type": "qwen4_exp",
            "language_model_only": True,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 48,
                "layer_types": [
                    "linear_attention" if (index + 1) % 4 else "full_attention"
                    for index in range(48)
                ],
                "max_position_embeddings": 262144,
                "mtp_num_hidden_layers": 0,
                "ngram_size": 3,
                "linear_conv_kernel_dim": 4,
                "ple_conv_kernel_size": 4,
                "indexer_compress_ratio": 4,
                "indexer_budget": 2048,
                "ple_layer_ids": [2],
            },
        }

    def test_official_layout_has_expected_hybrid_boundaries(self) -> None:
        layout = qwen4_cache_layout(self.config)
        self.assertEqual(layout.linear_layers, 36)
        self.assertEqual(layout.full_attention_layers, 12)
        self.assertEqual(layout.ple_layers, 1)
        self.assertEqual(layout.ngram_context_tokens, 2)
        self.assertEqual(layout.qsa_compress_ratio, 4)
        self.assertEqual(layout.qsa_token_budget, 2048)

    def test_prefill_and_token_decode_have_identical_bounded_state(self) -> None:
        layout = qwen4_cache_layout(self.config)
        prefill = advance_qwen4_cache(layout, empty_qwen4_cache(layout), [10, 20, 30, 40, 50])
        decode = empty_qwen4_cache(layout)
        for token in (10, 20, 30, 40, 50):
            decode = advance_qwen4_cache(layout, decode, [token])
        self.assertEqual(prefill, decode)
        self.assertEqual(prefill.ngram_tail, (40, 50))
        self.assertEqual(prefill.qsa_complete_blocks, 1)
        self.assertEqual(prefill.qsa_tail_tokens, 1)
        self.assertFalse(prefill.to_dict()["stores_tensor_values"])
        verify_qwen4_cache_state(layout, prefill)

    def test_restore_rejects_wrong_config_and_corrupt_boundaries(self) -> None:
        layout = qwen4_cache_layout(self.config)
        state = advance_qwen4_cache(layout, empty_qwen4_cache(layout), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            advance_qwen4_cache(replace(layout, config_fingerprint="0" * 64), state, [4])
        with self.assertRaisesRegex(ValueError, "invariants"):
            verify_qwen4_cache_state(layout, replace(state, qsa_tail_tokens=0))

    def test_static_fixture_compares_three_chunkings_without_storing_tokens(self) -> None:
        result = run_qwen4_cache_fixture(self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["chunkings_compared"], 3)
        self.assertFalse(result["stores_token_ids"])
        self.assertFalse(result["stores_tensor_values"])
        self.assertFalse(result["allocates_model_or_metal"])


if __name__ == "__main__":
    unittest.main()
