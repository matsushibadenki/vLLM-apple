import unittest

from vllm_apple.qwen3_vl_attention_coreml import ATTENTION_MAXIMUM_SCALED_ERROR


class Qwen3VLAttentionCoreMLTests(unittest.TestCase):
    def test_fp16_attention_error_gate_is_fixed(self):
        self.assertEqual(ATTENTION_MAXIMUM_SCALED_ERROR, 0.02)


if __name__ == "__main__":
    unittest.main()
