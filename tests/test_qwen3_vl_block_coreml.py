import unittest

from vllm_apple.qwen3_vl_block_coreml import BLOCK_MAXIMUM_SCALED_ERROR


class Qwen3VLBlockCoreMLTests(unittest.TestCase):
    def test_complete_block_error_gate_is_fixed(self):
        self.assertEqual(BLOCK_MAXIMUM_SCALED_ERROR, 0.03)


if __name__ == "__main__":
    unittest.main()
