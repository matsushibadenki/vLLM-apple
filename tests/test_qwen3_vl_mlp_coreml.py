import unittest

from vllm_apple.qwen3_vl_mlp_coreml import MLP_MAXIMUM_SCALED_ERROR


class Qwen3VLMLPCoreMLTests(unittest.TestCase):
    def test_fp16_backend_error_gate_is_fixed(self):
        self.assertEqual(MLP_MAXIMUM_SCALED_ERROR, 0.01)


if __name__ == "__main__":
    unittest.main()
