import unittest

from vllm_apple.qwen3_vl_deepstack_coreml import (
    DEEPSTACK_MAXIMUM_SCALED_ERROR,
    MAIN_MAXIMUM_SCALED_ERROR,
    _merger_reference,
)
from vllm_apple.qwen3_vl_mlp_coreml import _PREDICTION_PROGRAM


class Qwen3VLDeepStackCoreMLTests(unittest.TestCase):
    def test_main_and_deepstack_error_gates_are_independent(self):
        self.assertEqual(MAIN_MAXIMUM_SCALED_ERROR, 0.03)
        self.assertEqual(DEEPSTACK_MAXIMUM_SCALED_ERROR, 0.04)

    def test_prediction_program_accepts_digest_bound_segment_input(self):
        self.assertIn("CommandLine.arguments.count > 6", _PREDICTION_PROGRAM)
        self.assertIn("inputData.count == input.count", _PREDICTION_PROGRAM)

    def test_final_merger_normalizes_before_shuffle(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is available only in the isolated Core ML toolchain")

        hidden = np.arange(8, dtype=np.float16).reshape(4, 2)
        weights = {
            "norm.weight": np.ones(2, dtype=np.float16),
            "norm.bias": np.zeros(2, dtype=np.float16),
            "linear_fc1.weight": np.eye(4, dtype=np.float16),
            "linear_fc1.bias": np.zeros(4, dtype=np.float16),
            "linear_fc2.weight": np.eye(4, dtype=np.float16),
            "linear_fc2.bias": np.zeros(4, dtype=np.float16),
        }
        output = _merger_reference(
            hidden, weights, np, postshuffle_norm=False
        )
        self.assertEqual(output.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
