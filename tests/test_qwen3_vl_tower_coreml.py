import unittest

from vllm_apple.qwen3_vl_tower_coreml import TOWER_MAXIMUM_SCALED_ERROR


class Qwen3VLTowerCoreMLTests(unittest.TestCase):
    def test_tower_error_budget_does_not_grow_with_layer_count(self):
        self.assertEqual(TOWER_MAXIMUM_SCALED_ERROR, 0.03)


if __name__ == "__main__":
    unittest.main()
