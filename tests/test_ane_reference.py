import math
import unittest

from vllm_apple.ane_reference import representative_encoder_input, run_encoder_reference


class ANEReferenceTests(unittest.TestCase):
    def test_encoder_reference_is_deterministic_finite_and_bounded(self):
        values = representative_encoder_input(8)
        first = run_encoder_reference(values, layers=2)
        self.assertEqual(first, run_encoder_reference(values, layers=2))
        self.assertEqual(len(first), 8)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in first))

    def test_encoder_reference_rejects_unbounded_shape(self):
        with self.assertRaisesRegex(ValueError, "bound"):
            representative_encoder_input(1025)
        with self.assertRaisesRegex(ValueError, "bound"):
            run_encoder_reference((1.0,), layers=17)
