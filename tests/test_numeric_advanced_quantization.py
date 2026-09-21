import unittest

from vllm_apple.numeric_advanced_quantization import (
    double_quantize_scales,
    quantize_mixed_precision,
    quantize_with_sparse_residual,
)


class NumericAdvancedQuantizationTests(unittest.TestCase):
    def test_double_quantized_scales_are_bounded(self):
        scales = tuple(0.1 + index / 100 for index in range(32))
        result = double_quantize_scales(scales)
        restored = result.values()
        self.assertEqual(len(restored), len(scales))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(scales, restored)), result.step / 2)
        self.assertEqual(double_quantize_scales((0.5,) * 3).values(), (0.5,) * 3)

    def test_mixed_precision_selects_group_bits(self):
        values = (0.01, 0.02, -0.01, 0.0, 4.0, -3.0, 2.0, 1.0)
        result = quantize_mixed_precision(
            values, group_size=4, high_precision_threshold=1.0,
            low_bits=2, high_bits=8,
        )
        self.assertEqual(tuple(group.tensor.bits for group in result.groups), (2, 8))
        self.assertEqual(len(result.values()), len(values))

    def test_sparse_residual_restores_outliers_on_quantized_base(self):
        values = (0.1, 0.2, 10.0, -0.3, -12.0, 0.4)
        result = quantize_with_sparse_residual(
            values, bits=4, group_size=3, outlier_threshold=1.0
        )
        self.assertEqual(result.indices, (2, 4))
        restored = result.values()
        self.assertAlmostEqual(restored[2], 10.0)
        self.assertAlmostEqual(restored[4], -12.0)
        with self.assertRaises(ValueError):
            quantize_with_sparse_residual(
                values, bits=4, group_size=3, outlier_threshold=0
            )


if __name__ == "__main__":
    unittest.main()
