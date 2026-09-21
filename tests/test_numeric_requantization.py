import unittest

from vllm_apple.numeric_formats import NumericFormatDescriptor, convert_nvfp4_to_int8
from vllm_apple.numeric_requantization import (
    Int8ExecutionCapability,
    compare_nvfp4_int8_routes,
    requantize_symmetric_int8,
)


class NumericRequantizationTests(unittest.TestCase):
    def tensor(self):
        return convert_nvfp4_to_int8(
            NumericFormatDescriptor("nvfp4_e2m1", 16),
            bytes(range(0x10, 0x100, 0x22)), bytes((56,)), 1,
        )

    def test_preserving_route_is_exact_and_requantization_error_is_measured(self):
        comparison = compare_nvfp4_int8_routes(
            self.tensor(), requantization_group_size=16,
            capability=Int8ExecutionCapability(
                "int32", ("nvfp4_block", "tensor")
            ),
        )
        self.assertEqual(comparison.preserving.maximum_absolute_error, 0)
        self.assertTrue(comparison.preserving.compatible)
        self.assertTrue(comparison.requantized.compatible)
        self.assertGreater(comparison.requantized.maximum_absolute_error, 0)
        self.assertEqual(comparison.accumulator, "int32")

    def test_scale_granularity_and_signedness_fail_closed(self):
        comparison = compare_nvfp4_int8_routes(
            self.tensor(), requantization_group_size=4,
            capability=Int8ExecutionCapability("fp32", ("tensor",), signed=False),
        )
        self.assertFalse(comparison.preserving.compatible)
        self.assertEqual(
            comparison.preserving.incompatibility_reason, "signed_int8_unsupported"
        )
        self.assertFalse(comparison.requantized.compatible)

    def test_requantization_is_bounded_and_deterministic(self):
        values = tuple(index / 7 for index in range(-31, 32))
        first = requantize_symmetric_int8(values, group_size=8)
        second = requantize_symmetric_int8(values, group_size=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first.payload), len(values))
        with self.assertRaises(ValueError):
            requantize_symmetric_int8(values, group_size=0)


if __name__ == "__main__":
    unittest.main()
