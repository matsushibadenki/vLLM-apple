import unittest
from dataclasses import replace

from vllm_apple.numeric_formats import (
    NumericFormatDescriptor, conversion_plan, convert_nvfp4_to_int8, decode_nvfp4,
)


class NumericFormatTests(unittest.TestCase):
    def test_all_codes_and_finite_scale_codes_preserve_numeric_values(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        packed = bytes(range(0x10, 0x100, 0x22))
        expected = (0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6)
        self.assertEqual(decode_nvfp4(descriptor, packed, bytes([56]), 1), expected)
        for scale in range(127):
            with self.subTest(scale=scale):
                converted = convert_nvfp4_to_int8(descriptor, packed, bytes([scale]), .75)
                self.assertEqual(converted.reference_values(), decode_nvfp4(
                    descriptor, packed, bytes([scale]), .75
                ))
                self.assertEqual(len(converted.payload), 16)

    def test_partial_blocks_and_padding(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 17)
        converted = convert_nvfp4_to_int8(descriptor, bytes([0x22] * 8 + [2]), bytes([56, 64]), 1)
        self.assertEqual(converted.reference_values(), (1,) * 16 + (2,))
        with self.assertRaises(ValueError):
            convert_nvfp4_to_int8(descriptor, bytes([0x22] * 9), bytes([56, 64]), 1)

    def test_variant_identity_and_invalid_metadata(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        self.assertEqual(conversion_plan(descriptor).plan_id, conversion_plan(descriptor).plan_id)
        self.assertNotEqual(conversion_plan(descriptor).plan_id,
                            conversion_plan(replace(descriptor, elements=32)).plan_id)
        for change in ({"layout": "swizzled"}, {"block_size": 32}, {"encoding": "mxfp4"}):
            with self.assertRaises(ValueError):
                conversion_plan(replace(descriptor, **change))
        for scale, global_scale in ((127, 1), (128, 1), (56, float("nan")), (56, -1), (126, 1e308)):
            with self.assertRaises(ValueError):
                convert_nvfp4_to_int8(descriptor, bytes(8), bytes([scale]), global_scale)
        for count in (True, 0, 65537):
            with self.assertRaises(ValueError):
                NumericFormatDescriptor("nvfp4_e2m1", count)


if __name__ == "__main__":
    unittest.main()
