import unittest
from dataclasses import replace

from vllm_apple.numeric_formats import (
    NumericFormatDescriptor, conversion_plan, convert_nvfp4_to_int8, decode_nvfp4,
)


class NumericFormatTests(unittest.TestCase):
    def test_content_binding_and_source_verification(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 16)
        packed, scales = bytes([0x22] * 8), bytes([56])
        converted = convert_nvfp4_to_int8(descriptor, packed, scales, 1)
        converted.verify_source(descriptor, packed, scales, 1.0)
        self.assertEqual(converted, convert_nvfp4_to_int8(descriptor, packed, scales, 1.0))
        for payload, scale, global_scale in (
            (bytes(8), scales, 1), (packed, bytes([57]), 1), (packed, scales, 2)
        ):
            with self.assertRaises(ValueError):
                converted.verify_source(descriptor, payload, scale, global_scale)
        for change in (
            {"payload": bytes(16)}, {"payload": bytes(15)},
            {"block_scales": bytes([57])}, {"global_scale": 2},
            {"source_digest": "invalid"},
            {"plan": replace(converted.plan, adapter="unknown")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(converted, **change)
        forged = replace(converted, source_digest="0" * 64)
        with self.assertRaises(ValueError):
            forged.verify_source(descriptor, packed, scales, 1)

    def test_signed_zero_sources_remain_distinguishable(self):
        descriptor = NumericFormatDescriptor("nvfp4_e2m1", 2)
        positive = convert_nvfp4_to_int8(descriptor, bytes([0]), bytes([56]), 1)
        negative = convert_nvfp4_to_int8(descriptor, bytes([0x88]), bytes([56]), 1)
        self.assertEqual(positive.target_digest, negative.target_digest)
        self.assertNotEqual(positive.source_digest, negative.source_digest)
        with self.assertRaises(ValueError):
            positive.verify_source(descriptor, bytes([0x88]), bytes([56]), 1)

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
