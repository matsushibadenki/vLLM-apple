import math
import unittest

from vllm_apple.numeric_codecs import (
    NF4_CODEBOOK,
    decode_fp8,
    decode_nf4,
    encode_fp8,
    encode_nf4,
    quantize_groupwise_affine,
)


class NumericCodecTests(unittest.TestCase):
    def test_fp8_known_values_roundtrip_and_nonfinite_rejection(self):
        for variant, values in (
            ("e4m3fn", (0.0, -0.0, 1.0, -2.0, 448.0)),
            ("e5m2", (0.0, -0.0, 1.0, -2.0, 57344.0)),
        ):
            with self.subTest(variant=variant):
                payload = encode_fp8(values, variant)
                decoded = decode_fp8(payload, variant)
                self.assertEqual(decoded, values)
                self.assertEqual(math.copysign(1, decoded[1]), -1)
        with self.assertRaisesRegex(ValueError, "NaN"):
            decode_fp8(bytes((0x7F,)), "e4m3fn")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            decode_fp8(bytes((0x7C,)), "e5m2")

    def test_nf4_codebook_and_padding(self):
        payload = encode_nf4(NF4_CODEBOOK)
        self.assertEqual(decode_nf4(payload, 16), NF4_CODEBOOK)
        odd = encode_nf4((0.0, 1.0, -1.0))
        self.assertEqual(decode_nf4(odd, 3), (0.0, 1.0, -1.0))
        with self.assertRaisesRegex(ValueError, "padding"):
            decode_nf4(odd[:-1] + bytes((odd[-1] | 0xF0,)), 3)

    def test_signed_and_unsigned_groupwise_affine_precisions(self):
        values = tuple(index / 7 for index in range(-15, 16))
        for bits in (2, 4, 8):
            for signed in (False, True):
                with self.subTest(bits=bits, signed=signed):
                    tensor = quantize_groupwise_affine(
                        values, bits=bits, signed=signed, group_size=8
                    )
                    self.assertEqual(len(tensor.values()), len(values))
                    self.assertEqual(tensor.bits, bits)
        with self.assertRaises(ValueError):
            quantize_groupwise_affine(values, bits=3, signed=True, group_size=8)


if __name__ == "__main__":
    unittest.main()
