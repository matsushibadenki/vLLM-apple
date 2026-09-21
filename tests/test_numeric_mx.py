from __future__ import annotations

import math
import unittest

from vllm_apple.numeric_mx import (
    MXElementFormat,
    MXFormatAdapter,
    convert_mx_to_int8,
    decode_mx,
)


def _pack(codes: tuple[int, ...], bits: int) -> bytes:
    accumulator = 0
    available = 0
    output = bytearray()
    for code in codes:
        accumulator |= code << available
        available += bits
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator)
    return bytes(output)


class MXFormatAdapterTests(unittest.TestCase):
    def test_fp4_e2m1_values_and_e8m0_scale(self) -> None:
        adapter = MXFormatAdapter(MXElementFormat.FP4_E2M1, 4)
        self.assertEqual(
            decode_mx(adapter, _pack((0, 1, 7, 15), 4), bytes((128,))),
            (0.0, 1.0, 12.0, -12.0),
        )

    def test_all_published_element_encodings_are_finite_or_rejected_specials(self) -> None:
        for element_format in MXElementFormat:
            adapter = MXFormatAdapter(element_format, 1)
            special_count = 0
            for code in range(1 << adapter.element_bits):
                try:
                    value = decode_mx(adapter, _pack((code,), adapter.element_bits), b"\x7f")[0]
                except ValueError as error:
                    self.assertIn("NaN", str(error))
                    special_count += 1
                else:
                    self.assertTrue(math.isfinite(value))
            if element_format is MXElementFormat.FP8_E5M2:
                self.assertEqual(special_count, 8)
            elif element_format is MXElementFormat.FP8_E4M3:
                self.assertEqual(special_count, 2)
            else:
                self.assertEqual(special_count, 0)

    def test_rejects_nan_scale_and_nonzero_padding(self) -> None:
        adapter = MXFormatAdapter(MXElementFormat.FP6_E2M3, 1)
        with self.assertRaisesRegex(ValueError, "NaN scale"):
            decode_mx(adapter, b"\x00", b"\xff")
        with self.assertRaisesRegex(ValueError, "unused packing bits"):
            decode_mx(adapter, b"\xc0", b"\x7f")

    def test_mxfp6_bitstream_crosses_byte_boundaries(self) -> None:
        codes = tuple(range(16))
        adapter = MXFormatAdapter(MXElementFormat.FP6_E3M2, len(codes))
        values = decode_mx(adapter, _pack(codes, 6), b"\x7f")
        self.assertEqual(len(values), len(codes))
        self.assertEqual(values[0], 0.0)
        self.assertGreater(values[-1], values[1])

    def test_converts_each_mx_family_through_shared_symmetric_int8(self) -> None:
        for element_format in (
            MXElementFormat.FP4_E2M1,
            MXElementFormat.FP6_E2M3,
            MXElementFormat.FP8_E4M3,
        ):
            adapter = MXFormatAdapter(element_format, 32)
            codes = tuple(1 for _ in range(32))
            converted = convert_mx_to_int8(
                adapter,
                _pack(codes, adapter.element_bits),
                b"\x7f",
            )
            self.assertEqual(converted.elements, 32)
            self.assertEqual(converted.values(), decode_mx(adapter, _pack(codes, adapter.element_bits), b"\x7f"))


if __name__ == "__main__":
    unittest.main()
