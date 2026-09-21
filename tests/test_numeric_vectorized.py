from __future__ import annotations

import unittest

from vllm_apple.numeric_codecs import _pack, decode_fp8, encode_nf4
from vllm_apple.numeric_layout import (
    NumericLayoutDescriptor,
    repack_bytes,
    repack_packed_nibbles,
)
from vllm_apple.numeric_requantization import requantize_symmetric_int8
from vllm_apple.numeric_vectorized import (
    VectorizedNumericUnavailableError,
    vectorized_decode_fp8,
    vectorized_encode_nf4,
    vectorized_repack,
    vectorized_repack_bytes,
    vectorized_repack_packed_nibbles,
    vectorized_requantize_symmetric_int8,
    vectorized_unpack,
)


class VectorizedNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            vectorized_unpack(b"\x00", 2, 1)
        except VectorizedNumericUnavailableError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_unpack_and_repack_match_reference_for_all_widths(self) -> None:
        for bits in (2, 4, 8):
            mask = (1 << bits) - 1
            codes = tuple(index & mask for index in range(257))
            payload = _pack(codes, bits)
            self.assertEqual(vectorized_unpack(payload, bits, len(codes)), codes)
            self.assertEqual(vectorized_repack(payload, bits, bits, len(codes)), payload)

    def test_fp8_decode_matches_reference_for_all_finite_codes(self) -> None:
        for variant in ("e4m3fn", "e5m2"):
            finite = bytearray()
            for code in range(256):
                try:
                    decode_fp8(bytes((code,)), variant)
                except ValueError:
                    continue
                finite.append(code)
            payload = bytes(finite)
            self.assertEqual(vectorized_decode_fp8(payload, variant), decode_fp8(payload, variant))

    def test_nf4_and_requantization_match_scalar_bytes_and_digest(self) -> None:
        values = tuple(((index * 37) % 257 - 128) / 128 for index in range(4097))
        self.assertEqual(vectorized_encode_nf4(values), encode_nf4(values))
        for group_size in (1, 32, 128, len(values)):
            expected = requantize_symmetric_int8(values, group_size=group_size)
            actual = vectorized_requantize_symmetric_int8(values, group_size=group_size)
            self.assertEqual(actual, expected)

    def test_rejects_nonzero_padding_and_narrow_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "padding"):
            vectorized_unpack(b"\xfc", 2, 1)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            vectorized_repack(bytes((0xff,)), 8, 4, 1)

    def test_layout_repack_matches_scalar_for_all_layouts(self) -> None:
        layouts = (
            NumericLayoutDescriptor(7, 9),
            NumericLayoutDescriptor(7, 9, order="column_major"),
            NumericLayoutDescriptor(7, 9, order="row_padded", row_stride=12, padding_code=3),
            NumericLayoutDescriptor(7, 9, order="tile_row_major", tile_rows=4, tile_columns=4, padding_code=3),
            NumericLayoutDescriptor(7, 9, order="tile_morton", tile_rows=4, tile_columns=4, padding_code=3),
        )
        for layout in layouts:
            storage = [layout.padding_code] * layout.storage_elements
            for row in range(layout.rows):
                for column in range(layout.columns):
                    storage[layout.storage_index(row, column)] = (row * layout.columns + column) & 15
            byte_payload = bytes(storage)
            self.assertEqual(
                vectorized_repack_bytes(byte_payload, layout),
                repack_bytes(byte_payload, layout),
            )
            for order in ("low_nibble_first", "high_nibble_first"):
                packed = bytearray((len(storage) + 1) // 2)
                for index, code in enumerate(storage):
                    shift = 4 * (index % 2) if order == "low_nibble_first" else 4 * (1 - index % 2)
                    packed[index // 2] |= code << shift
                if len(storage) % 2:
                    shift = 4 if order == "low_nibble_first" else 0
                    packed[-1] |= layout.padding_code << shift
                self.assertEqual(
                    vectorized_repack_packed_nibbles(bytes(packed), layout, order),
                    repack_packed_nibbles(bytes(packed), layout, order),
                )


if __name__ == "__main__":
    unittest.main()
