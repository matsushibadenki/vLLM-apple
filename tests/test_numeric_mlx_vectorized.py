from __future__ import annotations

import os
import unittest

from vllm_apple.numeric_codecs import _pack, decode_fp8, encode_nf4
from vllm_apple.numeric_layout import (
    NumericLayoutDescriptor,
    repack_bytes,
    repack_packed_nibbles,
)
from vllm_apple.numeric_mlx_vectorized import (
    mlx_decode_fp8,
    mlx_encode_nf4,
    mlx_repack,
    mlx_repack_bytes,
    mlx_repack_packed_nibbles,
    mlx_requantize_symmetric_int8,
    mlx_unpack,
)
from vllm_apple.numeric_requantization import requantize_symmetric_int8


@unittest.skipUnless(os.environ.get("VLLM_APPLE_TEST_MLX_NUMERIC") == "1", "opt-in MLX test")
class MLXVectorizedNumericTests(unittest.TestCase):
    def test_packing_and_fp8_match_scalar(self) -> None:
        for bits in (2, 4, 8):
            codes = tuple(index & ((1 << bits) - 1) for index in range(257))
            payload = _pack(codes, bits)
            self.assertEqual(mlx_unpack(payload, bits, len(codes)), codes)
            self.assertEqual(mlx_repack(payload, bits, bits, len(codes)), payload)
        for variant in ("e4m3fn", "e5m2"):
            payload = bytes(code for code in range(256) if self._finite(code, variant))
            self.assertEqual(mlx_decode_fp8(payload, variant), decode_fp8(payload, variant))

    def test_nf4_matches_scalar(self) -> None:
        values = tuple(((index * 37) % 257 - 128) / 128 for index in range(4097))
        self.assertEqual(mlx_encode_nf4(values), encode_nf4(values))

    def test_symmetric_int8_requantization_matches_scalar(self) -> None:
        values = tuple(((index * 37) % 257 - 128) / 17 for index in range(4097))
        for group_size in (1, 32, 128, 4097):
            actual = mlx_requantize_symmetric_int8(values, group_size=group_size)
            expected = requantize_symmetric_int8(values, group_size=group_size)
            self.assertEqual(actual.payload, expected.payload)
            self.assertEqual(actual.scales, expected.scales)
            self.assertEqual(actual.output_sha256, expected.output_sha256)

    def test_layouts_match_scalar(self) -> None:
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
            self.assertEqual(mlx_repack_bytes(byte_payload, layout), repack_bytes(byte_payload, layout))
            for order in ("low_nibble_first", "high_nibble_first"):
                packed = bytearray((len(storage) + 1) // 2)
                for index, code in enumerate(storage):
                    shift = 4 * (index % 2) if order == "low_nibble_first" else 4 * (1 - index % 2)
                    packed[index // 2] |= code << shift
                if len(storage) % 2:
                    packed[-1] |= layout.padding_code << (4 if order == "low_nibble_first" else 0)
                self.assertEqual(
                    mlx_repack_packed_nibbles(bytes(packed), layout, order),
                    repack_packed_nibbles(bytes(packed), layout, order),
                )

    @staticmethod
    def _finite(code: int, variant: str) -> bool:
        try:
            decode_fp8(bytes((code,)), variant)
        except ValueError:
            return False
        return True


if __name__ == "__main__":
    unittest.main()
