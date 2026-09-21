import unittest

from vllm_apple.numeric_layout import (
    NumericExporterLayoutAdapter,
    NumericExporterLayoutRegistry,
    NumericLayoutDescriptor,
    repack_bytes,
    repack_packed_nibbles,
)


def pack(codes):
    result = bytearray((len(codes) + 1) // 2)
    for index, code in enumerate(codes):
        result[index // 2] |= code << (4 * (index % 2))
    return bytes(result)


class NumericLayoutTests(unittest.TestCase):
    def test_column_major_and_row_padding_repack(self):
        column = NumericLayoutDescriptor(2, 3, "column_major")
        self.assertEqual(repack_bytes(bytes((1, 4, 2, 5, 3, 6)), column), bytes(range(1, 7)))
        padded = NumericLayoutDescriptor(2, 3, "row_padded", row_stride=4, padding_code=0)
        self.assertEqual(repack_bytes(bytes((1, 2, 3, 0, 4, 5, 6, 0)), padded), bytes(range(1, 7)))
        with self.assertRaisesRegex(ValueError, "padding"):
            repack_bytes(bytes((1, 2, 3, 9, 4, 5, 6, 0)), padded)

    def test_tiled_morton_and_packed_nibble_repack(self):
        layout = NumericLayoutDescriptor(
            2, 2, "tile_morton", tile_rows=2, tile_columns=2
        )
        # Morton storage coordinates are (0,0), (0,1), (1,0), (1,1).
        self.assertEqual(repack_packed_nibbles(pack((1, 2, 3, 4)), layout,
                                               "low_nibble_first"), pack((1, 2, 3, 4)))
        high_packed = bytes((0x12, 0x34))
        self.assertEqual(repack_packed_nibbles(high_packed, layout,
                                               "high_nibble_first"), pack((1, 2, 3, 4)))
        odd = NumericLayoutDescriptor(1, 3, padding_code=0)
        with self.assertRaisesRegex(ValueError, "trailing padding"):
            repack_packed_nibbles(bytes((0x21, 0xF3)), odd, "low_nibble_first")

    def test_registry_requires_exact_exporter_and_recipe(self):
        layout = NumericLayoutDescriptor(1, 2)
        adapter = NumericExporterLayoutAdapter(
            "vendor.exporter", "nvfp4.v1", layout, layout, "low_nibble_first"
        )
        registry = NumericExporterLayoutRegistry((adapter,))
        self.assertEqual(registry.resolve("vendor.exporter", "nvfp4.v1"), adapter)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            registry.resolve("vendor.exporter", "unknown")
        self.assertEqual(len(adapter.adapter_id), 64)


if __name__ == "__main__":
    unittest.main()
