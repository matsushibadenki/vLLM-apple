"""Explicit exporter layout, padding and swizzle adapters for packed numerics."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

MAX_LAYOUT_ELEMENTS = 65_536


@dataclass(frozen=True, slots=True)
class NumericLayoutDescriptor:
    rows: int
    columns: int
    order: str = "contiguous_row_major"
    row_stride: int | None = None
    tile_rows: int | None = None
    tile_columns: int | None = None
    padding_code: int = 0

    def __post_init__(self) -> None:
        if (type(self.rows) is not int or type(self.columns) is not int
                or not 1 <= self.rows * self.columns <= MAX_LAYOUT_ELEMENTS
                or type(self.padding_code) is not int
                or not 0 <= self.padding_code <= 15):
            raise ValueError("invalid numeric layout shape")
        if self.order == "contiguous_row_major":
            valid = self.row_stride is None and self.tile_rows is None and self.tile_columns is None
        elif self.order == "column_major":
            valid = self.row_stride is None and self.tile_rows is None and self.tile_columns is None
        elif self.order == "row_padded":
            valid = (type(self.row_stride) is int and self.row_stride >= self.columns
                     and self.tile_rows is None and self.tile_columns is None)
        elif self.order in {"tile_row_major", "tile_morton"}:
            valid = (
                self.row_stride is None
                and type(self.tile_rows) is int and type(self.tile_columns) is int
                and 1 <= self.tile_rows <= 16 and 1 <= self.tile_columns <= 16
            )
            if self.order == "tile_morton":
                valid = (valid and self.tile_rows == self.tile_columns
                         and _power_of_two(self.tile_rows)
                         and _power_of_two(self.tile_columns))
        else:
            valid = False
        if not valid or self.storage_elements > MAX_LAYOUT_ELEMENTS * 4:
            raise ValueError("unsupported numeric layout")

    @property
    def logical_elements(self) -> int:
        return self.rows * self.columns

    @property
    def storage_elements(self) -> int:
        if self.order == "row_padded":
            assert self.row_stride is not None
            return self.rows * self.row_stride
        if self.order.startswith("tile_"):
            assert self.tile_rows is not None and self.tile_columns is not None
            return (math.ceil(self.rows / self.tile_rows)
                    * math.ceil(self.columns / self.tile_columns)
                    * self.tile_rows * self.tile_columns)
        return self.logical_elements

    def storage_index(self, row: int, column: int) -> int:
        if (type(row) is not int or type(column) is not int
                or not 0 <= row < self.rows or not 0 <= column < self.columns):
            raise ValueError("numeric layout coordinate is out of range")
        if self.order == "contiguous_row_major":
            return row * self.columns + column
        if self.order == "column_major":
            return column * self.rows + row
        if self.order == "row_padded":
            assert self.row_stride is not None
            return row * self.row_stride + column
        assert self.tile_rows is not None and self.tile_columns is not None
        tile_column_count = math.ceil(self.columns / self.tile_columns)
        tile_row, inner_row = divmod(row, self.tile_rows)
        tile_column, inner_column = divmod(column, self.tile_columns)
        tile_offset = (
            tile_row * tile_column_count + tile_column
        ) * self.tile_rows * self.tile_columns
        if self.order == "tile_row_major":
            return tile_offset + inner_row * self.tile_columns + inner_column
        return tile_offset + _morton(inner_row, inner_column)

    @property
    def layout_id(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class NumericExporterLayoutAdapter:
    exporter_id: str
    recipe_id: str
    value_layout: NumericLayoutDescriptor
    scale_layout: NumericLayoutDescriptor
    nibble_order: str

    def __post_init__(self) -> None:
        if (not _identifier(self.exporter_id) or not _identifier(self.recipe_id)
                or self.nibble_order not in {"low_nibble_first", "high_nibble_first"}):
            raise ValueError("invalid numeric exporter layout adapter")

    @property
    def adapter_id(self) -> str:
        value = asdict(self)
        return hashlib.sha256(_canonical(value)).hexdigest()

    def repack_values(self, packed: bytes) -> bytes:
        return repack_packed_nibbles(packed, self.value_layout, self.nibble_order)

    def repack_scales(self, scales: bytes) -> bytes:
        return repack_bytes(scales, self.scale_layout)


class NumericExporterLayoutRegistry:
    def __init__(self, adapters: tuple[NumericExporterLayoutAdapter, ...]) -> None:
        if (not adapters or len(adapters) > 256
                or len({(item.exporter_id, item.recipe_id) for item in adapters})
                != len(adapters)):
            raise ValueError("invalid numeric exporter layout registry")
        self._adapters = {
            (item.exporter_id, item.recipe_id): item for item in adapters
        }

    def resolve(self, exporter_id: str, recipe_id: str) -> NumericExporterLayoutAdapter:
        adapter = self._adapters.get((exporter_id, recipe_id))
        if adapter is None:
            raise ValueError("unsupported numeric exporter layout")
        return adapter


def repack_packed_nibbles(
    packed: bytes, layout: NumericLayoutDescriptor, nibble_order: str
) -> bytes:
    if (not isinstance(packed, bytes)
            or len(packed) != (layout.storage_elements + 1) // 2
            or nibble_order not in {"low_nibble_first", "high_nibble_first"}):
        raise ValueError("numeric packed layout payload is invalid")
    storage_codes = tuple(_nibble(packed, index, nibble_order)
                          for index in range(layout.storage_elements))
    if (layout.storage_elements % 2
            and _nibble(packed, layout.storage_elements, nibble_order)
            != layout.padding_code):
        raise ValueError("numeric layout trailing padding code mismatch")
    logical_indices = {
        layout.storage_index(row, column)
        for row in range(layout.rows) for column in range(layout.columns)
    }
    if any(storage_codes[index] != layout.padding_code
           for index in range(layout.storage_elements) if index not in logical_indices):
        raise ValueError("numeric layout padding code mismatch")
    logical = [
        storage_codes[layout.storage_index(row, column)]
        for row in range(layout.rows) for column in range(layout.columns)
    ]
    output = bytearray((len(logical) + 1) // 2)
    for index, code in enumerate(logical):
        output[index // 2] |= code << (4 * (index % 2))
    return bytes(output)


def repack_bytes(payload: bytes, layout: NumericLayoutDescriptor) -> bytes:
    if not isinstance(payload, bytes) or len(payload) != layout.storage_elements:
        raise ValueError("numeric byte layout payload is invalid")
    logical_indices = {
        layout.storage_index(row, column)
        for row in range(layout.rows) for column in range(layout.columns)
    }
    if any(payload[index] != layout.padding_code
           for index in range(layout.storage_elements) if index not in logical_indices):
        raise ValueError("numeric layout padding code mismatch")
    return bytes(
        payload[layout.storage_index(row, column)]
        for row in range(layout.rows) for column in range(layout.columns)
    )


def _nibble(payload: bytes, index: int, order: str) -> int:
    shift = 4 * (index % 2) if order == "low_nibble_first" else 4 * (1 - index % 2)
    return (payload[index // 2] >> shift) & 15


def _morton(row: int, column: int) -> int:
    result = 0
    for bit in range(4):
        result |= ((column >> bit) & 1) << (2 * bit)
        result |= ((row >> bit) & 1) << (2 * bit + 1)
    return result


def _power_of_two(value: int | None) -> bool:
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
