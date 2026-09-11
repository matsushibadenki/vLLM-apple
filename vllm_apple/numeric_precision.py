"""Scalar reference precision policy for the F32-mediated correctness bridge."""
import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumericPrecisionPolicy:
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    allow_underflow: bool = False

    def __post_init__(self) -> None:
        for value in (self.absolute_tolerance, self.relative_tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("precision tolerance must be finite and nonnegative")
        if type(self.allow_underflow) is not bool:
            raise ValueError("allow_underflow must be boolean")

    def checked_bytes(self, source: float, target_dtype: str) -> bytes:
        """Round via F32, enforce error against the original decoded scalar."""
        if target_dtype not in ("F32", "F16", "BF16"):
            raise ValueError("unsupported precision target")
        if not math.isfinite(source):
            raise ValueError("nonfinite precision source")
        try:
            raw = struct.pack("<f", source)
            intermediate = struct.unpack("<f", raw)[0]
            if target_dtype == "F16":
                raw = struct.pack("<e", intermediate)
                rounded = struct.unpack("<e", raw)[0]
            elif target_dtype == "BF16":
                bits = int.from_bytes(raw, "little")
                bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
                raw = struct.pack("<H", bits)
                rounded = struct.unpack("<f", b"\0\0" + raw)[0]
            else:
                rounded = intermediate
        except (OverflowError, struct.error) as error:
            raise ValueError("precision target overflow") from error
        if not math.isfinite(rounded):
            raise ValueError("precision target is nonfinite")
        if source != 0 and rounded == 0 and not self.allow_underflow:
            raise ValueError("precision policy rejects underflow to zero")
        # max(atol, rtol * abs(source)), not an additive tolerance.
        error = abs(rounded - source)
        if error > self.absolute_tolerance and (source == 0 or error / abs(source) > self.relative_tolerance):
            raise ValueError("precision tolerance exceeded")
        return raw
