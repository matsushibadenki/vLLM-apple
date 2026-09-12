"""Scalar reference precision policy for the F32-mediated correctness bridge."""
import math
import struct
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumericPrecisionPolicy:
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    allow_underflow: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported precision policy version")
        for value in (self.absolute_tolerance, self.relative_tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("precision tolerance must be finite and nonnegative")
        if type(self.allow_underflow) is not bool:
            raise ValueError("allow_underflow must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "absolute_tolerance": float(self.absolute_tolerance) or 0.0,
            "relative_tolerance": float(self.relative_tolerance) or 0.0,
            "allow_underflow": self.allow_underflow,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NumericPrecisionPolicy":
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "absolute_tolerance", "relative_tolerance", "allow_underflow"
        }:
            raise ValueError("invalid precision policy fields")
        return cls(**value)

    @property
    def policy_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()

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


@dataclass(frozen=True, slots=True)
class PrecisionExecutionContract:
    """Local diagnostic identity; not an execution authorization or signature."""

    tensor_digest: str
    target_dtype: str
    policy: NumericPrecisionPolicy
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported precision contract version")
        if (not isinstance(self.tensor_digest, str) or len(self.tensor_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.tensor_digest)):
            raise ValueError("invalid tensor digest")
        if self.target_dtype not in ("F32", "F16", "BF16"):
            raise ValueError("unsupported precision target")
        if not isinstance(self.policy, NumericPrecisionPolicy):
            raise ValueError("invalid precision policy")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "tensor_digest": self.tensor_digest,
                "target_dtype": self.target_dtype, "policy": self.policy.to_dict(),
                "bridge": "scaled_int8_via_f32_rne_v1"}

    @classmethod
    def from_dict(cls, value: object) -> "PrecisionExecutionContract":
        if (not isinstance(value, dict) or set(value) != {
            "schema_version", "tensor_digest", "target_dtype", "policy", "bridge"
        } or value["bridge"] != "scaled_int8_via_f32_rne_v1"):
            raise ValueError("invalid precision contract fields")
        return cls(value["tensor_digest"], value["target_dtype"],
                   NumericPrecisionPolicy.from_dict(value["policy"]), value["schema_version"])

    @property
    def contract_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
