"""CPU reference contracts; no native execution or performance qualification implied."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

MAX_REFERENCE_ELEMENTS = 65_536
_E2M1_TWICE = (0, 1, 2, 3, 4, 6, 8, 12)


@dataclass(frozen=True, slots=True)
class NumericFormatDescriptor:
    encoding: str
    elements: int
    block_size: int = 16
    packing: str = "low_nibble_first"
    scale_encoding: str = "e4m3fn"
    layout: str = "contiguous_1d"
    value_multiplier: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported descriptor version")
        if type(self.elements) is not int or not 0 < self.elements <= MAX_REFERENCE_ELEMENTS:
            raise ValueError("reference element count must be in 1..65536")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("invalid block size")
        if type(self.value_multiplier) not in (int, float) or not math.isfinite(self.value_multiplier) or self.value_multiplier <= 0:
            raise ValueError("invalid value multiplier")
        for value in (self.encoding, self.packing, self.scale_encoding, self.layout):
            if not isinstance(value, str) or not 0 < len(value) <= 64:
                raise ValueError("invalid format identifier")


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    source: NumericFormatDescriptor
    target: NumericFormatDescriptor
    adapter: str = "nvfp4_int8_cpu_reference_v1"
    schema_version: int = 1

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TensorGeometry:
    """Logical C-order shape and per-axis block scales; no stride/swizzle support.

    Scale storage follows C order with the blocked axis replaced by its ceiling
    block count. Blocks restart at each axis boundary, never across rows.
    """

    shape: tuple[int, ...]
    scale_axis: int
    block_size: int = 16
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported geometry version")
        if (not isinstance(self.shape, tuple) or not 1 <= len(self.shape) <= 8
                or any(type(size) is not int or size <= 0 for size in self.shape)):
            raise ValueError("shape must contain 1..8 positive integer dimensions")
        if math.prod(self.shape) > MAX_REFERENCE_ELEMENTS:
            raise ValueError("geometry exceeds reference element limit")
        if type(self.scale_axis) is not int or not 0 <= self.scale_axis < len(self.shape):
            raise ValueError("scale axis must be a nonnegative in-range integer")
        if type(self.block_size) is not int or not 0 < self.block_size <= MAX_REFERENCE_ELEMENTS:
            raise ValueError("invalid geometry block size")

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def scale_shape(self) -> tuple[int, ...]:
        return tuple((size + self.block_size - 1) // self.block_size
                     if axis == self.scale_axis else size
                     for axis, size in enumerate(self.shape))

    @property
    def scale_count(self) -> int:
        return math.prod(self.scale_shape)

    def scale_index(self, flat_index: int) -> int:
        if type(flat_index) is not int or not 0 <= flat_index < self.elements:
            raise ValueError("element index out of range")
        coordinates = [0] * len(self.shape)
        for axis in range(len(self.shape) - 1, -1, -1):
            flat_index, coordinates[axis] = divmod(flat_index, self.shape[axis])
        coordinates[self.scale_axis] //= self.block_size
        index = 0
        for size, coordinate in zip(self.scale_shape, coordinates):
            index = index * size + coordinate
        return index


@dataclass(frozen=True, slots=True)
class TensorConversionPlan:
    """Geometry-bound planning identity, not permission to flatten or execute."""

    format_plan: ConversionPlan
    geometry: TensorGeometry
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported tensor plan version")
        if not isinstance(self.format_plan, ConversionPlan) or not isinstance(self.geometry, TensorGeometry):
            raise ValueError("invalid tensor plan metadata")
        for descriptor in (self.format_plan.source, self.format_plan.target):
            if (descriptor.elements != self.geometry.elements
                    or descriptor.block_size != self.geometry.block_size):
                raise ValueError("format and geometry mismatch")

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ConversionAdapter:
    """Exact format templates for planning; registration grants no execution capability.

    Template element counts must be one. All other descriptor fields match exactly.
    """

    adapter_id: str
    source: NumericFormatDescriptor
    target: NumericFormatDescriptor

    def __post_init__(self) -> None:
        if (not isinstance(self.adapter_id, str) or not 0 < len(self.adapter_id) <= 128
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.adapter_id)):
            raise ValueError("invalid adapter identifier")
        for template in (self.source, self.target):
            if not isinstance(template, NumericFormatDescriptor) or template.elements != 1:
                raise ValueError("adapter templates must have one element")


@dataclass(frozen=True, slots=True)
class ConversionRegistry:
    """Immutable, explicitly supplied planning routes; never loads plugin code."""

    adapters: tuple[ConversionAdapter, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.adapters, tuple) or len(self.adapters) > 256:
            raise ValueError("registry requires at most 256 immutable adapter entries")
        if any(not isinstance(item, ConversionAdapter) for item in self.adapters):
            raise ValueError("invalid adapter entry")
        if len({item.adapter_id for item in self.adapters}) != len(self.adapters):
            raise ValueError("duplicate adapter identifier")

    def register(self, adapter: ConversionAdapter) -> ConversionRegistry:
        return ConversionRegistry(self.adapters + (adapter,))

    def plan_tensor(self, source: NumericFormatDescriptor, geometry: TensorGeometry, *,
                    target: NumericFormatDescriptor | None = None,
                    adapter_id: str | None = None) -> TensorConversionPlan:
        return TensorConversionPlan(self.plan(source, target=target, adapter_id=adapter_id), geometry)

    def plan(self, source: NumericFormatDescriptor, *,
             target: NumericFormatDescriptor | None = None,
             adapter_id: str | None = None) -> ConversionPlan:
        if not isinstance(source, NumericFormatDescriptor):
            raise ValueError("invalid source descriptor")
        if target is not None and not isinstance(target, NumericFormatDescriptor):
            raise ValueError("invalid target descriptor")
        candidates = [
            ConversionPlan(source, replace(item.target, elements=source.elements), item.adapter_id)
            for item in self.adapters
            if replace(source, elements=1) == item.source
            and (adapter_id is None or adapter_id == item.adapter_id)
        ]
        if target is not None:
            candidates = [plan for plan in candidates if plan.target == target]
        if not candidates:
            raise ValueError("unsupported numeric format route")
        if len(candidates) != 1:
            raise ValueError("ambiguous numeric format route; specify target or adapter_id")
        return candidates[0]


DEFAULT_CONVERSION_REGISTRY = ConversionRegistry((ConversionAdapter(
    "nvfp4_int8_cpu_reference_v1", NumericFormatDescriptor("nvfp4_e2m1", 1),
    NumericFormatDescriptor("scaled_int8", 1, packing="signed_byte", value_multiplier=0.5),
),))


def conversion_plan(source: NumericFormatDescriptor) -> ConversionPlan:
    """Plan the built-in CPU reference route, preserving the existing API and ID."""
    return DEFAULT_CONVERSION_REGISTRY.plan(source)


def _scale(code: int) -> float:
    if code >= 127:
        raise ValueError("block scale must be nonnegative finite E4M3FN")
    exponent, mantissa = code >> 3, code & 7
    return math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)


def _validate_inputs(descriptor, packed, scales, global_scale, geometry=None):
    plan = (conversion_plan(descriptor) if geometry is None else
            DEFAULT_CONVERSION_REGISTRY.plan_tensor(descriptor, geometry))
    if not isinstance(packed, bytes) or len(packed) != (descriptor.elements + 1) // 2:
        raise ValueError("packed payload size mismatch")
    scale_count = ((descriptor.elements + 15) // 16 if geometry is None
                   else geometry.scale_count)
    if not isinstance(scales, bytes) or len(scales) != scale_count:
        raise ValueError("block scale count mismatch")
    if type(global_scale) not in (float, int) or not math.isfinite(global_scale) or global_scale < 0:
        raise ValueError("invalid global scale")
    for code in scales:
        if not math.isfinite(_scale(code) * global_scale * 6):
            raise ValueError("reference scale overflow")
    if descriptor.elements % 2 and packed[-1] >> 4:
        raise ValueError("unused padding nibble must be zero")
    return plan


def _codes(packed: bytes, elements: int):
    for index in range(elements):
        yield (packed[index // 2] >> (4 * (index % 2))) & 15


def decode_nvfp4(
    descriptor: NumericFormatDescriptor, packed: bytes, scales: bytes, global_scale: float,
    *, geometry: TensorGeometry | None = None,
) -> tuple[float, ...]:
    _validate_inputs(descriptor, packed, scales, global_scale, geometry)
    return tuple(
        math.copysign(_E2M1_TWICE[code & 7] / 2, -1 if code & 8 else 1)
        * _scale(scales[index // 16 if geometry is None else geometry.scale_index(index)]) * global_scale
        for index, code in enumerate(_codes(packed, descriptor.elements))
    )


def _content_digest(plan, payload, scales, global_scale, role):
    # Versioned, framed metadata; keep scale sign (including negative zero).
    metadata = {
        "version": 1, "role": role, "plan_id": plan.plan_id,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "scales_sha256": hashlib.sha256(scales).hexdigest(),
        "global_scale": float(global_scale).hex(),
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ScaledInt8Tensor:
    plan: ConversionPlan | TensorConversionPlan
    payload: bytes
    block_scales: bytes
    global_scale: float
    source_digest: str
    target_digest: str

    def __post_init__(self) -> None:
        format_plan = self.plan.format_plan if isinstance(self.plan, TensorConversionPlan) else self.plan
        if not isinstance(format_plan, ConversionPlan) or format_plan != conversion_plan(format_plan.source):
            raise ValueError("unsupported conversion plan")
        if not isinstance(self.payload, bytes) or len(self.payload) != format_plan.target.elements:
            raise ValueError("INT8 payload size mismatch")
        allowed = {value & 255 for magnitude in _E2M1_TWICE for value in (magnitude, -magnitude)}
        if any(value not in allowed for value in self.payload):
            raise ValueError("invalid reference INT8 value")
        _validate_inputs(format_plan.source, bytes((format_plan.source.elements + 1) // 2),
                         self.block_scales, self.global_scale, self.geometry)
        if (not isinstance(self.source_digest, str) or len(self.source_digest) != 64
                or any(char not in "0123456789abcdef" for char in self.source_digest)):
            raise ValueError("invalid source digest")
        if self.target_digest != _content_digest(
            self.plan, self.payload, self.block_scales, self.global_scale, "target"
        ):
            raise ValueError("target content digest mismatch")

    @property
    def geometry(self) -> TensorGeometry | None:
        return self.plan.geometry if isinstance(self.plan, TensorConversionPlan) else None

    def verify_source(self, descriptor, packed, scales, global_scale, *, geometry=None) -> None:
        """Check provenance and conversion consistency, not cryptographic authenticity."""
        expected = convert_nvfp4_to_int8(descriptor, packed, scales, global_scale, geometry=geometry)
        if self != expected:
            raise ValueError("conversion source or output mismatch")

    def reference_values(self) -> tuple[float, ...]:
        # Preserve the original scale factors; divide the small integer first to
        # avoid introducing underflow by halving a very small global scale.
        return tuple(
            (value if value < 128 else value - 256) / 2
            * _scale(self.block_scales[index // 16 if self.geometry is None
                                      else self.geometry.scale_index(index)]) * self.global_scale
            for index, value in enumerate(self.payload)
        )


def convert_nvfp4_to_int8(
    descriptor: NumericFormatDescriptor, packed: bytes, scales: bytes, global_scale: float,
    *, geometry: TensorGeometry | None = None,
) -> ScaledInt8Tensor:
    """Convert globally packed C-order nibbles; no per-row nibble padding is allowed."""
    plan = _validate_inputs(descriptor, packed, scales, global_scale, geometry)
    payload = bytes(
        ((-1 if code & 8 else 1) * _E2M1_TWICE[code & 7]) & 255
        for code in _codes(packed, descriptor.elements)
    )
    # Signed zero is canonicalized by integer storage; numeric equality only.
    return ScaledInt8Tensor(
        plan, payload, scales, global_scale,
        _content_digest(plan, packed, scales, global_scale, "source"),
        _content_digest(plan, payload, scales, global_scale, "target"),
    )
