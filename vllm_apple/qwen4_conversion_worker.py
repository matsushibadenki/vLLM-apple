from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from .qwen4_component_loader import Qwen4ComponentLoader, Qwen4MemoryAdmission
from .qwen4_conversion_protocol import (
    QWEN4_CONVERSION_ABI_VERSION,
    parse_qwen4_conversion_request,
    parse_qwen4_conversion_response,
)
from .qwen4_load_plan import build_qwen4_component_load_plan
from .qwen4_tensor_reader import Qwen4TensorReader


@dataclass(frozen=True, slots=True)
class ConvertedTensorEvidence:
    backend: str
    backend_version: str
    output_shape: tuple[int, ...]
    output_bytes: int
    output_digest: str


class Qwen4TensorConverter(Protocol):
    def convert(
        self,
        chunks: Iterable[bytes],
        *,
        source_dtype: str,
        target_dtype: str,
        output_shape: tuple[int, ...],
        reserved_bytes: int,
    ) -> ConvertedTensorEvidence: ...


class _CountingChunks:
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self.consumed_bytes = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.consumed_bytes += len(chunk)
            yield chunk


def execute_qwen4_conversion_request(
    payload: object,
    converter: Qwen4TensorConverter,
) -> dict[str, object]:
    request = parse_qwen4_conversion_request(payload)
    requested_modes = tuple(request["requested_modes"])
    load_plan = build_qwen4_component_load_plan(
        request["stage_root"],
        maximum_artifact_bytes=request["maximum_artifact_bytes"],
        requested_modes=requested_modes,
        target_dtype=request["target_dtype"],
        scratch_bytes_per_tensor=request["scratch_bytes"],
    )
    if (
        load_plan["contract_id"] != request["contract_id"]
        or load_plan["load_plan_id"] != request["load_plan_id"]
    ):
        raise ValueError("Qwen4 conversion request does not match the rebuilt stage plan")
    reader = Qwen4TensorReader(
        request["stage_root"],
        maximum_artifact_bytes=request["maximum_artifact_bytes"],
        requested_modes=requested_modes,
    )
    admission = Qwen4MemoryAdmission(request["memory_capacity_bytes"])
    loader = Qwen4ComponentLoader(reader, admission)
    slice_value = request["axis0_slice"]
    if slice_value is None:
        context = loader.open_tensor(
            request["tensor_name"],
            target_dtype=request["target_dtype"],
            scratch_bytes=request["scratch_bytes"],
        )
    else:
        context = loader.open_tensor_axis0_slice(
            request["tensor_name"],
            start=slice_value["start"],
            count=slice_value["count"],
            target_dtype=request["target_dtype"],
            scratch_bytes=request["scratch_bytes"],
        )
    with context as lease:
        shape = lease.descriptor.get("shape")
        source_dtype = lease.descriptor.get("dtype")
        source_bytes = lease.descriptor.get("bytes")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) for value in shape)
            or not isinstance(source_dtype, str)
            or not isinstance(source_bytes, int)
        ):
            raise ValueError("Qwen4 conversion lease descriptor is invalid")
        counted = _CountingChunks(lease.chunks)
        converted = converter.convert(
            counted,
            source_dtype=source_dtype,
            target_dtype=request["target_dtype"],
            output_shape=tuple(shape),
            reserved_bytes=lease.reservation.reserved_bytes,
        )
        if counted.consumed_bytes != source_bytes:
            raise ValueError("Qwen4 converter did not consume the complete tensor slice")
        if converted.output_shape != tuple(shape):
            raise ValueError("Qwen4 converter changed the tensor shape")
        response = {
            "abi_version": QWEN4_CONVERSION_ABI_VERSION,
            "passed": True,
            "backend": converted.backend,
            "backend_version": converted.backend_version,
            "contract_id": request["contract_id"],
            "load_plan_id": request["load_plan_id"],
            "target_dtype": request["target_dtype"],
            "output_shape": list(converted.output_shape),
            "output_bytes": converted.output_bytes,
            "output_digest": converted.output_digest,
            "peak_reserved_bytes": lease.reservation.reserved_bytes,
            "stores_tensor_values": False,
        }
        return parse_qwen4_conversion_response(response)
