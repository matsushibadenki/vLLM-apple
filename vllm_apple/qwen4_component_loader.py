from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from .qwen4_tensor_reader import Qwen4TensorReader


MAX_ACTIVE_TENSOR_LOADS = 1024
_TARGET_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


@dataclass(frozen=True, slots=True)
class TensorLoadReservation:
    reservation_id: int
    tensor_name: str
    component: str
    source_stream_bytes: int
    destination_bytes: int
    scratch_bytes: int
    reserved_bytes: int


@dataclass(slots=True)
class TensorLoadLease:
    reservation: TensorLoadReservation
    descriptor: dict[str, object]
    chunks: Iterator[bytes]


class Qwen4MemoryAdmission:
    def __init__(
        self,
        capacity_bytes: int,
        *,
        component_limits: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(capacity_bytes, int) or isinstance(capacity_bytes, bool) or capacity_bytes <= 0:
            raise ValueError("Qwen4 loader memory capacity is invalid")
        limits = dict(component_limits or {})
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > capacity_bytes
            for name, value in limits.items()
        ):
            raise ValueError("Qwen4 loader component memory limit is invalid")
        self.capacity_bytes = capacity_bytes
        self.component_limits = limits
        self._lock = threading.Lock()
        self._next_id = 1
        self._reservations: dict[int, TensorLoadReservation] = {}
        self._reserved_bytes = 0
        self._component_bytes: dict[str, int] = {}

    def reserve(
        self,
        tensor_name: str,
        descriptor: Mapping[str, object],
        *,
        target_dtype: str,
        source_stream_bytes: int,
        scratch_bytes: int,
    ) -> TensorLoadReservation:
        component = descriptor.get("component")
        shape = descriptor.get("shape")
        if not isinstance(component, str) or not isinstance(shape, list):
            raise ValueError("Qwen4 loader tensor descriptor is invalid")
        if target_dtype not in _TARGET_DTYPE_BYTES:
            raise ValueError("Qwen4 loader target dtype is unsupported")
        if (
            not isinstance(source_stream_bytes, int)
            or isinstance(source_stream_bytes, bool)
            or source_stream_bytes < 0
            or not isinstance(scratch_bytes, int)
            or isinstance(scratch_bytes, bool)
            or scratch_bytes < 0
        ):
            raise ValueError("Qwen4 loader transient memory request is invalid")
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
                raise ValueError("Qwen4 loader tensor shape is invalid")
            elements *= dimension
            if elements * _TARGET_DTYPE_BYTES[target_dtype] > self.capacity_bytes:
                raise MemoryError("Qwen4 tensor destination exceeds the loader memory capacity")
        destination_bytes = elements * _TARGET_DTYPE_BYTES[target_dtype]
        reserved_bytes = source_stream_bytes + destination_bytes + scratch_bytes
        with self._lock:
            component_after = self._component_bytes.get(component, 0) + reserved_bytes
            component_limit = self.component_limits.get(component, self.capacity_bytes)
            if (
                len(self._reservations) >= MAX_ACTIVE_TENSOR_LOADS
                or self._reserved_bytes + reserved_bytes > self.capacity_bytes
                or component_after > component_limit
            ):
                raise MemoryError("Qwen4 tensor load memory admission rejected the reservation")
            reservation = TensorLoadReservation(
                reservation_id=self._next_id,
                tensor_name=tensor_name,
                component=component,
                source_stream_bytes=source_stream_bytes,
                destination_bytes=destination_bytes,
                scratch_bytes=scratch_bytes,
                reserved_bytes=reserved_bytes,
            )
            self._next_id += 1
            self._reservations[reservation.reservation_id] = reservation
            self._reserved_bytes += reserved_bytes
            self._component_bytes[component] = component_after
            return reservation

    def release(self, reservation: TensorLoadReservation) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current != reservation:
                raise ValueError("Qwen4 tensor load reservation is unknown or already released")
            self._reserved_bytes -= reservation.reserved_bytes
            component_after = self._component_bytes[reservation.component] - reservation.reserved_bytes
            if component_after:
                self._component_bytes[reservation.component] = component_after
            else:
                del self._component_bytes[reservation.component]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "capacity_bytes": self.capacity_bytes,
                "reserved_bytes": self._reserved_bytes,
                "remaining_bytes": self.capacity_bytes - self._reserved_bytes,
                "active_reservations": len(self._reservations),
                "component_bytes": dict(sorted(self._component_bytes.items())),
            }


class Qwen4ComponentLoader:
    def __init__(self, reader: Qwen4TensorReader, admission: Qwen4MemoryAdmission) -> None:
        self.reader = reader
        self.admission = admission

    @contextmanager
    def open_tensor(
        self,
        tensor_name: str,
        *,
        target_dtype: str,
        scratch_bytes: int = 0,
    ) -> Iterator[TensorLoadLease]:
        descriptor = self.reader.descriptor(tensor_name)
        if descriptor.get("active") is not True:
            raise ValueError("Qwen4 tensor is disabled for the requested mode")
        source_bytes = descriptor.get("bytes")
        if not isinstance(source_bytes, int) or isinstance(source_bytes, bool) or source_bytes < 0:
            raise ValueError("Qwen4 loader source tensor size is invalid")
        reservation = self.admission.reserve(
            tensor_name,
            descriptor,
            target_dtype=target_dtype,
            source_stream_bytes=min(source_bytes, self.reader.maximum_chunk_bytes),
            scratch_bytes=scratch_bytes,
        )
        chunks = self.reader.iter_tensor_chunks(tensor_name)
        try:
            yield TensorLoadLease(reservation=reservation, descriptor=descriptor, chunks=chunks)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
            self.admission.release(reservation)

    @contextmanager
    def open_tensor_axis0_slice(
        self,
        tensor_name: str,
        *,
        start: int,
        count: int,
        target_dtype: str,
        scratch_bytes: int = 0,
    ) -> Iterator[TensorLoadLease]:
        descriptor = self.reader.descriptor(tensor_name)
        shape = descriptor.get("shape")
        source_bytes = descriptor.get("bytes")
        if (
            descriptor.get("active") is not True
            or descriptor.get("component") != "mixture_of_experts"
            or ".experts." not in tensor_name
            or not isinstance(shape, list)
            or not shape
            or not isinstance(shape[0], int)
            or shape[0] <= 0
            or not isinstance(source_bytes, int)
            or source_bytes % shape[0] != 0
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or start < 0
            or count <= 0
            or start + count > shape[0]
        ):
            raise ValueError("Qwen4 component loader axis-0 slice is invalid")
        slice_descriptor = {
            **descriptor,
            "shape": [count, *shape[1:]],
            "bytes": source_bytes // shape[0] * count,
        }
        reservation = self.admission.reserve(
            tensor_name,
            slice_descriptor,
            target_dtype=target_dtype,
            source_stream_bytes=min(slice_descriptor["bytes"], self.reader.maximum_chunk_bytes),
            scratch_bytes=scratch_bytes,
        )
        chunks = self.reader.iter_tensor_axis0_slice(tensor_name, start=start, count=count)
        try:
            yield TensorLoadLease(
                reservation=reservation,
                descriptor=slice_descriptor,
                chunks=chunks,
            )
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
            self.admission.release(reservation)
