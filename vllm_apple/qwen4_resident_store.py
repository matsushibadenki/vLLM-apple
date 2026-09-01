from __future__ import annotations

import secrets
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from .qwen4_component_loader import (
    Qwen4ComponentLoader,
    Qwen4MemoryAdmission,
    TensorLoadReservation,
)
from .qwen4_conversion_protocol import _DTYPE_BYTES, _digest
from .qwen4_tensor_reader import Qwen4TensorReader


MAX_RESIDENT_TENSORS = 4096


@dataclass(frozen=True, slots=True)
class ResidentBackendAllocation:
    resource: object
    backend: str
    backend_version: str
    output_shape: tuple[int, ...]
    output_bytes: int
    output_digest: str


class Qwen4ResidentBackend(Protocol):
    def load(
        self,
        chunks: Iterable[bytes],
        *,
        source_dtype: str,
        target_dtype: str,
        output_shape: tuple[int, ...],
        reserved_bytes: int,
    ) -> ResidentBackendAllocation: ...

    def release(self, resource: object) -> None: ...


@dataclass(frozen=True, slots=True)
class _ResidentRecord:
    allocation: ResidentBackendAllocation
    reservation: TensorLoadReservation
    component: str
    target_dtype: str


class _CountingChunks:
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks = chunks
        self.consumed_bytes = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.consumed_bytes += len(chunk)
            yield chunk


class Qwen4ResidentStore:
    def __init__(
        self,
        reader: Qwen4TensorReader,
        admission: Qwen4MemoryAdmission,
        backend: Qwen4ResidentBackend,
    ) -> None:
        self.reader = reader
        self.admission = admission
        self.backend = backend
        self.loader = Qwen4ComponentLoader(reader, admission)
        self._lock = threading.Lock()
        self._records: dict[str, _ResidentRecord] = {}
        self._quarantined: dict[str, _ResidentRecord] = {}

    def load(
        self,
        tensor_name: str,
        *,
        target_dtype: str,
        scratch_bytes: int = 0,
        axis0_slice: tuple[int, int] | None = None,
    ) -> str:
        with self._lock:
            if len(self._records) >= MAX_RESIDENT_TENSORS:
                raise MemoryError("Qwen4 resident tensor handle limit reached")
            if axis0_slice is None:
                descriptor = self.reader.descriptor(tensor_name)
                chunks = self.reader.iter_tensor_chunks(tensor_name)
            else:
                descriptor = self.loader.axis0_slice_descriptor(
                    tensor_name, start=axis0_slice[0], count=axis0_slice[1]
                )
                chunks = self.reader.iter_tensor_axis0_slice(
                    tensor_name, start=axis0_slice[0], count=axis0_slice[1]
                )
            if descriptor.get("active") is not True:
                raise ValueError("Qwen4 resident tensor is disabled for the requested mode")
            shape = descriptor.get("shape")
            source_dtype = descriptor.get("dtype")
            source_bytes = descriptor.get("bytes")
            component = descriptor.get("component")
            if (
                not isinstance(shape, list)
                or not isinstance(source_dtype, str)
                or not isinstance(source_bytes, int)
                or not isinstance(component, str)
            ):
                raise ValueError("Qwen4 resident tensor descriptor is invalid")
            reservation = self.admission.reserve(
                tensor_name,
                descriptor,
                target_dtype=target_dtype,
                source_stream_bytes=min(source_bytes, self.reader.maximum_chunk_bytes),
                scratch_bytes=scratch_bytes,
            )
            allocation = None
            counted = _CountingChunks(chunks)
            try:
                allocation = self.backend.load(
                    counted,
                    source_dtype=source_dtype,
                    target_dtype=target_dtype,
                    output_shape=tuple(shape),
                    reserved_bytes=reservation.reserved_bytes,
                )
                expected_output_bytes = reservation.destination_bytes
                if (
                    counted.consumed_bytes != source_bytes
                    or allocation.output_shape != tuple(shape)
                    or allocation.output_bytes != expected_output_bytes
                    or allocation.output_bytes
                    != _shape_bytes(shape, target_dtype)
                    or not _digest(allocation.output_digest)
                ):
                    raise ValueError("Qwen4 resident backend allocation evidence is invalid")
                retained = self.admission.retain_destination(reservation)
                handle = secrets.token_hex(16)
                while handle in self._records:
                    handle = secrets.token_hex(16)
                self._records[handle] = _ResidentRecord(
                    allocation=allocation,
                    reservation=retained,
                    component=component,
                    target_dtype=target_dtype,
                )
                return handle
            except BaseException as load_error:
                if allocation is not None:
                    try:
                        self.backend.release(allocation.resource)
                    except BaseException as release_error:
                        quarantine_handle = secrets.token_hex(16)
                        self._quarantined[quarantine_handle] = _ResidentRecord(
                            allocation=allocation,
                            reservation=reservation,
                            component=component,
                            target_dtype=target_dtype,
                        )
                        raise RuntimeError(
                            "Qwen4 failed resident allocation was quarantined"
                        ) from release_error
                self.admission.release(reservation)
                raise load_error
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()

    def unload(self, handle: str) -> None:
        with self._lock:
            record = self._records.get(handle)
            if record is None:
                raise KeyError("unknown Qwen4 resident tensor handle")
            self.backend.release(record.allocation.resource)
            self.admission.release(record.reservation)
            del self._records[handle]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            components: dict[str, int] = {}
            for record in self._records.values():
                components[record.component] = components.get(record.component, 0) + 1
            return {
                "schema_version": 1,
                "resident_tensors": len(self._records),
                "quarantined_tensors": len(self._quarantined),
                "resident_components": dict(sorted(components.items())),
                "memory": self.admission.snapshot(),
                "stores_tensor_names": False,
            }

    def retry_quarantined_releases(self) -> int:
        released = 0
        with self._lock:
            for handle, record in tuple(self._quarantined.items()):
                try:
                    self.backend.release(record.allocation.resource)
                except BaseException:
                    continue
                self.admission.release(record.reservation)
                del self._quarantined[handle]
                released += 1
        return released

    def shutdown(self) -> dict[str, object]:
        with self._lock:
            for handle, record in tuple(self._records.items()):
                try:
                    self.backend.release(record.allocation.resource)
                except BaseException:
                    continue
                self.admission.release(record.reservation)
                del self._records[handle]
            for handle, record in tuple(self._quarantined.items()):
                try:
                    self.backend.release(record.allocation.resource)
                except BaseException:
                    continue
                self.admission.release(record.reservation)
                del self._quarantined[handle]
        return self.snapshot()


def _shape_bytes(shape: list[int], dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError("Qwen4 resident target dtype is invalid")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * _DTYPE_BYTES[dtype]
