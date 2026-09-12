"""Bounded ownership contracts for numeric tile streams."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass


MAX_NUMERIC_TILE_BYTES = 8 * 1024 * 1024
MAX_NUMERIC_STREAM_BYTES = 1 << 40
_ZERO_BLOCK = bytes(64 * 1024)


def _zero_buffer(buffer: bytearray) -> None:
    for offset in range(0, len(buffer), len(_ZERO_BLOCK)):
        length = min(len(_ZERO_BLOCK), len(buffer) - offset)
        buffer[offset:offset + length] = _ZERO_BLOCK[:length]


class NumericStreamingCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NumericStreamingPlan:
    source_digest: str
    source_bytes: int
    tile_bytes: int
    buffer_count: int = 2
    alignment_bytes: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported numeric streaming plan version")
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.source_digest)
        ):
            raise ValueError("numeric stream source digest is invalid")
        for value, label, maximum in (
            (self.source_bytes, "source size", MAX_NUMERIC_STREAM_BYTES),
            (self.tile_bytes, "tile size", MAX_NUMERIC_TILE_BYTES),
            (self.alignment_bytes, "alignment", MAX_NUMERIC_TILE_BYTES),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"numeric stream {label} is invalid")
        if self.tile_bytes > self.source_bytes:
            raise ValueError("numeric stream tile exceeds its source")
        if self.tile_bytes % self.alignment_bytes:
            raise ValueError("numeric stream tile does not satisfy its alignment")
        if type(self.buffer_count) is not int or self.buffer_count not in (1, 2):
            raise ValueError("numeric stream requires one or two buffers")

    @property
    def tile_count(self) -> int:
        return math.ceil(self.source_bytes / self.tile_bytes)

    @property
    def active_buffer_count(self) -> int:
        return min(self.buffer_count, self.tile_count)

    @property
    def working_set_bytes(self) -> int:
        return self.tile_bytes * self.active_buffer_count

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class NumericTileLease:
    __slots__ = ("_stream", "_slot", "_generation", "index", "offset", "length")

    def __init__(
        self,
        stream: NumericDoubleBufferStream,
        slot: int,
        generation: int,
        index: int,
        offset: int,
        length: int,
    ) -> None:
        self._stream = stream
        self._slot = slot
        self._generation = generation
        self.index = index
        self.offset = offset
        self.length = length

    def read(self) -> bytes:
        return self._stream._read(self._slot, self._generation, self.length)

    def view(self) -> memoryview:
        return self._stream._view(self._slot, self._generation, self.length)

    def release(self) -> None:
        self._stream._release(self._slot, self._generation)


class NumericDoubleBufferStream:
    """Copies source tiles into at most two explicitly leased reusable buffers."""

    def __init__(
        self,
        plan: NumericStreamingPlan,
        source: bytes,
        *,
        cancellation: threading.Event | None = None,
    ) -> None:
        if not isinstance(plan, NumericStreamingPlan) or not isinstance(source, bytes):
            raise ValueError("numeric stream inputs are invalid")
        if len(source) != plan.source_bytes:
            raise ValueError("numeric stream source size mismatch")
        if hashlib.sha256(source).hexdigest() != plan.source_digest:
            raise ValueError("numeric stream source digest mismatch")
        if cancellation is not None and not isinstance(cancellation, threading.Event):
            raise ValueError("numeric stream cancellation signal is invalid")
        self.plan = plan
        self._source = source
        self._cancellation = cancellation
        self._lock = threading.Lock()
        self._buffers = [bytearray(plan.tile_bytes) for _ in range(plan.active_buffer_count)]
        self._active = [False] * len(self._buffers)
        self._generations = [0] * len(self._buffers)
        self._next_index = 0
        self._closed = False
        self._cancelled = False

    def acquire_next(self) -> NumericTileLease | None:
        with self._lock:
            self._check_open_locked()
            if self._next_index >= self.plan.tile_count:
                return None
            try:
                slot = self._active.index(False)
            except ValueError as error:
                raise RuntimeError("numeric stream buffers are all leased") from error
            index = self._next_index
            offset = index * self.plan.tile_bytes
            length = min(self.plan.tile_bytes, self.plan.source_bytes - offset)
            buffer = self._buffers[slot]
            buffer[:length] = memoryview(self._source)[offset:offset + length]
            self._generations[slot] += 1
            self._active[slot] = True
            self._next_index += 1
            return NumericTileLease(
                self, slot, self._generations[slot], index, offset, length
            )

    def poll_cancellation(self) -> None:
        with self._lock:
            self._check_open_locked()

    def close(self) -> None:
        with self._lock:
            self._close_locked(cancelled=False)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1,
                "plan_id": self.plan.plan_id,
                "tile_count": self.plan.tile_count,
                "next_tile_index": self._next_index,
                "in_flight_tiles": sum(self._active),
                "buffer_count": len(self._buffers),
                "working_set_bytes": self.plan.working_set_bytes,
                "cancelled": self._cancelled,
                "closed": self._closed,
                "stores_tensor_values": False,
            }

    def _check_open_locked(self) -> None:
        if self._closed:
            raise ValueError("numeric stream is closed")
        if self._cancellation is not None and self._cancellation.is_set():
            self._close_locked(cancelled=True)
            raise NumericStreamingCancelled("numeric stream was cancelled")

    def _read(self, slot: int, generation: int, length: int) -> bytes:
        with self._lock:
            self._check_lease_locked(slot, generation)
            return bytes(self._buffers[slot][:length])

    def _view(self, slot: int, generation: int, length: int) -> memoryview:
        with self._lock:
            self._check_lease_locked(slot, generation)
            return memoryview(self._buffers[slot])[:length].toreadonly()

    def _release(self, slot: int, generation: int) -> None:
        with self._lock:
            self._check_lease_locked(slot, generation)
            _zero_buffer(self._buffers[slot])
            self._active[slot] = False

    def _check_lease_locked(self, slot: int, generation: int) -> None:
        if (
            self._closed
            or not 0 <= slot < len(self._buffers)
            or not self._active[slot]
            or self._generations[slot] != generation
        ):
            raise ValueError("numeric tile lease is invalid or already released")

    def _close_locked(self, *, cancelled: bool) -> None:
        if self._closed:
            return
        for index, buffer in enumerate(self._buffers):
            _zero_buffer(buffer)
            self._active[index] = False
            self._generations[index] += 1
        self._source = b""
        self._cancelled = cancelled
        self._closed = True

    def __enter__(self) -> NumericDoubleBufferStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
