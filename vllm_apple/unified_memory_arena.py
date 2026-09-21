"""Page-aware mmap arena for bounded zero-copy Unified Memory suballocations."""
from __future__ import annotations

import mmap
import threading
from dataclasses import dataclass

MAX_UNIFIED_MEMORY_ARENA_BYTES = 1 << 40
MAX_UNIFIED_MEMORY_ALLOCATIONS = 65_536


@dataclass(frozen=True, slots=True)
class UnifiedMemoryArenaSnapshot:
    capacity_bytes: int
    allocated_bytes: int
    free_bytes: int
    largest_free_range_bytes: int
    allocation_count: int
    peak_allocated_bytes: int
    failed_allocations: int
    fragmentation_bytes: int


class UnifiedMemoryLease:
    def __init__(
        self,
        arena: "UnifiedMemoryArena",
        allocation_id: int,
        offset: int,
        size: int,
        view: memoryview,
    ) -> None:
        self._arena = arena
        self.allocation_id = allocation_id
        self.offset = offset
        self.size = size
        self.view = view
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self.view.release()
        self._arena._release(self.allocation_id)
        self._released = True

    def __enter__(self) -> "UnifiedMemoryLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class UnifiedMemoryArena:
    def __init__(self, capacity_bytes: int, *, zero_on_release: bool = True) -> None:
        if (type(capacity_bytes) is not int
                or not 4096 <= capacity_bytes <= MAX_UNIFIED_MEMORY_ARENA_BYTES
                or type(zero_on_release) is not bool):
            raise ValueError("invalid Unified Memory arena capacity")
        self.capacity_bytes = capacity_bytes
        self._mapping = mmap.mmap(-1, capacity_bytes, access=mmap.ACCESS_WRITE)
        self._zero_on_release = zero_on_release
        self._free = [(0, capacity_bytes)]
        self._allocations: dict[int, tuple[int, int]] = {}
        self._next_id = 1
        self._allocated = self._peak = self._failed = 0
        self._closed = False
        self._lock = threading.RLock()

    def allocate(self, size: int, *, alignment: int = 4096) -> UnifiedMemoryLease:
        if (type(size) is not int or not 1 <= size <= self.capacity_bytes
                or type(alignment) is not int or not 1 <= alignment <= 65536
                or alignment & (alignment - 1)):
            raise ValueError("invalid Unified Memory allocation")
        with self._lock:
            if self._closed:
                raise RuntimeError("Unified Memory arena is closed")
            if len(self._allocations) >= MAX_UNIFIED_MEMORY_ALLOCATIONS:
                self._failed += 1
                raise ValueError("Unified Memory allocation table is full")
            for index, (start, length) in enumerate(self._free):
                offset = _align(start, alignment)
                prefix = offset - start
                if prefix + size > length:
                    continue
                suffix_start = offset + size
                suffix = start + length - suffix_start
                replacement = []
                if prefix:
                    replacement.append((start, prefix))
                if suffix:
                    replacement.append((suffix_start, suffix))
                self._free[index:index + 1] = replacement
                allocation_id = self._next_id
                self._next_id += 1
                self._allocations[allocation_id] = (offset, size)
                self._allocated += size
                self._peak = max(self._peak, self._allocated)
                return UnifiedMemoryLease(
                    self, allocation_id, offset, size,
                    memoryview(self._mapping)[offset:offset + size],
                )
            self._failed += 1
            raise ValueError("Unified Memory arena has no aligned free range")

    def trim_free_pages(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            page = mmap.PAGESIZE
            trimmed = 0
            advice = getattr(mmap, "MADV_FREE", getattr(mmap, "MADV_DONTNEED", None))
            if advice is None or not hasattr(self._mapping, "madvise"):
                return 0
            for start, length in self._free:
                aligned_start = _align(start, page)
                aligned_end = (start + length) // page * page
                if aligned_end <= aligned_start:
                    continue
                try:
                    self._mapping.madvise(advice, aligned_start, aligned_end - aligned_start)
                except (OSError, ValueError):
                    continue
                trimmed += aligned_end - aligned_start
            return trimmed

    def snapshot(self) -> UnifiedMemoryArenaSnapshot:
        with self._lock:
            free = sum(length for _, length in self._free)
            largest = max((length for _, length in self._free), default=0)
            return UnifiedMemoryArenaSnapshot(
                self.capacity_bytes, self._allocated, free, largest,
                len(self._allocations), self._peak, self._failed, free - largest,
            )

    def close(self) -> None:
        with self._lock:
            if self._allocations:
                raise RuntimeError("Unified Memory arena has active allocations")
            if not self._closed:
                self._mapping.close()
                self._closed = True

    def _release(self, allocation_id: int) -> None:
        with self._lock:
            allocation = self._allocations.pop(allocation_id, None)
            if allocation is None:
                raise RuntimeError("Unified Memory allocation is stale")
            offset, size = allocation
            if self._zero_on_release:
                zero = bytes(min(size, 1024 * 1024))
                written = 0
                while written < size:
                    count = min(len(zero), size - written)
                    self._mapping[offset + written:offset + written + count] = zero[:count]
                    written += count
            self._allocated -= size
            self._free.append((offset, size))
            self._free.sort()
            coalesced = []
            for start, length in self._free:
                if coalesced and coalesced[-1][0] + coalesced[-1][1] == start:
                    previous_start, previous_length = coalesced[-1]
                    coalesced[-1] = (previous_start, previous_length + length)
                else:
                    coalesced.append((start, length))
            self._free = coalesced


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
