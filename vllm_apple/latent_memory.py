"""Bounded reusable latent buffers backed by the Unified Memory arena."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .unified_memory_arena import UnifiedMemoryArena, UnifiedMemoryLease

MAX_LATENT_BUFFERS = 4096
MAX_LATENT_BYTES = 1 << 38


@dataclass(frozen=True, slots=True)
class LatentShape:
    dimensions: tuple[int, ...]
    bytes_per_element: int

    def __post_init__(self) -> None:
        if (not 1 <= len(self.dimensions) <= 5
                or any(type(value) is not int or not 1 <= value <= 1_000_000
                       for value in self.dimensions)
                or self.bytes_per_element not in (1, 2, 4)):
            raise ValueError("invalid latent shape")
        if self.byte_count > MAX_LATENT_BYTES:
            raise ValueError("latent shape exceeds byte bound")

    @property
    def byte_count(self) -> int:
        return math.prod(self.dimensions) * self.bytes_per_element


@dataclass(slots=True)
class _LatentEntry:
    shape: LatentShape
    arena_lease: UnifiedMemoryLease
    active: bool
    last_used: int


class LatentBufferLease:
    def __init__(self, manager: "LatentMemoryManager", entry_id: int, entry: _LatentEntry) -> None:
        self._manager = manager
        self._entry_id = entry_id
        self.shape = entry.shape
        self.view = entry.arena_lease.view
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._manager._release(self._entry_id)
        self._released = True

    def __enter__(self) -> "LatentBufferLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class LatentMemoryManager:
    def __init__(self, arena: UnifiedMemoryArena, *, maximum_buffers: int = 64) -> None:
        if (not isinstance(arena, UnifiedMemoryArena)
                or not 1 <= maximum_buffers <= MAX_LATENT_BUFFERS):
            raise ValueError("invalid latent memory configuration")
        self._arena = arena
        self._maximum_buffers = maximum_buffers
        self._entries: dict[int, _LatentEntry] = {}
        self._next_id = 1
        self._clock = 0
        self._hits = self._misses = self._evictions = self._rejections = 0
        self._lock = threading.RLock()

    def acquire(self, shape: LatentShape) -> LatentBufferLease:
        if not isinstance(shape, LatentShape):
            raise ValueError("invalid latent request")
        with self._lock:
            self._clock += 1
            reusable = next((
                (entry_id, entry) for entry_id, entry in self._entries.items()
                if not entry.active and entry.shape == shape
            ), None)
            if reusable is not None:
                entry_id, entry = reusable
                _zero_view(entry.arena_lease.view)
                entry.active = True
                entry.last_used = self._clock
                self._hits += 1
                return LatentBufferLease(self, entry_id, entry)
            self._misses += 1
            while len(self._entries) >= self._maximum_buffers:
                if not self._evict_oldest_idle():
                    self._rejections += 1
                    raise ValueError("latent buffer capacity is pinned")
            while True:
                try:
                    arena_lease = self._arena.allocate(shape.byte_count, alignment=4096)
                    break
                except ValueError:
                    if not self._evict_oldest_idle():
                        self._rejections += 1
                        raise ValueError("latent arena capacity is pinned") from None
            entry_id = self._next_id
            self._next_id += 1
            entry = _LatentEntry(shape, arena_lease, True, self._clock)
            self._entries[entry_id] = entry
            return LatentBufferLease(self, entry_id, entry)

    def trim(self) -> int:
        with self._lock:
            released = 0
            for entry_id in tuple(self._entries):
                entry = self._entries[entry_id]
                if not entry.active:
                    entry.arena_lease.release()
                    released += entry.shape.byte_count
                    del self._entries[entry_id]
                    self._evictions += 1
            self._arena.trim_free_pages()
            return released

    def close(self) -> None:
        with self._lock:
            if any(entry.active for entry in self._entries.values()):
                raise RuntimeError("latent memory has active leases")
            self.trim()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "buffers": len(self._entries),
                "active": sum(entry.active for entry in self._entries.values()),
                "resident_bytes": sum(entry.shape.byte_count for entry in self._entries.values()),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "rejections": self._rejections,
            }

    def _release(self, entry_id: int) -> None:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or not entry.active:
                raise RuntimeError("latent buffer lease is stale")
            self._clock += 1
            entry.active = False
            entry.last_used = self._clock

    def _evict_oldest_idle(self) -> bool:
        candidates = (
            (entry.last_used, entry_id) for entry_id, entry in self._entries.items()
            if not entry.active
        )
        victim = min(candidates, default=None)
        if victim is None:
            return False
        entry = self._entries.pop(victim[1])
        entry.arena_lease.release()
        self._evictions += 1
        return True


def _zero_view(view: memoryview) -> None:
    chunk = bytes(min(len(view), 1 << 20))
    offset = 0
    while offset < len(view):
        count = min(len(chunk), len(view) - offset)
        view[offset:offset + count] = chunk[:count]
        offset += count
