"""Bounded (layer, expert) working-set LRU for backend-owned residency."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

MAX_EXPERT_RESIDENCY_ENTRIES = 65_536
MAX_EXPERT_RESIDENCY_BYTES = 1 << 40


@dataclass(frozen=True, order=True, slots=True)
class ExpertKey:
    layer: int
    expert: int

    def __post_init__(self) -> None:
        if (type(self.layer) is not int or type(self.expert) is not int
                or not 0 <= self.layer < 1_000_000
                or not 0 <= self.expert < 1_000_000):
            raise ValueError("invalid expert key")


@dataclass(frozen=True, slots=True)
class ExpertResource:
    handle: object
    resident_bytes: int

    def __post_init__(self) -> None:
        if not 1 <= self.resident_bytes <= MAX_EXPERT_RESIDENCY_BYTES:
            raise ValueError("invalid expert resource")


class ExpertResidencyBackend(Protocol):
    def load_expert(self, key: ExpertKey) -> ExpertResource: ...
    def release_expert(self, resource: ExpertResource) -> None: ...


@dataclass(slots=True)
class _Entry:
    resource: ExpertResource
    leases: int
    last_used: int


class ExpertLease:
    def __init__(
        self, manager: "ExpertResidencyManager", key: ExpertKey, resource: ExpertResource
    ) -> None:
        self._manager = manager
        self.key = key
        self.resource = resource
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release_lease(self.key)

    def __enter__(self) -> "ExpertLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class ExpertResidencyManager:
    def __init__(
        self,
        backend: ExpertResidencyBackend,
        *,
        maximum_entries: int,
        maximum_bytes: int,
    ) -> None:
        if (not 1 <= maximum_entries <= MAX_EXPERT_RESIDENCY_ENTRIES
                or not 1 <= maximum_bytes <= MAX_EXPERT_RESIDENCY_BYTES):
            raise ValueError("invalid expert residency bounds")
        self._backend = backend
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._pending_bounds: tuple[int, int] | None = None
        self._entries: dict[ExpertKey, _Entry] = {}
        self._resident_bytes = 0
        self._clock = 0
        self._hits = self._misses = self._evictions = self._rejections = 0
        self._lock = threading.RLock()

    def acquire(self, key: ExpertKey) -> ExpertLease:
        if not isinstance(key, ExpertKey):
            raise ValueError("invalid expert residency request")
        with self._lock:
            self._clock += 1
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases += 1
                entry.last_used = self._clock
                self._hits += 1
                return ExpertLease(self, key, entry.resource)
            self._misses += 1
            resource = self._backend.load_expert(key)
            if not isinstance(resource, ExpertResource):
                raise ValueError("expert backend returned an invalid resource")
            victims = self._victims_for(resource.resident_bytes, additional_entries=1)
            if victims is None:
                self._rejections += 1
                self._backend.release_expert(resource)
                raise ValueError("expert residency capacity is pinned")
            try:
                for victim in victims:
                    removed = self._entries.pop(victim)
                    self._backend.release_expert(removed.resource)
                    self._resident_bytes -= removed.resource.resident_bytes
                    self._evictions += 1
            except BaseException:
                self._backend.release_expert(resource)
                raise
            self._entries[key] = _Entry(resource, 1, self._clock)
            self._resident_bytes += resource.resident_bytes
            return ExpertLease(self, key, resource)

    def resize(self, *, maximum_entries: int, maximum_bytes: int) -> bool:
        if (not 1 <= maximum_entries <= MAX_EXPERT_RESIDENCY_ENTRIES
                or not 1 <= maximum_bytes <= MAX_EXPERT_RESIDENCY_BYTES):
            raise ValueError("invalid expert residency bounds")
        with self._lock:
            old = (self._maximum_entries, self._maximum_bytes)
            self._maximum_entries, self._maximum_bytes = maximum_entries, maximum_bytes
            victims = self._victims_for(0, additional_entries=0)
            if victims is None:
                self._maximum_entries, self._maximum_bytes = old
                self._pending_bounds = (maximum_entries, maximum_bytes)
                return False
            for victim in victims:
                removed = self._entries.pop(victim)
                self._backend.release_expert(removed.resource)
                self._resident_bytes -= removed.resource.resident_bytes
                self._evictions += 1
            self._pending_bounds = None
            return True

    def close(self) -> None:
        with self._lock:
            if any(entry.leases for entry in self._entries.values()):
                raise RuntimeError("expert residency has active leases")
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._resident_bytes = 0
            for entry in entries:
                self._backend.release_expert(entry.resource)

    def snapshot(self) -> dict[str, int | bool | None]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "resident_bytes": self._resident_bytes,
                "maximum_entries": self._maximum_entries,
                "maximum_bytes": self._maximum_bytes,
                "active_leases": sum(entry.leases for entry in self._entries.values()),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "rejections": self._rejections,
                "resize_pending": self._pending_bounds is not None,
                "pending_maximum_entries": (
                    self._pending_bounds[0] if self._pending_bounds else None
                ),
                "pending_maximum_bytes": (
                    self._pending_bounds[1] if self._pending_bounds else None
                ),
            }

    def _release_lease(self, key: ExpertKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases <= 0:
                raise RuntimeError("expert lease is stale")
            entry.leases -= 1
            entry.last_used = self._clock = self._clock + 1
            pending = self._pending_bounds
        if pending is not None:
            self.resize(maximum_entries=pending[0], maximum_bytes=pending[1])

    def _victims_for(
        self, incoming_bytes: int, *, additional_entries: int
    ) -> tuple[ExpertKey, ...] | None:
        if incoming_bytes > self._maximum_bytes:
            return None
        count = len(self._entries) + additional_entries
        size = self._resident_bytes + incoming_bytes
        available = sorted(
            ((entry.last_used, key) for key, entry in self._entries.items()
             if entry.leases == 0),
            key=lambda item: (item[0], item[1]),
        )
        victims = []
        for _, key in available:
            if count <= self._maximum_entries and size <= self._maximum_bytes:
                break
            victims.append(key)
            count -= 1
            size -= self._entries[key].resource.resident_bytes
        if count > self._maximum_entries or size > self._maximum_bytes:
            return None
        return tuple(victims)
