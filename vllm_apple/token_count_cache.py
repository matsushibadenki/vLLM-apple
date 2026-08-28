from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenCountCacheSnapshot:
    capacity: int
    entries: int
    hits: int
    misses: int
    evictions: int
    expirations: int


class TokenCountCache:
    """Bounded TTL/LRU cache keyed only by an opaque request fingerprint."""

    def __init__(
        self,
        capacity: int = 256,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0 or ttl_seconds <= 0:
            raise ValueError("token count cache limits must be positive")
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, fingerprint: str) -> int | None:
        with self._lock:
            entry = self._entries.get(fingerprint)
            if entry is None:
                self._misses += 1
                return None
            count, created_at = entry
            if self._clock() - created_at >= self.ttl_seconds:
                del self._entries[fingerprint]
                self._misses += 1
                self._expirations += 1
                return None
            self._entries.move_to_end(fingerprint)
            self._hits += 1
            return count

    def put(self, fingerprint: str, count: int) -> None:
        if not fingerprint or count < 0:
            raise ValueError("invalid token count cache entry")
        with self._lock:
            self._entries[fingerprint] = (count, self._clock())
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
                self._evictions += 1

    def snapshot(self) -> TokenCountCacheSnapshot:
        with self._lock:
            return TokenCountCacheSnapshot(
                self.capacity,
                len(self._entries),
                self._hits,
                self._misses,
                self._evictions,
                self._expirations,
            )
