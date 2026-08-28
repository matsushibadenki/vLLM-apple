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


@dataclass(frozen=True, slots=True)
class TokenCountSingleFlightSnapshot:
    capacity: int
    active: int
    leaders: int
    followers: int
    bypasses: int
    timeouts: int


@dataclass(slots=True)
class _FlightState:
    event: threading.Event
    result: int | None = None


@dataclass(frozen=True, slots=True)
class TokenCountFlight:
    fingerprint: str
    leader: bool
    state: _FlightState | None


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


class TokenCountSingleFlight:
    """Bounds and coalesces concurrent work for identical opaque fingerprints."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("single-flight capacity must be positive")
        self.capacity = capacity
        self._entries: dict[str, _FlightState] = {}
        self._lock = threading.Lock()
        self._leaders = 0
        self._followers = 0
        self._bypasses = 0
        self._timeouts = 0

    def join(self, fingerprint: str) -> TokenCountFlight:
        if not fingerprint:
            raise ValueError("single-flight fingerprint cannot be empty")
        with self._lock:
            state = self._entries.get(fingerprint)
            if state is not None:
                self._followers += 1
                return TokenCountFlight(fingerprint, False, state)
            if len(self._entries) >= self.capacity:
                self._bypasses += 1
                return TokenCountFlight(fingerprint, True, None)
            state = _FlightState(threading.Event())
            self._entries[fingerprint] = state
            self._leaders += 1
            return TokenCountFlight(fingerprint, True, state)

    def wait(self, flight: TokenCountFlight, timeout: float) -> int | None:
        if flight.leader or flight.state is None or timeout <= 0:
            raise ValueError("invalid single-flight follower wait")
        if not flight.state.event.wait(timeout):
            with self._lock:
                self._timeouts += 1
            return None
        return flight.state.result

    def complete(self, flight: TokenCountFlight, result: int | None) -> None:
        if not flight.leader:
            raise ValueError("only a single-flight leader can complete work")
        if flight.state is None:
            return
        with self._lock:
            current = self._entries.get(flight.fingerprint)
            if current is not flight.state:
                return
            flight.state.result = result
            del self._entries[flight.fingerprint]
            flight.state.event.set()

    def snapshot(self) -> TokenCountSingleFlightSnapshot:
        with self._lock:
            return TokenCountSingleFlightSnapshot(
                self.capacity,
                len(self._entries),
                self._leaders,
                self._followers,
                self._bypasses,
                self._timeouts,
            )
