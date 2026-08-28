from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

from .semantic_cache import SemanticAnchor, SemanticAnchorCache, SemanticAnchorKind


MAX_PENDING_RELEASES = 1_024


@dataclass(frozen=True, slots=True)
class BackendStateReference:
    handle: str
    state_bytes: int


class SemanticStateBackend(Protocol):
    def capture_semantic_state(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
        token_position: int,
        kind: SemanticAnchorKind,
    ) -> BackendStateReference | None: ...

    def restore_semantic_state(self, handle: str) -> bool: ...

    def release_semantic_state(self, handle: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SemanticRestoreResult:
    reused: bool
    token_position: int | None = None
    kind: SemanticAnchorKind | None = None


class SemanticStateCoordinator:
    """Coordinates bounded anchor metadata with backend-owned KV/recurrent state."""

    def __init__(
        self,
        backend: SemanticStateBackend,
        cache: SemanticAnchorCache,
        event_publisher: Callable[[str, dict[str, object]], object] | None = None,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._event_publisher = event_publisher
        self._lock = threading.Lock()
        self._captures = 0
        self._hits = 0
        self._misses = 0
        self._restore_failures = 0
        self._release_failures = 0
        self._dropped_release_handles = 0
        self._pending: OrderedDict[str, None] = OrderedDict()

    def capture(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
        token_position: int,
        kind: SemanticAnchorKind,
    ) -> SemanticAnchor | None:
        reference = self._backend.capture_semantic_state(
            session_fingerprint,
            prefix_fingerprint,
            token_position,
            kind,
        )
        if reference is None:
            return None
        try:
            anchor = SemanticAnchor(
                session_fingerprint=session_fingerprint,
                prefix_fingerprint=prefix_fingerprint,
                token_position=token_position,
                kind=kind,
                state_handle=reference.handle,
                state_bytes=reference.state_bytes,
            )
            evicted = self._cache.put(anchor)
        except Exception:
            self._release_handle(reference.handle)
            raise
        self._release(evicted)
        with self._lock:
            self._captures += 1
        self._publish("semantic_cache.capture", anchor, len(evicted))
        return anchor

    def restore_deepest(
        self,
        session_fingerprint: str,
        boundaries: tuple[tuple[int, str], ...],
    ) -> SemanticRestoreResult:
        anchor = self._cache.deepest_reusable(session_fingerprint, boundaries)
        if anchor is None:
            with self._lock:
                self._misses += 1
            return SemanticRestoreResult(False)
        if self._backend.restore_semantic_state(anchor.state_handle):
            with self._lock:
                self._hits += 1
            self._publish("semantic_cache.hit", anchor, 0)
            return SemanticRestoreResult(True, anchor.token_position, anchor.kind)
        removed = self._cache.discard(
            anchor.session_fingerprint,
            anchor.prefix_fingerprint,
        )
        if removed is not None:
            self._release((removed,))
        with self._lock:
            self._misses += 1
            self._restore_failures += 1
        return SemanticRestoreResult(False)

    def resize(self, capacity_entries: int, capacity_bytes: int) -> int:
        evicted = self._cache.resize(capacity_entries, capacity_bytes)
        self._release(evicted)
        return len(evicted)

    def retry_pending_releases(self) -> int:
        with self._lock:
            handles = tuple(self._pending)
        released = 0
        for handle in handles:
            try:
                self._backend.release_semantic_state(handle)
            except Exception:
                continue
            with self._lock:
                self._pending.pop(handle, None)
            released += 1
        return released

    def close(self) -> None:
        self._release(self._cache.clear())
        self.retry_pending_releases()

    def snapshot(self) -> dict[str, int | bool]:
        cache = self._cache.snapshot()
        with self._lock:
            return {
                "enabled": True,
                "capacity_entries": cache.capacity_entries,
                "capacity_bytes": cache.capacity_bytes,
                "entry_count": cache.entry_count,
                "resident_bytes": cache.resident_bytes,
                "evictions": cache.evictions,
                "captures": self._captures,
                "hits": self._hits,
                "misses": self._misses,
                "restore_failures": self._restore_failures,
                "release_failures": self._release_failures,
                "pending_releases": len(self._pending),
                "dropped_release_handles": self._dropped_release_handles,
            }

    def _release(self, anchors: tuple[SemanticAnchor, ...]) -> None:
        for anchor in anchors:
            self._release_handle(anchor.state_handle)

    def _release_handle(self, handle: str) -> None:
        try:
            self._backend.release_semantic_state(handle)
        except Exception:
            with self._lock:
                self._release_failures += 1
                self._pending[handle] = None
                self._pending.move_to_end(handle)
                while len(self._pending) > MAX_PENDING_RELEASES:
                    self._pending.popitem(last=False)
                    self._dropped_release_handles += 1

    def _publish(self, name: str, anchor: SemanticAnchor, evicted: int) -> None:
        if self._event_publisher is not None:
            self._event_publisher(
                name,
                {
                    "kind": anchor.kind.value,
                    "token_position": anchor.token_position,
                    "state_bytes": anchor.state_bytes,
                    "evicted": evicted,
                },
            )


def disabled_semantic_state_snapshot() -> dict[str, int | bool]:
    return {
        "enabled": False,
        "capacity_entries": 0,
        "capacity_bytes": 0,
        "entry_count": 0,
        "resident_bytes": 0,
        "evictions": 0,
        "captures": 0,
        "hits": 0,
        "misses": 0,
        "restore_failures": 0,
        "release_failures": 0,
        "pending_releases": 0,
        "dropped_release_handles": 0,
    }
