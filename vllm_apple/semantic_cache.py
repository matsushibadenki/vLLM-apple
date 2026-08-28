from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum


MAX_ANCHOR_ENTRIES = 4_096
MAX_ANCHOR_STATE_BYTES = 1 << 40
MAX_PREFIX_TOKENS = 1_048_576
MAX_STATE_HANDLE_BYTES = 512


class SemanticAnchorKind(str, Enum):
    TURN = "turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


@dataclass(frozen=True, slots=True)
class SemanticAnchor:
    session_fingerprint: str
    prefix_fingerprint: str
    token_position: int
    kind: SemanticAnchorKind
    state_handle: str
    state_bytes: int

    def __post_init__(self) -> None:
        _validate_sha256("session fingerprint", self.session_fingerprint)
        _validate_sha256("prefix fingerprint", self.prefix_fingerprint)
        if not 1 <= self.token_position <= MAX_PREFIX_TOKENS:
            raise ValueError("semantic anchor token position is invalid")
        if not isinstance(self.kind, SemanticAnchorKind):
            raise ValueError("semantic anchor kind is invalid")
        if (
            not self.state_handle
            or len(self.state_handle.encode("utf-8")) > MAX_STATE_HANDLE_BYTES
            or not 1 <= self.state_bytes <= MAX_ANCHOR_STATE_BYTES
        ):
            raise ValueError("semantic anchor state metadata is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "session_fingerprint": self.session_fingerprint,
            "prefix_fingerprint": self.prefix_fingerprint,
            "token_position": self.token_position,
            "kind": self.kind.value,
            "state_handle": self.state_handle,
            "state_bytes": self.state_bytes,
        }


@dataclass(frozen=True, slots=True)
class SemanticCacheSnapshot:
    capacity_entries: int
    capacity_bytes: int
    entry_count: int
    resident_bytes: int
    evictions: int


class SemanticAnchorCache:
    """Thread-safe metadata cache; callers own and release backend state handles."""

    def __init__(self, capacity_entries: int, capacity_bytes: int) -> None:
        _validate_capacity(capacity_entries, capacity_bytes)
        self._capacity_entries = capacity_entries
        self._capacity_bytes = capacity_bytes
        self._resident_bytes = 0
        self._evictions = 0
        self._entries: OrderedDict[tuple[str, str], SemanticAnchor] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, anchor: SemanticAnchor) -> tuple[SemanticAnchor, ...]:
        if anchor.state_bytes > self._capacity_bytes:
            raise ValueError("semantic anchor exceeds cache byte capacity")
        key = (anchor.session_fingerprint, anchor.prefix_fingerprint)
        evicted: list[SemanticAnchor] = []
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._resident_bytes -= previous.state_bytes
                if previous.state_handle != anchor.state_handle:
                    evicted.append(previous)
            self._entries[key] = anchor
            self._resident_bytes += anchor.state_bytes
            evicted.extend(self._evict_to_budget())
        return tuple(evicted)

    def deepest_reusable(
        self,
        session_fingerprint: str,
        boundaries: tuple[tuple[int, str], ...],
    ) -> SemanticAnchor | None:
        _validate_sha256("session fingerprint", session_fingerprint)
        previous_position = 0
        for position, fingerprint in boundaries:
            if not previous_position < position <= MAX_PREFIX_TOKENS:
                raise ValueError("semantic prefix boundaries must be strictly increasing")
            _validate_sha256("prefix fingerprint", fingerprint)
            previous_position = position
        with self._lock:
            for position, fingerprint in reversed(boundaries):
                key = (session_fingerprint, fingerprint)
                anchor = self._entries.get(key)
                if anchor is not None and anchor.token_position == position:
                    self._entries.move_to_end(key)
                    return anchor
        return None

    def resize(
        self,
        capacity_entries: int,
        capacity_bytes: int,
    ) -> tuple[SemanticAnchor, ...]:
        _validate_capacity(capacity_entries, capacity_bytes)
        with self._lock:
            self._capacity_entries = capacity_entries
            self._capacity_bytes = capacity_bytes
            return tuple(self._evict_to_budget())

    def discard(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
    ) -> SemanticAnchor | None:
        _validate_sha256("session fingerprint", session_fingerprint)
        _validate_sha256("prefix fingerprint", prefix_fingerprint)
        with self._lock:
            anchor = self._entries.pop((session_fingerprint, prefix_fingerprint), None)
            if anchor is not None:
                self._resident_bytes -= anchor.state_bytes
            return anchor

    def clear(self) -> tuple[SemanticAnchor, ...]:
        with self._lock:
            evicted = tuple(self._entries.values())
            self._entries.clear()
            self._resident_bytes = 0
            self._evictions += len(evicted)
            return evicted

    def snapshot(self) -> SemanticCacheSnapshot:
        with self._lock:
            return SemanticCacheSnapshot(
                capacity_entries=self._capacity_entries,
                capacity_bytes=self._capacity_bytes,
                entry_count=len(self._entries),
                resident_bytes=self._resident_bytes,
                evictions=self._evictions,
            )

    def _evict_to_budget(self) -> list[SemanticAnchor]:
        evicted: list[SemanticAnchor] = []
        while (
            len(self._entries) > self._capacity_entries
            or self._resident_bytes > self._capacity_bytes
        ):
            _, anchor = self._entries.popitem(last=False)
            self._resident_bytes -= anchor.state_bytes
            self._evictions += 1
            evicted.append(anchor)
        return evicted


def semantic_prefix_fingerprint(token_ids: tuple[int, ...]) -> str:
    if (
        not token_ids
        or len(token_ids) > MAX_PREFIX_TOKENS
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < 1 << 64
            for token_id in token_ids
        )
    ):
        raise ValueError("semantic prefix tokens are invalid or unbounded")
    digest = hashlib.sha256(b"vllm-apple-semantic-prefix-v1\0")
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be SHA-256") from error


def _validate_capacity(capacity_entries: int, capacity_bytes: int) -> None:
    if (
        not isinstance(capacity_entries, int)
        or isinstance(capacity_entries, bool)
        or not 1 <= capacity_entries <= MAX_ANCHOR_ENTRIES
        or not isinstance(capacity_bytes, int)
        or isinstance(capacity_bytes, bool)
        or not 1 <= capacity_bytes <= MAX_ANCHOR_STATE_BYTES
    ):
        raise ValueError("semantic cache capacity is invalid")
