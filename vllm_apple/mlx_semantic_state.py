from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Callable

from .mlx_server import bounded_cache_nbytes
from .semantic_cache import (
    MAX_ANCHOR_STATE_BYTES,
    MAX_PREFIX_TOKENS,
    SemanticAnchorKind,
)
from .semantic_state import BackendStateReference


MAX_MLX_STATE_ENTRIES = 4_096


@dataclass(slots=True)
class _OwnedMLXState:
    snapshot: object
    state_bytes: int


class MLXPromptCacheStateAdapter:
    """Owns immutable MLX prompt-cache snapshots behind opaque handles."""

    def __init__(
        self,
        capture_snapshot: Callable[[], object],
        restore_snapshot: Callable[[object], bool],
        release_snapshot: Callable[[object], None],
        *,
        capacity_entries: int,
        capacity_bytes: int,
    ) -> None:
        if (
            not 1 <= capacity_entries <= MAX_MLX_STATE_ENTRIES
            or not 1 <= capacity_bytes <= MAX_ANCHOR_STATE_BYTES
        ):
            raise ValueError("MLX state adapter capacity is invalid")
        self._capture_snapshot = capture_snapshot
        self._restore_snapshot = restore_snapshot
        self._release_snapshot = release_snapshot
        self._capacity_entries = capacity_entries
        self._capacity_bytes = capacity_bytes
        self._resident_bytes = 0
        self._states: dict[str, _OwnedMLXState] = {}
        self._lock = threading.RLock()

    def capture_semantic_state(
        self,
        session_fingerprint: str,
        prefix_fingerprint: str,
        token_position: int,
        kind: SemanticAnchorKind,
    ) -> BackendStateReference | None:
        _validate_capture_metadata(
            session_fingerprint, prefix_fingerprint, token_position, kind
        )
        with self._lock:
            if len(self._states) >= self._capacity_entries:
                return None
            snapshot = self._capture_snapshot()
            state_bytes, complete = bounded_cache_nbytes(snapshot)
            if (
                not complete
                or not 1 <= state_bytes <= MAX_ANCHOR_STATE_BYTES
                or self._resident_bytes + state_bytes > self._capacity_bytes
            ):
                self._release_snapshot(snapshot)
                return None
            handle = self._new_handle()
            self._states[handle] = _OwnedMLXState(snapshot, state_bytes)
            self._resident_bytes += state_bytes
            return BackendStateReference(handle, state_bytes)

    def restore_semantic_state(self, handle: str) -> bool:
        _validate_handle(handle)
        with self._lock:
            owned = self._states.get(handle)
            if owned is None:
                return False
            return self._restore_snapshot(owned.snapshot) is True

    def release_semantic_state(self, handle: str) -> None:
        _validate_handle(handle)
        with self._lock:
            owned = self._states.pop(handle, None)
            if owned is None:
                return
            try:
                self._release_snapshot(owned.snapshot)
            except Exception:
                self._states[handle] = owned
                raise
            self._resident_bytes -= owned.state_bytes

    def close(self) -> None:
        with self._lock:
            handles = tuple(self._states)
        for handle in handles:
            self.release_semantic_state(handle)

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "backend": "mlx_lm",
                "capacity_entries": self._capacity_entries,
                "capacity_bytes": self._capacity_bytes,
                "entry_count": len(self._states),
                "resident_bytes": self._resident_bytes,
            }

    def _new_handle(self) -> str:
        for _ in range(8):
            handle = f"mlx-{secrets.token_hex(24)}"
            if handle not in self._states:
                return handle
        raise RuntimeError("could not allocate a unique MLX state handle")


def _validate_capture_metadata(
    session_fingerprint: str,
    prefix_fingerprint: str,
    token_position: int,
    kind: SemanticAnchorKind,
) -> None:
    for name, value in (
        ("session fingerprint", session_fingerprint),
        ("prefix fingerprint", prefix_fingerprint),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{name} must be SHA-256") from error
    if not 1 <= token_position <= MAX_PREFIX_TOKENS:
        raise ValueError("semantic token position is invalid")
    if not isinstance(kind, SemanticAnchorKind):
        raise ValueError("semantic anchor kind is invalid")


def _validate_handle(handle: str) -> None:
    if (
        not isinstance(handle, str)
        or len(handle) != 52
        or not handle.startswith("mlx-")
    ):
        raise ValueError("MLX state handle is invalid")
    try:
        int(handle[4:], 16)
    except ValueError as error:
        raise ValueError("MLX state handle is invalid") from error
