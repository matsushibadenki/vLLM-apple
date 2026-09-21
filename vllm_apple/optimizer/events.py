from __future__ import annotations

import json
import math
import os
import stat
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

from .types import OPTIMIZER_SCHEMA_VERSION


class OptimizerState(str, Enum):
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class OptimizerEvent:
    event_id: str
    timestamp: str
    plan_id: str
    stage: str
    state: OptimizerState
    progress: float
    message_key: str
    error_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["schema_version"] = OPTIMIZER_SCHEMA_VERSION
        result["state"] = self.state.value
        return result


class OptimizerEventBus:
    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("optimizer event capacity must be positive")
        self.capacity = capacity
        self._sequence = 0
        self._events: deque[OptimizerEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def publish(
        self,
        plan_id: str,
        stage: str,
        state: OptimizerState,
        progress: float,
        message_key: str,
        error_code: str | None = None,
    ) -> OptimizerEvent:
        if not plan_id or not stage or not message_key:
            raise ValueError("optimizer event labels cannot be empty")
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("optimizer progress must be finite and between zero and one")
        with self._lock:
            self._sequence += 1
            event = OptimizerEvent(
                event_id=str(self._sequence),
                timestamp=datetime.now(timezone.utc).isoformat(),
                plan_id=plan_id,
                stage=stage,
                state=state,
                progress=progress,
                message_key=message_key,
                error_code=error_code,
            )
            self._events.append(event)
            return event

    def snapshot(self) -> tuple[OptimizerEvent, ...]:
        with self._lock:
            return tuple(self._events)


class OptimizerEventJournal(OptimizerEventBus):
    """Bounded owner-only JSONL event transport for companion clients."""

    def __init__(self, path: str, capacity: int = 128, maximum_bytes: int = 1024 * 1024) -> None:
        super().__init__(capacity)
        if not 1024 <= maximum_bytes <= 1024 * 1024:
            raise ValueError("event journal limit must be between 1 KiB and 1 MiB")
        candidate = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(candidate)
        parent_info = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) & 0o077
        ):
            raise ValueError("event journal parent must be a private owned directory")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        self._descriptor = os.open(candidate, flags, 0o600)
        self._maximum_bytes = maximum_bytes
        self._written = 0
        self._journal_lock = threading.Lock()

    def publish(
        self,
        plan_id: str,
        stage: str,
        state: OptimizerState,
        progress: float,
        message_key: str,
        error_code: str | None = None,
    ) -> OptimizerEvent:
        event = super().publish(plan_id, stage, state, progress, message_key, error_code)
        encoded = (json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._journal_lock:
            if self._written + len(encoded) > self._maximum_bytes:
                raise ValueError("optimizer event journal exceeded its byte limit")
            view = memoryview(encoded)
            while view:
                written = os.write(self._descriptor, view)
                if written <= 0:
                    raise OSError("optimizer event journal write made no progress")
                view = view[written:]
            os.fsync(self._descriptor)
            self._written += len(encoded)
        return event

    def close(self) -> None:
        with self._journal_lock:
            if self._descriptor >= 0:
                os.close(self._descriptor)
                self._descriptor = -1
