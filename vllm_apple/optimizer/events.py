from __future__ import annotations

import math
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
