from __future__ import annotations

import threading
from dataclasses import dataclass
from contextlib import AbstractContextManager
from typing import Any, BinaryIO, Protocol

from .events import EventBus
from .profile import build_profile
from .scheduler import BasicScheduler
from .types import GIB, RuntimeProfile, RuntimeState


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceEngine(Protocol):
    @property
    def ready(self) -> bool: ...

    def models(self) -> list[dict[str, Any]]: ...

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def open_chat_stream(
        self, request: dict[str, Any]
    ) -> AbstractContextManager[BinaryIO]: ...


class UnavailableInferenceEngine:
    @property
    def ready(self) -> bool:
        return False

    def models(self) -> list[dict[str, Any]]:
        return []

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailableError("no inference backend is loaded")

    def open_chat_stream(
        self, request: dict[str, Any]
    ) -> AbstractContextManager[BinaryIO]:
        raise InferenceUnavailableError("no inference backend is loaded")


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    state: RuntimeState
    control_ready: bool
    inference_ready: bool
    profile: RuntimeProfile
    scheduler: dict[str, int]
    last_error: str | None
    events: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "control_ready": self.control_ready,
            "inference_ready": self.inference_ready,
            "profile": self.profile.to_dict(),
            "scheduler": self.scheduler,
            "last_error": self.last_error,
            "events": self.events,
        }


class RuntimeService:
    def __init__(
        self,
        engine: InferenceEngine | None = None,
        profile: RuntimeProfile | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state = RuntimeState.STARTING
        self._last_error: str | None = None
        self.events = event_bus or EventBus()
        self.profile = profile or build_profile()
        memory = self.profile.hardware.memory
        # Scheduler reservations cover transient work only. Retain the larger of
        # 1 GiB or 8% as an unreservable emergency margin.
        emergency_margin = max(GIB, int(memory.total_bytes * 0.08))
        scheduler_capacity = max(0, memory.available_bytes - emergency_margin)
        self.scheduler = BasicScheduler(self.profile.hardware, scheduler_capacity)
        self.engine = engine or UnavailableInferenceEngine()
        self._state = RuntimeState.READY if self.engine.ready else RuntimeState.DEGRADED
        self.events.publish(
            "runtime.state",
            {"state": self._state.value, "inference_ready": self.engine.ready},
        )

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def set_state(self, state: RuntimeState) -> None:
        with self._lock:
            self._state = state
            if state != RuntimeState.FAILED:
                self._last_error = None
        self.events.publish(
            "runtime.state",
            {"state": state.value, "inference_ready": self.engine.ready},
        )

    def set_failure(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._state = RuntimeState.FAILED
        self.events.publish("runtime.failure", {"state": RuntimeState.FAILED.value, "message": message})

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                state=self._state,
                control_ready=self._state not in {RuntimeState.FAILED, RuntimeState.STOPPED},
                inference_ready=self.engine.ready,
                profile=self.profile,
                scheduler=self.scheduler.memory.snapshot(),
                last_error=self._last_error,
                events=self.events.snapshot(),
            )
