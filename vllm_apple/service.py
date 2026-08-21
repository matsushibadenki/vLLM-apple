from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

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


class UnavailableInferenceEngine:
    @property
    def ready(self) -> bool:
        return False

    def models(self) -> list[dict[str, Any]]:
        return []

    def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailableError("no inference backend is loaded")


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    state: RuntimeState
    control_ready: bool
    inference_ready: bool
    profile: RuntimeProfile
    scheduler: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "control_ready": self.control_ready,
            "inference_ready": self.inference_ready,
            "profile": self.profile.to_dict(),
            "scheduler": self.scheduler,
        }


class RuntimeService:
    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self._lock = threading.RLock()
        self._state = RuntimeState.STARTING
        self.profile = build_profile()
        memory = self.profile.hardware.memory
        # Scheduler reservations cover transient work only. Retain the larger of
        # 1 GiB or 8% as an unreservable emergency margin.
        emergency_margin = max(GIB, int(memory.total_bytes * 0.08))
        scheduler_capacity = max(0, memory.available_bytes - emergency_margin)
        self.scheduler = BasicScheduler(self.profile.hardware, scheduler_capacity)
        self.engine = engine or UnavailableInferenceEngine()
        self._state = RuntimeState.READY if self.engine.ready else RuntimeState.DEGRADED

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def set_state(self, state: RuntimeState) -> None:
        with self._lock:
            self._state = state

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                state=self._state,
                control_ready=self._state not in {RuntimeState.FAILED, RuntimeState.STOPPED},
                inference_ready=self.engine.ready,
                profile=self.profile,
                scheduler=self.scheduler.memory.snapshot(),
            )

