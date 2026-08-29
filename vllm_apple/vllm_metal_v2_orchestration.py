from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .scheduler import BasicScheduler
from .vllm_metal_v2_tuning import VLLMMetalV2TuningProfile


@dataclass(frozen=True, slots=True)
class V2IdleTuningSnapshot:
    status: str
    run_id: int
    profile_id: str | None
    error_code: str | None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "error_code": self.error_code,
        }


class NativeV2IdleTuningCoordinator:
    """Runs one bounded tuning job under an exclusive scheduler idle lease."""

    _OWNER = "native_v2_observed_shape_tuning"

    def __init__(
        self,
        scheduler: BasicScheduler,
        *,
        publish: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._publish = publish or (lambda _name, _payload: None)
        self._lock = threading.Lock()
        self._status = "idle"
        self._run_id = 0
        self._profile_id: str | None = None
        self._error_code: str | None = None
        self._thread: threading.Thread | None = None

    def start(
        self,
        tune: Callable[[], VLLMMetalV2TuningProfile],
        apply: Callable[[VLLMMetalV2TuningProfile], None],
    ) -> bool:
        with self._lock:
            if self._status == "running":
                return False
            if not self._scheduler.begin_idle_maintenance(self._OWNER):
                self._status = "waiting_for_idle"
                self._error_code = None
                self._publish("runtime.native_v2_tuning", {"status": self._status})
                return False
            self._run_id += 1
            run_id = self._run_id
            self._status = "running"
            self._profile_id = None
            self._error_code = None
            thread = threading.Thread(
                target=self._run,
                args=(run_id, tune, apply),
                daemon=True,
                name="vllm-apple-native-v2-tuning",
            )
            self._thread = thread
            thread.start()
        self._publish("runtime.native_v2_tuning", {"status": "running", "run_id": run_id})
        return True

    def _run(
        self,
        run_id: int,
        tune: Callable[[], VLLMMetalV2TuningProfile],
        apply: Callable[[VLLMMetalV2TuningProfile], None],
    ) -> None:
        profile: VLLMMetalV2TuningProfile | None = None
        error_code: str | None = None
        try:
            profile = tune()
            apply(profile)
        except Exception as error:
            error_code = type(error).__name__
        finally:
            self._scheduler.end_idle_maintenance(self._OWNER)
        with self._lock:
            self._status = "applied" if error_code is None else "failed"
            self._profile_id = profile.profile_id if profile is not None else None
            self._error_code = error_code
            self._thread = None
            snapshot = self.snapshot_unlocked()
        self._publish("runtime.native_v2_tuning", snapshot.to_dict())

    def snapshot(self) -> V2IdleTuningSnapshot:
        with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> V2IdleTuningSnapshot:
        return V2IdleTuningSnapshot(
            self._status,
            self._run_id,
            self._profile_id,
            self._error_code,
        )

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()
