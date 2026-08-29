from __future__ import annotations

import threading
import hashlib
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .scheduler import BasicScheduler
from .vllm_metal_v2_tuning import VLLMMetalV2TuningProfile

MAX_MONITORED_OBSERVATION_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class V2IdleTuningSnapshot:
    enabled: bool
    status: str
    run_id: int
    profile_id: str | None
    error_code: str | None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "enabled": self.enabled,
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
        self._enabled = True
        self._run_id = 0
        self._profile_id: str | None = None
        self._error_code: str | None = None
        self._thread: threading.Thread | None = None
        self._last_tune: Callable[[], VLLMMetalV2TuningProfile] | None = None
        self._last_apply: Callable[[VLLMMetalV2TuningProfile], None] | None = None

    def start(
        self,
        tune: Callable[[], VLLMMetalV2TuningProfile],
        apply: Callable[[VLLMMetalV2TuningProfile], None],
    ) -> bool:
        with self._lock:
            self._last_tune = tune
            self._last_apply = apply
            if not self._enabled:
                self._status = "disabled"
                return False
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
            self._status = (
                "disabled"
                if not self._enabled and error_code is None
                else "applied" if error_code is None else "failed"
            )
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
            self._enabled,
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

    def set_enabled(self, enabled: bool) -> V2IdleTuningSnapshot:
        with self._lock:
            self._enabled = enabled
            if self._status != "running":
                self._status = "idle" if enabled else "disabled"
                self._error_code = None
            snapshot = self.snapshot_unlocked()
        self._publish("runtime.native_v2_tuning", snapshot.to_dict())
        return snapshot

    def retry(self) -> bool:
        with self._lock:
            tune = self._last_tune
            apply = self._last_apply
            enabled = self._enabled
        if not enabled or tune is None or apply is None:
            return False
        return self.start(tune, apply)


class NativeV2ObservationMonitor:
    """Bounded polling monitor with content debounce and duplicate suppression."""

    def __init__(
        self,
        path: Path,
        trigger: Callable[[], bool],
        *,
        interval_seconds: float = 5.0,
        debounce_seconds: float = 10.0,
        prime_existing: bool = False,
    ) -> None:
        if interval_seconds <= 0 or debounce_seconds < 0:
            raise ValueError("observation monitor timing is invalid")
        self._path = path
        self._trigger = trigger
        self._interval = interval_seconds
        self._debounce = debounce_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._consumed = self._read_digest() if prime_existing else None

    def poll_once(self, *, now: float | None = None) -> bool:
        digest = self._read_digest()
        if digest is None or digest == self._consumed:
            self._candidate = None
            return False
        current = time.monotonic() if now is None else now
        if digest != self._candidate:
            self._candidate = digest
            self._candidate_since = current
            return False
        if current - self._candidate_since < self._debounce:
            return False
        if not self._trigger():
            return False
        self._consumed = digest
        self._candidate = None
        return True

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("native v2 observation monitor was already started")
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-apple-native-v2-observation-monitor",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self._interval * 2))

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def _read_digest(self) -> str | None:
        try:
            attributes = self._path.lstat()
            if (
                not stat.S_ISREG(attributes.st_mode)
                or attributes.st_uid != os.getuid()
                or attributes.st_mode & 0o077
                or not 1 <= attributes.st_size <= MAX_MONITORED_OBSERVATION_BYTES
            ):
                return None
            payload = self._path.read_bytes()
        except OSError:
            return None
        if len(payload) != attributes.st_size:
            return None
        return hashlib.sha256(payload).hexdigest()
