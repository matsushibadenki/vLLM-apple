from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .hardware import detect_power_mode, detect_thermal_state
from .types import PowerMode, ThermalState


@dataclass(frozen=True, slots=True)
class OperatingState:
    thermal_state: ThermalState
    power_mode: PowerMode

    def to_dict(self) -> dict[str, str]:
        return {
            "thermal_state": self.thermal_state.value,
            "power_mode": self.power_mode.value,
        }


def detect_operating_state() -> OperatingState:
    return OperatingState(detect_thermal_state(), detect_power_mode())


class OperatingStateMonitor:
    """Bounded periodic thermal/power probe with duplicate coalescing."""

    def __init__(
        self,
        handler: Callable[[ThermalState, PowerMode], object],
        *,
        probe: Callable[[], OperatingState] = detect_operating_state,
        interval_seconds: float = 15.0,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0 or interval_seconds > 3600:
            raise ValueError("interval_seconds must be between 0 and 3600")
        self._handler = handler
        self._probe = probe
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_state: OperatingState | None = None
        self._polls = 0
        self._notifications = 0
        self._failures = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("operating state monitor was already started")
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="vllm-apple-operating-state-monitor",
            )
            self._thread.start()

    def poll_once(self) -> OperatingState | None:
        with self._poll_lock:
            return self._poll_once_serialized()

    def _poll_once_serialized(self) -> OperatingState | None:
        if self._stop.is_set():
            return None
        failed = False
        try:
            state = self._probe()
            if not isinstance(state, OperatingState) or not (
                isinstance(state.thermal_state, ThermalState)
                and isinstance(state.power_mode, PowerMode)
            ):
                raise ValueError("invalid operating state probe result")
        except Exception:
            failed = True
            state = OperatingState(ThermalState.UNKNOWN, PowerMode.UNKNOWN)
            with self._lock:
                self._failures += 1
        if self._stop.is_set():
            return None
        with self._lock:
            self._polls += 1
            if state == self._last_state:
                return None if failed else state
        try:
            self._handler(state.thermal_state, state.power_mode)
        except Exception:
            with self._lock:
                self._failures += 1
            return None if failed else state
        with self._lock:
            self._last_state = state
            self._notifications += 1
        return None if failed else state

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self._interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(self._interval_seconds + 1.0, 5.0))

    def snapshot(self) -> dict[str, int | float | str | bool | None]:
        with self._lock:
            state = self._last_state
            thread = self._thread
            return {
                "running": thread is not None and thread.is_alive(),
                "interval_seconds": self._interval_seconds,
                "thermal_state": state.thermal_state.value if state else None,
                "power_mode": state.power_mode.value if state else None,
                "polls": self._polls,
                "notifications": self._notifications,
                "failures": self._failures,
            }
