from __future__ import annotations

import ctypes
import ctypes.util
import platform
import threading
from collections.abc import Callable

from .types import MemoryPressure

DISPATCH_MEMORYPRESSURE_NORMAL = 0x01
DISPATCH_MEMORYPRESSURE_WARN = 0x02
DISPATCH_MEMORYPRESSURE_CRITICAL = 0x04
DISPATCH_MEMORYPRESSURE_ALL = 0x07


def pressure_from_dispatch_data(data: int) -> MemoryPressure:
    if data & DISPATCH_MEMORYPRESSURE_CRITICAL:
        return MemoryPressure.CRITICAL
    if data & DISPATCH_MEMORYPRESSURE_WARN:
        return MemoryPressure.WARNING
    if data & DISPATCH_MEMORYPRESSURE_NORMAL:
        return MemoryPressure.NORMAL
    return MemoryPressure.UNKNOWN


class DarwinMemoryPressureSource:
    """Minimal libdispatch bridge with no polling and one retained callback."""

    def __init__(self) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("memory pressure dispatch source requires macOS")
        library_path = ctypes.util.find_library("System")
        if library_path is None:
            raise RuntimeError("libSystem was not found")
        self._library = ctypes.CDLL(library_path)
        self._configure_functions()
        type_symbol = ctypes.c_byte.in_dll(
            self._library, "_dispatch_source_type_memorypressure"
        )
        source_type = ctypes.c_void_p(ctypes.addressof(type_symbol))
        queue = self._library.dispatch_get_global_queue(0, 0)
        self._source = self._library.dispatch_source_create(
            source_type, 0, DISPATCH_MEMORYPRESSURE_ALL, queue
        )
        if not self._source:
            raise RuntimeError("memory pressure dispatch source creation failed")
        self._callback: object | None = None
        self._started = False

    def _configure_functions(self) -> None:
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self._callback_type = callback_type
        self._library.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]
        self._library.dispatch_get_global_queue.restype = ctypes.c_void_p
        self._library.dispatch_source_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        self._library.dispatch_source_create.restype = ctypes.c_void_p
        self._library.dispatch_source_set_event_handler_f.argtypes = [
            ctypes.c_void_p,
            callback_type,
        ]
        self._library.dispatch_source_get_data.argtypes = [ctypes.c_void_p]
        self._library.dispatch_source_get_data.restype = ctypes.c_ulong
        self._library.dispatch_resume.argtypes = [ctypes.c_void_p]
        self._library.dispatch_source_cancel.argtypes = [ctypes.c_void_p]

    def start(self, handler: Callable[[MemoryPressure], None]) -> None:
        if self._started:
            raise RuntimeError("memory pressure source was already started")

        def receive(_context: ctypes.c_void_p) -> None:
            pressure = pressure_from_dispatch_data(
                int(self._library.dispatch_source_get_data(self._source))
            )
            if pressure != MemoryPressure.UNKNOWN:
                handler(pressure)

        self._callback = self._callback_type(receive)
        self._library.dispatch_source_set_event_handler_f(self._source, self._callback)
        self._library.dispatch_resume(self._source)
        self._started = True

    def stop(self) -> None:
        if self._started:
            self._library.dispatch_source_cancel(self._source)
            self._started = False


class MemoryPressureMonitor:
    def __init__(
        self,
        handler: Callable[[MemoryPressure], object],
        source: DarwinMemoryPressureSource | None = None,
    ) -> None:
        self._handler = handler
        self._source = source or DarwinMemoryPressureSource()
        self._lock = threading.Lock()
        self._last_pressure: MemoryPressure | None = None
        self._notifications = 0

    def start(self) -> None:
        self._source.start(self._receive)

    def _receive(self, pressure: MemoryPressure) -> None:
        with self._lock:
            if pressure == self._last_pressure:
                return
            self._last_pressure = pressure
            self._notifications += 1
        self._handler(pressure)

    def stop(self) -> None:
        self._source.stop()

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "last_pressure": self._last_pressure.value if self._last_pressure else None,
                "notifications": self._notifications,
            }
