import unittest

from vllm_apple.memory_pressure import (
    DISPATCH_MEMORYPRESSURE_CRITICAL,
    DISPATCH_MEMORYPRESSURE_NORMAL,
    DISPATCH_MEMORYPRESSURE_WARN,
    MemoryPressureMonitor,
    pressure_from_dispatch_data,
)
from vllm_apple.memory_telemetry import UnifiedMemoryTelemetry
from vllm_apple.service import RuntimeService
from vllm_apple.types import MemoryPressure


class FakePressureSource:
    def __init__(self) -> None:
        self.handler = None
        self.stopped = False

    def start(self, handler) -> None:
        self.handler = handler

    def emit(self, pressure: MemoryPressure) -> None:
        self.handler(pressure)

    def stop(self) -> None:
        self.stopped = True


class MemoryPressureMonitorTests(unittest.TestCase):
    def test_dispatch_flags_prioritize_critical(self) -> None:
        self.assertEqual(
            pressure_from_dispatch_data(DISPATCH_MEMORYPRESSURE_NORMAL), MemoryPressure.NORMAL
        )
        self.assertEqual(
            pressure_from_dispatch_data(DISPATCH_MEMORYPRESSURE_WARN), MemoryPressure.WARNING
        )
        self.assertEqual(
            pressure_from_dispatch_data(
                DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL
            ),
            MemoryPressure.CRITICAL,
        )

    def test_monitor_deduplicates_notifications_and_updates_runtime(self) -> None:
        source = FakePressureSource()
        service = RuntimeService(
            memory_telemetry=UnifiedMemoryTelemetry(
                16 * 1024**3,
                12 * 1024**3,
                "test-fixture",
                MemoryPressure.NORMAL,
            )
        )
        monitor = MemoryPressureMonitor(
            service.apply_memory_pressure, source=source  # type: ignore[arg-type]
        )
        monitor.start()
        source.emit(MemoryPressure.WARNING)
        source.emit(MemoryPressure.WARNING)
        source.emit(MemoryPressure.CRITICAL)
        monitor.stop()
        self.assertTrue(source.stopped)
        self.assertEqual(monitor.snapshot()["notifications"], 2)
        telemetry = service.snapshot().memory_telemetry
        self.assertEqual(telemetry["pressure"], "critical")
        self.assertEqual(telemetry["pressure_notifications"], 2)
        self.assertFalse(service.snapshot().elastic_memory["enabled"])


if __name__ == "__main__":
    unittest.main()
