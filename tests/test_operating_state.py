import unittest

from vllm_apple.operating_state import OperatingState, OperatingStateMonitor
from vllm_apple.service import RuntimeService
from vllm_apple.types import PowerMode, ThermalState


class OperatingStateMonitorTests(unittest.TestCase):
    def test_poll_coalesces_duplicates_and_updates_runtime_profile(self) -> None:
        states = iter(
            (
                OperatingState(ThermalState.NOMINAL, PowerMode.AUTOMATIC),
                OperatingState(ThermalState.NOMINAL, PowerMode.AUTOMATIC),
                OperatingState(ThermalState.FAIR, PowerMode.LOW_POWER),
            )
        )
        service = RuntimeService()
        before_events = service.events.snapshot()["latest_sequence"]
        monitor = OperatingStateMonitor(
            service.apply_operating_state,
            probe=lambda: next(states),
            interval_seconds=60,
        )

        monitor.poll_once()
        monitor.poll_once()
        monitor.poll_once()

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["polls"], 3)
        self.assertEqual(snapshot["notifications"], 2)
        self.assertEqual(snapshot["failures"], 0)
        self.assertEqual(service.profile.hardware.thermal_state, ThermalState.FAIR)
        self.assertEqual(service.profile.hardware.power_mode, PowerMode.LOW_POWER)
        self.assertGreater(service.events.snapshot()["latest_sequence"], before_events)

    def test_probe_failure_is_bounded_and_recoverable(self) -> None:
        attempts = 0

        def probe() -> OperatingState:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("unavailable")
            return OperatingState(ThermalState.UNKNOWN, PowerMode.UNKNOWN)

        received: list[tuple[ThermalState, PowerMode]] = []
        monitor = OperatingStateMonitor(
            lambda thermal, power: received.append((thermal, power)),
            probe=probe,
        )
        self.assertIsNone(monitor.poll_once())
        self.assertIsNotNone(monitor.poll_once())
        self.assertEqual(monitor.snapshot()["failures"], 1)
        self.assertEqual(received, [(ThermalState.UNKNOWN, PowerMode.UNKNOWN)])

    def test_interval_and_double_start_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OperatingStateMonitor(lambda _thermal, _power: None, interval_seconds=0)
        monitor = OperatingStateMonitor(
            lambda _thermal, _power: None,
            probe=lambda: OperatingState(ThermalState.NOMINAL, PowerMode.AUTOMATIC),
            interval_seconds=60,
        )
        monitor.start()
        try:
            with self.assertRaises(RuntimeError):
                monitor.start()
        finally:
            monitor.stop()
        self.assertFalse(monitor.snapshot()["running"])


if __name__ == "__main__":
    unittest.main()
