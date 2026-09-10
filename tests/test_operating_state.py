import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from vllm_apple.operating_state import OperatingState, OperatingStateMonitor
from vllm_apple.service import RuntimeService
from vllm_apple.types import PowerMode, ThermalState


class OperatingStateMonitorTests(unittest.TestCase):
    def test_service_publication_failure_preserves_snapshot_for_retry(self) -> None:
        service = RuntimeService()
        service.apply_operating_state(ThermalState.NOMINAL, PowerMode.AUTOMATIC)
        before = service.snapshot().profile
        monitor = OperatingStateMonitor(service.apply_operating_state, probe=lambda: OperatingState(
            ThermalState.SERIOUS, PowerMode.LOW_POWER
        ))
        with patch.object(service.events, "publish", side_effect=RuntimeError("unavailable")):
            monitor.poll_once()
        self.assertEqual(service.snapshot().profile, before)
        sequence = service.events.snapshot()["latest_sequence"]
        monitor.poll_once()
        self.assertEqual(service.snapshot().profile.hardware.thermal_state, ThermalState.SERIOUS)
        self.assertEqual(service.events.snapshot()["latest_sequence"], sequence + 1)
        self.assertEqual(monitor.snapshot()["notifications"], 1)

    def test_concurrent_updates_keep_event_transition_chain_consistent(self) -> None:
        service = RuntimeService()
        service.apply_operating_state(ThermalState.NOMINAL, PowerMode.AUTOMATIC)
        sequence = service.events.snapshot()["latest_sequence"]
        states = [ThermalState.FAIR, ThermalState.SERIOUS, ThermalState.NOMINAL] * 20
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda state: service.apply_operating_state(
                state, PowerMode.AUTOMATIC
            ), states))
        count = service.events.snapshot()["latest_sequence"] - sequence
        subscription = service.events.subscribe(after_sequence=sequence)
        previous = "nominal"
        try:
            for _ in range(count):
                event = next(subscription)
                self.assertEqual(event.payload["previous_thermal_state"], previous)
                previous = event.payload["thermal_state"]
        finally:
            subscription.close()
        self.assertGreater(count, 0)
        self.assertEqual(service.snapshot().profile.hardware.thermal_state.value, previous)

    def test_failure_invalidates_stale_state_and_recovery_is_delivered(self) -> None:
        healthy = OperatingState(ThermalState.NOMINAL, PowerMode.AUTOMATIC)
        outcomes = iter([healthy, OSError("probe unavailable"), None, healthy])

        def probe():
            value = next(outcomes)
            if isinstance(value, Exception):
                raise value
            return value

        received = []
        monitor = OperatingStateMonitor(lambda *state: received.append(state), probe=probe)
        for _ in range(4):
            monitor.poll_once()
        self.assertEqual(received, [
            (ThermalState.NOMINAL, PowerMode.AUTOMATIC),
            (ThermalState.UNKNOWN, PowerMode.UNKNOWN),
            (ThermalState.NOMINAL, PowerMode.AUTOMATIC),
        ])
        self.assertEqual(monitor.snapshot()["failures"], 2)

    def test_failed_handler_is_retried_for_unchanged_state(self) -> None:
        attempts = []

        def handler(*state):
            attempts.append(state)
            if len(attempts) == 1:
                raise RuntimeError("temporary sink failure")

        monitor = OperatingStateMonitor(handler, probe=lambda: OperatingState(
            ThermalState.FAIR, PowerMode.LOW_POWER
        ))
        monitor.poll_once()
        monitor.poll_once()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(monitor.snapshot()["notifications"], 1)

    def test_stop_during_probe_discards_result(self) -> None:
        entered, release = threading.Event(), threading.Event()
        received = []

        def probe():
            entered.set()
            release.wait(timeout=3)
            return OperatingState(ThermalState.FAIR, PowerMode.LOW_POWER)

        monitor = OperatingStateMonitor(lambda *state: received.append(state), probe=probe)
        worker = threading.Thread(target=monitor.poll_once)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            monitor.stop()
        finally:
            release.set()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received, [])
        self.assertIsNone(monitor.poll_once())

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
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                OperatingStateMonitor(lambda _thermal, _power: None, interval_seconds=interval)
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
