import unittest

from vllm_apple.service import RuntimeService
from vllm_apple.startup_progress import StartupProgress
from vllm_apple.types import RuntimeState


class StartupProgressTests(unittest.TestCase):
    def test_state_transitions_publish_bounded_structured_progress(self) -> None:
        service = RuntimeService()
        service.set_state(RuntimeState.PROFILING)
        service.set_state(RuntimeState.LOADING_MODEL)
        progress = service.snapshot().startup_progress
        self.assertEqual(progress["stage"], "loading_model")
        self.assertEqual(progress["percent"], 66)
        subscription = service.events.subscribe(after_sequence=1, heartbeat=0.1)
        try:
            events = [next(subscription) for _ in range(4)]
        finally:
            subscription.close()
        progress_events = [event for event in events if event and event.type == "runtime.startup_progress"]
        self.assertEqual(
            [event.payload["stage"] for event in progress_events],
            ["profiling", "loading_model"],
        )

    def test_progress_rejects_invalid_units_and_message_keys(self) -> None:
        for values in (
            ("loading_model", 7, 6, "startup.loading_model"),
            ("unknown", 1, 6, "startup.unknown"),
            ("loading_model", 1, 6, ""),
        ):
            with self.assertRaises(ValueError):
                StartupProgress(1, *values)


if __name__ == "__main__":
    unittest.main()
