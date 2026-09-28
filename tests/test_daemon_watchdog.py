import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.backend import BackendStartupError
from vllm_apple.daemon import build_backend_watchdog_event_handler
from vllm_apple.service import RuntimeService


class BackendWatchdogBridgeTests(unittest.TestCase):
    def test_restart_failure_persists_diagnostic_without_failing_service(self) -> None:
        class Backend:
            @staticmethod
            def recent_logs() -> tuple[str, ...]:
                return ("private model path",)

        service = RuntimeService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            handler = build_backend_watchdog_event_handler(
                service, Backend(), diagnostic_root=root
            )
            handler(
                "restart_failed",
                {"attempt": 1},
                BackendStartupError("worker exited", code="backend_exited"),
            )
            diagnostics = list(root.glob("*.json"))
            self.assertEqual(len(diagnostics), 1)
            self.assertIsNone(service.snapshot().failure)
            self.assertNotIn("private model path", diagnostics[0].read_text())

    def test_restart_exhaustion_sets_failure_and_persists_diagnostic(self) -> None:
        backend = type("Backend", (), {"recent_logs": lambda self: ()})()
        service = RuntimeService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            handler = build_backend_watchdog_event_handler(
                service, backend, diagnostic_root=root
            )
            handler("restart_exhausted", {"restart_count": 2}, None)
            failure = service.snapshot().failure
            self.assertIsNotNone(failure)
            self.assertEqual(failure["code"], "backend_exited")
            payload = json.loads(next(root.glob("*.json")).read_text())
            self.assertEqual(payload["failure"]["code"], "backend_exited")


if __name__ == "__main__":
    unittest.main()
